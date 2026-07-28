"""Reconstruct a network topology graph from one RouteNet-Gauss dataset sample.

The dataset has no explicit adjacency/edge-list field. Topology is reconstructed
from the routing sequences that the model itself consumes (see ``models.py``,
``RouteNetGauss.call``):

- ``link_to_path[flow]``: ordered sequence of link ids traversed by ``flow``.
- ``queue_to_link[link]``: the queue(s) feeding the transmit side of ``link``.
- ``node_groupings_inversed[queue]``: the physical node that owns ``queue``.

For a flow's path ``[l0, l1, ..., lk]``, link ``l_i`` is a directed edge from
``node_of(l_i)`` to ``node_of(l_{i+1})`` — packets sent on ``l_i`` arrive at the
node where they queue for ``l_{i+1}``. The last link of a path only reveals a
destination if some *other* flow later uses it as an interior hop; links that are
never resolved that way (true by construction for links ending at a leaf/end-host)
get a synthetic sink node instead of being silently dropped.

Part of ``visualization/`` — added by Claude Code for data-insight/visualization
purposes only, not part of the model or training. See visualization/README.md.
"""

from collections import Counter

import networkx as nx
import numpy as np

from . import tensor_utils
from .tensor_utils import flatten_floats as _flatten_floats
from .tensor_utils import flatten_ints as _flatten_ints


def get_capacity_array(plain_x: dict) -> list:
    """Mirrors models.py:213-221 exactly, so reported stats match what the model
    actually trains on: prefer the unified ``link_capacity`` field; otherwise
    concatenate router-then-switch capacities, deliberately excluding
    ``link_tg_capacity`` (an existing quirk of the model, not something to fix
    here)."""
    if "link_capacity" in plain_x:
        return _flatten_floats(plain_x["link_capacity"])
    r = _flatten_floats(plain_x.get("link_r_capacity"))
    s = _flatten_floats(plain_x.get("link_s_capacity"))
    return r + s


def get_pkt_header_array(plain_x: dict) -> list:
    if "link_pkt_header_size" in plain_x:
        return _flatten_floats(plain_x["link_pkt_header_size"])
    r = _flatten_floats(plain_x.get("link_r_pkt_header_size"))
    s = _flatten_floats(plain_x.get("link_s_pkt_header_size"))
    return r + s


def node_of_link(link_id: int, queue_to_link: list, node_groupings_inversed: list):
    """Source node of ``link_id``: the node owning the first queue that feeds it.
    Returns ``None`` if the link has no recorded feeding queue."""
    if link_id >= len(queue_to_link):
        return None
    queues = queue_to_link[link_id]
    if not queues:
        return None
    q = int(queues[0])
    if q >= len(node_groupings_inversed):
        return None
    return int(node_groupings_inversed[q])


def classify_tiers(plain_x: dict, num_nodes: int) -> dict:
    """Best-effort node_id -> {'router', 'switch', 'traffic_generator'} using the
    same routers-then-switches(-then-traffic-generators) ordering models.py itself
    relies on for concatenating link_r_capacity/link_s_capacity. This can't be
    verified without executing real TensorFlow, so any inconsistency (counts that
    don't add up to the actual node count) falls back to labeling every node
    'unknown' rather than guessing.

    A dataset can carry `num_traffic_generators`/tg tensors even when that tier is
    empty (e.g. mawi_pcaps), so the tg tier is only counted when it actually has
    members.
    """
    try:
        num_routers = tensor_utils.scalar(plain_x["num_routers"])
        num_switches = tensor_utils.scalar(plain_x["num_switches"])
        num_tg = tensor_utils.scalar(plain_x.get("num_traffic_generators", 0))
    except (KeyError, ValueError, TypeError):
        return {i: "unknown" for i in range(num_nodes)}

    tg_groupings = plain_x.get("tg_groupings")
    if num_tg <= 0 or not tg_groupings:
        num_tg = 0

    if num_routers < 0 or num_switches < 0 or num_tg < 0:
        return {i: "unknown" for i in range(num_nodes)}
    if num_routers + num_switches + num_tg != num_nodes:
        return {i: "unknown" for i in range(num_nodes)}

    tiers = {}
    for i in range(num_nodes):
        if i < num_routers:
            tiers[i] = "router"
        elif i < num_routers + num_switches:
            tiers[i] = "switch"
        else:
            tiers[i] = "traffic_generator"
    return tiers


def buffer_type_distribution(plain_x: dict) -> Counter:
    """Count of queues per buffer_type category (0/1/2)."""
    return Counter(_flatten_ints(plain_x["buffer_type"]))


def _count_parallel_edge_pairs(graph: nx.MultiDiGraph) -> int:
    pair_counts = Counter((u, v) for u, v, _ in graph.edges(keys=True))
    return sum(1 for count in pair_counts.values() if count > 1)


def link_count_matrix(graph: nx.MultiDiGraph) -> tuple:
    """Link counts per (source node, destination node) pair, as a dense matrix.

    A node-link drawing of this data is inherently hard to read -- many distinct
    links collapse onto a few node pairs, so a matrix view is what actually makes
    the distribution legible. Built from the reconstructed graph (rather than
    re-deriving adjacency) so both views always agree.

    Synthetic sink nodes are excluded (they aren't real devices); the diagonal is
    kept, so self-loops show up as diagonal cells.

    Returns
    -------
    tuple
        ``(node_labels, matrix)`` -- ordered real node ids, and ``matrix[i][j]`` =
        number of links from ``node_labels[i]`` to ``node_labels[j]``.
    """
    real_nodes = _ordered_real_nodes(graph)
    index = {n: i for i, n in enumerate(real_nodes)}
    matrix = [[0] * len(real_nodes) for _ in real_nodes]
    for u, v, _key in graph.edges(keys=True):
        if u in index and v in index:
            matrix[index[u]][index[v]] += 1
    return real_nodes, matrix


def _ordered_real_nodes(graph: nx.MultiDiGraph) -> list:
    """Real (non-synthetic) node ids in a stable display order. Shared by both matrix
    builders so the count view and the capacity view can never disagree on ordering."""
    return sorted(
        (n for n in graph.nodes if not graph.nodes[n].get("is_synthetic_sink")),
        key=lambda n: (not isinstance(n, int), n),
    )


def link_capacity_matrix(graph: nx.MultiDiGraph) -> tuple:
    """Per-pair link capacities, as both the individual values and their sum.

    The count matrix answers "how many links", but not "how fat are they" -- this
    carries the actual capacities so the matrix panel can show them per node pair.

    Returns
    -------
    tuple
        ``(node_labels, capacity_lists, totals)`` where ``capacity_lists[i][j]`` is a
        list of every link's capacity (Gbps) from ``node_labels[i]`` to
        ``node_labels[j]``, and ``totals[i][j]`` is their sum (0 for empty cells).
        Links with no recorded capacity are skipped in the lists.
    """
    real_nodes = _ordered_real_nodes(graph)
    index = {n: i for i, n in enumerate(real_nodes)}
    size = len(real_nodes)
    capacity_lists = [[[] for _ in range(size)] for _ in range(size)]
    for u, v, _key, data in graph.edges(keys=True, data=True):
        if u not in index or v not in index:
            continue
        capacity = data.get("capacity_gbps")
        if capacity is None:
            continue
        capacity_lists[index[u]][index[v]].append(capacity)
    totals = [[sum(cell) for cell in row] for row in capacity_lists]
    return real_nodes, capacity_lists, totals


def flow_traffic_totals(plain_x: dict) -> list:
    """Offered traffic per flow, summed over all temporal segments.

    ``flow_traffic`` is shaped (num_flows, seg_num, 1), so this collapses the time
    axis. Units are the dataset's own; dividing by ``capacity * 1e9`` gives the
    dimensionless load the model computes (see ``models.py``'s ``load = ... /
    expanded_capacity``).
    """
    traffic = plain_x.get("flow_traffic")
    if traffic is None:
        return []
    arr = np.asarray(traffic, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        return [float(v) for v in arr]
    return [float(v) for v in arr.reshape(arr.shape[0], -1).sum(axis=1)]


def traffic_matrix(plain_x: dict, graph: nx.MultiDiGraph) -> tuple:
    """Offered traffic aggregated per (flow origin node -> flow last-hop node).

    This is the sample's traffic matrix: how much traffic each source device offers
    toward each destination device. Note the column node is the flow's **last modeled
    hop**, not its ultimate destination -- the true endpoint lies outside the modeled
    network, which is the same reason synthetic sink nodes exist.

    Returns
    -------
    tuple
        ``(node_labels, matrix, unassigned)`` -- ordered real node ids,
        ``matrix[i][j]`` = summed offered traffic, and ``unassigned`` = traffic whose
        endpoints could not be resolved to real nodes (0.0 when everything mapped).
    """
    endpoints = graph.graph.get("flow_endpoints") or []
    totals = flow_traffic_totals(plain_x)
    labels = _ordered_real_nodes(graph)
    index = {n: i for i, n in enumerate(labels)}
    matrix = [[0.0] * len(labels) for _ in labels]

    unassigned = 0.0
    for flow, (src, dst) in enumerate(endpoints):
        if flow >= len(totals):
            break
        if src in index and dst in index:
            matrix[index[src]][index[dst]] += totals[flow]
        else:
            unassigned += totals[flow]
    return labels, matrix, unassigned


def flow_traffic_per_window(plain_x: dict) -> list:
    """Offered traffic per flow **per window**, i.e. ``flow_traffic`` with only the
    trailing singleton axis collapsed. Shape ``(num_flows, seg_num)``."""
    traffic = plain_x.get("flow_traffic")
    if traffic is None:
        return []
    arr = np.asarray(traffic, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        return [[float(v)] for v in arr]
    return arr.reshape(arr.shape[0], arr.shape[1], -1).sum(axis=2).tolist()


def link_load_per_window(plain_x: dict) -> list:
    """Per-link offered load as a fraction of capacity, **per temporal window**.

    This is the quantity ``models.py`` actually computes: it indexes ``flow_traffic`` by
    window and divides by capacity, so load is per-window. Summing traffic across all
    ``seg_num`` windows first (what ``link_load`` below does) inflates the figure by
    roughly ``seg_num`` -- on a real 40-window sample that reported every link at
    1.4-30x capacity, which is why this per-window version exists.

    Returns a list of ``seg_num``-long load lists, one per link.
    """
    path_to_link = plain_x.get("path_to_link")
    capacity = get_capacity_array(plain_x)
    per_window = flow_traffic_per_window(plain_x)
    if path_to_link is None or not capacity or not per_window:
        return []

    num_windows = len(per_window[0])
    loads = []
    for link, entries in enumerate(path_to_link):
        if link >= len(capacity):
            break
        offered = [0.0] * num_windows
        for entry in entries:
            flow = int(entry[0]) if hasattr(entry, "__len__") else int(entry)
            if 0 <= flow < len(per_window):
                for w in range(num_windows):
                    offered[w] += per_window[flow][w]
        denominator = capacity[link] * 1e9
        loads.append(
            [value / denominator for value in offered] if denominator else [0.0] * num_windows
        )
    return loads


def link_load(plain_x: dict) -> list:
    """Per-link offered load as a fraction of capacity.

    Mirrors the quantity ``models.py`` feeds into its link embedding: sum the traffic
    of every flow crossing a link, then divide by that link's capacity in bit/s
    (``capacity * 1e9``). ``path_to_link[link]`` lists the flows on that link with the
    flow index in column 0, exactly as the model gathers it.
    """
    path_to_link = plain_x.get("path_to_link")
    capacity = get_capacity_array(plain_x)
    totals = flow_traffic_totals(plain_x)
    if path_to_link is None or not capacity or not totals:
        return []

    loads = []
    for link, entries in enumerate(path_to_link):
        if link >= len(capacity):
            break
        offered = 0.0
        for entry in entries:
            flow = int(entry[0]) if hasattr(entry, "__len__") else int(entry)
            if 0 <= flow < len(totals):
                offered += totals[flow]
        denominator = capacity[link] * 1e9
        loads.append(offered / denominator if denominator else 0.0)
    return loads


def _truncate_to_flow_length(link_to_path: list, plain_x: dict) -> list:
    """Defensively truncate each flow's path to its true ``flow_length``.

    ``models.py`` never trusts a dense-converted path without re-trimming to the
    true length (e.g. ``tf.RaggedTensor.from_tensor(occupancy_gather,
    lengths=length)`` right after the readout step) — it treats ``flow_length`` as
    the sole source of truth for how many hops a flow really has. If
    ``link_to_path`` ever carries trailing padding beyond that (e.g. because the
    underlying tensor isn't as cleanly ragged as assumed), reading it untruncated
    would silently fabricate self-loops out of repeated padding values and
    overwrite the true trailing hops of a path — exactly where a last-hop access
    switch would typically sit. This is a no-op if the data was already clean.
    """
    flow_length = _flatten_ints(plain_x.get("flow_length"))
    if len(flow_length) != len(link_to_path):
        return [[int(v) for v in row] for row in link_to_path]
    return [
        [int(v) for v in row[:length]]
        for row, length in zip(link_to_path, flow_length)
    ]


def build_topology_graph(plain_x: dict) -> nx.MultiDiGraph:
    """Reconstruct the topology of one already-``to_plain``'d sample dict."""
    link_to_path = _truncate_to_flow_length(plain_x["link_to_path"], plain_x)
    queue_to_link = plain_x["queue_to_link"]
    node_groupings = plain_x["node_groupings"]
    node_groupings_inversed = _flatten_ints(plain_x["node_groupings_inversed"])
    capacity = get_capacity_array(plain_x)
    pkt_header = get_pkt_header_array(plain_x)
    buffer_type = _flatten_ints(plain_x["buffer_type"])
    num_links = len(capacity)
    num_nodes = len(node_groupings)

    graph = nx.MultiDiGraph()

    tiers = classify_tiers(plain_x, num_nodes)
    for node_id in range(num_nodes):
        graph.add_node(
            node_id,
            tier=tiers.get(node_id, "unknown"),
            buffer_types=[],
            is_synthetic_sink=False,
            is_flow_origin=False,
        )

    link_source_node = {
        link_id: node_of_link(link_id, queue_to_link, node_groupings_inversed)
        for link_id in range(num_links)
    }

    def add_link_edge(link_id, u, v, is_synthetic):
        cap = capacity[link_id] if link_id < len(capacity) else None
        hdr = pkt_header[link_id] if link_id < len(pkt_header) else None
        graph.add_edge(
            u,
            v,
            key=link_id,
            link_id=link_id,
            capacity_gbps=cap,
            pkt_header_size=hdr,
            is_synthetic=is_synthetic,
            self_loop=(u == v),
        )

    resolved_links = set()
    flow_origin_nodes = set()
    # Per-flow (origin node, last modeled hop node), recorded here because
    # link_source_node is only available during the build. The second element is the
    # node owning the flow's LAST link, not its true destination -- that lies beyond
    # the modeled network (the same reason synthetic sinks exist).
    flow_endpoints = []
    for flow_links in link_to_path:
        if flow_links:
            origin = link_source_node.get(flow_links[0])
            flow_endpoints.append((origin, link_source_node.get(flow_links[-1])))
            if origin is not None:
                flow_origin_nodes.add(origin)
        else:
            flow_endpoints.append((None, None))
        for i in range(len(flow_links) - 1):
            l_i, l_next = flow_links[i], flow_links[i + 1]
            u = link_source_node.get(l_i)
            v = link_source_node.get(l_next)
            if u is None or v is None:
                continue
            resolved_links.add(l_i)
            add_link_edge(l_i, u, v, is_synthetic=False)

    terminal_links = {int(fl[-1]) for fl in link_to_path if len(fl) > 0}
    synthetic_sinks = 0
    for link_id in terminal_links:
        if link_id in resolved_links:
            continue
        u = link_source_node.get(link_id)
        if u is None:
            continue
        sink_node = f"sink::{link_id}"
        graph.add_node(
            sink_node, tier="sink", buffer_types=[], is_synthetic_sink=True, is_flow_origin=False
        )
        add_link_edge(link_id, u, sink_node, is_synthetic=True)
        resolved_links.add(link_id)
        synthetic_sinks += 1

    for node_id, queues in enumerate(node_groupings):
        types = [buffer_type[q] for q in queues if q < len(buffer_type)]
        graph.nodes[node_id]["buffer_types"] = types

    for node_id in flow_origin_nodes:
        if node_id in graph.nodes:
            graph.nodes[node_id]["is_flow_origin"] = True

    isolated_by_tier = Counter(
        graph.nodes[n].get("tier", "unknown")
        for n in graph.nodes
        if graph.degree(n) == 0
    )

    # Counted from the finished graph, not from path-traversal occurrences: edges are
    # keyed by link_id, so one link revisited by many flows is a single edge. Counting
    # occurrences instead (the earlier approach) overstated this and disagreed with the
    # matrix panel's diagonal.
    self_loop_edges = sum(1 for u, v, _k in graph.edges(keys=True) if u == v)

    # Which nodes absorb the most edges after reconstruction -- a real topology
    # shouldn't have edges overwhelmingly concentrated on 1-2 nodes; a sharp skew
    # here is a signal worth investigating (e.g. a possible indexing issue in
    # node_of_link), not merely a rendering nuisance.
    top_nodes_by_degree = sorted(
        ((n, graph.degree(n)) for n in graph.nodes if not graph.nodes[n].get("is_synthetic_sink")),
        key=lambda item: -item[1],
    )[:5]

    graph.graph["diagnostics"] = {
        "num_links": num_links,
        "resolved_links": len(resolved_links),
        "unresolved_links": num_links - len(resolved_links),
        "synthetic_sinks": synthetic_sinks,
        "self_loops": self_loop_edges,
        "parallel_edge_node_pairs": _count_parallel_edge_pairs(graph),
        "tier_fallback": num_nodes == 0 or all(t == "unknown" for t in tiers.values()),
        "flow_origin_nodes": len(flow_origin_nodes),
        "flow_exit_nodes": synthetic_sinks,
        "isolated_nodes_by_tier": dict(isolated_by_tier),
        "top_nodes_by_degree": top_nodes_by_degree,
    }
    graph.graph["flow_endpoints"] = flow_endpoints
    return graph