"""Single-sample "data insight" entry point: reconstruct the topology of one
dataset sample, draw it, and print/save a text summary.

This is what train.py calls in place of a placeholder comment that used to say
"give me some insight about the data" (see train.py, right after ds_train/ds_val
are built).

Part of ``visualization/`` — added by Claude Code for data-insight/visualization
purposes only, not part of the model or training. See visualization/README.md.
"""

import os
import random
from collections import Counter

import numpy as np

from . import graph_builder, tensor_utils


def pick_sample(ds, index: int = None, unknown_size_guess: int = 256) -> tuple:
    """Pick one ``(x, y)`` element from a dataset, at random by default.

    Parameters
    ----------
    ds
        A ``tf.data.Dataset`` of ``(features, label)`` pairs, as returned by
        ``utils.load_dataset`` (before any ``.map(prepare_targets_and_mask(...))``).
    index : int, optional
        Position to take. ``None`` (the default) picks uniformly at random, so each
        run shows a different topology. Pass an int to reproduce a specific one.

    Returns
    -------
    tuple
        ``(x, y, index)`` -- the chosen element and the position it came from.

    Notes
    -----
    Randomness comes from ``random.SystemRandom`` (os.urandom), NOT the ``random``
    module's global stream, for two reasons: ``train.py`` calls ``random.seed(SEED)``
    for reproducibility, so a seeded draw would return the *same* "random" sample on
    every run; and drawing from that stream would perturb it for anything downstream.
    """
    # Imported lazily on purpose: this package keeps TensorFlow confined to
    # tensor_utils' duck-typed conversions, so the rest of it (and its tests) can be
    # exercised without TensorFlow installed.
    import tensorflow as tf

    rng = random.SystemRandom()
    total = int(tf.data.experimental.cardinality(ds))

    if index is None:
        # cardinality returns negative sentinels for unknown/infinite datasets
        index = rng.randrange(total) if total > 0 else rng.randrange(unknown_size_guess)

    # With an unknown size the guess can overshoot the end; back off instead of failing.
    candidate = index
    while True:
        for element in ds.skip(candidate).take(1):
            return element[0], element[1], candidate
        if candidate == 0:
            raise ValueError("dataset appears to be empty")
        candidate //= 2


def _format_si(value: float) -> str:
    """Compact fixed-width number for table cells (1.2M, 950k, 3.4G)."""
    if value == 0:
        return "."
    for threshold, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= threshold:
            return f"{value / threshold:.3g}{suffix}"
    return f"{value:.3g}"


def _stats(values: list, show_total: bool = False) -> str:
    """Summarise a list of values. ``show_total`` only for genuinely additive
    quantities -- summing packet sizes, delays or loss rates produces a number with
    no physical meaning, so those omit it."""
    if not values:
        return "n/a"
    total = sum(values)
    text = (
        f"min={_format_si(min(values))}, max={_format_si(max(values))}, "
        f"mean={_format_si(total / len(values))}"
    )
    return f"{text}, total={_format_si(total)}" if show_total else text


def _format_matrix_table(labels: list, matrix: list, cell_width: int = 9) -> list:
    """Render a square numeric matrix as aligned monospace text rows."""
    header = " " * 8 + "".join(f"{str(n):>{cell_width}}" for n in labels)
    lines = [f"{'to →':>8}" + header[8:], f"{'from ↓':>8}" + "-" * (cell_width * len(labels))]
    for label, row in zip(labels, matrix):
        cells = "".join(f"{_format_si(v):>{cell_width}}" for v in row)
        lines.append(f"{str(label):>8}" + cells)
    return lines


def _masked_target_values(plain_x: dict, field: str, mask_field: str) -> list:
    """Values of a per-flow/per-segment target, restricted to valid windows.

    Falls back to every value when the mask's shape doesn't line up, since the exact
    layouts can't be verified without a live TensorFlow run.
    """
    raw = plain_x.get(field)
    if raw is None:
        return []
    values = tensor_utils.flatten_floats(raw)
    mask = plain_x.get(mask_field)
    if mask is None:
        return values
    flags = tensor_utils.flatten_ints(mask)
    if len(flags) != len(values):
        return values
    return [v for v, keep in zip(values, flags) if keep]


def window_duration_ms(plain_x: dict) -> tuple:
    """What the data does and does not say about absolute window duration.

    The dataset has no timestamp and no window-duration field. ``flow_packets_per_ms``
    (carried only by ``mawi_pcaps``) looked like a way to recover one, but measuring it
    on real data settles the question negatively:

    * layout is ``(flows, windows, ragged[...], 1)`` -- **flow-major**, matching
      ``flow_packets`` ``(flows, windows, 1)``, with a variable-length series per
      (flow, window) and a trailing singleton axis;
    * that series length varies enormously between (flow, window) pairs (e.g. 100..7434
      in one real sample), so it cannot be "milliseconds per window" if the windows are
      equal length. It behaves like a per-flow variable-length series whose unit is
      undocumented.

    So the honest answer is that ``seg_num`` fixes the *number* of windows and nothing in
    the data fixes their *duration*. Returns ``(verdict, ragged_axis_range)``.

    Two earlier attempts were wrong and are recorded so they are not retried: dividing
    ``flow_packets`` by ``flow_packets_per_ms`` (not element-wise comparable -- different
    rank, and it crashed on the ragged axis), and reading the deepest nested length
    (which measured the trailing singleton axis and produced a bogus "~1 ms").
    """
    per_ms = plain_x.get("flow_packets_per_ms")
    if per_ms is None:
        return ("flow_packets_per_ms not present (only mawi_pcaps carries it); nothing in "
                "this sample constrains the absolute window duration", None)

    structure = tensor_utils.describe_structure(per_ms)
    packets_structure = tensor_utils.describe_structure(plain_x.get("flow_packets"))
    layout = (
        f"observed layouts: flow_packets={packets_structure}, "
        f"flow_packets_per_ms={structure}"
    )

    ragged = next(
        (size for size in tensor_utils.axis_sizes(per_ms) if isinstance(size, tuple)), None
    )
    if ragged is None:
        return (
            "flow_packets_per_ms has no variable-length axis in this sample, so it "
            f"carries no window-length information. {layout}", None
        )

    lo, hi = ragged
    if lo == hi:
        verdict = (
            f"flow_packets_per_ms carries a constant {lo}-entry series per (flow, window). "
            f"That is consistent with (but does not prove) a window of {lo} units of "
            f"whatever the series step is. {layout}"
        )
    else:
        verdict = (
            f"NOT DERIVABLE: flow_packets_per_ms carries a variable-length series per "
            f"(flow, window), {lo}..{hi} entries. Windows are equal in count but this "
            f"series length varies {hi / max(lo, 1):.0f}x between flows, so it is not "
            f"'milliseconds per window'. seg_num fixes how MANY windows there are; no "
            f"field fixes how LONG one is. {layout}"
        )
    return verdict, ragged


# One-line description of what each field carries. Sourced from how models.py uses the
# tensor plus the field naming; see data/mawi_pcaps/README.md for the long form.
FIELD_CONTENT = {
    "link_to_path": "per flow, the ORDERED link ids it traverses (the routing table)",
    "path_to_link": "per link, the flows crossing it (flow index in column 0)",
    "queue_to_link": "per link, the queue(s) feeding its transmit side",
    "node_groupings": "per node, the queues it owns",
    "node_groupings_inversed": "per queue, the node owning it",
    "routers_groupings": "per router, its queues",
    "routers_groupings_inversed": "per queue, its router",
    "switches_groupings": "per switch, its queues",
    "switches_groupings_inversed": "per queue, its switch",
    "path_to_r_link": "per router-link, the flows crossing it",
    "path_to_s_link": "per switch-link, the flows crossing it",
    "r_queue_to_r_link": "per router-link, its feeding queue(s)",
    "s_queue_to_s_link": "per switch-link, its feeding queue(s)",
    "link_capacity": "link capacity in Gbps (models.py multiplies by 1e9 for bit/s)",
    "link_r_capacity": "router-link capacities (Gbps)",
    "link_s_capacity": "switch-link capacities (Gbps)",
    "link_pkt_header_size": "per-packet L1/L2 framing overhead (bytes)",
    "link_r_pkt_header_size": "router-link framing overhead (bytes)",
    "link_s_pkt_header_size": "switch-link framing overhead (bytes)",
    "buffer_type": "per queue, categorical buffer/scheduling class in {0,1,2}",
    "flow_traffic": "offered traffic per flow per window (z-scored inside the model)",
    "flow_packets": "packet rate per flow per window (z-scored inside the model)",
    "flow_packet_size": "mean packet size, used for the transmission-delay term",
    "flow_has_traffic": "mask: did this flow send anything in this window",
    "flow_length": "hop count of the flow's path (the model's authority on length)",
    "flow_packets_per_ms": "mawi-only variable-length per-flow packet-rate series",
    "flow_id": "flow identifier",
    "flow_seg_membership": "which temporal window(s) a flow belongs to",
    "seg_num": "number of temporal windows in this sample",
    "flow_avg_delay": "TARGET: mean delay per flow per window",
    "flow_p50_delay": "TARGET: median delay",
    "flow_p75_delay": "delay 75th percentile (available; unused by train.py)",
    "flow_p90_delay": "TARGET: 90th percentile delay",
    "flow_p95_delay": "TARGET: 95th percentile delay",
    "flow_p99_delay": "TARGET: 99th percentile delay",
    "flow_max_delay": "maximum delay (available; unused by train.py)",
    "flow_avg_jitter": "mean jitter (target when target='jitter')",
    "flow_p50_jitter": "median jitter",
    "flow_p75_jitter": "jitter 75th percentile",
    "flow_p90_jitter": "90th percentile jitter",
    "flow_p95_jitter": "95th percentile jitter",
    "flow_p99_jitter": "99th percentile jitter",
    "flow_max_jitter": "maximum jitter",
    "flow_has_delay": "validity mask for delay labels (needs >=1 packet)",
    "flow_has_jitter": "validity mask for jitter labels (needs >=2 packets)",
    "flow_total_avg_delay": "delay averaged over the whole sample, not per window",
    "loss_rate_per_seg": "packet loss rate per temporal window",
    "total_loss_rate": "sample-level loss figure (units undocumented — see CAUTION above)",
    "flow_total_packets": "total packets offered by the flow",
    "flow_total_trans_pkts": "total packets actually transmitted",
    "flow_trans_pkts_per_seg": "transmitted packets per window",
    "num_routers": "router count", "num_switches": "switch count",
    "num_traffic_generators": "traffic-generator count (0 for mawi_pcaps)",
    "num_r_links": "router-link count", "num_s_links": "switch-link count",
    "num_flows": "flow count", "sample_idx": "unique sample id within the dataset",
}


def _axis_label(size, counts: dict) -> str:
    """Name one axis by matching its size against known entity counts.

    Deliberately reports *every* matching entity when sizes coincide (links and queues
    are both 50 in real samples) instead of picking one. Axis-order assumptions have
    been wrong twice here, so ambiguity is surfaced rather than guessed.
    """
    if isinstance(size, tuple):
        return f"ragged[{size[0]}..{size[1]}] (variable-length series)"
    if size == 1:
        return "1 (trailing singleton)"
    matches = sorted(name for name, count in counts.items() if count == size)
    if matches:
        return f"{size} = {' or '.join(matches)}"
    return f"{size} (no matching entity count)"


def describe_loaded_tensors(plain_x: dict, graph) -> list:
    """Per-field structure, per-dimension meaning, and contents."""
    diagnostics = graph.graph.get("diagnostics", {})
    num_flows = len(tensor_utils.flatten_ints(plain_x.get("flow_length"))) or None
    counts = {
        "flows": num_flows,
        "windows": tensor_utils.scalar(plain_x["seg_num"]) if "seg_num" in plain_x else None,
        "links": diagnostics.get("num_links") or None,
        "queues": len(tensor_utils.flatten_ints(plain_x.get("buffer_type"))) or None,
        "nodes": len(plain_x.get("node_groupings") or []) or None,
    }
    counts = {name: value for name, value in counts.items() if value}

    lines = [
        "entity counts in this sample: "
        + ", ".join(f"{name}={value}" for name, value in sorted(counts.items())),
        "axis sizes are matched against those counts; where two entities share a size "
        "both are listed rather than guessed.",
        "",
    ]
    for name in sorted(plain_x):
        value = plain_x[name]
        structure = tensor_utils.describe_structure(value)
        sizes = tensor_utils.axis_sizes(value)
        used = "model input" if name in _MODEL_INPUTS else "not read by models.py"
        lines.append(f"{name}  {structure}  [{used}]")
        content = FIELD_CONTENT.get(name)
        if content:
            lines.append(f"    contains: {content}")
        if sizes:
            for axis, size in enumerate(sizes):
                lines.append(f"    dim{axis}: {_axis_label(size, counts)}")
        else:
            lines.append("    dim: scalar (no axes)")
    return lines


# Kept in sync with the grep of models.py's inputs["..."] accesses.
_MODEL_INPUTS = {
    "buffer_type", "flow_has_traffic", "flow_length", "flow_packet_size", "flow_packets",
    "flow_traffic", "link_capacity", "link_pkt_header_size", "link_r_capacity",
    "link_r_pkt_header_size", "link_s_capacity", "link_s_pkt_header_size", "link_to_path",
    "node_groupings", "node_groupings_inversed", "path_to_link", "queue_to_link", "seg_num",
}


def _capacity_stats(capacity: list) -> dict:
    if not capacity:
        return {"min_gbps": None, "max_gbps": None, "mean_gbps": None}
    return {
        "min_gbps": min(capacity),
        "max_gbps": max(capacity),
        "mean_gbps": sum(capacity) / len(capacity),
    }


def summarize_sample(plain_x: dict, graph) -> str:
    diagnostics = graph.graph.get("diagnostics", {})
    tier_counts = Counter(data.get("tier", "unknown") for _, data in graph.nodes(data=True))

    capacity = graph_builder.get_capacity_array(plain_x)
    cap_stats = _capacity_stats(capacity)
    buffer_counts = graph_builder.buffer_type_distribution(plain_x)

    flow_length = tensor_utils.flatten_ints(plain_x.get("flow_length"))
    num_flows = len(flow_length) if flow_length else tensor_utils.scalar(
        plain_x.get("num_flows", 0)
    )

    lines = []
    lines.append("=== RouteNet-Gauss data insight: single-sample deep dive ===")
    if "sample_idx" in plain_x:
        lines.append(f"sample_idx: {tensor_utils.scalar(plain_x['sample_idx'])}")

    lines.append("")
    lines.append("-- Topology --")
    lines.append(f"nodes: {sum(tier_counts.values())} " + ", ".join(
        f"{tier}={count}" for tier, count in sorted(tier_counts.items())
    ))
    lines.append(f"links: {diagnostics.get('num_links', 0)}")
    lines.append(f"flows: {num_flows}")
    if flow_length:
        lines.append(
            f"path length (hops): min={min(flow_length)}, "
            f"max={max(flow_length)}, mean={sum(flow_length) / len(flow_length):.2f}"
        )

    lines.append("")
    lines.append("-- Link capacities --")
    if cap_stats["min_gbps"] is not None:
        lines.append(
            f"capacity (Gbps): min={cap_stats['min_gbps']:.3f}, "
            f"max={cap_stats['max_gbps']:.3f}, mean={cap_stats['mean_gbps']:.3f}"
        )
    else:
        lines.append("capacity: no link_capacity/link_r_capacity/link_s_capacity field found")

    lines.append("")
    lines.append("-- Temporal structure & granularity --")
    lines.append(
        "NO TIMESTAMPS: this dataset carries no wall-clock time field of any kind, so "
        "no absolute date/time can be recovered — only relative window structure."
    )
    seg_num = plain_x.get("seg_num")
    if seg_num is not None:
        segments = tensor_utils.scalar(seg_num)
        lines.append(f"temporal windows per sample (seg_num): {segments}")
    else:
        segments = None
        lines.append("seg_num field absent")

    verdict, _ratios = window_duration_ms(plain_x)
    lines.append(f"window duration: {verdict}")

    # Offered traffic per window shows how bursty the sample is over time.
    traffic = plain_x.get("flow_traffic")
    if traffic is not None and segments:
        arr = np.asarray(traffic, dtype=float)
        if arr.ndim >= 2 and arr.shape[1] == segments:
            per_window = arr.reshape(arr.shape[0], segments, -1).sum(axis=(0, 2))
            lines.append(
                "offered traffic per window: "
                + ", ".join(f"w{i}={_format_si(v)}" for i, v in enumerate(per_window))
            )
        else:
            lines.append(
                f"flow_traffic shape {arr.shape} does not expose seg_num={segments} on "
                "axis 1; per-window totals skipped"
            )

    per_seg_loss = tensor_utils.flatten_floats(plain_x.get("loss_rate_per_seg"))
    if per_seg_loss:
        lines.append(f"loss_rate_per_seg: {_stats(per_seg_loss)}")

    lines.append("")
    lines.append("-- Traffic matrix (offered traffic, summed over all time windows) --")
    lines.append(
        "rows = flow origin node, columns = the flow's LAST MODELED HOP (not its true "
        "destination, which lies outside the modeled network). '.' = no flows."
    )
    tm_labels, tm_matrix, tm_unassigned = graph_builder.traffic_matrix(plain_x, graph)
    if tm_labels and any(any(row) for row in tm_matrix):
        lines.extend(_format_matrix_table(tm_labels, tm_matrix))
        row_totals = [sum(row) for row in tm_matrix]
        col_totals = [sum(col) for col in zip(*tm_matrix)]
        busiest_src = max(range(len(row_totals)), key=lambda i: row_totals[i])
        busiest_dst = max(range(len(col_totals)), key=lambda i: col_totals[i])
        lines.append(
            f"largest source: node {tm_labels[busiest_src]} "
            f"({_format_si(row_totals[busiest_src])}); "
            f"largest destination: node {tm_labels[busiest_dst]} "
            f"({_format_si(col_totals[busiest_dst])})"
        )
        if tm_unassigned:
            lines.append(
                f"NOTE: {_format_si(tm_unassigned)} of offered traffic belongs to flows "
                "whose endpoints did not resolve to real nodes and is not in the table."
            )
    else:
        lines.append("no per-flow traffic available (flow_traffic field missing/empty)")

    lines.append("")
    lines.append("-- Offered traffic & load --")
    flow_totals = graph_builder.flow_traffic_totals(plain_x)
    lines.append(f"per-flow offered traffic: {_stats(flow_totals, show_total=True)}")
    packets = tensor_utils.flatten_floats(plain_x.get("flow_packets"))
    if packets:
        lines.append(f"per-flow packet rate (per window): {_stats(packets, show_total=True)}")
    pkt_size = tensor_utils.flatten_floats(plain_x.get("flow_packet_size"))
    if pkt_size:
        lines.append(f"packet size: {_stats(pkt_size)}")

    # Per WINDOW, matching models.py. Summing traffic over all seg_num windows first
    # inflates load by ~seg_num and made every link look oversubscribed.
    per_window_loads = graph_builder.link_load_per_window(plain_x)
    if per_window_loads:
        peaks = [max(link) for link in per_window_loads]
        means = [sum(link) / len(link) for link in per_window_loads]
        hottest = sorted(range(len(peaks)), key=lambda i: -peaks[i])[:5]
        lines.append(
            "per-link load per window (offered traffic / capacity, as models.py computes "
            f"it): mean over links={sum(means) / len(means):.4f}, "
            f"peak over any link+window={max(peaks):.4f}"
        )
        lines.append(
            "highest-peak links: "
            + ", ".join(f"link {i}: {peaks[i]:.4f}" for i in hottest)
        )
        over = [i for i, v in enumerate(peaks) if v > 1.0]
        if over:
            shown = ", ".join(str(i) for i in over[:10])
            suffix = f" (first 10 of {len(over)} shown)" if len(over) > 10 else ""
            lines.append(
                f"NOTE: {len(over)} link(s) peak above 1.0 (more offered traffic than "
                f"capacity in at least one window): {shown}{suffix}"
            )

    loss = plain_x.get("total_loss_rate")
    if loss is not None:
        loss_values = tensor_utils.flatten_floats(loss)
        if loss_values:
            lines.append(f"total_loss_rate (raw field): {_stats(loss_values)}")
            per_seg = tensor_utils.flatten_floats(plain_x.get("loss_rate_per_seg"))
            if per_seg and min(loss_values) > 0.9 and sum(per_seg) / len(per_seg) < 0.1:
                lines.append(
                    "  CAUTION: total_loss_rate sits near 1.0 while loss_rate_per_seg "
                    "averages far below it, so total_loss_rate is NOT a [0,1] loss "
                    "fraction on the same scale — its units are undocumented; treat it "
                    "as an uninterpreted raw value."
                )

    lines.append("")
    lines.append("-- Prediction targets (what the model is trained to output) --")
    for metric in ("delay", "jitter"):
        mask_field = f"flow_has_{metric}"
        reported = False
        for stat in ("avg", "p50", "p90", "p95", "p99"):
            field = f"flow_{stat}_{metric}"
            values = _masked_target_values(plain_x, field, mask_field)
            if values:
                lines.append(f"{field}: {_stats(values)}")
                reported = True
        if not reported:
            lines.append(f"no {metric} targets present in this sample")

    lines.append("")
    lines.append("-- Queue info (buffer_type) --")
    total_queues = sum(buffer_counts.values())
    if total_queues:
        for buffer_type, count in sorted(buffer_counts.items()):
            pct = 100.0 * count / total_queues
            lines.append(f"buffer_type {buffer_type}: {count} queues ({pct:.1f}%)")
    else:
        lines.append("no queues found")

    lines.append("")
    lines.append("-- What the diagram's markers mean --")
    lines.append(
        "TRAFFIC SOURCE (thick magenta ring) marks a device where a flow's traffic is "
        "first observed entering a queue — the actual sending host isn't itself "
        "represented in the data, only the first modeled hop is shown."
    )
    lines.append(
        "'flow destination' (gray node, dashed edge) is a SYNTHETIC placeholder, not a "
        "real device: it marks where a flow's path ends with no further modeled hop, "
        "i.e. where traffic leaves the modeled portion of the network."
    )

    lines.append("")
    lines.append("-- Reconstruction diagnostics --")
    lines.append(f"resolved links: {diagnostics.get('resolved_links', 0)}")
    lines.append(f"unresolved links (never seen in any flow path): {diagnostics.get('unresolved_links', 0)}")
    lines.append(f"synthetic sink nodes (unresolved terminal links): {diagnostics.get('synthetic_sinks', 0)}")
    lines.append(
        f"self-loop links (both ends resolve to one node): {diagnostics.get('self_loops', 0)}"
    )
    lines.append(f"node pairs with parallel links: {diagnostics.get('parallel_edge_node_pairs', 0)}")
    lines.append(f"distinct flow-origin nodes: {diagnostics.get('flow_origin_nodes', 0)}")
    lines.append(f"distinct flow-exit (synthetic sink) nodes: {diagnostics.get('flow_exit_nodes', 0)}")

    isolated = diagnostics.get("isolated_nodes_by_tier") or {}
    total_isolated = sum(isolated.values())
    if total_isolated:
        breakdown = ", ".join(f"{tier}={count}" for tier, count in sorted(isolated.items()))
        lines.append(
            f"NOTE: {total_isolated} node(s) have no edges at all in this sample ({breakdown}) "
            "— these devices own a queue but that queue's link never appears in any flow's "
            "path here. This can mean the link genuinely carries no traffic in this sample, "
            "not necessarily a bug; see visualization/README.md."
        )

    top_nodes = diagnostics.get("top_nodes_by_degree") or []
    if top_nodes:
        breakdown = ", ".join(f"node {n}: {d} edges" for n, d in top_nodes)
        lines.append(f"top nodes by edge count: {breakdown}")
        if (
            len(top_nodes) >= 2
            and top_nodes[0][1] >= 10
            and top_nodes[0][1] >= 3 * max(top_nodes[-1][1], 1)
        ):
            lines.append(
                "NOTE: edges are heavily concentrated on a small number of nodes above. "
                "A prior fix (truncating link_to_path to flow_length) did not change these "
                "counts at all for this dataset, so it is likely NOT padding -- this "
                "concentration may reflect a real property of the data, or an unverified "
                "assumption in node_of_link's queue_to_link indexing (see graph_builder.py); "
                "treat it as an open question rather than a resolved one."
            )

    if diagnostics.get("tier_fallback"):
        lines.append(
            "WARNING: router/switch/traffic-generator node-id partition could not be "
            "confirmed for this sample; topology plot uses a single untiered color/layout "
            "instead of the tiered one."
        )

    lines.append("")
    lines.append("-- Loaded tensors (shape & meaning of each dimension) --")
    lines.extend(describe_loaded_tensors(plain_x, graph))

    return "\n".join(lines)


def run_sample_deep_dive(
    x: dict,
    y=None,
    sample_index: int = 0,
    dataset_name: str = "dataset",
    output_dir: str = "visualization/output",
) -> dict:
    """Reconstruct, plot, and summarize the topology of one raw dataset sample.

    Parameters
    ----------
    x : dict
        Raw sample feature dict, as yielded by ``utils.load_dataset`` (before any
        ``.map(prepare_targets_and_mask(...))``).
    y
        Unused; accepted so callers can pass the full ``(x, y)`` tuple elements
        without unpacking selectively.
    sample_index : int
        Used only to name the output files (e.g. picking element 0 of the dataset).
    dataset_name : str
        Used to namespace the output directory, e.g. the value of ``ds_name`` in
        train.py.
    output_dir : str
        Base directory; files are written to ``{output_dir}/{dataset_name}/``.

    Returns
    -------
    dict
        ``{"graph", "summary_text", "png_path", "txt_path"}``.
    """
    from . import plotting

    plain_x = tensor_utils.sample_to_plain_dict(x)
    graph = graph_builder.build_topology_graph(plain_x)
    summary_text = summarize_sample(plain_x, graph)

    sample_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(sample_dir, exist_ok=True)
    png_path = os.path.join(sample_dir, f"sample_{sample_index}_topology.png")
    txt_path = os.path.join(sample_dir, f"sample_{sample_index}_summary.txt")

    plotting.plot_topology(
        graph, png_path, title=f"{dataset_name} — sample {sample_index} topology"
    )
    with open(txt_path, "w") as fh:
        fh.write(summary_text + "\n")

    print(summary_text)
    print(f"\n[visualization] topology plot saved to {png_path}")
    print(f"[visualization] summary saved to {txt_path}")

    return {
        "graph": graph,
        "summary_text": summary_text,
        "png_path": png_path,
        "txt_path": txt_path,
    }
