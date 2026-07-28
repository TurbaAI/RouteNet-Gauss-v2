"""Render a reconstructed topology graph (see graph_builder.py) to a PNG file.

Part of ``visualization/`` — added by Claude Code for data-insight/visualization
purposes only, not part of the model or training. See visualization/README.md.
"""

import matplotlib

matplotlib.use("Agg")  # headless: train.py never calls plt.show()

import math
from collections import Counter

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from . import graph_builder

TIER_COLOR = {
    "router": "#4C72B0",
    "switch": "#DD8452",
    "traffic_generator": "#55A868",
    "sink": "#B0B0B0",
    "unknown": "#8C8C8C",
}
TIER_LABEL = {
    "router": "router",
    "switch": "switch",
    "traffic_generator": "traffic generator",
    "sink": "flow destination (outside the modeled queues)",
    "unknown": "unknown",
}
# Magenta deliberately sits OUTSIDE the viridis capacity colormap
# (dark purple -> blue -> teal -> green -> yellow). A categorical marker drawn in a
# color the continuous colormap also produces is unreadable: the previous green ring
# (#2E7D32) was nearly the same as a ~10 Gbps edge (#1F968B), so source nodes visually
# competed with green arrows. It is also distinct from routers (#4C72B0) and switches
# (#DD8452). See _assert_ring_color_distinct for the guard that keeps it that way.
FLOW_ORIGIN_RING_COLOR = "#D81B60"
FLOW_ORIGIN_RING_WIDTH = 5.5

# Single source of truth for node marker AREA (matplotlib node_size is points^2, not a
# diameter -- 4x here is 2x the visible width). Arrow shrink is DERIVED from this (see
# _arrow_shrink) -- when these were independent, raising node_size left the hardcoded
# shrink too small and arrowheads were drawn underneath the node circles.
NODE_SIZE = 3400

# Radius of the ring the nodes sit on. Note this value alone does NOT change apparent
# spacing: the axes autoscale to the data extent, so scaling every coordinate just
# rescales pixels-per-unit and every node lands on the same pixel. Visible spacing comes
# from the panel's pixel budget (see _figsize / width_ratios).
CIRCLE_RADIUS = 1.5

# Sinks sit outside the ring so their dashed links point radially outward instead of
# cutting back across the circle. DERIVED from CIRCLE_RADIUS on purpose: as an absolute
# constant it silently fell inside the ring the moment the ring grew past it.
SINK_RADIUS = CIRCLE_RADIUS * 1.45

# Curvature applied to every pair arrow, same value in both directions (see the
# comment at its use site: sign-flipping it makes reciprocal arrows coincide).
# The two arcs of a reciprocal pair end up rad*|chord| apart at their midpoints.
EDGE_ARC_RAD = 0.18

# Font sizes are DERIVED from the figure width, not fixed. Point sizes are absolute, so
# every time the canvas grew (22in -> 26in -> 34in over successive rounds) fixed sizes
# occupied a smaller fraction of it and the text silently shrank. FONT_BASE holds the
# sizes intended at FONT_REFERENCE_WIDTH; _font_sizes() scales them with the actual
# figure so a future size change cannot dilute the text again.
FONT_REFERENCE_WIDTH = 34.0
FONT_BASE = {
    "title": 40,
    "panel_title": 30,
    "caption": 22,
    "node_label": 34,
    "legend": 18,
    "axis_label": 26,
    "tick": 22,
    "edge_label": 20,
    "matrix_cell": 12,
    "annotation": 22,
}


def _font_sizes(fig_width: float = FONT_REFERENCE_WIDTH) -> dict:
    """Font sizes for a figure of the given width, scaled off FONT_BASE."""
    scale = fig_width / FONT_REFERENCE_WIDTH
    return {name: size * scale for name, size in FONT_BASE.items()}


# On-edge text is only legible when there are few edge groups. Past that, the labels
# drift off their curves and stack into an unreadable column -- and the matrix panel
# already gives exact per-pair counts, so suppressing the text loses no information.
MAX_LABELED_GROUPS = 10
MAX_MATRIX_CELL_LABELS = 15
# Kept deliberately short and hard-wrapped: this text is laid out at caption size
# on one line each, and `bbox_inches="tight"` grows the whole figure to fit the widest
# element -- a long caption silently stretched the canvas and shrank the actual plots.
# The legend defines every mark, so the caption only needs to orient the reader.
CAPTION = (
    "LEFT: one arrow per node pair — color = mean link capacity, width ∝ number of links.\n"
    "RIGHT: every link's capacity per node pair; cell shade = the pair's total capacity."
)


def _arrow_shrink(node_size: float = NODE_SIZE, ring_width: float = FLOW_ORIGIN_RING_WIDTH,
                  gap: float = 4.0) -> float:
    """Distance (points) an arrow tip must stop short of a node's center so the
    arrowhead lands *outside* the visible marker.

    Derived from the marker area rather than hardcoded: matplotlib's ``node_size`` is
    an area in points^2, so the radius is ``sqrt(node_size/pi)``. The traffic-source
    ring is stroked centered on the marker edge, so half its linewidth extends beyond
    it. A literal value here silently buried every arrowhead once node_size grew.
    """
    radius = math.sqrt(node_size / math.pi)
    return radius + ring_width / 2 + gap


def _nearest_neighbor_median_distance(pos: dict) -> float:
    """Typical local spacing between nodes, used to size self-loops so they stay
    visually attached to their node regardless of how skewed/elongated the overall
    layout is (a fixed fraction of the *global* bounding box breaks badly on
    elongated layouts — see plotting.py history)."""
    coords = list(pos.values())
    if len(coords) < 2:
        return 0.2
    nearest = []
    for i, (x1, y1) in enumerate(coords):
        best = min(
            math.hypot(x1 - x2, y1 - y2) for j, (x2, y2) in enumerate(coords) if j != i
        )
        nearest.append(best)
    nearest.sort()
    mid = len(nearest) // 2
    if len(nearest) % 2:
        return nearest[mid]
    return (nearest[mid - 1] + nearest[mid]) / 2


def _chord_set(graph: nx.MultiDiGraph, circle_nodes: list) -> list:
    """Undirected node pairs that have at least one link between them.

    Deduped to undirected because the panel draws one arrow per *ordered* pair and the
    two opposite directions follow essentially the same chord across the circle.
    Self-loops are excluded -- they render as rings beside a node, not as chords.
    """
    members = set(circle_nodes)
    pairs = set()
    for u, v in graph.edges():
        if u != v and u in members and v in members:
            pairs.add(frozenset((u, v)))
    return [tuple(pair) for pair in pairs]


def _count_crossings(order: list, chords: list) -> int:
    """Number of chord pairs that cross, for nodes placed in ``order`` on a circle.

    Two chords of a circle cross iff their endpoints interleave. With endpoint indices
    normalised so ``a < b`` and ``c < d``, that is: exactly one of ``c, d`` lies
    strictly inside ``(a, b)``. Chords sharing an endpoint meet at a node rather than
    crossing, so those pairs are skipped -- without that check the interleave test
    would count every adjacent chord pair as a crossing.
    """
    index = {node: i for i, node in enumerate(order)}
    spans = []
    for u, v in chords:
        i, j = index[u], index[v]
        spans.append((min(i, j), max(i, j)))

    crossings = 0
    for p in range(len(spans)):
        a, b = spans[p]
        for q in range(p + 1, len(spans)):
            c, d = spans[q]
            if len({a, b, c, d}) < 4:  # share a node: meeting, not crossing
                continue
            if (a < c < b) != (a < d < b):
                crossings += 1
    return crossings


def _bfs_seed_order(graph: nx.MultiDiGraph, circle_nodes: list) -> list:
    """Deterministic starting order that already puts neighbours near each other.

    Seeded at the highest-degree node (ties broken by id) and expanded breadth-first,
    visiting each frontier's neighbours in descending-degree order. A much better
    starting point for the swap search than the graph's arbitrary iteration order.
    """
    members = set(circle_nodes)
    undirected = nx.Graph()
    undirected.add_nodes_from(circle_nodes)
    for u, v in graph.edges():
        if u != v and u in members and v in members:
            undirected.add_edge(u, v)

    def sort_key(node):
        return (-undirected.degree(node), str(node))

    remaining = sorted(circle_nodes, key=sort_key)
    order, seen = [], set()
    for start in remaining:
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in sorted(undirected.neighbors(node), key=sort_key):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return order


def _minimize_crossings(order: list, chords: list, max_passes: int = 40) -> list:
    """Reduce crossings by accepting any pairwise swap that strictly improves them.

    Deterministic hill-climbing (no randomness), so a given sample always renders
    identically. Cheap at this scale: a handful of nodes and ~10-20 chords.
    """
    best = list(order)
    best_score = _count_crossings(best, chords)
    for _ in range(max_passes):
        improved = False
        for i in range(len(best)):
            for j in range(i + 1, len(best)):
                candidate = list(best)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                score = _count_crossings(candidate, chords)
                if score < best_score:
                    best, best_score = candidate, score
                    improved = True
        if not improved:
            break
    return best


def _compute_layout(graph: nx.MultiDiGraph) -> dict:
    """Place nodes on a circle, ordered so that links cross as little as possible.

    A plain circular layout spreads nodes out but does nothing about crossings -- the
    *order* around the circumference is what determines them, so the order is
    optimised explicitly (see _minimize_crossings).

    Synthetic sink nodes are placed OUTSIDE the ring, at the angle of the real node
    they attach to, so their dashed links point radially outward and cannot cross the
    ring or any chord.
    """
    sinks = [n for n in graph.nodes if graph.nodes[n].get("is_synthetic_sink")]
    circle_nodes = [n for n in graph.nodes if n not in set(sinks)]
    if not circle_nodes:
        return nx.circular_layout(graph) if graph.number_of_nodes() else {}

    chords = _chord_set(graph, circle_nodes)
    order = _minimize_crossings(_bfs_seed_order(graph, circle_nodes), chords)

    pos = {}
    count = len(order)
    angles = {}
    for i, node in enumerate(order):
        angle = 2 * math.pi * i / count
        angles[node] = angle
        pos[node] = (CIRCLE_RADIUS * math.cos(angle), CIRCLE_RADIUS * math.sin(angle))

    # Sinks sit outside the ring, aligned with whichever real node feeds them.
    fanned = {}
    for sink in sinks:
        anchors = [p for p in graph.predecessors(sink) if p in angles]
        if not anchors:
            anchors = [n for n in graph.neighbors(sink) if n in angles]
        base = angles[anchors[0]] if anchors else 0.0
        # several sinks on one anchor would overlap, so fan them apart slightly
        seen = fanned.get(base, 0)
        fanned[base] = seen + 1
        angle = base + (seen - (seen // 2)) * 0.10
        pos[sink] = (SINK_RADIUS * math.cos(angle), SINK_RADIUS * math.sin(angle))
    return pos


def _figsize(num_nodes: int) -> tuple:
    width = min(max(34.0, 2.0 * num_nodes), 58.0)
    height = min(max(16.0, 1.0 * num_nodes), 34.0)
    return width, height


def _draw_isolated_strip(ax, graph, isolated: list, fonts: dict = None) -> None:
    """Park zero-degree nodes in their own captioned band below the main drawing.

    These are real devices whose queue's link never appears in any flow path for
    this sample. Left in the main layout they read as a rendering failure; here
    they're explicitly accounted for.
    """
    fonts = fonts or _font_sizes()
    if not isolated:
        return
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    y_span = y_hi - y_lo
    band_top = y_lo - 0.04 * y_span
    band_bottom = y_lo - 0.20 * y_span
    band_mid = (band_top + band_bottom) / 2

    ax.add_patch(
        plt.Rectangle(
            (x_lo, band_bottom), x_hi - x_lo, band_top - band_bottom,
            facecolor="#F2F2F2", edgecolor="#CCCCCC", linewidth=1.0, zorder=0,
        )
    )
    count_by_tier = {}
    for n in isolated:
        tier = graph.nodes[n].get("tier", "unknown")
        count_by_tier[tier] = count_by_tier.get(tier, 0) + 1
    breakdown = ", ".join(f"{c} {t}" for t, c in sorted(count_by_tier.items()))
    # Sits just ABOVE the band, not inside its top edge -- at the larger annotation
    # font it previously straddled the border and collided with the band outline.
    # Kept short: the full explanation is in the printed summary and README, and a
    # longer string overflowed the axes and was clipped mid-word.
    ax.text(
        (x_lo + x_hi) / 2, band_top + 0.012 * y_span,
        f"not carrying traffic in this sample ({breakdown})",
        ha="center", va="bottom", fontsize=fonts["annotation"], color="#555555",
        style="italic", zorder=4,
    )

    step = (x_hi - x_lo) / (len(isolated) + 1)
    for i, node in enumerate(isolated, start=1):
        x = x_lo + i * step
        color = TIER_COLOR.get(graph.nodes[node].get("tier", "unknown"), TIER_COLOR["unknown"])
        ax.scatter(
            # NODE_SIZE, not a literal: these must stay the same size as the nodes in
            # the circle above, or a NODE_SIZE change silently leaves them mismatched.
            [x], [band_mid - 0.02 * y_span], s=NODE_SIZE, c=[color],
            edgecolors="black", linewidths=0.8, zorder=3,
        )
        text = ax.text(
            x, band_mid - 0.02 * y_span, str(node), ha="center", va="center",
            fontsize=fonts["node_label"], fontweight="bold", color="white", zorder=4,
        )
        text.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="black")])
    ax.set_ylim(band_bottom - 0.02 * y_span, y_hi)


def _format_capacity_cell(caps: list) -> str:
    """Group a cell's link capacities into compact, lossless text.

    A literal list is unusable here: real cells hold 20-25 links. But only a handful
    of *distinct* capacities occur, so ``[1,1,1,10]`` becomes ``"3×1G\\n10G"`` -- every
    value still shown, nothing truncated. The ``×1`` multiplier is dropped for a lone
    link since ``"1×10G"`` reads worse than ``"10G"``.
    """
    if not caps:
        return ""
    counts = Counter(round(c, 1) for c in caps)
    parts = []
    for value in sorted(counts):
        count = counts[value]
        magnitude = _format_gbps(value)
        parts.append(f"{count}×{magnitude}G" if count > 1 else f"{magnitude}G")
    return "\n".join(parts)


def _draw_link_matrix(ax, graph, fonts: dict = None) -> None:
    """Right panel: the capacities of every link between each pair of nodes.

    Cell background = the pair's TOTAL capacity (so the fattest aggregate pipes stand
    out); cell text lists the individual link capacities as grouped counts.
    """
    fonts = fonts or _font_sizes()
    node_labels, capacity_lists, totals = graph_builder.link_capacity_matrix(graph)
    if not node_labels:
        ax.axis("off")
        return

    max_total = max((max(row) for row in totals if row), default=0)
    # aspect="auto" (not "equal"): square cells forced the grid into a narrow band with
    # large empty margins, and cells here must hold 2-3 stacked text lines, so cells
    # wider than tall both fill the panel and suit the text.
    image = ax.imshow(
        totals, cmap="Blues", vmin=0, vmax=max(max_total, 1), aspect="auto",
    )
    ax.set_xticks(range(len(node_labels)))
    ax.set_yticks(range(len(node_labels)))
    ax.set_xticklabels([str(n) for n in node_labels], fontsize=fonts["tick"])
    ax.set_yticklabels([str(n) for n in node_labels], fontsize=fonts["tick"])
    ax.set_xlabel("to node", fontsize=fonts["axis_label"])
    ax.set_ylabel("from node", fontsize=fonts["axis_label"])
    ax.set_title("Link capacities per node pair", fontsize=fonts["panel_title"], pad=34)
    # Spell out the cell notation here, panel-local, rather than in the global CAPTION:
    # the caption is laid out as full-width single lines and bbox_inches="tight" grows
    # the whole figure to fit the widest element, so a long caption line silently
    # stretches the canvas and shrinks both plots.
    ax.text(
        0.5, 1.01, "N×XG = N separate links of X Gbps  ·  shade = pair's total capacity",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=fonts["annotation"] - 2, color="#555555",
    )
    ax.set_xticks([i - 0.5 for i in range(1, len(node_labels))], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, len(node_labels))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    if len(node_labels) <= MAX_MATRIX_CELL_LABELS:
        for i in range(len(node_labels)):
            for j in range(len(node_labels)):
                text = _format_capacity_cell(capacity_lists[i][j])
                if not text:
                    continue
                # contrast-aware: dark text on light cells, white on saturated ones
                color = "white" if totals[i][j] > 0.6 * max(max_total, 1) else "#222222"
                ax.text(
                    j, i, text, ha="center", va="center",
                    fontsize=fonts["matrix_cell"], color=color, fontweight="bold",
                    linespacing=1.15,
                )

    cbar = ax.figure.colorbar(image, ax=ax, shrink=0.6)
    cbar.set_label(
        "total capacity of all links, row-node → col-node (Gbps)", fontsize=fonts["annotation"]
    )
    cbar.ax.tick_params(labelsize=fonts["tick"])


def _edge_width(num_links: int) -> float:
    """Line width for an aggregated edge. Thicker still means more links, but the
    whole scale is deliberately fine so the arrows read as lines rather than bars.
    Shared by the self-loop and pair-edge branches so they cannot drift apart."""
    return min(0.7 + 0.3 * num_links, 4.5)


def _cap_stats(group: list) -> tuple:
    caps = [d["capacity_gbps"] for d in group if d.get("capacity_gbps") is not None]
    if not caps:
        return None, None, None
    return min(caps), max(caps), sum(caps) / len(caps)


def _curve_midpoint(ax, p0: tuple, p1: tuple, rad: float) -> tuple:
    """Midpoint of the quadratic Bezier that matplotlib's ``arc3`` connectionstyle
    actually draws, so a label sits on the curve it names.

    Critically, this is computed in **display** coordinates and converted back:
    ``connectionstyle`` builds its path in display space, so doing the same math in
    data space (the previous approach) misses the drawn curve whenever the x and y
    data-to-pixel scales differ -- which put labels far outside the drawing on long,
    near-vertical edges.
    """
    (dx0, dy0), (dx1, dy1) = ax.transData.transform([p0, p1])
    mid = ((dx0 + dx1) / 2, (dy0 + dy1) / 2)
    vx, vy = dx1 - dx0, dy1 - dy0
    # Must match matplotlib's convention exactly: Arc3 places the control point at
    # mid + rad*(dy, -dx). The negated form used here previously mirrored every label
    # across its chord -- invisible while reciprocal arcs coincided, but it would now
    # park a label against the opposite arrow.
    control = (mid[0] + rad * vy, mid[1] - rad * vx)
    bezier_mid = (0.5 * mid[0] + 0.5 * control[0], 0.5 * mid[1] + 0.5 * control[1])
    return tuple(ax.transData.inverted().transform(bezier_mid))


def _nudge_off_nodes(ax, point: tuple, pos: dict, min_px: float = 22.0) -> tuple:
    """Nudge a label just clear of any node it would land on top of.

    Distances are evaluated in display space so the threshold means the same thing
    regardless of data scaling. Without this, an edge label can completely cover a
    node's number (node "1" was fully hidden in an earlier real render).

    Deliberately a single gentle pass, not an iterative relaxation: repeatedly
    pushing a label out of every nearby node's radius walks it far away from the
    edge it labels, which is worse than a slight overlap.
    """
    inv = ax.transData.inverted()
    px, py = ax.transData.transform(point)
    worst = None
    for node_point in pos.values():
        nx_px, ny_px = ax.transData.transform(node_point)
        dist = math.hypot(px - nx_px, py - ny_px)
        if dist < min_px and (worst is None or dist < worst[0]):
            worst = (dist, nx_px, ny_px)
    if worst is not None:
        dist, nx_px, ny_px = worst
        if dist < 1e-6:
            px += min_px
        else:
            scale = (min_px - dist) + 2.0
            px += (px - nx_px) / dist * scale
            py += (py - ny_px) / dist * scale
    return tuple(inv.transform((px, py)))


def _format_gbps(value: float) -> str:
    return f"{value:.0f}" if value >= 1 or value == 0 else f"{value:.2g}"


def _capacity_legend_entries(drawn_capacities: list, norm, cmap, max_entries: int = 4) -> list:
    """Legend swatches tying the *actual* arrow colors to capacity numbers.

    A continuous colorbar alone does not answer "what's the difference between the
    purple and the green arrows?" -- these entries name the specific colors on the
    canvas. Colors are produced by the same ``cmap(norm(value))`` used to draw the
    edges, so a swatch can never disagree with its edge.
    """
    values = sorted({round(c, 1) for c in drawn_capacities if c is not None})
    if not values:
        return []

    # Group by the *rendered* color, not by number: on a log scale 10G and 11G map to
    # #1F968B and #1E9B89, which are indistinguishable to the eye. Two swatches that
    # look identical defeat the point of the legend, so they collapse into one entry
    # labeled with the range they cover.
    groups = []
    for value in values:
        rgb = cmap(norm(value))[:3]
        quantized = tuple(round(channel, 1) for channel in rgb)
        if groups and groups[-1]["quantized"] == quantized:
            groups[-1]["values"].append(value)
        else:
            groups.append({"quantized": quantized, "values": [value], "color": rgb})

    if len(groups) > max_entries:
        # keep the extremes and spread the rest evenly across the sorted groups
        picks = [0]
        step = (len(groups) - 1) / (max_entries - 1)
        picks += [round(i * step) for i in range(1, max_entries)]
        groups = [groups[i] for i in sorted(set(picks))]

    entries = []
    for group in groups:
        lo, hi = min(group["values"]), max(group["values"])
        if lo == hi:
            label = f"link ≈{_format_gbps(lo)} Gbps"
        else:
            label = f"link ≈{_format_gbps(lo)}–{_format_gbps(hi)} Gbps"
        entries.append(Line2D([0], [0], color=group["color"], lw=4.0, label=label))
    return entries


def _assert_ring_color_distinct(norm, cmap, drawn_capacities: list, min_distance: float = 0.35):
    """Guard against the traffic-source ring color drifting back into the capacity
    colormap's range. Returns a warning string if it collides, else None."""
    ring = matplotlib.colors.to_rgb(FLOW_ORIGIN_RING_COLOR)
    for cap in drawn_capacities:
        if cap is None:
            continue
        edge = cmap(norm(cap))[:3]
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(ring, edge)))
        if distance < min_distance:
            return (
                f"traffic-source ring {FLOW_ORIGIN_RING_COLOR} is too close to the edge "
                f"color for {cap}G (RGB distance {distance:.2f})"
            )
    return None


def _group_label(count: int, cap_mean) -> str:
    """Only aggregated groups (count > 1) get a text label -- a single link's
    capacity is already fully encoded by edge color/width against the colorbar,
    so labeling it too just adds text clutter with nothing new to say.

    Uses the group's *mean* capacity: a min-max range reads as "1-80G" for nearly
    every group in real data (capacities are heavily skewed), so the range carried
    no information while the mean actually differs between groups.
    """
    if count <= 1:
        return ""
    if cap_mean is None:
        return f"×{count} links"
    return f"×{count} links, ~{cap_mean:.0f}G avg"


def plot_topology(graph: nx.MultiDiGraph, out_path: str, title: str = None) -> None:
    num_nodes = graph.number_of_nodes()

    # Isolated nodes get their own captioned strip, so exclude them from the main
    # layout -- otherwise they stretch the drawing area and read as broken.
    isolated = sorted(
        (n for n in graph.nodes if graph.degree(n) == 0),
        key=lambda n: (not isinstance(n, int), n),
    )
    connected = [n for n in graph.nodes if graph.degree(n) > 0]
    layout_graph = graph.subgraph(connected) if connected else graph
    pos = _compute_layout(layout_graph)

    fig_width, fig_height = _figsize(num_nodes)
    fonts = _font_sizes(fig_width)
    fig, (ax, ax_matrix) = plt.subplots(
        1, 2, figsize=(fig_width, fig_height), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )

    node_list = list(pos.keys())
    node_colors = [
        TIER_COLOR.get(graph.nodes[n].get("tier", "unknown"), TIER_COLOR["unknown"])
        for n in node_list
    ]
    node_edgecolors = [
        FLOW_ORIGIN_RING_COLOR if graph.nodes[n].get("is_flow_origin") else "black"
        for n in node_list
    ]
    node_linewidths = [
        FLOW_ORIGIN_RING_WIDTH if graph.nodes[n].get("is_flow_origin") else 0.8
        for n in node_list
    ]
    node_collection = nx.draw_networkx_nodes(
        layout_graph, pos, nodelist=node_list, ax=ax, node_color=node_colors,
        node_size=NODE_SIZE, edgecolors=node_edgecolors, linewidths=node_linewidths,
    )
    # Arrows are drawn after the axes limits settle (they need final transforms for the
    # short-edge guard), so layering is set by explicit zorder rather than draw order.
    node_collection.set_zorder(3)

    all_edges = [
        (u, v, k, d)
        for u, v, k, d in graph.edges(keys=True, data=True)
        if u in pos and v in pos
    ]
    capacities = [d["capacity_gbps"] for *_, d in all_edges if d.get("capacity_gbps") is not None]
    if capacities:
        # Log scale: real capacities are heavily skewed (e.g. min 1G, mean 3.6G,
        # max 80G), so a linear scale crushes almost every link into the bottom few
        # percent of the colormap and every edge renders the same dark color.
        positive = [c for c in capacities if c > 0]
        if positive and max(positive) / min(positive) >= 10:
            norm = LogNorm(vmin=min(positive), vmax=max(positive))
        else:
            norm = Normalize(vmin=min(capacities), vmax=max(capacities))
    else:
        norm = Normalize(0, 1)
    cmap = matplotlib.colormaps["viridis"]

    # Many distinct links resolve onto the same (u, v) pair (or the same node, as a
    # self-loop) once topology is reconstructed from routing data -- drawing one arc
    # per link there produces an unreadable stack of near-identical curves. Instead,
    # aggregate each group into a single edge/loop, colored by mean capacity, labeled
    # with how many links it represents.
    pair_groups = {}
    loop_groups = {}
    for u, v, _key, data in all_edges:
        if u == v:
            loop_groups.setdefault(u, []).append(data)
        else:
            pair_groups.setdefault((u, v), []).append(data)

    loop_radius = max(0.20 * _nearest_neighbor_median_distance(pos), 1e-3)
    # centre of the circular layout, used to push self-loop rings radially outward
    center_x = sum(p[0] for p in pos.values()) / len(pos)
    center_y = sum(p[1] for p in pos.values()) / len(pos)
    total_groups = len(pair_groups) + len(loop_groups)
    show_labels = total_groups <= MAX_LABELED_GROUPS

    # Labels are positioned only after the axes limits are final, because both
    # _curve_midpoint and _nudge_off_nodes work through ax.transData, which changes
    # whenever the limits do (the isolated strip below extends ylim).
    pending_labels = []
    # Mean capacities actually rendered, so the legend can name the exact colors drawn.
    drawn_capacities = []

    self_loop_extents = []
    for node, group in loop_groups.items():
        _cap_min, _cap_max, cap_mean = _cap_stats(group)
        drawn_capacities.append(cap_mean)
        color = cmap(norm(cap_mean)) if cap_mean is not None else "#999999"
        width = _edge_width(len(group))
        style = "dashed" if any(d.get("is_synthetic") for d in group) else "solid"

        cx, cy = pos[node]
        # Push the ring radially OUTWARD from the layout's centre so it sits outside
        # the circle of nodes. A fixed diagonal offset (the previous approach) shoved
        # roughly half of them into the interior, on top of the chords.
        norm_len = math.hypot(cx - center_x, cy - center_y)
        if norm_len > 1e-9:
            ux, uy = (cx - center_x) / norm_len, (cy - center_y) / norm_len
        else:
            ux, uy = 1 / math.sqrt(2), 1 / math.sqrt(2)
        center = (cx + loop_radius * ux, cy + loop_radius * uy)
        ax.add_patch(
            Circle(
                center, loop_radius, fill=False, edgecolor=color, linewidth=width,
                linestyle=style, zorder=1,
            )
        )
        label = _group_label(len(group), cap_mean)
        if show_labels and label:
            pending_labels.append(("loop", (center[0] + loop_radius, center[1]), label))
        # add_patch() does not participate in axis autoscaling, so without this the
        # loop can be silently clipped by the axes boundary near a layout edge.
        self_loop_extents.append((center[0] + loop_radius, center[1] + loop_radius))
        self_loop_extents.append((center[0] - loop_radius, center[1] - loop_radius))

    if self_loop_extents:
        ex, ey = zip(*self_loop_extents)
        ax.scatter(ex, ey, s=0, alpha=0)

    pending_arrows = []
    for (u, v), group in pair_groups.items():
        _cap_min, _cap_max, cap_mean = _cap_stats(group)
        drawn_capacities.append(cap_mean)
        color = cmap(norm(cap_mean)) if cap_mean is not None else "#999999"
        width = _edge_width(len(group))
        style = "dashed" if any(d.get("is_synthetic") for d in group) else "solid"
        # The SAME rad for both directions -- deliberately not sign-flipped by node
        # order. arc3 builds its control point perpendicular to the *travel direction*
        # (mid + rad*(dy, -dx)), so reversing travel already flips which side the curve
        # bows to. Flipping the sign as well cancels that out and makes A->B and B->A
        # trace the identical curve (verified against ConnectionStyle.Arc3.connect:
        # both produced control point (0.5, -0.12)). With one constant rad they bow to
        # opposite sides and read as the usual lens shape for a bidirectional pair.
        rad = EDGE_ARC_RAD
        pending_arrows.append((u, v, color, width, style, rad))
        label = _group_label(len(group), cap_mean)
        if show_labels and label:
            pending_labels.append(("edge", (pos[u], pos[v], rad), label))

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85)
    cbar.set_label("Link capacity (Gbps, log scale)", fontsize=fonts["axis_label"])
    cbar.ax.tick_params(labelsize=fonts["tick"])

    present_tiers = sorted({graph.nodes[n].get("tier", "unknown") for n in graph.nodes})
    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w",
            label=TIER_LABEL.get(tier, tier.replace("_", " ")),
            markerfacecolor=TIER_COLOR.get(tier, TIER_COLOR["unknown"]),
            markersize=12, markeredgecolor="black",
        )
        for tier in present_tiers
    ]
    if any(graph.nodes[n].get("is_flow_origin") for n in graph.nodes):
        legend_elements.append(
            Line2D(
                [0], [0], marker="o", color="w",
                label="TRAFFIC SOURCE — where a flow's traffic enters",
                markerfacecolor="white", markeredgecolor=FLOW_ORIGIN_RING_COLOR,
                markeredgewidth=FLOW_ORIGIN_RING_WIDTH, markersize=13,
            )
        )

    # Capacity swatches intentionally NOT added here: the matrix panel now prints the
    # actual per-pair capacities, and the vertical colorbar still explains arrow color.
    # _capacity_legend_entries is retained (and still regression-tested) in case the
    # discrete swatches are wanted back -- re-add one extend() call to restore them.

    if loop_groups:
        legend_elements.append(
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="none",
                markeredgecolor="#555555", markeredgewidth=2.0, markersize=15,
                label="oval = self-loop (both link ends on same node)",
            )
        )

    if any(d.get("is_synthetic") for *_, d in all_edges):
        # Neutral gray, and the label says color still means capacity: synthetic links
        # are drawn capacity-colored, so a black swatch here would misdescribe them.
        legend_elements.append(
            Line2D(
                [0], [0], color="#888888", lw=2.0, linestyle="dashed",
                label="dashed = synthetic flow-exit link",
            )
        )
    ax.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.03),
        ncol=min(len(legend_elements), 3), fontsize=fonts["legend"], framealpha=0.9,
    )

    ax.set_title("Topology (reconstructed from flow paths)", fontsize=fonts["panel_title"], pad=10)
    ax.axis("off")

    # Isolated nodes extend ylim, so this must happen before any transData-based
    # label placement below.
    _draw_isolated_strip(ax, graph, isolated, fonts)

    # Axis limits are final now -- draw arrows and resolve deferred label positions.
    fig.canvas.draw()

    base_shrink = _arrow_shrink()
    for u, v, color, width, style, rad in pending_arrows:
        # Guard short edges: an arrow whose two endpoints are closer together than the
        # combined shrink would vanish or render inverted, so scale the shrink down to
        # fit rather than dropping the edge.
        (ux, uy), (vx, vy) = ax.transData.transform([pos[u], pos[v]])
        separation = math.hypot(vx - ux, vy - uy)
        shrink = min(base_shrink, max(separation * 0.35, 2.0))
        ax.annotate(
            "",
            xy=pos[v], xycoords="data",
            xytext=pos[u], textcoords="data",
            zorder=1,
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=width, linestyle=style,
                connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink,
                # Heads scale with line width: a thick aggregated edge with a
                # default-size head reads as a blunt stub rather than an arrow.
                mutation_scale=24 + 1.5 * width,
            ),
        )

    for kind, payload, label in pending_labels:
        if kind == "loop":
            point = payload
        else:
            p_u, p_v, rad = payload
            point = _curve_midpoint(ax, p_u, p_v, rad)
        lx, ly = _nudge_off_nodes(ax, point, pos)
        ax.text(
            lx, ly, label, fontsize=fonts["edge_label"], color="black", zorder=5,
            ha="center", va="center",
            bbox=dict(boxstyle="round", fc="white", ec="#BBBBBB", alpha=0.9),
        )

    # Node numbers drawn last so they always sit above edges and edge labels.
    node_labels = {n: str(n) for n in pos if not graph.nodes[n].get("is_synthetic_sink")}
    label_artists = nx.draw_networkx_labels(
        layout_graph, pos, labels=node_labels, ax=ax,
        font_size=fonts["node_label"], font_weight="bold", font_color="white",
    )
    for text in label_artists.values():
        text.set_zorder(6)
        text.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="black")])

    _draw_link_matrix(ax_matrix, graph, fonts)

    collision = _assert_ring_color_distinct(norm, cmap, drawn_capacities)
    if collision:
        print(f"[visualization] WARNING: {collision}")

    # Figure-level (not axes-relative) title/caption: axes-relative placement (the
    # previous approach) scales with the axes bounding box, which varies enough
    # across graph sizes that the two text elements ended up overlapping.
    fig.suptitle(title or "Reconstructed network topology", fontsize=fonts["title"], y=0.985)
    fig.text(0.5, 0.938, CAPTION, ha="center", va="top", fontsize=fonts["caption"], color="#333333")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
