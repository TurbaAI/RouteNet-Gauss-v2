# `visualization/`

> **Note:** This entire package was added by Claude Code (an AI coding assistant), at the
> user's request, purely as a data-insight/visualization aid — it is **not** part of the
> RouteNet-Gauss paper, model, or training logic. It only reads dataset samples; it never
> touches the model, the loss, or `model.fit(...)`. `train.py` calls it in exactly one place
> (see the clearly marked "Added by Claude" block, right after `ds_train`/`ds_val` are built)
> and that call can be deleted without affecting training in any way.

## What it does

Given one raw sample from a RouteNet-Gauss dataset (as returned by `utils.load_dataset`, before
`.map(prepare_targets_and_mask(...))`), this package:

1. Reconstructs the network topology. The dataset has no explicit adjacency/edge-list field —
   topology is inferred from the same routing tensors the model itself consumes
   (`link_to_path`, `queue_to_link`, `node_groupings_inversed`; see `graph_builder.py`'s module
   docstring for the exact algorithm).
2. Renders it to a PNG: nodes colored/laid out by tier (router / switch / traffic-generator,
   when that can be determined), edges colored by link capacity with a colorbar, dashed edges
   for links whose destination couldn't be resolved from routing data (see "Diagnostics" below).
3. Writes a text summary: node/link/flow counts, link capacity stats (min/max/mean, Gbps), the
   `buffer_type` (queue discipline) distribution, path-length stats, and:
   - a **traffic matrix** — offered traffic per (flow origin node → flow's last modeled hop),
     summed over all time windows, plus the largest source/destination;
   - **offered traffic & load** — per-flow traffic, packet rate and packet size stats, per-link
     load (`offered traffic / capacity`, the same quantity `models.py` feeds its link embedding),
     the most loaded links, and any link whose offered load exceeds 1.0;
   - **prediction targets** — the delay/jitter avg and p50–p99 fields the model is trained to
     output, restricted to valid windows via the `flow_has_*` mask, plus `seg_num`.

Both files are written to `visualization/output/<dataset_name>/sample_<n>_{topology.png,summary.txt}`
(gitignored — generated, not source) and the summary is also printed to stdout.

## Files

| File | Responsibility |
|---|---|
| `tensor_utils.py` | The only module that knows about TensorFlow tensor types (`tf.Tensor`/`tf.RaggedTensor` → plain Python/NumPy). Everything else works on plain data. |
| `graph_builder.py` | Reconstructs the topology as an `networkx.MultiDiGraph` from a raw sample dict; classifies nodes into router/switch/traffic-generator tiers on a best-effort basis. |
| `plotting.py` | Renders a `MultiDiGraph` to a PNG (matplotlib, headless `Agg` backend — never calls `plt.show()`). |
| `report.py` | Text summary + `run_sample_deep_dive(...)`, the one public entry point `train.py` calls. |

## Reading the diagram

The PNG has **two panels**. A node-link drawing alone is a poor fit for this data — many
distinct links collapse onto a few node pairs — so the matrix panel is where you read the
actual numbers, and the graph panel is for the overall shape.

- **Left panel (topology):** nodes sit on a **circle**, and their order around it is chosen to
  minimise how often links cross (a plain circular layout spreads nodes out but does nothing about
  crossings — the circumferential *order* is what determines them, so `_minimize_crossings` searches
  for a better one). Flow destinations (sinks) are placed **outside** the ring at the angle of the
  node feeding them, so their dashed links point radially outward rather than cutting back across
  the circle. One arrow per *node pair*, no matter how many links it carries.
  Arrow color = that pair's mean link capacity on a **log** scale (capacities are heavily skewed —
  e.g. min 1G, mean 3.6G, max 80G — so a linear scale renders almost every arrow the same color).
  Arrow width ∝ number of links. The vertical colorbar is the legend for that color.
- **Right panel (link capacities per node pair):** each cell lists the capacity of *every*
  individual link from that row-node to that col-node, as grouped counts — `22×1G` / `3×10G`
  means 25 links, 22 of them 1 Gbps and 3 of them 10 Gbps. This is lossless: no value is hidden.
  A literal list is not used because real cells hold 20-25 links, but only a handful of *distinct*
  capacities occur, so grouping stays compact. The **cell shade** is the pair's **total** capacity
  (sum over its links), so the fattest aggregate pipes stand out. The diagonal holds self-loops.
  This panel is also the reliable way to see how links are distributed — it's what makes a
  lopsided distribution (a couple of nodes absorbing most links) obvious at a glance.
- **Router / switch / traffic-generator nodes** (blue / orange / green) are real physical
  devices from the dataset — each owns at least one queue.
- **A thick magenta outline on a node** marks a **traffic source**: a flow's traffic is *first
  observed* entering that device's queue. The actual sending host is **not** itself represented in
  the data anywhere — only the first modeled hop is shown. This is the honest answer to "how does
  traffic get injected, is it from outside?": yes, conceptually, from a point the dataset doesn't
  model at all. Magenta is chosen deliberately because it sits **outside** the viridis capacity
  colormap (dark purple → blue → teal → green → yellow); an earlier green ring was nearly
  identical to a ~10 Gbps arrow (`#1F968B`), which made source nodes hard to pick out.
  `plotting._assert_ring_color_distinct` guards against that collision returning.
- **A large oval next to a node** is a **self-loop**: links whose two ends both resolve to that
  same node. These are the matrix's diagonal cells.
- **A gray node with a dashed edge** ("flow destination") is a **synthetic placeholder**, not
  a real device — it's added by this code, not present in the dataset. It marks a link that is
  the last hop of some flow's path and never resolves to a real next node from any other flow's
  routing data either — i.e. where the packet leaves the modeled portion of the network.
- **Nodes in the gray strip at the bottom** have no edges at all: each owns a queue, but that
  queue's link never appears in *any* flow's path in this sample. They're parked there (rather
  than floating unconnected in the main drawing, where they read as a rendering failure) and the
  strip's caption states how many there are. This can simply mean those links carry no traffic in
  this sample — not every physical link must carry traffic in every snapshot. The printed summary
  reports the same thing under "Reconstruction diagnostics". If most/all nodes of one tier are
  isolated across many samples, that's worth investigating further (see "Known limitations").
- **Self-loops** (a small ring drawn just outside a node) mean two *consecutive* hops in some
  flow's path resolved to the same physical node. This can be a legitimate artifact of how the
  path is segmented in the data — it does not necessarily indicate an error.
- **On-edge `"×N links, ~XG avg"` text** appears only when the graph has few enough edge groups
  (≤ 10) for the text to stay legible. Beyond that it's suppressed on purpose: the labels drift
  off their curves and stack into an unreadable column, and the matrix panel already gives the
  exact counts, so nothing is lost.

## Known limitations

- The dataset's router/switch/traffic-generator node-id partition is inferred by analogy with
  how `models.py` concatenates `link_r_capacity`/`link_s_capacity` (routers-then-switches). This
  assumption is checked for internal consistency (expected node count vs. actual) and falls back
  to an untiered plot rather than mis-labeling nodes if it doesn't hold — check the printed
  summary's "Reconstruction diagnostics" section (`tier_fallback` warning) to see whether it held
  for a given sample.
- **Open question, not yet resolved:** on the one real sample inspected so far (`mawi_pcaps`
  sample 0), edges after reconstruction are heavily concentrated on 2-3 router nodes, and all 3
  switch nodes are fully isolated (own a queue, but that queue's link never appears in any of
  the 40 flows' paths). A first hypothesis — that `link_to_path` carried unread trailing padding
  beyond each flow's true `flow_length` — was tested and **disproven**: truncating to
  `flow_length` in `graph_builder.py` changed none of these diagnostics at all for that sample.
  So this concentration/isolation is either a genuine property of that specific sample, or a
  still-unverified assumption elsewhere in `node_of_link` (e.g. the direction/semantics of
  `queue_to_link` indexing, never confirmed against live tensor values in this environment). The
  summary's "top nodes by edge count" line is meant to make this pattern visible and quantified
  across future samples/datasets rather than leaving it as a one-off visual observation.
