"""Dump every field of a few dataset samples, and answer the granularity question.

Unlike the rest of ``visualization/``, this module imports TensorFlow at the top: it is
a command-line entry point, not library code, so it does not need to stay importable in
a TF-free environment.

Run from the repository root (data paths are relative, same as ``train.py``)::

    python -m visualization.describe_dataset
    python -m visualization.describe_dataset --dataset trex_synthetic --partition test
    python -m visualization.describe_dataset --update-readme

``--update-readme`` rewrites the generated block in ``data/<dataset>/README.md`` so the
documentation carries real values instead of placeholders.

Part of ``visualization/`` — added by Claude Code for data-insight/visualization
purposes only, not part of the model or training. See visualization/README.md.
"""

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

from utils import load_dataset

from . import graph_builder, report, tensor_utils

BEGIN_MARKER = "<!-- BEGIN GENERATED: SAMPLES -->"
END_MARKER = "<!-- END GENERATED: SAMPLES -->"

# Fields models.py actually reads (grep 'inputs["..."]' in models.py). Everything else
# is carried by the dataset but unused by the model.
MODEL_INPUTS = {
    "buffer_type", "flow_has_traffic", "flow_length", "flow_packet_size", "flow_packets",
    "flow_traffic", "link_capacity", "link_pkt_header_size", "link_r_capacity",
    "link_r_pkt_header_size", "link_s_capacity", "link_s_pkt_header_size", "link_to_path",
    "node_groupings", "node_groupings_inversed", "path_to_link", "queue_to_link", "seg_num",
}


def _describe_raw(tensor) -> str:
    """Shape/dtype/kind of a field, read from the RAW tensor.

    Must happen before tensor_utils.to_plain(), which discards dtype and flattens
    ragged structure into nested lists.
    """
    kind = "ragged" if hasattr(tensor, "to_list") else "dense"
    shape = getattr(tensor, "shape", None)
    dtype = getattr(tensor, "dtype", None)
    dtype_name = getattr(dtype, "name", str(dtype)) if dtype is not None else "?"
    return f"{kind:<7} shape={str(shape):<22} dtype={dtype_name}"


def _summarize_values(plain) -> str:
    """Compact value summary: numeric range, or the first few index entries."""
    try:
        flat = tensor_utils.flatten_floats(plain)
    except (TypeError, ValueError):
        flat = []
    if flat:
        lo, hi = min(flat), max(flat)
        mean = sum(flat) / len(flat)
        return f"n={len(flat):<6} min={lo:<12.6g} max={hi:<12.6g} mean={mean:.6g}"
    # ragged index tensors: show structure instead
    if isinstance(plain, list):
        preview = str(plain[:2])
        if len(preview) > 70:
            preview = preview[:67] + "..."
        return f"rows={len(plain):<6} first={preview}"
    return f"value={plain}"


def describe_sample(x, index: int) -> list:
    """Human-readable description of every field in one sample."""
    plain = tensor_utils.sample_to_plain_dict(x)
    graph = graph_builder.build_topology_graph(plain)

    lines = [f"### Sample at position {index}", ""]

    sample_idx = plain.get("sample_idx")
    if sample_idx is not None:
        lines.append(f"`sample_idx` = {tensor_utils.scalar(sample_idx)}")
        lines.append("")

    verdict, ratios = report.window_duration_ms(plain)
    lines.append(f"**Temporal granularity:** {verdict}")
    if ratios:
        lines.append(
            f"(ratio over {len(ratios)} entries: min={min(ratios):.6g}, max={max(ratios):.6g})"
        )
    lines.append("")

    lines.append("| field | used by models.py | kind / shape / dtype | values |")
    lines.append("|---|---|---|---|")
    for name in sorted(x.keys()):
        used = "yes" if name in MODEL_INPUTS else "—"
        raw_desc = _describe_raw(x[name])
        values = _summarize_values(plain.get(name))
        lines.append(f"| `{name}` | {used} | `{raw_desc}` | `{values}` |")
    lines.append("")

    lines.append("Derived view of this sample (same code the plot/summary uses):")
    lines.append("")
    lines.append("```")
    lines.extend(report.summarize_sample(plain, graph).splitlines())
    lines.append("```")
    lines.append("")
    return lines


def _update_readme(path: str, body: str) -> bool:
    if not os.path.exists(path):
        print(f"[describe_dataset] {path} does not exist; skipping README update")
        return False
    with open(path) as fh:
        text = fh.read()
    if BEGIN_MARKER not in text or END_MARKER not in text:
        print(f"[describe_dataset] markers not found in {path}; skipping README update")
        return False
    head, rest = text.split(BEGIN_MARKER, 1)
    _stale, tail = rest.split(END_MARKER, 1)
    with open(path, "w") as fh:
        fh.write(f"{head}{BEGIN_MARKER}\n\n{body}\n{END_MARKER}{tail}")
    print(f"[describe_dataset] updated generated block in {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mawi_pcaps")
    parser.add_argument("--partition", default="test")
    parser.add_argument(
        "--samples", type=int, nargs="+", default=[0, 1, 2],
        help="dataset positions to describe (fixed by default so docs are reproducible)",
    )
    parser.add_argument(
        "--random", action="store_true",
        help="pick positions at random instead of using --samples",
    )
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(f"{args.dataset}/{args.partition}")

    sections = [
        f"_Generated by `python -m visualization.describe_dataset --dataset {args.dataset} "
        f"--partition {args.partition}`._",
        "",
    ]
    for wanted in args.samples:
        x, _y, actual = report.pick_sample(ds, index=None if args.random else wanted)
        sections.extend(describe_sample(x, actual))

    body = "\n".join(sections)
    print(body)

    if args.update_readme:
        _update_readme(os.path.join("data", args.dataset, "README.md"), body)


if __name__ == "__main__":
    main()
