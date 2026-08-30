"""Conversion boundary between TensorFlow tensor types and plain Python/NumPy data.

This is the only module in the ``visualization`` package that needs to know about
TensorFlow's tensor types. Everything downstream (``graph_builder``, ``plotting``,
``report``) works on plain nested lists / NumPy arrays, so it can be reasoned about
(and exercised with synthetic stand-ins) without a TensorFlow runtime.

Part of ``visualization/`` — added by Claude Code for data-insight/visualization
purposes only, not part of the model or training. See visualization/README.md.
"""

import numpy as np


def to_plain(value):
    """Convert a single dataset feature value into plain Python/NumPy data.

    ``tf.RaggedTensor`` is converted with ``.to_list()`` rather than ``.numpy()``:
    ``.numpy()`` on a ragged tensor returns an object-dtype NumPy array whose rows
    are themselves arrays of possibly different lengths, which is easy to misuse
    as if it supported regular vectorized indexing. ``.to_list()`` gives a clean
    nested Python list instead.

    Detection is duck-typed (``to_list`` before ``numpy``) so this function has no
    hard import-time dependency on ``tensorflow`` and works on plain stand-ins
    (lists, NumPy arrays, Python scalars) unchanged.

    PyTorch: works unchanged — ``torch.Tensor`` has ``.numpy()`` and
    ``torch_ragged.Ragged`` has ``.to_list()``; string fields arrive as plain lists.
    """
    if hasattr(value, "to_list"):
        return value.to_list()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def sample_to_plain_dict(x: dict) -> dict:
    """Apply :func:`to_plain` to every value of a raw sample feature dict."""
    return {key: to_plain(value) for key, value in x.items()}


def scalar(value) -> int:
    """Extract a plain Python int from a 0-d/1-element value of any of the shapes
    that show up for scalar dataset fields (Python int, 0-d array, 1-element list
    or array, e.g. after :func:`to_plain`)."""
    arr = np.asarray(value)
    return int(arr.reshape(-1)[0])


def _flatten_recursive(value, dtype, out):
    """Depth-first flatten of arbitrarily nested, possibly ragged sequences."""
    if isinstance(value, np.ndarray):
        for item in value.reshape(-1):
            _flatten_recursive(item, dtype, out)
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten_recursive(item, dtype, out)
        return out
    try:
        out.append(dtype(value))
    except (TypeError, ValueError):
        pass  # non-numeric leaf (e.g. bytes): nothing useful to contribute
    return out


def flatten_dense(value, dtype):
    """Flatten a per-entity field into a plain flat list of ``dtype``.

    Fast path is ``np.asarray`` for genuinely dense fields (e.g. shape ``(n, 1)``).
    Ragged fields ARE supported: several fields in these datasets are inhomogeneous
    (``queue_to_link``, ``node_groupings``, ``link_to_path``, and notably
    ``flow_packets_per_ms``, which is ``(windows, flows, variable)``). NumPy raises
    ``ValueError`` on such input, so this falls back to a recursive walk rather than
    propagating — the earlier "dense only" contract crashed real callers.
    """
    if value is None:
        return []
    try:
        arr = np.asarray(value, dtype=dtype)
    except (ValueError, TypeError):
        return _flatten_recursive(value, dtype, [])
    if arr.dtype == object:
        return _flatten_recursive(value, dtype, [])
    if arr.size == 0:
        return []
    return [dtype(v) for v in arr.reshape(-1)]


def describe_structure(value, _depth: int = 0) -> str:
    """Compact description of a nested field's shape, marking ragged axes.

    e.g. ``"(40, 20, 1)"`` for a dense field vs ``"(20, 40, ragged[18..22])"``. These
    datasets mix flow-major and segment-major layouts and document neither, so the
    summary shows the observed structure instead of assuming one.
    """
    if value is None:
        return "absent"
    if isinstance(value, np.ndarray) and value.dtype != object:
        return "(" + ", ".join(str(d) for d in value.shape) + ")" if value.shape else "scalar"
    if not isinstance(value, (list, tuple, np.ndarray)):
        return "scalar"

    return "(" + ", ".join(_axis_descriptions(value)) + ")"


def axis_sizes(value) -> list:
    """Per-axis size of a nested value: an int for uniform axes, ``(lo, hi)`` for ragged.

    Used to label what each dimension means by matching sizes against known entity
    counts, rather than assuming an axis order (assuming one has been wrong twice).
    """
    sizes = []
    current = value
    while True:
        if isinstance(current, np.ndarray) and current.dtype != object:
            sizes.extend(int(d) for d in current.shape)
            return sizes
        if not isinstance(current, (list, tuple, np.ndarray)):
            return sizes
        items = list(current)
        if not items:
            sizes.append(0)
            return sizes
        children = [
            len(item) if isinstance(item, (list, tuple, np.ndarray)) else None
            for item in items
        ]
        sizes.append(len(items))
        if all(length is None for length in children):
            return sizes
        lengths = [length for length in children if length is not None]
        lo, hi = min(lengths), max(lengths)
        if lo != hi or len(lengths) != len(items):
            sizes.append((lo, hi))
            # Descend through the longest child so axes *below* a ragged one are still
            # reported -- truncating here hid a trailing singleton axis and caused a
            # ragged series length to be mis-measured as 1.
            current = max(
                (item for item in items if isinstance(item, (list, tuple, np.ndarray))),
                key=len,
            )
            deeper = axis_sizes(current)
            sizes.extend(deeper[1:] if deeper else [])
            return sizes
        current = items[0]


def _axis_descriptions(value) -> list:
    out = []
    for size in axis_sizes(value):
        out.append(f"ragged[{size[0]}..{size[1]}]" if isinstance(size, tuple) else str(size))
    return out


def flatten_ints(value):
    return flatten_dense(value, int)


def flatten_floats(value):
    return flatten_dense(value, float)
