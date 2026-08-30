"""
Copyright 2025 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# One-time, lossless conversion of the TensorFlow datasets under data/ into TF-free PyTorch
# files under data_torch/. This is the ONLY place where the PyTorch pipeline touches
# TensorFlow: the shipped datasets are `tf.data.Dataset.save` snapshots that can only be read
# with TF. After conversion, training/evaluation in PyTorch never imports TensorFlow.
#
# Layout mirrors the TF shards one-to-one so `utils.load_dataset` stays a literal translation:
#
#   data/<dataset>/<partition>/<k>/             (TF snapshot shard k)
#   data_torch/<dataset>/<partition>/<k>.pt.gz  (same scenarios, same order)
#   data_torch/<dataset>/<partition>/manifest.json
#
# Each shard is a gzip-compressed `torch.save` stream (gzip halves the footprint; torch.save
# itself does not compress). Load it with
#   torch.load(io.BytesIO(gzip.open(path, "rb").read()), weights_only=True)
# (`utils.load_dataset` does this). The object is
#   {"format_version": 1, "dataset", "partition", "shard", "source", "dropped_fields",
#    "samples": [ {"x": {field: value, ...}, "y": Tensor}, ... ] }
# where a field value is
#   - a torch.Tensor with the TF dtype preserved (float32/int32/bool, ...), or
#   - {"__ragged__": True, "values": Tensor, "row_splits": [Tensor(int64), ...]}
#     for tf.RaggedTensor (outermost row_splits first; ragged_rank == len(row_splits)), or
#   - {"__strings__": True, "data": [str, ...], "shape": [...]} for tf.string tensors.
#
# By default `flow_packets_per_ms` is dropped: it is a mawi-only, variable-length per-flow
# packet series that nothing in the model or the evaluation reads, and it is ~90% of the bytes
# (keeping it would make the mawi files ~3.8 GB instead of ~200 MB). Pass --keep-all-fields to
# keep it. Every written shard is re-read and compared field-by-field, bit-for-bit, against a
# fresh pass over the TF source (--verify none disables that).
#
# Run from the repo root in an env that has BOTH tensorflow(-cpu) and torch (conda env RG_torch):
#   python convert_data_to_torch.py                     # all datasets, all partitions
#   python convert_data_to_torch.py --datasets mawi_pcaps,trex_multiburst

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import torch

FORMAT_VERSION = 1
DEFAULT_DROP = ["flow_packets_per_ms"]
GZIP_LEVEL = 6  # -1: 1.30s/11.3MB, -6: 2.05s/10.8MB, -9: 3.49s/10.8MB on a 19.7MB shard
GITHUB_WARN_BYTES = 50 * 1024 * 1024
GITHUB_HARD_BYTES = 100 * 1024 * 1024


def _tensor(arr, dtype=None):
    # np.array(...) copies and PRESERVES 0-d shapes (np.ascontiguousarray would promote a scalar
    # to shape (1,), which the round-trip check catches as a shape mismatch).
    return torch.from_numpy(np.array(arr, dtype=dtype))


def tf_value_to_torch(v):
    """Convert one dataset feature value (tf.Tensor or tf.RaggedTensor) losslessly."""
    if isinstance(v, tf.RaggedTensor):
        return {
            "__ragged__": True,
            "values": tf_value_to_torch(v.flat_values),
            "row_splits": [_tensor(rs.numpy(), np.int64) for rs in v.nested_row_splits],
        }
    if v.dtype == tf.string:
        arr = np.asarray(v.numpy())
        return {"__strings__": True, "data": [s.decode("utf-8") for s in arr.reshape(-1).tolist()], "shape": list(arr.shape)}
    return _tensor(v.numpy())


def load_shard(path):
    """Read one converted shard (gzip-compressed torch.save stream) safely."""
    with gzip.open(path, "rb") as f:
        return torch.load(io.BytesIO(f.read()), weights_only=True)


def values_equal(converted, v):
    """Bit-for-bit comparison of a converted value against its TF source (dtype included)."""
    if isinstance(v, tf.RaggedTensor):
        if not (isinstance(converted, dict) and converted.get("__ragged__")):
            return False
        if len(converted["row_splits"]) != len(v.nested_row_splits):
            return False
        for c_rs, t_rs in zip(converted["row_splits"], v.nested_row_splits):
            if not np.array_equal(c_rs.numpy(), t_rs.numpy().astype(np.int64)):
                return False
        return values_equal(converted["values"], v.flat_values)
    if v.dtype == tf.string:
        arr = np.asarray(v.numpy())
        return (isinstance(converted, dict) and converted.get("__strings__")
                and converted["shape"] == list(arr.shape)
                and converted["data"] == [s.decode("utf-8") for s in arr.reshape(-1).tolist()])
    src = v.numpy()
    dst = converted.numpy()
    return dst.dtype == src.dtype and dst.shape == src.shape and np.array_equal(dst, src)


def spec_to_json(spec):
    out = {}
    x_spec, y_spec = spec
    for k, s in x_spec.items():
        if isinstance(s, tf.RaggedTensorSpec):
            out[k] = {"dtype": s.dtype.name, "shape": [d for d in s.shape.as_list()], "ragged": True,
                      "ragged_rank": int(s.ragged_rank), "row_splits_dtype": s.row_splits_dtype.name}
        else:
            out[k] = {"dtype": s.dtype.name, "shape": (s.shape.as_list() if s.shape.rank is not None else None), "ragged": False}
    out["__y__"] = {"dtype": y_spec.dtype.name, "shape": y_spec.shape.as_list(), "ragged": False}
    return out


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_shards(part_dir):
    return sorted((d for d in os.listdir(part_dir) if d.isdigit() and os.path.isdir(os.path.join(part_dir, d))), key=int)


def convert_partition(dataset, partition, args, log):
    src_part = os.path.join(args.data_path, dataset, partition)
    dst_part = os.path.join(args.out_path, dataset, partition)
    os.makedirs(dst_part, exist_ok=True)
    shards = numeric_shards(src_part)
    assert shards == [str(i) for i in range(len(shards))], f"non-contiguous shards in {src_part}: {shards}"
    drop = [] if args.keep_all_fields else list(DEFAULT_DROP)

    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset": dataset,
        "partition": partition,
        "source": src_part,
        "source_git_commit": args.git_commit,
        "converter": os.path.basename(__file__),
        "tensorflow_version": tf.__version__,
        "torch_version": torch.__version__,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dropped_fields": drop,
        "element_spec": None,
        "n_shards": len(shards),
        "n_samples": 0,
        "shards": [],
        "verified_bit_exact": None if args.verify == "none" else True,
    }
    all_idx = []
    for k in shards:
        t0 = time.time()
        src = os.path.join(src_part, k)
        ds = tf.data.Dataset.load(src, compression="GZIP")
        if manifest["element_spec"] is None:
            manifest["element_spec"] = spec_to_json(ds.element_spec)
        samples, idxs = [], []
        for x, y in ds:
            xs = {f: tf_value_to_torch(v) for f, v in x.items() if f not in drop}
            samples.append({"x": xs, "y": tf_value_to_torch(y)})
            idxs.append(int(x["sample_idx"]))
        out_path = os.path.join(dst_part, f"{k}.pt.gz")
        buf = io.BytesIO()
        torch.save({"format_version": FORMAT_VERSION, "dataset": dataset, "partition": partition,
                    "shard": int(k), "source": src, "dropped_fields": drop, "samples": samples}, buf)
        raw_bytes = buf.tell()
        with gzip.open(out_path, "wb", compresslevel=GZIP_LEVEL) as f:
            f.write(buf.getvalue())
        size = os.path.getsize(out_path)

        verified = None
        if args.verify != "none":
            back = load_shard(out_path)
            assert back["format_version"] == FORMAT_VERSION and len(back["samples"]) == len(samples)
            verified, failure = True, None
            for i, (x, y) in enumerate(tf.data.Dataset.load(src, compression="GZIP")):
                bx, by = back["samples"][i]["x"], back["samples"][i]["y"]
                if set(bx) != set(f for f in x if f not in drop):
                    failure = f"scenario {i}: field set differs"
                elif not values_equal(by, y):
                    failure = f"scenario {i}: y differs"
                else:
                    bad = [f for f, v in x.items() if f not in drop and not values_equal(bx[f], v)]
                    if bad:
                        failure = f"scenario {i}: fields differ: {bad}"
                if failure:
                    verified = False
                    break
            manifest["verified_bit_exact"] = manifest["verified_bit_exact"] and verified
            if not verified:
                log(f"  !!! VERIFICATION FAILED for {out_path}: {failure}")

        manifest["shards"].append({"shard": int(k), "file": f"{k}.pt.gz", "n_samples": len(samples),
                                   "bytes": size, "raw_bytes": raw_bytes, "sha256": sha256_of(out_path),
                                   "sample_idx": idxs, "verified_bit_exact": verified})
        manifest["n_samples"] += len(samples)
        all_idx += idxs
        flag = "" if size < GITHUB_WARN_BYTES else ("  (>50 MB: GitHub will warn)" if size < GITHUB_HARD_BYTES else "  !!! >100 MB: GitHub will REJECT this file")
        log(f"  shard {k}: {len(samples)} scenarios -> {out_path} ({size / 1e6:.1f} MB gz, {raw_bytes / 1e6:.1f} MB raw, verified={verified}) in {time.time() - t0:.0f}s{flag}")

    assert len(all_idx) == len(set(all_idx)), f"duplicate sample_idx within {dataset}/{partition}"
    with open(os.path.join(dst_part, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def parse_args():
    p = argparse.ArgumentParser(description="Convert the TF datasets under data/ to PyTorch files under data_torch/.")
    p.add_argument("--data-path", default="data")
    p.add_argument("--out-path", default="data_torch")
    p.add_argument("--datasets", default=None, help="comma-separated subset (default: every directory under --data-path)")
    p.add_argument("--partitions", default=None, help="comma-separated subset (default: every partition present)")
    p.add_argument("--keep-all-fields", action="store_true", help=f"also keep {DEFAULT_DROP} (large; not meant to be committed)")
    p.add_argument("--verify", choices=["reread", "none"], default="reread",
                   help="reread: re-read every TF shard and compare bit-for-bit with the written file (default)")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        args.git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        args.git_commit = None
    datasets = args.datasets.split(",") if args.datasets else sorted(
        d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d)))

    def log(msg):
        print(msg, flush=True)

    log(f"[convert] tf {tf.__version__} | torch {torch.__version__} | datasets: {datasets} | drop={[] if args.keep_all_fields else DEFAULT_DROP} | verify={args.verify}")
    summary = {}
    t_all = time.time()
    for dataset in datasets:
        parts = args.partitions.split(",") if args.partitions else sorted(
            d for d in os.listdir(os.path.join(args.data_path, dataset)) if os.path.isdir(os.path.join(args.data_path, dataset, d)))
        for partition in parts:
            log(f"[convert] {dataset}/{partition}")
            m = convert_partition(dataset, partition, args, log)
            summary[f"{dataset}/{partition}"] = {"n_shards": m["n_shards"], "n_samples": m["n_samples"],
                                                 "bytes": sum(s["bytes"] for s in m["shards"]),
                                                 "verified_bit_exact": m["verified_bit_exact"]}
    os.makedirs(args.out_path, exist_ok=True)
    with open(os.path.join(args.out_path, "conversion_summary.json"), "w") as f:
        json.dump({"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_git_commit": args.git_commit,
                   "dropped_fields": [] if args.keep_all_fields else DEFAULT_DROP, "partitions": summary}, f, indent=1)
    log(f"[convert] ===== SUMMARY ({time.time() - t_all:.0f}s) =====")
    total = 0
    for k, v in summary.items():
        total += v["bytes"]
        log(f"[convert] {k:40s} shards={v['n_shards']:2d} scenarios={v['n_samples']:5d} {v['bytes'] / 1e6:8.1f} MB verified={v['verified_bit_exact']}")
    log(f"[convert] total {total / 1e6:.1f} MB -> {args.out_path}/ ; all verified: {all(v['verified_bit_exact'] for v in summary.values())}")
    if not all(v["verified_bit_exact"] in (True, None) for v in summary.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()