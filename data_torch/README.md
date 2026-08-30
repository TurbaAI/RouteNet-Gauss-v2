# `data_torch/` — the datasets in a TF-free PyTorch format

Lossless, one-to-one conversion of the TensorFlow datasets in `data/` (which are
`tf.data.Dataset.save` snapshots and can only be read with TensorFlow). Produced by
`convert_data_to_torch.py`; loaded by `utils.load_dataset` (PyTorch version). **Same scenarios,
same order, same shard numbering** as the TF shards, so `data_torch/<dataset>/<partition>/<k>.pt.gz`
holds exactly the scenarios of `data/<dataset>/<partition>/<k>/`.

| | |
|---|---|
| source | `data/` at commit `2e30d5d` (unchanged since the datasets were added) |
| converter | `convert_data_to_torch.py` (TF 2.15.0, torch 2.13.0) |
| verification | every shard re-read from the TF source and compared field-by-field, bit-for-bit (dtype, shape, values, ragged row splits, strings) — `verified_bit_exact: true` in every `manifest.json` |
| dropped field | `flow_packets_per_ms` (mawi_pcaps only): a variable-length per-flow packet-rate series that nothing in the model or evaluation reads and that is ~90 % of the bytes. `python convert_data_to_torch.py --keep-all-fields` re-creates the files with it (not meant to be committed). |
| size | 912 MB gzip-compressed (≈1.7 GB raw); largest shard 54 MB (GitHub warns above 50 MB, rejects above 100 MB) |

## File format

Each `<k>.pt.gz` is a gzip stream of a `torch.save` object, loadable **without** trusting pickle:

```python
import gzip, io, torch
shard = torch.load(io.BytesIO(gzip.open("data_torch/trex_multiburst/test/0.pt.gz", "rb").read()), weights_only=True)
shard["samples"][0]["x"]["flow_traffic"]        # torch.float32 [n_flows, seg_num, 1]
shard["samples"][0]["x"]["link_to_path"]        # {"__ragged__": True, "values": int32 [N], "row_splits": [int64 [n_flows+1]]}
shard["samples"][0]["x"]["flow_id"]             # {"__strings__": True, "data": [...], "shape": [n_flows]}
shard["samples"][0]["y"]                        # torch.float32 [n_flows * seg_num, 1] (raw label, replaced by prepare_targets_and_mask)
```

`utils.load_dataset` turns the ragged dicts into `torch_ragged.Ragged` objects and string dicts
into Python lists, which is the form `models.RouteNetGauss.forward` consumes.

Field value encodings:
- dense `tf.Tensor` → `torch.Tensor`, TF dtype preserved (`float32`, `int32`, `bool`), scalars stay 0-d;
- `tf.RaggedTensor` → `{"__ragged__": True, "values": Tensor, "row_splits": [Tensor(int64), ...]}` (outermost splits first; `ragged_rank == len(row_splits)`);
- `tf.string` → `{"__strings__": True, "data": [str, ...], "shape": [...]}`.

`manifest.json` per partition records the TF `element_spec`, per-shard sample counts, the
`sample_idx` list of every shard, sizes, SHA-256 of each file and the verification flag.
`conversion_summary.json` at the top level aggregates all partitions.

## Regenerating

```bash
conda activate RG_torch          # needs tensorflow-cpu AND torch (see requirements-torch.txt)
python convert_data_to_torch.py  # ~5 min; idempotent
```
