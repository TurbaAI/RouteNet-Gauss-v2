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

# PyTorch stand-ins for the tf.RaggedTensor operations RouteNet-Gauss relies on.
#
# TensorFlow represents variable-length per-row data (a flow's sequence of hops, the queues
# of a node, the (flow, position) pairs crossing a link) as tf.RaggedTensor and the model
# uses tf.gather / tf.gather_nd / tf.reduce_sum / Keras RNN directly on them. PyTorch has no
# ragged tensor type, so this module provides the minimal equivalent: a `Ragged` container
# and the handful of operations models.py needs. Each helper names the TF op it replaces, so
# models.py can stay a line-by-line translation.
#
# Representation: a Ragged holds EITHER the flat form (`values` [N, ...] + int64
# `row_splits` [nrows+1] — exactly TensorFlow's layout) OR the padded form (`padded`
# [nrows, T, ...] + `row_lengths`), and derives the other lazily on first use, caching it.
# The message-passing loop alternates between the two views (GRU over padded sequences,
# gathers/sums over flat values), so this avoids re-deriving the same tensors hundreds of
# times per scenario. All operations are ordinary autograd-friendly tensor code and
# deterministic on CPU and (with torch.use_deterministic_algorithms(True)) on CUDA.

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


class Ragged:
    """A 2-level ragged tensor: `nrows` rows of variable length. Mirrors
    tf.RaggedTensor(ragged_rank=1); row i is values[row_splits[i]:row_splits[i+1]]."""

    __slots__ = ("_values", "_row_splits", "_padded", "_row_lengths", "_row_ids", "_mask", "_max_len",
                 "_n_values", "_flat_index")

    def __init__(self, values: Optional[Tensor] = None, row_splits: Optional[Tensor] = None, *,
                 padded: Optional[Tensor] = None, row_lengths: Optional[Tensor] = None,
                 n_values: Optional[int] = None):
        assert (values is not None and row_splits is not None) or (padded is not None and row_lengths is not None)
        self._values = values
        self._row_splits = row_splits
        self._padded = padded
        self._row_lengths = row_lengths
        self._row_ids = None
        self._mask = None
        self._flat_index = None
        self._max_len = None if padded is None else int(padded.shape[1])
        # number of flat values; a Python int so that derived index tensors can be built
        # without synchronising with the device
        self._n_values = int(values.shape[0]) if values is not None else n_values

    # ----- construction ---------------------------------------------------------------
    @staticmethod
    def from_row_lengths(values: Tensor, row_lengths: Tensor) -> "Ragged":
        """tf.RaggedTensor.from_row_lengths"""
        row_lengths = row_lengths.to(torch.int64)
        splits = torch.zeros(row_lengths.numel() + 1, dtype=torch.int64, device=values.device)
        splits[1:] = torch.cumsum(row_lengths, 0)
        r = Ragged(values, splits)
        r._row_lengths = row_lengths
        return r

    @staticmethod
    def from_padded(dense: Tensor, row_lengths: Tensor, n_values: Optional[int] = None) -> "Ragged":
        """tf.RaggedTensor.from_tensor(dense, lengths=row_lengths): row i keeps its first
        row_lengths[i] entries (flat order is row-major, i.e. the same as TF's)."""
        return Ragged(padded=dense, row_lengths=row_lengths.to(torch.int64), n_values=n_values)

    @staticmethod
    def from_dict(d: Dict) -> "Ragged":
        """From the on-disk form written by convert_data_to_torch.py
        ({"__ragged__": True, "values": Tensor, "row_splits": [Tensor]}); only ragged_rank 1
        is used by the model."""
        assert d.get("__ragged__") and len(d["row_splits"]) == 1, "only ragged_rank=1 supported"
        return Ragged(d["values"], d["row_splits"][0].to(torch.int64))

    # ----- the two views (derived lazily, cached) ------------------------------------------
    @property
    def n_values(self) -> int:
        if self._n_values is None:
            self._n_values = int(self.row_lengths().sum())  # one sync, only if never provided
        return self._n_values

    @property
    def values(self) -> Tensor:
        """Flat values [N, ...] in row-major order (tf.RaggedTensor.values)."""
        if self._values is None:
            p = self._padded
            self._values = p.reshape((p.shape[0] * p.shape[1],) + tuple(p.shape[2:]))[self.flat_index()]
        return self._values

    def flat_index(self) -> Tensor:
        """Index of every flat value inside the flattened padded form (row * T + position)."""
        if self._flat_index is None:
            rows = self.row_ids()
            pos = torch.arange(self.n_values, device=self.device) - self.row_splits[rows]
            self._flat_index = rows * self.max_row_length() + pos
        return self._flat_index

    @property
    def row_splits(self) -> Tensor:
        if self._row_splits is None:
            lengths = self.row_lengths()
            splits = torch.zeros(lengths.numel() + 1, dtype=torch.int64, device=lengths.device)
            splits[1:] = torch.cumsum(lengths, 0)
            self._row_splits = splits
        return self._row_splits

    def to_padded(self, pad_value=0) -> Tensor:
        """tf.RaggedTensor.to_tensor(): dense [nrows, max_row_length, ...] padded with 0."""
        if self._padded is None:
            T = self.max_row_length()
            out = self._values.new_full((self.nrows * T,) + tuple(self._values.shape[1:]), pad_value)
            if self._n_values > 0:
                out = out.index_copy(0, self.flat_index(), self._values)
            self._padded = out.reshape((self.nrows, T) + tuple(self._values.shape[1:]))
        return self._padded

    # ----- shape queries --------------------------------------------------------------
    @property
    def nrows(self) -> int:
        return self._row_lengths.numel() if self._row_lengths is not None else self._row_splits.numel() - 1

    def row_lengths(self) -> Tensor:
        """tf.RaggedTensor.row_lengths()"""
        if self._row_lengths is None:
            self._row_lengths = self._row_splits[1:] - self._row_splits[:-1]
        return self._row_lengths

    def row_ids(self) -> Tensor:
        """tf.RaggedTensor.value_rowids(): the row index of every flat value."""
        if self._row_ids is None:
            self._row_ids = torch.repeat_interleave(
                torch.arange(self.nrows, device=self.device), self.row_lengths(),
                output_size=self.n_values,
            )
        return self._row_ids

    def max_row_length(self) -> int:
        if self._max_len is None:
            self._max_len = int(self.row_lengths().max()) if self.nrows > 0 else 0
        return self._max_len

    def padding_mask(self) -> Tensor:
        """bool [nrows, max_row_length]: True where a padded position holds a real value."""
        if self._mask is None:
            self._mask = torch.arange(self.max_row_length(), device=self.device)[None, :] < self.row_lengths()[:, None]
        return self._mask

    @property
    def device(self):
        return (self._values if self._values is not None else self._padded).device

    # ----- derived raggeds ------------------------------------------------------------------
    def with_values(self, values: Tensor) -> "Ragged":
        """Same row structure, new flat values (e.g. after an elementwise op on .values)."""
        r = Ragged(values, self.row_splits)
        r._row_lengths, r._row_ids, r._max_len, r._flat_index, r._mask = (
            self._row_lengths, self._row_ids, self._max_len, self._flat_index, self._mask)
        return r

    def inner_slice_from(self, start: int) -> "Ragged":
        """ragged[:, start:] — drop the first `start` entries of every row (all rows are
        assumed to have at least `start` entries, which holds for the flow sequences: every
        flow has >= 1 hop and the sequence carries one extra leading element)."""
        return Ragged(padded=self.to_padded()[:, start:], row_lengths=self.row_lengths() - start,
                      n_values=self.n_values - start * self.nrows)

    def to(self, device) -> "Ragged":
        if self._values is not None:
            return Ragged(self._values.to(device), self.row_splits.to(device))
        return Ragged(padded=self._padded.to(device), row_lengths=self._row_lengths.to(device), n_values=self._n_values)

    def long(self) -> "Ragged":
        return self.with_values(self.values.to(torch.int64))

    def to_list(self):
        """tf.RaggedTensor.to_list(): nested Python lists (used by visualization/tensor_utils)."""
        vals, splits = self.values.cpu(), self.row_splits.cpu().tolist()
        return [vals[splits[i]:splits[i + 1]].tolist() for i in range(self.nrows)]

    def __repr__(self):
        return f"Ragged(nrows={self.nrows}, values={tuple(self.values.shape)}, dtype={self.values.dtype})"


# ----- the tf ops used by models.py -----------------------------------------------------

def ragged_gather(params: Tensor, indices: Ragged) -> Ragged:
    """tf.gather(params, ragged_indices): params[i] for every flat index; keeps the row
    structure of `indices`. Result values: [N, *params.shape[1:]]."""
    return indices.with_values(params[indices.values])


def ragged_gather_nd(padded: Tensor, indices: Ragged) -> Ragged:
    """tf.gather_nd(ragged_2d_sequence, ragged_indices) where every index is a
    (row, position) pair: `padded` is the dense [rows, T, ...] form of the sequence and the
    result keeps the row structure of `indices`."""
    idx = indices.values
    return indices.with_values(padded[idx[:, 0], idx[:, 1]])


def ragged_reduce_sum(r: Ragged) -> Tensor:
    """tf.reduce_sum(ragged, axis=1): sum each row's values. Result: [nrows, ...]."""
    values = r.values
    out = values.new_zeros((r.nrows,) + tuple(values.shape[1:]))
    return out.index_add(0, r.row_ids(), values)


def ragged_prepend(first: Tensor, r: Ragged) -> Ragged:
    """tf.concat([tf.expand_dims(first, 1), ragged], axis=1): insert one element per row at
    position 0 (first: [nrows, ...])."""
    return Ragged(padded=torch.cat([first.unsqueeze(1), r.to_padded()], dim=1), row_lengths=r.row_lengths() + 1,
                  n_values=r.n_values + r.nrows)


def run_gru_over_ragged(gru: torch.nn.GRU, inputs: Ragged, initial_state: Tensor) -> Tuple[Ragged, Tensor]:
    """tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)(ragged_inputs,
    initial_state=...): run the GRU along every row's sequence (rows are the batch).

    Keras masks the ragged rows so a row's state stops updating once its sequence ends. A GRU
    is causal, so the padded (zero) steps appended after a row's real hops cannot influence
    its real outputs; the state after the last real hop is therefore simply the output at
    position row_length-1, which is what we return as the final state (rows of length 0 keep
    the initial state). The padded positions of the returned sequence hold values that are
    never read (Ragged only exposes the real positions). One fused sequence call replaces the
    per-step cell loop, which is what makes the model fast in eager PyTorch.
    Returns (states after every real step, as Ragged with the rows' structure; final state
    per row [nrows, H])."""
    x = inputs.to_padded()  # [rows, T, in]
    lengths = inputs.row_lengths()
    if x.shape[1] == 0:
        return Ragged.from_row_lengths(initial_state.new_zeros((0, initial_state.shape[1])), lengths), initial_state
    # (cuDNN requires a contiguous h0; initial_flow_state[seg] is a slice of a permuted tensor)
    out, _ = gru(x, initial_state.unsqueeze(0).contiguous())  # out: [rows, T, H]
    last = out[torch.arange(out.shape[0], device=out.device), torch.clamp(lengths - 1, min=0)]
    h = torch.where((lengths > 0)[:, None], last, initial_state)
    return Ragged(padded=out, row_lengths=lengths, n_values=inputs.n_values), h


# ----- sample-dict utilities (used by utils.py / experiment.py) -----------------------

def decode_sample(x: Dict) -> Dict:
    """Turn a scenario dict as stored in data_torch/ (plain tensors, ragged dicts, string
    dicts) into the form the model consumes: tensors and `Ragged` objects; string fields
    become plain Python lists."""
    out = {}
    for k, v in x.items():
        if isinstance(v, dict) and v.get("__ragged__"):
            out[k] = Ragged.from_dict(v) if len(v["row_splits"]) == 1 else v
        elif isinstance(v, dict) and v.get("__strings__"):
            out[k] = v["data"]
        else:
            out[k] = v
    return out


def sample_to_device(x, device):
    """Move every tensor / Ragged in a (nested) sample structure to `device`."""
    if isinstance(x, Ragged) or isinstance(x, Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: sample_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(sample_to_device(v, device) for v in x)
    return x
