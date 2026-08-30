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

# PyTorch translation. Every original TensorFlow line is kept as a `#TF:` comment with its
# translation below; the frozen TF original is importable as `tf_reference.utils`.
#
# The TF pipeline is `tf.data.Dataset` chains: load_dataset(...).map(fn).shuffle(...).repeat().
# PyTorch has no equivalent object, so `ListDataset` below provides the same chainable
# interface (map / shuffle / repeat / prefetch / take / cardinality / iteration) over an
# in-memory list of scenarios, with tf.data's *buffered* shuffle algorithm (a buffer of
# `buffer_size` elements, one drawn at random per step, refilled from the stream; reshuffled
# on every pass) driven by a seeded torch.Generator. Semantics match; the random numbers do
# not, which is why the exact-replay experiments feed the recorded TF order instead
# (`ListDataset.from_order`).

import gzip
import io
import os

# Default to CPU-only (as the original repo does) but DON'T clobber an explicit choice made
# by the caller before import — experiment.py / run_experiments.py set CUDA_VISIBLE_DEVICES=0
# to run on the GPU, and an unconditional "=-1" here would silently force them back to CPU.
# PyTorch: not needed — TF grabs every visible GPU at import time, torch only uses a GPU when a
# tensor is explicitly moved to it (`--device` in experiment.py), so the default stays untouched.
#TF: os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

#TF: import tensorflow as tf
import numpy as np
import torch
from typing import List

from torch_ragged import decode_sample


def seg_to_global_reshape(tensor, num_dims=3):
    assert num_dims > 1
    perms = [1, 0] + list(range(2, num_dims))
    #TF: total_flows = tf.shape(tensor)[0] * tf.shape(tensor)[1]
    total_flows = tensor.shape[0] * tensor.shape[1]
    if num_dims == 2:
        new_shape = (total_flows,)
    else:
        #TF: new_shape = tf.concat([[total_flows], tf.shape(tensor)[2:]], axis=0)
        new_shape = (total_flows,) + tuple(tensor.shape[2:])
    #TF: return tf.reshape(tf.transpose(tensor, perms), new_shape)
    return tensor.permute(*perms).reshape(new_shape)


def prepare_targets_and_mask(targets: List[str], mask: str) -> callable:
    """Obtains map function to prepare the targets of the dataset for the current model.

    Parameters
    ----------
    targets : List[str]
        List of features to be selected as targets, in that order.
    mask : str
        Mask feature used to determine which windows in the temporal dimension are
        valid.

    Returns
    -------
    callable
        Map function to be called by the tf.data.Dataset.map method.
    """

    def modified_target_map(x, y):
        #TF: reshaped_mask = tf.expand_dims(seg_to_global_reshape(x[mask], num_dims=2), 1)
        reshaped_mask = seg_to_global_reshape(x[mask], num_dims=2).unsqueeze(1)

        #TF: return x, tf.concat(
        #TF:     [
        #TF:         tf.reshape(
        #TF:             tf.boolean_mask(seg_to_global_reshape(x[target]), reshaped_mask),
        #TF:             (-1, 1),
        #TF:         )
        #TF:         for target in targets
        #TF:     ],
        #TF:     axis=1,
        #TF: )
        return x, torch.cat(
            [
                seg_to_global_reshape(x[target])[reshaped_mask].reshape(-1, 1)
                for target in targets
            ],
            dim=1,
        )

    return modified_target_map


class ListDataset:
    """Minimal in-memory stand-in for the tf.data.Dataset chains used by this repo.

    Elements are (x, y) scenario tuples. `map` is lazy (applied on iteration); `shuffle`
    reproduces tf.data's buffered-shuffle algorithm with a seeded torch.Generator; `repeat`
    makes iteration endless with a fresh reshuffle on every pass (reshuffle_each_iteration).
    """

    def __init__(self, items, map_fns=(), shuffle=None, repeat=False, gen=None):
        self._items = items
        self._map_fns = tuple(map_fns)
        self._shuffle = shuffle  # None or (buffer_size, seed)
        self._repeat = repeat
        # tf.data keeps ONE seed generator per shuffle() call, shared by every iterator created
        # from that dataset (or from datasets derived from it): each new iterator continues the
        # random stream, so e.g. the z-score pass and model.fit see different permutations.
        # The torch.Generator created in shuffle() plays that role and is shared by clones.
        self._gen = gen

    def _clone(self, **kw):
        d = dict(items=self._items, map_fns=self._map_fns, shuffle=self._shuffle, repeat=self._repeat, gen=self._gen)
        d.update(kw)
        return ListDataset(**d)

    # --- tf.data.Dataset API subset -------------------------------------------------
    def map(self, fn):
        return self._clone(map_fns=self._map_fns + (fn,))

    def shuffle(self, buffer_size, seed=None, reshuffle_each_iteration=True):
        assert reshuffle_each_iteration, "only reshuffle_each_iteration=True is implemented (tf.data default)"
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            gen.seed()
        return self._clone(shuffle=(int(buffer_size), seed), gen=gen)

    def repeat(self):
        return self._clone(repeat=True)

    def prefetch(self, buffer_size=None):
        return self  # no-op: everything is already in memory

    def take(self, n):
        return ListDataset(list(self._iter_raw(limit=n)), map_fns=())

    def skip(self, n):
        assert not (self._shuffle or self._repeat), "skip() is only implemented for plain (unshuffled) datasets"
        return self._clone(items=self._items[n:])

    def cardinality(self):
        return -1 if self._repeat else len(self._items)

    def __len__(self):
        return len(self._items)

    def concatenate(self, other):
        assert not (self._map_fns or self._shuffle or self._repeat or other._map_fns or other._shuffle or other._repeat)
        return ListDataset(self._items + other._items)

    @staticmethod
    def from_order(items, sample_idx_order):
        """Exact-replay helper: the elements of `items` (an un-shuffled ListDataset or list of
        (x, y)) emitted in the given sample_idx order (one pass over `sample_idx_order`; wrap
        in `.repeat()`-free training loops that count steps)."""
        base = items._items if isinstance(items, ListDataset) else list(items)
        by_idx = {int(x["sample_idx"]): (x, y) for x, y in base}
        return ListDataset([by_idx[int(i)] for i in sample_idx_order])

    # --- iteration ----------------------------------------------------------------------
    def _apply(self, item):
        x, y = item
        for fn in self._map_fns:
            x, y = fn(x, y)
        return x, y

    def _one_pass(self, gen):
        items = self._items
        if self._shuffle is None:
            for it in items:
                yield it
            return
        # tf.data ShuffleDataset: fill a buffer, then repeatedly emit a uniformly random buffer
        # slot and refill it with the next upstream element; drain the buffer at the end.
        buffer_size, _ = self._shuffle
        buf = []
        pos = 0
        while pos < len(items) and len(buf) < buffer_size:
            buf.append(items[pos])
            pos += 1
        while buf:
            j = int(torch.randint(len(buf), (1,), generator=gen))
            if pos < len(items):
                out, buf[j] = buf[j], items[pos]
                pos += 1
            else:
                out = buf.pop(j)
            yield out

    def index_order(self, n):
        """sample_idx of the first n elements this dataset would emit (consumes the shared
        shuffle stream exactly like iterating would). Used by experiment.py to materialise
        the training order up front, which makes runs resumable and auditable
        (results/.../sample_order_used.npy)."""
        return np.fromiter((int(x["sample_idx"]) for x, _ in self._iter_raw(limit=n)), dtype=np.int32, count=n)

    def _iter_raw(self, limit=None):
        gen = self._gen
        n = 0
        while True:
            for it in self._one_pass(gen):
                if limit is not None and n >= limit:
                    return
                yield it
                n += 1
            if not self._repeat:
                return

    def __iter__(self):
        for it in self._iter_raw():
            yield self._apply(it)


def _load_shard(path):
    """Read one converted shard written by convert_data_to_torch.py (gzip'd torch.save)."""
    with gzip.open(path, "rb") as f:
        payload = torch.load(io.BytesIO(f.read()), weights_only=True)
    return [(decode_sample(s["x"]), s["y"]) for s in payload["samples"]]


#TF: def load_dataset(name: str, data_path: str = "data") -> tf.data.Dataset:
def load_dataset(name: str, data_path: str = "data_torch") -> ListDataset:
    """Function to unshard and load a dataset from the data folder.

    Parameters
    ----------
    name : str
        Name of the dataset and partition [training/validation/test] to load, in format
        '{name}/{partition}'.

    data_path : str
        Path to the data folder. By default, it is 'data_torch' (the TF-free conversion of
        'data' produced by convert_data_to_torch.py), which assumes the working directory
        is the root of the project.

    Returns
    -------
    ListDataset
        The dataset loaded from the shards (same scenarios, same order as the TF shards).
    """
    path = os.path.join(data_path, name)
    #TF: shards = os.listdir(path)
    shards = [f for f in os.listdir(path) if f.endswith(".pt.gz")]
    assert len(shards) > 0, f"Invalid dataset: {name}"

    #TF: ds = tf.data.Dataset.load(path + "/0", compression="GZIP")
    ds = ListDataset(_load_shard(path + "/0.pt.gz"))
    for ii in range(1, len(shards)):
        #TF: ds = ds.concatenate(tf.data.Dataset.load(path + f"/{ii}", compression="GZIP"))
        ds = ds.concatenate(ListDataset(_load_shard(path + f"/{ii}.pt.gz")))

    #TF: return ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds.prefetch()
