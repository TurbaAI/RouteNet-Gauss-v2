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

# Shared, importable helpers used by both train.py (single-run script) and
# experiment.py (job-matrix runner). These were originally defined inline in
# train.py; they were extracted here so they can be reused without importing
# train.py (which runs training at import time).

import os
import pickle

import numpy as np
import tensorflow as tf


def get_z_scores_dict(
    ds,
    params,
    include_y=None,
    flatten=False,
    summarize=-1,
    store_res_path=None,
    check_existing=False,
):
    """
    Get the mean and the std for different parameters of a dataset. Later used by the
    models for the z-score normalization.

    Parameters
    ----------
    ds
        tensorflow.data.Dataset
    params
        list of strings indicating the parameters to extract the features from
    include_y, optional
        Indicates if to also extract the features of the output variable. Inputs
        indicate the string key used on the return dict. If None, it is not included.
    flatten, optional
        If true, mean and std are computed globally for all dimensions in each feature.
        Otherwise, the values are computed for each dimension separately.
    summarize, optional
        If > 0, only uses the first n samples to compute the mean and std.
    store_res_path, optional
        If not None, the results are stored in the path indicated by the string.
        The dictionary is stored using the pickle library.

    Returns
    -------
    dict
        Dictionary containing the min and the max-min for each parameter.
    """
    # If check_existing is True, check if the file exists and return the dict (if so)
    if store_res_path is not None and check_existing:
        if os.path.exists(store_res_path):
            with open(store_res_path, "rb") as ff:
                return pickle.load(ff)

    # Use first sample to get the shape of the tensors
    iter_ds = iter(ds)
    next_sample = next(iter_ds)
    sample, label = next_sample[0], next_sample[1]
    params_lists = {param: sample[param].numpy() for param in params}
    if include_y:
        params_lists[include_y] = label.numpy()

    if summarize > 0:
        max_samples = summarize - 1

    # Include the rest of the samples
    for ii, (sample, label) in enumerate(map(lambda x: (x[0], x[1]), iter_ds)):
        if summarize > 0 and ii > max_samples:
            break
        for param in params:
            params_lists[param] = np.concatenate(
                (params_lists[param], sample[param].numpy()), axis=0
            )
        if include_y:
            params_lists[include_y] = np.concatenate(
                (params_lists[include_y], label.numpy()), axis=0
            )

    scores = dict()
    axis = None if flatten else 0
    for param, param_list in params_lists.items():
        scores[param] = [np.mean(param_list, axis=axis), np.std(param_list, axis=axis)]
        # Check if std is 0
        if scores[param][1].size == 1 and scores[param][1] == 0:
            print(f"Z-score normalization Warning: {param} has a std of 0.")
            scores[param][1] = 1
        elif scores[param][1].size > 1 and np.any(scores[param][1] == 0):
            print(
                f"Z-score normalization Warning: Several values of {param} has a std of 0."
            )
            scores[param][1][scores[param][1] == 0] = 1

    if store_res_path is not None:
        store_res_dir, _ = os.path.split(store_res_path)
        os.makedirs(store_res_dir, exist_ok=True)
        with open(store_res_path, "wb") as ff:
            pickle.dump(scores, ff)

    return scores


def get_positional_mape(pos: int, name: str) -> callable:
    """Returns the MAPE metric for a specific feature at a given position.

    Parameters
    ----------
    pos : int
        Position of the feature in the output tensor.
    name : str
        Name of the feature.

    Returns
    -------
    callable
        Function to calculate the MAPE metric for a specific feature, to be added to
        model metrics to be included in the tensorboard logs.
    """

    def mape_metric(y_true, y_pred):
        y_true = y_true[:, pos]
        y_pred = y_pred[:, pos]
        return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100

    mape_metric.__name__ = f"mape_{name}_metric"
    return mape_metric


def r2_score(y_true: tf.TensorSpec, y_pred: tf.TensorSpec) -> float:
    """Returns the R2 score between the true and predicted values.

    Parameters
    ----------
    y_true : tf.TensorSpec
        True values
    y_pred : tf.TensorSpec
        Predicted values

    Returns
    -------
    float
        R2 score
    """
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2


def get_positional_r2(pos: int, name: str) -> callable:
    """Returns the R2 metric for a specific feature at a given position.

    Parameters
    ----------
    pos : int
        Position of the feature in the output tensor.
    name : str
        Name of the feature.

    Returns
    -------
    callable
        Function to calculate the R2 metric for a specific feature, to be added to
        model metrics to be included in the tensorboard logs.
    """

    def r2_metric(y_true, y_pred):
        y_true = y_true[:, pos]
        y_pred = y_pred[:, pos]
        return r2_score(y_true, y_pred)

    r2_metric.__name__ = f"r2_{name}_metric"
    return r2_metric


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Writes the optimizer's current learning rate into the epoch `logs` dict.

    Adding it to the callback list means the LR shows up in the CSVLogger output and
    the TensorBoard scalars alongside loss/metrics, which is useful for inspecting the
    effect of ReduceLROnPlateau on these short runs.
    """

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        lr = self.model.optimizer.learning_rate
        # learning_rate may be a schedule or a variable; resolve to a float either way.
        if callable(lr):
            lr = lr(self.model.optimizer.iterations)
        logs["lr"] = float(tf.keras.backend.get_value(lr))
