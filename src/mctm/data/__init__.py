# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : __init__.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2025-01-28 17:08:47 (Marcel Arpogaus)
# changed : 2025-01-28 17:26:29 (Marcel Arpogaus)

# %% License ###################################################################

# %% Description ###############################################################
"""Provides data loaders for different datasets."""

# %% imports ##########################################################################
import tensorflow as tf

from .benchmark import get_dataset as get_benchmark_dataset
from .malnutrion import get_dataset as get_malnutrition_dataset
from .sklearn_datasets import get_dataset as get_sim_dataset


# %% private functions #########################################################
def _make_dataset(ds, batch_size):
    return (
        ds.shuffle(min(2**14, batch_size * 100))
        .batch(batch_size, drop_remainder=True)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )


def _get_batch_size(fit_kwargs):
    if isinstance(fit_kwargs, dict):
        batch_size = fit_kwargs.pop("batch_size")
    elif isinstance(fit_kwargs, list):
        batch_size = []
        for fkw in fit_kwargs:
            batch_size.append(fkw.pop("batch_size"))
    return batch_size


# %% public functions ##########################################################
def get_dataset(dataset_type, dataset_name, test_mode, fit_kwargs, **dataset_kwargs):
    if dataset_type == "benchmark":
        get_dataset_fn = get_benchmark_dataset
        get_dataset_kwargs = {
            "dataset_name": dataset_name,
            # "test_mode": test_mode,
        }
        batch_size = _get_batch_size(fit_kwargs)

        def mk_ds(data, batch_size):
            return _make_dataset(
                tf.data.Dataset.from_tensor_slices((tf.ones_like(data), data)),
                batch_size,
            )

        def preprocess_dataset(data, model) -> dict:
            if isinstance(batch_size, list):
                bs = batch_size
            else:
                bs = batch_size[1] if model.joint_trainable else batch_size[0]
            return {
                "x": mk_ds(data[0], bs),
                "validation_data": mk_ds(data[1], bs),
            }

    elif dataset_type == "malnutrition":
        get_dataset_fn = get_malnutrition_dataset
        get_dataset_kwargs = {
            **dataset_kwargs,
            "test_mode": test_mode,
        }
        batch_size = _get_batch_size(fit_kwargs)

        def mk_ds(data, batch_size):
            return _make_dataset(
                tf.data.Dataset.from_tensor_slices((data[0], data[1])),
                batch_size,
            )

        def preprocess_dataset(data, model) -> dict:
            if isinstance(batch_size, list):
                bs = batch_size
            else:
                bs = batch_size[1] if model.joint_trainable else batch_size[0]
            return {
                "x": mk_ds(data[0], bs),
                "validation_data": mk_ds(data[1], bs),
            }

    elif dataset_type == "sim":
        get_dataset_fn = get_sim_dataset
        get_dataset_kwargs = {
            **dataset_kwargs,
            "dataset_name": dataset_name,
            "test_mode": test_mode,
        }

        def preprocess_dataset(data, model) -> dict:
            return {
                "x": tf.convert_to_tensor(data[1], dtype=model.dtype),
                "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
            }
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    return get_dataset_fn, get_dataset_kwargs, preprocess_dataset
