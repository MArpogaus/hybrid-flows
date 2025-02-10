# %% Description ###############################################################
"""Provides data loaders for different datasets."""

# %% imports ##########################################################################
import tensorflow as tf

from .benchmark import load_data as load_benchmark_data
from .malnutrion import load_data as load_malnutrition_data
from .sklearn_datasets import gen_data as get_sim_data


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
    else:
        batch_size = 32
    return batch_size


def _get_preprocess_dataset(batch_size, get_dataset_fn):
    def preprocess_data(data, model) -> dict:
        if isinstance(batch_size, list):
            bs = batch_size[1] if model.joint_trainable else batch_size[0]
        else:
            bs = batch_size
        return {
            "x": get_dataset_fn(data[0], bs),
            "validation_data": get_dataset_fn(data[1], bs),
        }

    return preprocess_data


# %% public functions ##########################################################
def make_malnutrition_dataset(data, batch_size):
    """Return malnutrition data as TensorFlow dataset."""
    return _make_dataset(
        tf.data.Dataset.from_tensor_slices((tf.squeeze(data[0]), data[1])),
        batch_size,
    )


def make_benchmark_dataset(data, batch_size):
    """Return benchmark data as TensorFlow dataset."""
    return _make_dataset(
        tf.data.Dataset.from_tensor_slices((tf.ones_like(data), data)),
        batch_size,
    )


def get_dataset(
    dataset_type, dataset_name, test_mode, fit_kwargs=None, **dataset_kwargs
):
    """Load dataset of given type and prepare it for training."""
    if dataset_type == "benchmark":
        get_data_fn = load_benchmark_data
        get_data_fn_kwargs = {
            "dataset_name": dataset_name,
            # "test_mode": test_mode,
        }
        batch_size = _get_batch_size(fit_kwargs)

        preprocess_data_fn = _get_preprocess_dataset(batch_size, make_benchmark_dataset)

    elif dataset_type == "malnutrition":
        get_data_fn = load_malnutrition_data
        get_data_fn_kwargs = {
            **dataset_kwargs,
            "test_mode": test_mode,
        }
        batch_size = _get_batch_size(fit_kwargs)

        preprocess_data_fn = _get_preprocess_dataset(
            batch_size, make_malnutrition_dataset
        )

    elif dataset_type == "sim":
        get_data_fn = get_sim_data
        get_data_fn_kwargs = {
            **dataset_kwargs,
            "dataset_name": dataset_name,
            "test_mode": test_mode,
        }

        def preprocess_data_fn(data, model) -> dict:
            return {
                "x": tf.convert_to_tensor(data[1], dtype=model.dtype),
                "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
            }
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    return get_data_fn, get_data_fn_kwargs, preprocess_data_fn
