"""Provide access to malnutrition dataset."""

from functools import partial

import pandas as pd
import tensorflow as tf
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mctm.utils.tensorflow import set_seed


def get_dataset(
    data_path,
    targets,
    scale=True,
    test_size=0.1,
    val_size=0.1,
    covariates=None,
    stratify=False,
    dtype=tf.float32,
    seed=1,
    column_transformers=[],
    test_mode=False,
):
    """Load and preprocesse a dataset for training, validation, and testing.

    :param str data_path: The path to the dataset file.
    :param list targets: List of target column names.
    :param float test_size: The proportion of the dataset to include in the
    test split.
    :param float val_size: The proportion of the dataset to include in the
    validation split.
    :param list covariates: List of covariate column names. If None, inferred
    from the dataset.
    :param tf.dtypes.DType dtype: The data type for the tensors.
    :param int seed: Random seed for reproducibility.
    :return: Tuple containing the training, validation, and test data tensors
    along with the number of targets.
    :rtype: tuple
    """
    # Load dataset
    data = pd.read_csv(data_path, sep=r"\s+")

    # Infer covariates if not provided
    if covariates is None:
        covariates = data.columns[~data.columns.isin(targets)].to_list()
        print(f"{covariates=}")

    if test_mode:
        data = data.groupby(covariates).sample(frac=0.2, random_state=1)

    # Split the dataset into train, validation, and test sets
    set_seed(seed)
    train_val_data, test_data = train_test_split(
        data,
        test_size=test_size,
        shuffle=True,
        stratify=data[covariates] if stratify else None,
    )
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size,
        shuffle=True,
        stratify=train_val_data[covariates] if stratify else None,
    )

    # Apply data scaling using column transformer
    if scale:
        column_transformers = [(StandardScaler(), targets)] + column_transformers
        ct = make_column_transformer(
            *column_transformers,
            remainder=StandardScaler(),
            verbose_feature_names_out=False,
        )
        ct.set_output(transform="pandas")

        train_data = ct.fit_transform(train_data)
        val_data = ct.transform(val_data)
        test_data = ct.transform(test_data)

    # Ensure no overlap in indices between sets
    assert not train_data.index.isin(test_data.index).any()
    assert not train_data.index.isin(val_data.index).any()
    assert not val_data.index.isin(test_data.index).any()

    # Convert data to TensorFlow tensors
    t = partial(tf.convert_to_tensor, dtype=dtype)

    train_x, train_y = t(train_data[covariates]), t(train_data[targets])
    val_x, val_y = t(val_data[covariates]), t(val_data[targets])
    test_x, test_y = t(test_data[covariates]), t(test_data[targets])

    return ((train_x, train_y), (val_x, val_y), (test_x, test_y)), len(targets)
