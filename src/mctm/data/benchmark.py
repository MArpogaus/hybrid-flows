"""Access functions for benchmark datasets."""

import os

import numpy as np


def load_data(dataset_name, dataset_path="datasets/benchmark", test_mode=False):
    """Provide access to the specified dataset.

    :param str dataset_name: Name of the dataset (e.g., "POWER", "GAS", etc.).
    :return: Tuple containing the training, validation, and test data arrays.
    :rtype: tuple
    """
    train_data, validation_data, test_data = (
        np.load(os.path.join(dataset_path, dataset_name.lower() + "_train.npy")),
        np.load(os.path.join(dataset_path, dataset_name.lower() + "_validate.npy")),
        np.load(os.path.join(dataset_path, dataset_name.lower() + "_test.npy")),
    )

    dims = train_data.shape[-1]

    if test_mode:
        return (train_data[:1000], validation_data[:1000], test_data[:1000]), dims
    else:
        return (train_data, validation_data, test_data), dims


if __name__ == "__main__":
    import yaml

    shift_and_scale = {}
    for dataset_name in ("POWER", "GAS", "HEPMASS", "MINIBOONE", "BSDS300"):
        print(f"=== {dataset_name} ===")
        (train_data, validation_data, _), dims = load_data(dataset_name)
        print(f"{dims=}")
        for d, split in zip((train_data, validation_data), ("train", "validate")):
            print(f"--- {split} ---")
            print(f"{type(d)=}")
            print(f"{d.shape=}")
            print(f"{d.min()=}")
            print(f"{d.max()=}")
            print(f"{d.std()=}")
            print(f"{d.mean()=}")

        d = train_data
        # data_min = np.min([train_data.min(), validation_data.min()])
        # data_max = np.max([train_data.max(), validation_data.max()])
        data_min = np.min([train_data.min(0), validation_data.min(0)], 0)
        data_max = np.max([train_data.max(0), validation_data.max(0)], 0)

        eps = 0.1
        domain_min = np.floor(data_min).astype(float) - eps
        domain_max = np.ceil(data_max).astype(float) + eps
        print(f"{domain_min=}")
        print(f"{domain_max=}")

        shift = (-domain_min + eps).round(3)
        scale = ((1 - 2 * eps) / (domain_max - domain_min)).round(3)

        print(f"{shift=}")
        print(f"{scale=}")

        scaled_train_data = (train_data + shift) * scale
        scaled_val_data = (validation_data + shift) * scale

        print(scaled_train_data.min(), scaled_train_data.max())
        print(scaled_val_data.min(), scaled_val_data.max())

        assert scaled_train_data.min() >= 0
        assert scaled_train_data.max() <= 1
        assert scaled_val_data.min() >= 0
        assert scaled_val_data.max() <= 1

        print(f"domain: {[domain_min.tolist(), domain_max.tolist()]}")

        shift_and_scale[dataset_name.lower()] = dict(
            scale=scale.tolist(),
            shift=shift.tolist(),
        )

    print(yaml.safe_dump({"benchmark_datasets": shift_and_scale}))
