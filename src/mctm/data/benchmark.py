"""Access functions for benchmark datasets."""
import hashlib
import os
import tarfile
import urllib.request
from collections import Counter

import h5py
import numpy as np
import pandas as pd

# Define the URLs for downloading the preprocessed data archive and its MD5 checksum
DATA_ARCHIVE_URL = "https://zenodo.org/record/1161203/files/data.tar.gz"
MD5_CHECKSUM = "9b9c9b0375315ad270eba4ce80c093ab"


def download_and_verify_data(download_path):
    """Download the preprocessed data archive and verifies its MD5 checksum.

    :return: Path to the downloaded data archive.
    :rtype: str
    """
    # Create the dataset directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # Calculate MD5 checksum of the downloaded file
    md5 = hashlib.md5()
    data_archive_file = os.path.join(download_path, "data.tar.gz")

    # Download the data archive if it doesn't exist
    if not os.path.exists(data_archive_file):
        print("Downloading data archive...")
        urllib.request.urlretrieve(DATA_ARCHIVE_URL, data_archive_file)

    # Verify MD5 checksum
    with open(data_archive_file, "rb") as f:
        while chunk := f.read(8192):
            md5.update(chunk)

    if md5.hexdigest() != MD5_CHECKSUM:
        raise ValueError("MD5 checksum does not match.")

    return data_archive_file


def extract_dataset(dataset_name, dataset_path):
    """Extract dataset."""
    data_archive_file = download_and_verify_data(dataset_path)
    dataset_dir = os.path.join(dataset_path, dataset_name.lower())

    # Extract the dataset if it hasn't been extracted yet
    if not os.path.exists(dataset_dir) or (
        os.path.exists(dataset_dir)
        and not list(
            filter(lambda fn: not fn.endswith(".dvc"), os.listdir(dataset_dir))
        )
    ):
        print(f"Extracting {dataset_name} dataset...")
        # Define the allowed file extensions
        allowed_extensions = (".csv", ".pickle", ".npy", ".hdf5")
        # Define the prefix for files to be extracted
        file_prefix = f"DATA/{dataset_name}"

        # Open the data archive
        with tarfile.open(data_archive_file, "r:gz") as tar:
            # Iterate through files in the archive
            for member in tar.getmembers():
                if (
                    member.isfile()
                    and member.name.upper().startswith(file_prefix)
                    and "/." not in member.name
                    and member.name.endswith(allowed_extensions)
                ):
                    # Extract the file to the dataset directory
                    member.name = os.path.basename(member.name)  # Remove leading path
                    print(f"Extracting {member.name} to {dataset_dir}...", end="")
                    tar.extract(member, path=dataset_dir)
                    print("done.")

    return dataset_dir


def load_and_preprocess_data(dataset_name, dataset_path):
    """Load and preprocesse the specified dataset.

    :param str dataset_name: Name of the dataset (e.g., "POWER", "GAS", etc.).
    :return: Tuple containing the training, validation, and test data arrays.
    :rtype: tuple
    """
    dataset_name = dataset_name.upper()

    # Load the data and apply preprocessing as per the original code
    if dataset_name == "POWER":
        dataset_dir = extract_dataset(dataset_name, dataset_path)
        return load_and_process_power_data(dataset_dir)
    elif dataset_name == "GAS":
        dataset_dir = extract_dataset(dataset_name, dataset_path)
        return load_and_process_gas_data(dataset_dir)
    elif dataset_name == "HEPMASS":
        dataset_dir = extract_dataset(dataset_name, dataset_path)
        return load_and_process_hepmass_data(dataset_dir)
    elif dataset_name == "MINIBOONE":
        dataset_dir = extract_dataset(dataset_name, dataset_path)
        return load_and_process_miniboone_data(dataset_dir)
    elif dataset_name == "BSDS300":
        dataset_dir = extract_dataset(dataset_name, dataset_path)
        return load_and_process_bsds300_data(dataset_dir)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")


def load_and_process_power_data(dataset_dir):
    """Load data."""
    # https://github.com/gpapamak/maf/blob/master/datasets/power.py

    def load_data_split_with_noise(data):
        rng = np.random.RandomState(42)

        rng.shuffle(data)
        N = data.shape[0]

        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        ############################
        # Add noise
        ############################
        # global_intensity_noise = 0.1*rng.rand(N, 1)
        voltage_noise = 0.01 * rng.rand(N, 1)
        # grp_noise = 0.001*rng.rand(N, 1)
        gap_noise = 0.001 * rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data = data + noise

        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised(data):
        data_train, data_validate, data_test = load_data_split_with_noise(data)
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_validate, data_test

    data = np.load(dataset_dir + "/data.npy")
    data_train, data_validate, data_test = load_data_normalised(data)
    return (data_train, data_validate, data_test), data_train.shape[-1]


def load_and_process_gas_data(dataset_dir):
    """Load data."""

    # https://github.com/gpapamak/maf/blob/master/datasets/gas.py
    def load_data(file):
        data = pd.read_pickle(file)
        # data = pd.read_pickle(file).sample(frac=0.25)
        # data.to_pickle(file)
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)
        return data

    def get_correlation_numbers(data):
        C = data.corr()
        A = C > 0.98
        B = A.as_matrix().sum(axis=1)
        return B

    def load_data_and_clean(file):
        data = load_data(file)
        B = get_correlation_numbers(data)

        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = get_correlation_numbers(data)
        # print(data.corr())
        data = (data - data.mean()) / data.std()

        return data

    def load_data_and_clean_and_split(file):
        data = load_data_and_clean(file).as_matrix()
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data_train = data[0:-N_test]
        N_validate = int(0.1 * data_train.shape[0])
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test

    file = dataset_dir + "/ethylene_CO.pickle"
    data_train, data_validate, data_test = load_data_and_clean_and_split(file)
    return (data_train, data_validate, data_test), data_train.shape[-1]


def load_and_process_hepmass_data(dataset_dir):
    """Load data."""

    # https://github.com/gpapamak/maf/blob/master/datasets/hepmass.py
    # https://github.com/bayesiains/nsf/blob/master/data/hepmass.py
    def load_data(path):
        data_train = pd.read_csv(
            filepath_or_buffer=os.path.join(path, "1000_train.csv"), index_col=False
        )
        data_test = pd.read_csv(
            filepath_or_buffer=os.path.join(path, "1000_test.csv"), index_col=False
        )

        return data_train, data_test

    def load_data_no_discrete(path):
        """Load the positive class examples from the first 10% of the dataset."""
        data_train, data_test = load_data(path)

        # Gets rid of any background noise examples i.e. class label 0.
        data_train = data_train[data_train[data_train.columns[0]] == 1]
        data_train = data_train.drop(data_train.columns[0], axis=1)
        data_test = data_test[data_test[data_test.columns[0]] == 1]
        data_test = data_test.drop(data_test.columns[0], axis=1)
        # Because the data_ set is messed up!
        data_test = data_test.drop(data_test.columns[-1], axis=1)

        return data_train, data_test

    def load_data_no_discrete_normalised(path):
        data_train, data_test = load_data_no_discrete(path)
        mu = data_train.mean()
        s = data_train.std()
        data_train = (data_train - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_test

    def load_data_no_discrete_normalised_as_array(path):
        data_train, data_test = load_data_no_discrete_normalised(path)
        data_train, data_test = data_train.values, data_test.values

        i = 0
        # Remove any features that have too many re-occurring real values.
        features_to_remove = []
        for feature in data_train.T:
            c = Counter(feature)
            max_count = np.array([v for k, v in sorted(c.items())])[0]
            if max_count > 5:
                features_to_remove.append(i)
            i += 1
        data_train = data_train[
            :,
            np.array(
                [i for i in range(data_train.shape[1]) if i not in features_to_remove]
            ),
        ]
        data_test = data_test[
            :,
            np.array(
                [i for i in range(data_test.shape[1]) if i not in features_to_remove]
            ),
        ]

        N = data_train.shape[0]
        N_validate = int(N * 0.1)
        data_validate = data_train[-N_validate:]
        data_train = data_train[0:-N_validate]

        return data_train, data_validate, data_test

    path = dataset_dir
    data_train, data_validate, data_test = load_data_no_discrete_normalised_as_array(
        path
    )
    return (data_train, data_validate, data_test), data_train.shape[-1]


def load_and_process_miniboone_data(dataset_dir):
    """Load data."""

    def load_data(root_path):
        data = np.load(root_path)
        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised(root_path):
        data_train, data_validate, data_test = load_data(root_path)
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_validate, data_test

    file = dataset_dir + "/data.npy"
    data_train, data_validate, data_test = load_data_normalised(file)
    return (data_train, data_validate, data_test), data_train.shape[-1]


def load_and_process_bsds300_data(dataset_dir):
    """Load data."""
    # load dataset
    f = h5py.File(dataset_dir + "/BSDS300.hdf5", "r")

    data_train, data_validate, data_test = (
        f["train"][...],
        f["validation"][...],
        f["test"][...],
    )
    return (data_train, data_validate, data_test), data_train.shape[-1]


def get_dataset(dataset_name, dataset_path="datasets"):
    """Provide access to the specified dataset.

    :param str dataset_name: Name of the dataset (e.g., "POWER", "GAS", etc.).
    :return: Tuple containing the training, validation, and test data arrays.
    :rtype: tuple
    """
    return load_and_preprocess_data(dataset_name, dataset_path)


def compute_shift_and_scale(datasets=["POWER", "HEPMASS", "MINIBOONE", "BSDS300"]):
    shift_and_scale = {}
    for dataset_name in datasets:
        print(f"=== {dataset_name} ===")
        (train_data, validation_data, _), dims = get_dataset(dataset_name)
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
        eps = 0.01
        data_min = np.min([train_data.min(0), validation_data.min(0)], 0)
        data_max = np.max([train_data.max(0), validation_data.max(0)], 0)
        shift = (-data_min + eps).round(3)
        scale = ((1 - 2 * eps) / (data_max - data_min)).round(3)

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

        shift_and_scale[dataset_name.lower()] = dict(
            scale=scale.tolist(), shift=shift.tolist()
        )

    return shift_and_scale


if __name__ == "__main__":
    import yaml

    shift_and_scale = compute_shift_and_scale()

    print(yaml.safe_dump({"benchmark_datasets": shift_and_scale}))
