"""Functions for preprocessing and loading.

The 'preprocessing' module provides functions for data preprocessing and
loading.

It includes functions for loading a pre-trained autoencoder model, creating
encoded datasets, normalizing images, and loading MNIST data.

This module is used to prepare data for further processing and modeling.
"""

import logging

import mlflow
import mlflow.pyfunc
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

logging.getLogger("tensorflow").setLevel(logging.NOTSET)


def load_pretrained_autoencoder_model(runid):
    """Load pre-trained autoencoder models from MLflow.

    :param str runid: The MLflow run ID associated with the models.
    :return: A tuple containing the generator (`g`), decoder (`f`), and latent
             dimension (`latentdim`).
    :rtype: tuple
    """
    g = mlflow.pyfunc.load_model(model_uri=f"runs:/{runid}/g")
    f = mlflow.pyfunc.load_model(model_uri=f"runs:/{runid}/f")
    latentdim = f.metadata.signature.inputs.to_dict()[0]["tensor-spec"]["shape"][1]

    return g, f, latentdim


def create_encoded_dataset(g, ds):  # -> module data -> file mnist
    """Create an encoded dataset using a generator (autoencoder) and a dataset.

    :param mlflow.pyfunc.PyFuncModel g: The generator model.
    :param tf.data.Dataset ds: The input dataset.
    :return: A tuple containing the encoded dataset, xmin, xmax, and denom.
    :rtype: tuple
    """
    _ys = []
    _xs = []
    # with io.capture_output() as captured:
    tf.get_logger().setLevel("ERROR")
    for x, y in tfds.as_numpy(ds):
        _ys.append(y.copy())
        _xs.append(
            g.predict(x).copy()
        )  # TODO: why does predict not recognize verbose=0?
    __xs = tf.data.Dataset.from_tensor_slices(
        (list(map(tf.constant, _xs[:-1])))
    )  # -1 last image is incomplete
    __ys = tf.data.Dataset.from_tensor_slices((list(map(tf.constant, _ys[:-1]))))
    ds_train_encoded = tf.data.Dataset.zip((__ys, __xs))  # .prefetch(1) # 1 batch size
    ds_train_encoded

    # min max skalierung im latenten raum zu 0 und 1
    xmax = np.concatenate(_xs, 0).max(0)
    xmax

    xmin = np.concatenate(_xs, 0).min(0)
    xmin

    denom = xmax - xmin
    ds_train_encoded_scaled = ds_train_encoded.map(lambda x, y: (x, (y - xmin) / denom))
    return ds_train_encoded_scaled, xmin, xmax, denom


def normalize_img(image, label):
    """Normalize images from `uint8` to `float32`.

    :param tf.Tensor image: The image tensor.
    :param label: The label associated with the image.
    :return: A tuple containing the normalized image and label.
    :rtype: tuple
    """
    return tf.cast(image, tf.float32) / 255.0, label


def load_mnist_data():
    """Load and preprocess the MNIST dataset.

    :return: A tuple containing training dataset, testing dataset,
             and dataset information.
    :rtype: tuple
    """
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples, seed=1)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_train

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128, drop_remainder=True)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    ds_test

    return ds_train, ds_test, ds_info


def get_preprocessed_mnist_data(encoder_id):
    """Load pretrained autoencoder.
    
    Load pre-trained autoencoder models and preprocessed MNIST data for
    experimentation.

    :param str encoder_id: The MLflow run ID associated with the
                          autoencoder model.
    :return: A tuple containing the generator (`g`), decoder (`f`), latent
             dimension (`latentdim`), encoded training dataset, xmin, xmax,
             and denom.
    :rtype: tuple
    """
    g, f, latentdim = load_pretrained_autoencoder_model(encoder_id)
    ds_train, ds_test, ds_info = load_mnist_data()
    ds_train_encoded, xmin, xmax, denom = create_encoded_dataset(g, ds_train)

    return g, f, latentdim, ds_train_encoded, xmin, xmax, denom
