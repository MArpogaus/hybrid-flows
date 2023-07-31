import mlflow
import mlflow.pyfunc
from mlflow import log_artifacts, log_figure, log_metric, log_param, log_params
from mlflow.models import infer_signature
from mlflow.keras import autolog, log_model
import tensorflow_datasets as tfds
import numpy as np
import logging
logging.getLogger("tensorflow").setLevel(logging.NOTSET)
import tensorflow as tf

import os

def load_pretrained_autoencoder_model(runid):
    g = mlflow.pyfunc.load_model(model_uri=f"runs:/{runid}/g")
    f = mlflow.pyfunc.load_model(model_uri=f"runs:/{runid}/f")
    latentdim = f.metadata.signature.inputs.to_dict()[0]['tensor-spec']['shape'][1]
    
    return g, f, latentdim

def create_encoded_dataset(g, ds): #-> module data -> file mnist
    _ys = []
    _xs = []
    #with io.capture_output() as captured:
    tf.get_logger().setLevel('ERROR')
    for (x, y) in tfds.as_numpy(ds):
        _ys.append(y.copy())
        _xs.append(g.predict(x).copy()) #TODO: why does predict not recognize verbose=0?
    __xs = tf.data.Dataset.from_tensor_slices((list(map(tf.constant,_xs[:-1])))) # -1 last image is incomplete
    __ys = tf.data.Dataset.from_tensor_slices((list(map(tf.constant,_ys[:-1]))))
    ds_train_encoded = tf.data.Dataset.zip((__ys,__xs))#.prefetch(1) # 1 batch size
    ds_train_encoded
    
    # min max skalierung im latenten raum zu 0 und 1
    xmax = np.concatenate(_xs, 0).max(0)
    xmax
    
    xmin = np.concatenate(_xs, 0).min(0)
    xmin
    
    denom = xmax-xmin
    ds_train_encoded_scaled = ds_train_encoded.map(lambda x,y: (x,(y-xmin)/denom))
    return ds_train_encoded_scaled, xmin, xmax, denom

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label

def load_mnist_data():
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
    g, f, latentdim = load_pretrained_autoencoder_model(encoder_id)
    ds_train, ds_test, ds_info = load_mnist_data()
    ds_train_encoded, xmin, xmax, denom = create_encoded_dataset(g, ds_train)
    
    return g, f, latentdim, ds_train_encoded, xmin, xmax, denom