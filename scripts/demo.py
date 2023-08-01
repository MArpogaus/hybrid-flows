import numpy as np
import tensorflow as tf

from mctm.distributions import mctm_lambda
from mctm.preprocessing import get_preprocessed_mnist_data
from mctm.utils import pipeline

# TODO: why is run not found?
g, f, latentdim, ds_train_encoded, xmin, xmax, denom = get_preprocessed_mnist_data(
    "027463004629487ebee3ebc8e6a2cdcf"
)

# TODO: kw arguments aus parameters / DVC params
# https://dvc.org/doc/api-reference/params_show)
# factor out isometric autoencoder
# tox formatting

# mail wenn läuft
dist_keywords = {
    "dims": latentdim,
    "M": 20,
    "thetas_constrain_fn": tf.math.softplus,
    "xmin": xmin,
    "denom": denom,
}
M = dist_keywords["M"]
output_shape = latentdim * M + np.sum(np.arange(latentdim + 1))
model, hist = pipeline(
    mctm_lambda, dist_keywords, output_shape, ds_train_encoded, lambda x: x
)
