from mctm.distributions import multivariate_normal_lambda, mctm_lambda
from mctm.utils import pipeline
from mctm.preprocessing import get_preprocessed_mnist_data

#TODO: why is run not found?
g, f, latentdim, ds_train_encoded, xmin, xmax, denom = get_preprocessed_mnist_data("027463004629487ebee3ebc8e6a2cdcf")

# TODO: kw arguments aus parameters / DVC params (https://dvc.org/doc/api-reference/params_show)
# factor out isometric autoencoder
# tox formatting

# mail wenn läuft

model, hist =  pipeline(mctm_lambda, latentdim, ds_train_encoded, lambda x: x)