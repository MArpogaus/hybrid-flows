"""Demostrate pipeline use."""
# %%

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from mctm.preprocessing import get_preprocessed_mnist_data

# TODO: why is run not found?
g, f, latentdim, ds_train_encoded, xmin, xmax, denom = get_preprocessed_mnist_data(
    "027463004629487ebee3ebc8e6a2cdcf"
)

data = list(tfds.as_numpy(ds_train_encoded))
X, Y = [], []
for d in data:
    X.append(d[0])
    Y.append(d[1])

X = np.array(X)
Y = np.array(Y)

print(X.shape, Y.shape)

# TODO: kw arguments aus parameters / DVC params
# https://dvc.org/doc/api-reference/params_show)
# factor out isometric autoencoder
# tox formatting

# %%
dist_keywords = {
    "dims": latentdim,
    "M": 20,
    "thetas_constrain_fn": tf.math.softplus,
    "xmin": xmin,
    "denom": denom,
}
M = dist_keywords["M"]
output_shape = latentdim * M + np.sum(np.arange(latentdim + 1))
# model, hist = pipeline(
#     __get_multivariate_normal_lambda__,
# dist_keywords, output_shape,
# ds_train_encoded,
# lambda x: x
# )


# def get_parameter_model(
#     input_shape, hidden_layers, activation, batch_norm, output_shape, dist_lambda
# ):
#     inputs = K.Input(input_shape)
#     if batch_norm:
#         inputs = K.layers.BatchNormalization(name="batch_norm")(inputs)
#     for i, h in enumerate(hidden_layers):
#         x = K.layers.Dense(h, activation=activation, name=f"hidden{i}")(inputs)
#     # x = K.layers.Dense(32, activation="relu", name="hidden2")(x)
#     pv = K.layers.Dense(output_shape, activation="linear", name="pv")(x)
#     dist = tfp.layers.DistributionLambda(dist_lambda)(pv)
#     param_model = K.Model(inputs=inputs, outputs=dist)
#     param_model.summary()
#     return param_model

#  -> get_simple_fully_connected_parameter_network_lambda -> bei conditional = true

# DensityRegressionModel mit multivariate normal und anderer input shape


# def build_model(ds):
#     model_params = {
#         "input_shape": (1,),
#         "hidden_layers": [16, 16],
#         "activation": "relu",
#         "batch_norm": False,
#     }
#     P = partial(get_parameter_model, **model_params)
#     model = P(
#         output_shape=output_shape,
#         dist_lambda=__get_multivariate_normal_lambda__(ds[0].shape[-1]),
#     )
#     return model


# pipeline(
#     experiment_name="mctm_demo",
#     run_name="demo",
#     results_path="results/demo",
#     log_file="mctm_demo_log.log",
#     test_mode=False,
#     seed=1,
#     get_dataset=lambda: (X, Y),
#     get_model=build_model,
#     fit_kwds={},
#     params=dist_keywords,
#     plot_data=None,
#     plot_samples=None,
# )
