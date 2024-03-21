# %% import

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from mctm.models import DensityRegressionModel
from mctm.utils import pipeline
from mctm.utils.tensorflow import fit_distribution, set_seed
from mctm.utils.visualisation import (
    get_figsize,
    plot_2d_data,
    plot_flow,
    plot_samples,
    setup_latex,
)
from mctm.parameters import get_parameter_vector_fn
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    tensor_util,
)


# %% functions
def gen_data(t, scale):
    t1 = t
    y1 = 1.0 * t1
    y1 += np.random.normal(0, 0.05 * np.abs(t1))

    t2 = t
    y2 = -0.2 * t2
    y2 += np.random.normal(0, 0.2 * np.abs(t2))

    t = np.concatenate([t1, t2])
    y = np.concatenate([y1, y2])

    if scale:
        y_min = y.min()
        y_max = y.max()
        y -= y_min
        y /= y_max - y_min
    return t[..., np.newaxis], y[..., np.newaxis]


def gen_test_data(n_samples, observations, **kwds):
    t = np.linspace(0, 1, n_samples, dtype=np.float32)
    t = np.repeat([t], observations)

    return gen_data(t, **kwds)


def gen_train_data(n_samples, **kwds):
    t = np.random.uniform(0, 1, n_samples).astype(np.float32)

    return gen_data(t, **kwds)


def nll_loss(y, dist):
    return -dist.log_prob(y)


def thetas_constrain_fn(diff):
    fn = tf.abs

    dtype = dtype_util.common_dtype([diff], dtype_hint=tf.float32)
    shift = tf.math.log(2.0) * tf.cast(prefer_static.shape(diff)[-1], dtype=dtype) / 2

    diff = tensor_util.convert_nonref_to_tensor(diff, name="diff", dtype=dtype)
    low_theta = diff[..., :1]
    diff = diff[..., 1:] + 1e-6

    diff_positive = fn(diff)
    c = tf.concat(
        (
            low_theta,
            diff_positive,
        ),
        axis=-1,
    )
    thetas_constrained = tf.cumsum(c, axis=-1, name="theta") - shift

    return thetas_constrained


def plot_dists(dist, test_x, test_t, test_y):
    # breakpoint()
    yy = np.linspace(-1, 1.5, 1000, dtype=np.float32)[..., None, None]
    ps = dist.prob(yy)

    fig, ax = plt.subplots(
        len(test_x), figsize=(4, len(test_x) * 3), constrained_layout=True
    )
    fig.suptitle("Learned Distributions", fontsize=16)

    for i, x in enumerate(test_x):
        # breakpoint()
        ax[i].set_title(f"x={x}")
        sampl = test_y[(test_t.flatten() == x)].flatten()
        ax[i].scatter(sampl, [0] * len(sampl), marker="|")
        ax[i].plot(yy.flatten(), ps[:, i], label="flow")
        ax[i].set_xlabel("y")
        ax[i].set_ylabel(f"p(y|x={x})")
        ax[i].legend()
    return fig


# %% Params

seed = 1
distribution = "elementwise_flow"
dataset_kwds = {"n_samples": 2000, "scale": True}
distribution_kwds = {
    "bijector_name": "bernstein_poly",
    "order": 20,
    "low": -4,
    "high": 4,
    "eps": 0.001,
    "smooth_bounds": False,
    "allow_flexible_bounds": False,
    "shift": False,
    "scale": False,
    "fn": tf.abs
    # "base_distribution_kwds": {"distribution_type": "uniform", "low": 0, "high": 1},
}
distribution_kwds = {
    "bijector_name": "bernstein_poly",
    "order": 50,
    "shift": False,
    "scale": False,
    "parameter_constrain_fn": thetas_constrain_fn
    # "base_distribution_kwds": {"distribution_type": "uniform", "low": 0, "high": 1},
}
parameter_kwds = {"dtype": "float", "polynomial_order": 1, "conditional_event_dims": 1}
fit_kwds = {
    "epochs": 2000,
    "validation_split": 0.1,
    "batch_size": 128,
    "learning_rate": 0.001,
    # tf.keras.optimizers.schedules.CosineDecayRestarts(
    #    initial_learning_rate=0.01, first_decay_steps=20
    # ),
    "lr_patience": 50,
    "reduce_lr_on_plateau": False,
    "early_stopping": True,
    "verbose": True,
    "monitor": "val_loss",
}

# %%
model_kwds = dict(
    distribution=distribution,
    distribution_kwds=distribution_kwds,
    parameter_kwds=parameter_kwds,
)

get_model = DensityRegressionModel

preprocess_dataset = lambda data, model: {
    "x": tf.convert_to_tensor(data[0], dtype=model.dtype),
    "y": tf.convert_to_tensor(data[1], dtype=model.dtype),
}
results_path = "./results/" + distribution


# %% Load data
set_seed(seed)
dims = 1
data = gen_train_data(**dataset_kwds)

# %%
plt.scatter(*data)


# %% Init Model
def get_polynomial_parameter_lambda(
    parameter_shape, polynomial_order, conditional_event_dims, dtype
):
    parameter_shape = [
        conditional_event_dims,
        polynomial_order + 1,
    ] + parameter_shape
    _, parameter_vector = get_parameter_vector_fn(
        parameters_shape=parameter_shape, dtype=dtype
    )

    def get_parameter_lambda(conditional_input=None, **kwds):
        xx = tf.stack(
            [conditional_input**i for i in range(0, polynomial_order + 1)], -1
        )
        yy = xx[..., None, None] * parameter_vector
        return lambda *_: tf.reduce_sum(yy, axis=[1, 2])

    return get_parameter_lambda, parameter_vector


model = DensityRegressionModel(
    dims=dims, get_parameter_lambda_fn=get_polynomial_parameter_lambda, **model_kwds
)

# execute training
preprocessed = preprocess_dataset(data, model)

model.

# %%
hist = fit_distribution(
    model=model,
    seed=seed,
    results_path=results_path,
    loss=nll_loss,
    **preprocessed,
    **fit_kwds,
)

pd.DataFrame(hist.history).plot()

# %% Samples
x, y = preprocessed.values()
dist = model(x)

samples = dist.sample(seed=1).numpy()

plt.scatter(x, samples.flatten())
plt.scatter(*data)

# %%
model.trainable_parameters

# %%
from bernstein_flow.activations import get_thetas_constrain_fn

# tcf = get_thetas_constrain_fn(
#     low=distribution_kwds["low"],
#     high=distribution_kwds["high"],
#     smooth_bounds=distribution_kwds["smooth_bounds"],
#     allow_flexible_bounds=distribution_kwds["allow_flexible_bounds"],
#     eps=distribution_kwds["eps"],
#     fn=distribution_kwds["fn"],
# )
tcf = thetas_constrain_fn

# %%
t = np.linspace(0.0, 1.0, 200, dtype="float32")
pv_u = model.parameter_fn(t[..., None])().numpy().squeeze()
pv = tcf(pv_u)

# %%
fig = plt.plot(t, pv_u)

# %%
fig = plt.plot(t, pv)


# %% Plot dist
test_t, test_y = gen_test_data(5, 200, scale=True)
test_x = np.unique(test_t)


plot_dists(model(test_x[..., None]), test_x, test_t, test_y)
model.parameter_fn(test_x[..., None])()
model.trainable_parameters
