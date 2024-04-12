# %% import
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from mctm.models import DensityRegressionModel
from mctm.parameters import get_parameter_vector_fn
from mctm.utils.tensorflow import fit_distribution, set_seed
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    tensor_util,
)


# %% functions
def gen_data(t):
    x = t + 0.3
    y = 0.3 * x
    y1 = y + np.random.normal(0.3, 0.05 * np.ones_like(t)) * 0.5 * x
    y2 = y + np.random.normal(-0.3, 0.1 * np.ones_like(t)) * 0.5 * x

    t = np.concatenate([t, t])
    y = np.concatenate([y1, y2])

    return t[..., np.newaxis], y[..., np.newaxis]


def gen_test_data(n_samples, observations, **kwargs):
    t = np.linspace(0, 1, n_samples, dtype=np.float32)
    t = np.repeat(t, observations)

    return gen_data(t, **kwargs)


def gen_train_data(n_samples, **kwargs):
    t = np.random.uniform(0, 1, n_samples).astype(np.float32)

    return gen_data(t, **kwargs)


def nll_loss(y, dist):
    return -dist.log_prob(y)


def thetas_constrain_fn(diff):
    fn = tf.abs

    dtype = dtype_util.common_dtype([diff], dtype_hint=tf.float32)
    eps = dtype_util.eps(dtype)

    # shift = tf.math.log(2.0) * tf.cast(prefer_static.shape(diff)[-1], dtype=dtype) / 2

    diff = tensor_util.convert_nonref_to_tensor(diff, name="diff", dtype=dtype)
    low_theta = diff[..., :1]
    diff = diff[..., 1:]

    diff_positive = fn(diff) + eps
    c = tf.concat(
        (low_theta, diff_positive[..., :1], diff_positive, diff_positive[..., -1:]),
        axis=-1,
    )
    thetas_constrained = tf.cumsum(c, axis=-1, name="theta")  # - shift

    return thetas_constrained


def thetas_constrain_fn2(diff):
    fn = tf.abs
    dtype = dtype_util.common_dtype([diff], dtype_hint=tf.float32)
    eps = dtype_util.eps(dtype)

    diff = tensor_util.convert_nonref_to_tensor(diff, name="diff", dtype=dtype)
    low_theta = -4 * tf.ones_like(diff[..., :1])

    diff_positive = fn(diff)
    diff_positive /= tf.reduce_sum(diff_positive, axis=-1)[..., None]
    diff_positive *= 8 - tf.cast(prefer_static.shape(diff)[-1], dtype=dtype) * eps
    c = tf.concat(
        (
            low_theta,
            diff_positive + eps,
        ),
        axis=-1,
    )
    thetas_constrained = tf.cumsum(c, axis=-1, name="theta")

    return thetas_constrained


def get_polynomial_parameter_fn(
    parameter_shape, polynomial_order, conditional_event_shape, dtype
):
    parameter_shape = [
        conditional_event_shape,
        polynomial_order + 1,
    ] + parameter_shape
    _, parameter_vector = get_parameter_vector_fn(
        parameter_shape=parameter_shape,
        dtype=dtype,  #  initializer=tf.ones
    )

    def parameter_fn(conditional_input=None, **kwargs):
        xx = tf.stack(
            [conditional_input**i for i in range(0, polynomial_order + 1)], -1
        )
        yy = xx[..., None, None] * parameter_vector
        # \vartheta_i = a_i*x^3+b_i*x^2...
        return tf.reduce_sum(yy, axis=[-4, -3])

    return parameter_fn, parameter_vector


def get_additive_parameter_fn(parameter_shape, conditional_event_shape, dtype):
    dims = parameter_shape[0]
    order = parameter_shape[1] - 2
    _, theta_parameter_vector = get_parameter_vector_fn(
        parameter_shape=[dims, order],
        dtype=dtype,  #  initializer=tf.ones
    )
    (
        scale_and_shift_parameter_fn,
        scale_and_shift_parameters,
    ) = get_polynomial_parameter_fn([dims, 2], 1, conditional_event_shape, dtype)

    def parameter_fn(conditional_input, **kwargs):
        return tf.concat(
            [
                scale_and_shift_parameter_fn(conditional_input, **kwargs),
                tf.ones_like(conditional_input)[..., None]
                * theta_parameter_vector[None, ...],
            ],
            -1,
        )

    return parameter_fn, [theta_parameter_vector, scale_and_shift_parameters]


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
epochs = 100
seed = 1
distribution = "elementwise_flow"
dataset_kwargs = {"n_samples": 2**14}
# distribution_kwargs = {
#     "bijector_name": "bernstein_poly",
#     "order": 20,
#     "low": -4,
#     "high": 4,
#     "eps": 0.001,
#     "smooth_bounds": False,
#     "allow_flexible_bounds": False,
#     "shift": False,
#     "scale": False,
#     "fn": tf.abs,
#     # "base_distribution_kwargs": {"distribution_type": "uniform", "low": 0, "high": 1},
# }
distribution_kwargs = {
    "bijector_name": "bernstein_poly",
    "order": 25,
    "shift": True,
    "scale": True,
    "parameter_constrain_fn": thetas_constrain_fn2,
    # "base_distribution_kwargs": {"distribution_type": "uniform", "low": 0, "high": 1},
}
parameter_kwargs = {
    "dtype": "float",
    # "polynomial_order": 1,
    "conditional_event_shape": 1,
    # "conditional": True,
    # "hidden_units": [16, 16],
    # "activation": "relu",
    # "batch_norm": False,
    # "dropout": False,
}
initial_learning_rate = 0.01
scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate,
    decay_steps=epochs,
    end_learning_rate=0.00001,
    power=1,
)
fit_kwargs = {
    "epochs": epochs,
    "validation_split": 0.25,
    "batch_size": 128,
    "learning_rate": initial_learning_rate,
    "callbacks": [tf.keras.callbacks.LearningRateScheduler(scheduler)],
    "lr_patience": 50,
    "reduce_lr_on_plateau": False,
    "early_stopping": False,
    "verbose": True,
    "monitor": "val_loss",
}
model_kwargs = dict(
    distribution=distribution,
    distribution_kwargs=distribution_kwargs,
    parameter_kwargs=parameter_kwargs,
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
data = gen_train_data(**dataset_kwargs)
test_t, test_y = gen_test_data(5, 200)
test_x = np.unique(test_t)

fig = plt.figure()
plt.scatter(*data, alpha=0.05, label="train")
plt.scatter(test_t.flatten(), test_y.flatten(), alpha=0.05, label="test")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Simulated Data")
plt.legend()
fig.savefig("./org/gfx/1d_sim_data.png")

# %% Init Model
model = DensityRegressionModel(
    dims=dims,
    # get_parameter_fn=get_polynomial_parameter_fn,
    get_parameter_fn=get_additive_parameter_fn,
    **model_kwargs,
)


# %% inital values
tcf = distribution_kwargs["parameter_constrain_fn"]

t = np.linspace(0.0, 1.0, 200, dtype="float32")
pv_u = model.parameter_fn(t[..., None]).numpy().squeeze()
pv_poly = tcf(pv_u[..., 2:])

fig, axs = plt.subplots(3, sharex=True)
axs[0].plot(t, pv_u)
axs[0].set_title("unconstrained")
axs[1].plot(t, pv_poly)
axs[1].set_title("constrained pv poly")
axs[2].plot(t, tf.math.abs(pv_u[..., 0]), label="scale")
axs[2].plot(t, pv_u[..., 1], label="shift")
axs[2].set_title("constrained scale and shift")
axs[2].legend()
fig.tight_layout()

# %% fit model
preprocessed = preprocess_dataset(data, model)
hist = fit_distribution(
    model=model,
    seed=seed,
    results_path=results_path,
    loss=nll_loss,
    **preprocessed,
    **fit_kwargs,
)

# %% Learning curve
pd.DataFrame(hist.history).plot()

# %% Samples
x, y = preprocessed.values()
set_seed(1)
idx = np.random.permutation(len(x))[:3000]
x, y = x.numpy()[idx], y.numpy()[idx]
dist = model(x)

samples = dist.sample(seed=1).numpy()

fig = plt.figure()
plt.scatter(x, y, alpha=0.05, label="data")
plt.scatter(x, samples.flatten(), alpha=0.1, label="model")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sampled data")
plt.legend()
fig.savefig("./org/gfx/1d_bimodal_affine_samples.png")

# %%
t = np.linspace(0.0, 1.0, 200, dtype="float32")
pv_u = model.parameter_fn(t[..., None]).numpy().squeeze()
pv_poly = tcf(pv_u[..., 2:])

fig, axs = plt.subplots(3, sharex=True)
axs[0].plot(t, pv_u)
axs[0].set_title("unconstrained")
axs[1].plot(t, pv_poly)
axs[1].set_title("constrained pv poly")
axs[2].plot(t, tf.math.abs(pv_u[..., 0]), label="scale")
axs[2].plot(t, pv_u[..., 1], label="shift")
axs[2].set_title("constrained scale and shift")
axs[2].legend()
fig.tight_layout()
fig.savefig("./org/gfx/1d_bimodal_affine_parameters.png")

# %% Plot dist


fig = plot_dists(model(test_x[..., None]), test_x, test_t, test_y)
fig.savefig("./org/gfx/1d_bimodal_affine_dist.png")
# model.parameter_fn(test_x[..., None])

# %% plot flow
from bernstein_flow.util.visualization import plot_flow

fig = plot_flow(model(0.5), bijector_name="bernstein_poly")
fig.savefig("./org/gfx/1d_affine_flow.png")
# %%
