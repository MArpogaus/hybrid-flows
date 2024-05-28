# %% import

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from mctm.data.sklearn_datasets import get_dataset
from mctm.models import DensityRegressionModel, HybridDenistyRegressionModel
from mctm.utils.tensorflow import fit_distribution, set_seed
from mctm.utils.visualisation import plot_2d_data, plot_samples
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    tensor_util,
)


# %% functions
def nll_loss(y, dist):
    return -dist.log_prob(y)


def preprocess_dataset(data, model):
    return {
        "x": tf.convert_to_tensor(data[1][..., None], dtype=model.dtype),
        "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
    }


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


def thetas_constrain_fn2(diff, fn=tf.abs):
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


# %% elemntwise Flow
epochs = 100
seed = 1
distribution = "elementwise_flow"
dataset_kwargs = {"n_samples": 2**14, "scale": True}
distribution_kwargs = {
    "bijector_name": "bernstein_poly",
    "order": 30,
    "shift": False,
    "scale": False,
    "bounds": "linear",
    # "parameter_constrain_fn": thetas_constrain_fn2,
    # "base_distribution_kwargs": {"distribution_type": "uniform", "low": 0, "high": 1},
}
parameter_kwargs = {
    "dtype": "float",
    "conditional": False,
    # "polynomial_order": 1,
    # "conditional_event_shape": 1,
    # "conditional": True,
    # "hidden_units": [16, 16],
    # "activation": "relu",
    # "batch_norm": False,
    # "dropout": False,
}
initial_learning_rate = 0.05
# scheduler = K.optimizers.schedules.PolynomialDecay(
#     initial_learning_rate,
#     decay_steps=epochs,
#     end_learning_rate=0.00001,
#     power=1,
# )
fit_kwargs = {
    "epochs": epochs,
    "validation_split": 0.1,
    "batch_size": 128,
    "learning_rate": initial_learning_rate,
    # "callbacks": [K.callbacks.LearningRateScheduler(scheduler)],
    "lr_patience": 50,
    "reduce_lr_on_plateau": False,
    "early_stopping": 10,
    "verbose": True,
    "monitor": "val_loss",
}
model_kwargs = dict(
    distribution=distribution,
    distribution_kwargs=distribution_kwargs,
    parameter_kwargs=parameter_kwargs,
)

get_model = DensityRegressionModel
results_path = "./results/moons_" + distribution


# %% Load data
set_seed(seed)
data, dims = get_dataset("moons", n_samples=2**14, noise=0.05, scale=(0.01, 0.99))
(Y, X) = data

fig = plot_2d_data(Y, X)

# %% Init Model
marginal_model = DensityRegressionModel(
    dims=dims,
    **model_kwargs,
)
preprocessed = preprocess_dataset(data, marginal_model)

# %% fit model
hist = fit_distribution(
    model=marginal_model,
    seed=seed,
    results_path=results_path,
    loss=nll_loss,
    **preprocessed,
    **fit_kwargs,
    compile_kwargs={"jit_compile": True},
)

# %% Learning curve
pd.DataFrame(hist.history).plot()

# %% Samples
x, y = preprocessed.values()
marginal_dist = marginal_model(x)
# plot_samples(
#     marginal_dist,
#     y,
# )

# %% Plot dist
# samples = marginal_dist.sample(2000)
t = np.linspace(0, 1, 200, dtype=np.float32)
p = marginal_dist.prob(t[..., None])

fig, ax = plt.subplots(2, sharex=True)

ax[0].scatter(
    y[:, 0],
    np.zeros_like(y[:, 0]),
    marker="|",
    alpha=0.01,
    color="gray",
)
ax[1].scatter(
    y[:, 1],
    np.zeros_like(y[:, 1]),
    marker="|",
    alpha=0.01,
    color="gray",
)
sns.kdeplot(y[:, 0], ax=ax[0], label="KDE(y_1)")
sns.kdeplot(y[:, 1], ax=ax[1], label="KDE(y_2)")

ax[0].plot(t, p[:, 0], label="$P(y_1)$")
ax[1].plot(t, p[:, 1], label="$P(y_2)$")
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel("y1")
ax[1].set_xlabel("y2")

fig.tight_layout()

# %% bpoly
z = marginal_dist.bijector.inverse(y)
pit = marginal_dist.cdf(y)

df = pd.DataFrame(
    columns=[
        "$y1$",
        "$y2$",
        "$z_{1,1}$",
        "$z_{1,2}$",
        "$F_1(y_1)$",
        "$F_2(y_2)$",
        "$x$",
    ],
    data=np.concatenate([y, z, pit, x], -1),
)

g = sns.JointGrid(data=df, x="$z_{1,1}$", y="$y1$")
g.plot(sns.lineplot, sns.kdeplot)

# %% data
g = sns.jointplot(data=df, x="$y1$", y="$y2$")
g.figure.savefig("./org/gfx/moons.png")

# %% normalized data
g = sns.jointplot(data=df, x="$z_{1,1}$", y="$z_{1,2}$")
g.figure.savefig("./org/gfx/moons_T1.png")

# %% PIT
g = sns.jointplot(
    df,
    x="$F_1(y_1)$",
    y="$F_2(y_2)$",
)
g.figure.savefig("./org/gfx/moons_pit.png")

# %% hybrid dist
joint_distribution = "coupling_flow"
joint_model = HybridDenistyRegressionModel(
    dims=dims,
    distribution=joint_distribution,
    base_distribution=marginal_model,
    distribution_kwargs={
        "bijector_name": "bernstein_poly",
        "order": 32,
        "shift": False,
        "scale": False,
        # "bounds": "linear",
        # "parameter_constrain_fn": thetas_constrain_fn2,
        "low": 0,
        "high": 1,
        "num_layers": 1,
        # "base_distribution_kwargs": {"distribution_type": "uniform", "low": 0, "high": 1},
    },
    parameter_kwargs=dict(
        activation="tanh",
        batch_norm=False,
        dropout=False,
        hidden_units=[8, 16],
        # res_blocks=None,
        # res_block_units=None,
    ),
    freeze_base_model=True,
)

# %% fit model
epoachs = 40
initial_learning_rate = 0.01
scheduler = K.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps=epochs,
    # end_learning_rate=0.00001,
    # power=1,
)
fit_kwargs = {
    "epochs": epochs,
    "validation_split": 0.1,
    "batch_size": 128,
    "learning_rate": initial_learning_rate,
    "callbacks": [K.callbacks.LearningRateScheduler(scheduler)],
    "lr_patience": 50,
    "reduce_lr_on_plateau": False,
    "early_stopping": 10,
    "verbose": True,
    "monitor": "val_loss",
}

hist = fit_distribution(
    model=joint_model,
    seed=seed,
    results_path="./results/moons_" + joint_distribution,
    loss=nll_loss,
    **preprocessed,
    **fit_kwargs,
    compile_kwargs={"jit_compile": True},
)

# %% Learning curve
pd.DataFrame(hist.history).plot()

# %% samples
x, y = preprocessed.values()
joint_dist = joint_model(x)
plot_samples(
    joint_dist,
    y,
)

# %% bpoly
z2 = joint_dist.bijector.inverse(y)
z1 = marginal_dist.bijector.inverse(z2)

df = pd.DataFrame(
    columns=["$y1$", "$y2$", "$z_{2,1}$", "$z_{2,2}$", "$z_{1,1}$", "$z_{1,2}$", "$x$"],
    data=np.concatenate([y, z2, z1, x], -1),
)
# df=df.sort_values("$y2$")
g = sns.JointGrid(data=df, x="$z_{1,2}$", y="$y2$")
g.plot(sns.scatterplot, sns.kdeplot)

# %% decorrelated data
g = sns.jointplot(data=df, x="$z_{2,1}$", y="$z_{2,2}$")
g.figure.savefig("./org/gfx/moons_T2.png")

# %% normalized data
g = sns.jointplot(data=df, x="$z_{1,1}$", y="$z_{1,2}$")
g.figure.savefig("./org/gfx/moons_T2T1.png")

# %% PIT
pit = marginal_dist.cdf(y)

sns.jointplot(x=pit[:, 0], y=pit[:, 1])

# %% PDF
n = 200
x = np.linspace(0, 1, n, dtype=np.float32)
xx, yy = np.meshgrid(x, x)
grid = np.stack([xx.flatten(), yy.flatten()], -1)

p_y = joint_dist.prob(grid).numpy()
p_z1 = tf.reduce_prod(marginal_dist.prob(grid), -1).numpy()

# c(y) = p_y(y) / p_z1(y)
c = p_y / p_z1
c = np.where(p_z1 < 1e-4, 0, c)  # for numerical stability

fig = plt.figure(figsize=plt.figaspect(0.3))
ax = fig.add_subplot(131, projection="3d")
ax.plot_surface(
    xx,
    yy,
    p_y.reshape(-1, n),
    cmap="plasma",
    linewidth=0,
    antialiased=False,
    alpha=0.5,
)
ax = fig.add_subplot(132, projection="3d")
ax.plot_surface(
    xx,
    yy,
    p_z1.reshape(-1, n),
    cmap="plasma",
    linewidth=0,
    antialiased=False,
    alpha=0.5,
)
ax = fig.add_subplot(133, projection="3d")
ax.plot_surface(
    xx,
    yy,
    c.reshape(-1, n),
    cmap="plasma",
    linewidth=0,
    antialiased=False,
    alpha=0.5,
)
