# %% import
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as K
from matplotlib import pyplot as plt
from mctm.data.sklearn_datasets import get_dataset
from mctm.models import DensityRegressionModel, HybridDenistyRegressionModel
from mctm.utils.tensorflow import fit_distribution, set_seed
from mctm.utils.visualisation import plot_2d_data, plot_samples, setup_latex
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    tensor_util,
)

# %%
print(f"{tf.__version__=}\n{tfp.__version__=}")


# %% functions
def independent_nll(y, dist):
    return -tfd.Independent(dist, 1).log_prob(y)


def nll(y, dist):
    return -dist.log_prob(y)


def preprocess_dataset(data, model):
    return {
        "x": tf.convert_to_tensor(data[1][..., None], dtype=model.dtype),
        "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
    }


def thetas_constrain_fn(diff, fn=tf.math.softplus):
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


def thetas_constrain_fn2(diff, fn=tf.math.softplus):
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


# %% setup latex for plotting
setup_latex(fontsize=10)

# %% elemntwise Flow
epochs = 500
seed = 1
distribution = "elementwise_flow"
dataset_kwargs = {"n_samples": 2**14, "scale": True}
distribution_kwargs = {
    "bijector_name": "bernstein_poly",
    "order": 100,
    "shift": False,
    "scale": False,
    # "bounds": None,
    # "low": None,
    # "high": None,
    # "min_slope": 1e-12,
    "bounds": "smooth",
    "allow_flexible_bounds": True,
    # "parameter_constrain_fn": thetas_constrain_fn,
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
initial_learning_rate = 0.01
# scheduler = K.optimizers.schedules.PolynomialDecay(
#     initial_learning_rate,
#     decay_steps=epochs,
#     end_learning_rate=0.0001,
#     power=0.5,
# )
fit_kwargs = {
    "epochs": epochs,
    "validation_split": 0.1,
    "batch_size": 128,
    "learning_rate": initial_learning_rate,
    "weight_decay": 0.004,
    # "callbacks": [K.callbacks.LearningRateScheduler(scheduler)],
    "lr_patience": 50,
    "reduce_lr_on_plateau": False,
    "early_stopping": 15,
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
    loss=independent_nll,
    **preprocessed,
    **fit_kwargs,
    compile_kwargs={"jit_compile": True},
)

# %% Learning curve
pd.DataFrame(hist.history).plot()

# %% marginal dist
x, y = preprocessed.values()
marginal_dist = marginal_model(x)

# %% samples
plot_samples(
    marginal_dist,
    y,
)

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
g = sns.JointGrid(data=df, x="$y1$", y="$y2$", height=2)
g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
g.plot_marginals(sns.kdeplot)
g.figure.savefig("../paper/gfx/moons.pdf")

# %% normalized data
g = sns.JointGrid(data=df, x="$z_{1,1}$", y="$z_{1,2}$", height=2)
g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
g.plot_marginals(sns.kdeplot)
g.figure.savefig("../paper/gfx/moons_h1.pdf")

# %% PIT
g = sns.jointplot(df, x="$F_1(y_1)$", y="$F_2(y_2)$", height=2, s=4, alpha=0.5)
g.figure.savefig("../paper/gfx/moons_pit.pdf")

# %% hybrid dist
joint_distribution = "coupling_flow"
joint_distribution_kwargs = {
    "bijector_name": "bernstein_poly",
    "order": 128,
    # "shift": True,
    # "scale": True,
    # "bounds": "linear",
    # "parameter_constrain_fn": thetas_constrain_fn2,
    # "min_slope": 0.001,
    # "analytic_jacobian": True,
    "low": 0,
    "high": 1,
    "bounds": False,
    "num_layers": 1,
    # "allow_flexible_bounds": True,
    # "base_distribution_kwargs": {"distribution_type": "uniform", "low": 0, "high": 1},
}
joint_model = HybridDenistyRegressionModel(
    dims=dims,
    distribution=joint_distribution,
    distribution_kwargs=joint_distribution_kwargs,
    base_distribution=marginal_model,
    parameter_kwargs=dict(
        activation="sigmoid",
        batch_norm=False,
        dropout=False,
        hidden_units=[128] * 4,
        # res_blocks=2,
        # res_block_units=32,
    ),
    freeze_base_model=True,
)

# %% fit model
epochs = 200
initial_learning_rate = 0.001
scheduler = K.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps=epochs,
    # end_learning_rate=0.00001,
    # power=1,
)
fit_kwargs = {
    "epochs": epochs,
    "validation_split": 0.1,
    "batch_size": 256,
    "learning_rate": initial_learning_rate,
    "weight_decay": initial_learning_rate / 10,
    "callbacks": [K.callbacks.LearningRateScheduler(scheduler)],
    # "lr_patience": 50,
    "reduce_lr_on_plateau": False,
    "early_stopping": False,
    "verbose": True,
    "monitor": "val_loss",
}

hist = fit_distribution(
    model=joint_model,
    seed=seed,
    results_path="./results/moons_" + joint_distribution,
    loss=nll,
    **preprocessed,
    **fit_kwargs,
    compile_kwargs={"jit_compile": True},
)

# %% Learning curve
pd.DataFrame(hist.history).plot()

# %% dist
x, y = preprocessed.values()
joint_dist = joint_model(x)

# %% samples
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
g = sns.JointGrid(data=df, x="$z_{2,1}$", y="$z_{2,2}$", height=2)
g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
g.plot_marginals(sns.kdeplot)
g.figure.savefig("../paper/gfx/moons_h2.pdf")

# %% normalized data
g = sns.JointGrid(data=df, x="$z_{1,1}$", y="$z_{1,2}$", height=2)
g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
g.plot_marginals(sns.kdeplot)
g.figure.savefig("../paper/gfx/moons_T2T1.pdf")

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

# %% Bijector
n = 400
x = np.linspace(0, 1, n, dtype=np.float32)
xx, yy = np.meshgrid(x, x)
grid = np.stack([xx.flatten(), yy.flatten()], -1)

# z2 = joint_dist.bijector.inverse(grid).numpy()
# ldj = joint_dist.bijector.forward_log_det_jacobian(grid).numpy()
[z2, tfp_grads] = tfp.math.value_and_gradient(joint_dist.bijector.inverse, grid)
z2 = z2.numpy()
tfp_grads = tfp_grads.numpy()
np_grad = np.gradient(z2[:, 1], grid[:, 0])

fig = plt.figure(figsize=plt.figaspect(0.3))
ax = fig.add_subplot(131, projection="3d")
ax.plot_surface(
    xx,
    yy,
    z2[:, 1].reshape(-1, n),
    cmap="plasma",
    linewidth=0,
    antialiased=False,
    alpha=0.75,
)
ax = fig.add_subplot(132, projection="3d")
ax.plot_surface(
    xx,
    yy,
    # np.diff(z2[:,1].reshape(-1, n)),
    # np_grad.reshape(-1, n),
    tfp_grads[:, 1].reshape(-1, n),
    cmap="plasma",
    linewidth=0,
    antialiased=False,
    alpha=0.75,
)
ax = fig.add_subplot(133, projection="3d")
ax.plot_surface(
    xx,
    yy,
    np.log(tfp_grads[:, 1]).reshape(-1, n),
    cmap="plasma",
    linewidth=0,
    antialiased=False,
    alpha=0.75,
)

# %% J
z2 = joint_dist.bijector.inverse(grid).numpy()
# ldj = joint_dist.bijector.forward_log_det_jacobian(grid).numpy()
J = [
    [np.gradient(z2[:, 0], grid[:, 0]), np.gradient(z2[:, 0], grid[:, 1])],
    [np.gradient(z2[:, 1], grid[:, 0]), np.gradient(z2[:, 1], grid[:, 1])],
]
J

# %% np gradient
# x21=y1
# x22=h(y2|y1)

import tensorflow_probability as tfp

n = 200
x = np.linspace(0, 1, n, dtype=np.float32)
xx, yy = np.meshgrid(x, x)
grid = np.stack([xx.flatten(), yy.flatten()], -1)

flow_parametrization_fn = (
    joint_dist.bijector.bijectors[0]._bijector_fn.__closure__[0].cell_contents
)
param_net = joint_dist.bijector.bijectors[0]._bijector_fn.__closure__[1].cell_contents
thetas_u = param_net(x)  # grid[:, :1])
flow = flow_parametrization_fn(thetas_u)

# %%
from mctm.distributions import _get_flow_parametrization_fn

joint_distribution_kwargs2 = joint_distribution_kwargs.copy()
joint_distribution_kwargs2.pop("num_layers")
joint_distribution_kwargs2["analytic_jacobian"] = False
flow_fn, _ = _get_flow_parametrization_fn(**joint_distribution_kwargs2, min_slope=0)

flow_z22 = flow_fn(thetas_u)

# %% thetas
thetas = flow_z22.bijector.thetas.numpy().squeeze()
print(thetas.min(), thetas.max())
plt.plot(x, thetas, alpha=0.5)

# %%
thetas_u = param_net(grid[:, :1])
flow_z22 = flow_fn(thetas_u)
z22 = flow_z22.inverse(grid[:, 1:]).numpy()
z2 = np.concatenate([grid[:, :1], z22], 1)

# ldj = joint_dist.bijector.forward_log_det_jacobian(grid).numpy()
# np_grad = np.gradient(z2[:, 1], grid[:, 0])
[_, grads] = tfp.math.value_and_gradient(flow_z22.inverse, grid[:, 1:])
ildj_z22 = flow_z22.inverse_log_det_jacobian(grid[:, 1:]).numpy()
ildj_z2 = np.concatenate([tf.ones_like(ildj_z22), ildj_z22], 1)
p_y = tf.reduce_prod(marginal_dist.prob(z2) * np.exp(ildj_z2), -1).numpy()

fig = plt.figure(figsize=plt.figaspect(0.3))
ax = fig.add_subplot(131, projection="3d")
ax.plot_surface(
    xx,
    yy,
    z22.reshape(-1, n),
    cmap="plasma",
    linewidth=0,
    antialiased=False,
    alpha=0.5,
)
ax = fig.add_subplot(132, projection="3d")
ax.plot_surface(
    xx,
    yy,
    ildj_z22.reshape(-1, n),
    # np.log(grads).reshape(-1, n),
    cmap="plasma",
    linewidth=0,
    antialiased=False,
    alpha=0.5,
)
ax = fig.add_subplot(133, projection="3d")
ax.plot_surface(
    xx,
    yy,
    p_y.reshape(-1, n),
    cmap="plasma",
    linewidth=0,
    antialiased=False,
    alpha=0.5,
)


# %%
import tensorflow_probability as tfp

n = 200
x = np.linspace(0, 1, n, dtype=np.float32)
xx, yy = np.meshgrid(x, x)
grid = np.stack([xx.flatten(), yy.flatten()], -1)

# p_y = joint_dist.prob(grid).numpy()
p_z1 = tf.reduce_prod(marginal_dist.prob(grid), -1).numpy()

z2 = joint_dist.bijector.inverse(grid).numpy()
# ldj = joint_dist.bijector.forward_log_det_jacobian(grid).numpy()
np_grad = np.gradient(z2[:, 1], grid[:, 0])
[funval, grads] = tfp.math.value_and_gradient(joint_dist.bijector.inverse, grid)
ildj = joint_dist.bijector.inverse_log_det_jacobian(grid)

# p_y = tf.reduce_prod(marginal_dist.prob(z2), -1).numpy() *grads[:,1].numpy()
p_y = tf.reduce_prod(marginal_dist.prob(z2), -1).numpy() * np.exp(ildj)
# * np.abs(
#     np_grad
# )

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
