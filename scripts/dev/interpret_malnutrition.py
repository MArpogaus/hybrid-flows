# A rich database available from Demographic and Health Surveys (DHS, https://dhsprogram.com/)
# provides nationally representative information about the health and nutritional status
# of populations in many of those countries. Here we use data from India that were
# collected in 1998.
#
# We used three indicators, stunting, wasting and underweight, as the response vector
# - stunting :: stunted growth, measured as an insufficient height with respect to the childs age
# - wasting and underweight :: refer to insufficient weight for height and insufficient weight for age

# %% import
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from mctm.data.malnutrion import get_dataset
from mctm.models import DensityRegressionModel
from mctm.parameters import get_parameter_vector_fn
from mctm.utils.tensorflow import fit_distribution, set_seed
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    tensor_util,
)


# %% functions
def nll_loss(y, dist):
    return -dist.log_prob(y)


def plot_grid(data, **kwargs):
    """Plot sns.PairGrid."""
    sns.set_theme(style="white")
    g = sns.PairGrid(data, diag_sharey=False, **kwargs)
    g.map_upper(sns.scatterplot, s=15)

    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)

    return g


def plot_data(train_data, targets, covariates, frac=0.1, **kwargs):
    """Plot data."""
    data = np.concatenate([train_data[0][..., None], train_data[1]], -1)
    df = pd.DataFrame(data, columns=targets + covariates).sample(frac=frac)

    g = plot_grid(df, **kwargs)
    return g.figure


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


def get_polynomial_parameter_lambda(
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

    def get_parameter_lambda(conditional_input=None, **kwargs):
        xx = tf.stack(
            [(conditional_input / 35) ** i for i in range(0, polynomial_order + 1)], -1
        )
        yy = xx[..., None, None] * parameter_vector
        return tf.reduce_sum(yy, axis=[-4, -3])

    return get_parameter_lambda, parameter_vector


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
dataset_kwargs = {
    "data_path": "datasets/malnutrition/india.raw",
    "covariates": ["cage"],
    "targets": ["stunting", "wasting", "underweight"],
}
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
    "order": 8,
    "shift": False,
    "scale": False,
    "parameter_constrain_fn": thetas_constrain_fn2,
    # "base_distribution_kwargs": {"distribution_type": "uniform", "low": 0, "high": 1},
}
parameter_kwargs = {
    "dtype": "float",
    "polynomial_order": 3,
    "conditional_event_shape": 1,
    # "conditional": True,
    # "hidden_units": [16, 16],
    # "activation": "relu",
    # "batch_norm": False,
    # "dropout": False,
}
initial_learning_rate = 0.01
scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps=epochs,
    # end_learning_rate=0.00001,
    # power=1,
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

preprocess_dataset = lambda data, model: {
    "x": data[0][0],
    "y": data[0][1],
    "validation_data": data[1],
}
results_path = "./results/" + distribution


# %% Load unscaled data
# raw_data = pd.read_csv(dataset_kwargs["data_path"], sep=r"\s+")

# %% Load unscaled data
set_seed(seed)
targets = dataset_kwargs["targets"]
covariates = dataset_kwargs["covariates"]
(unscaled_train_data, _, _), dims = get_dataset(**dataset_kwargs, scale=False)
df = pd.DataFrame(
    np.concatenate([unscaled_train_data[1], unscaled_train_data[0]], -1),
    columns=targets + covariates,
)
fig = plot_grid(df.sample(frac=0.2), vars=targets, hue="cage")
fig.savefig("./org/gfx/malnutrition_data.png")

# %% kde maginal plot

fig, axs = plt.subplots(1, len(targets), figsize=(8, 2))

for i, c in enumerate(targets):
    sns.kdeplot(
        df.sample(frac=0.5),
        x=c,
        hue="cage",  # df.mage // 128,
        ax=axs[i],
        common_norm=False,
        alpha=0.1,
        fill=True,
        palette="crest",
        linewidth=0,
        legend=False,
    )

fig.tight_layout()
# %% ecdf maginal plot
fig, axs = plt.subplots(
    1, len(targets), figsize=(8, 2), sharey=True, sharex=True, layout="constrained"
)
palette = "rocket_r"

for i, c in enumerate(targets):
    sns.ecdfplot(
        df,
        x=c,
        hue="cage",  # df.mage // 128,
        ax=axs[i],
        alpha=0.5,
        legend=False,
        palette=palette,
    )

axs[0].set_ylabel(r"$ECDF(y|\text{age})$")

fig.colorbar(
    plt.cm.ScalarMappable(
        cmap=palette, norm=plt.Normalize(df.cage.min(), df.cage.max())
    ),
    ax=axs,
    label="age",
)
# fig.tight_layout()
fig.savefig("./org/gfx/malnutrition_ecdf.png")

# %% scaled data
set_seed(seed)
data, dims = get_dataset(
    **dataset_kwargs,
    scale=True,
    column_transformers=[
        ("passthrough", covariates),
    ],
)

# %% Init Model
model = DensityRegressionModel(
    dims=dims,
    get_parameter_fn=get_polynomial_parameter_lambda,
    **model_kwargs,
)

# %% inital values
tcf = distribution_kwargs["parameter_constrain_fn"]

t = np.linspace(-1, 1, 20, dtype="float32")
pv_u = model.parameter_fn(t[..., None]).numpy().squeeze()
pv = tcf(pv_u)

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
for i, c in enumerate(targets):
    axs[0][i].plot(t, pv_u[:, i])
    axs[0][i].set_title(c)
    axs[1][i].plot(t, pv[:, i])
    axs[1][i].set_xlabel("x")
axs[0][0].set_ylabel(r"unconstrained $\theta$")
axs[1][0].set_ylabel(r"constrained $\theta$")
fig.tight_layout()

# %% fit model
preprocessed = preprocess_dataset(data, model)
X, Y, validation_data = preprocessed.values()
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

# %% params after fit
tcf = distribution_kwargs["parameter_constrain_fn"]

t = np.unique(X)  # np.linspace(-1, 1, 20, dtype="float32")
pv_u = model.parameter_fn(t[..., None]).numpy().squeeze()
pv = tcf(pv_u)

fig, axs = plt.subplots(2, 3, sharex=True)
for i, c in enumerate(targets):
    axs[0][i].plot(t, pv_u[:, i])
    axs[0][i].set_title(c)
    axs[1][i].plot(t, pv[:, i])
    axs[1][i].set_xlabel("cage")
axs[0][0].set_ylabel(r"unconstrained $\theta$")
axs[1][0].set_ylabel(r"constrained $\theta$")
axs[1][0].set_xticks((t.min(), t.max()))
fig.tight_layout()
fig.savefig("./org/gfx/malnutrition_params.png")

# %% marginal cdf
palette = "rocket_r"
cages = np.arange(0, 35, 8, dtype=np.float32)
cages = np.unique(X)
colors = sns.color_palette("rocket_r", as_cmap=True)(
    np.linspace(0, 1, len(cages))
).tolist()
dists = model(cages[..., None])

y = np.linspace(0, 1, 20)[..., None, None]

cdf = dists.cdf(y)
fig, axs = plt.subplots(
    2, len(targets), figsize=(8, 4), sharey=True, sharex=True, layout="constrained"
)

for i, c in enumerate(targets):
    sns.ecdfplot(
        x=Y[..., i],
        hue=X.numpy().flatten(),  # df.mage // 128,
        ax=axs[0, i],
        alpha=0.5,
        legend=False,
        palette=palette,
    )
    axs[1, i].set_prop_cycle("color", colors)
    axs[1, i].plot(y.flatten(), cdf[..., i], alpha=0.5)
    axs[1, i].set_xlabel(c)

axs[0, 0].set_ylabel(r"$ECDF(y|\text{age})$")
axs[1, 0].set_ylabel(r"$F(y|\text{age})$")

fig.colorbar(
    plt.cm.ScalarMappable(
        cmap=palette, norm=plt.Normalize(df.cage.min(), df.cage.max())
    ),
    ax=axs[:, -1],
    label="age",
    shrink=0.5,
)

fig.savefig("./org/gfx/malnutrition_cdf.png")

# %% Samples
X, Y, validation_data = preprocessed.values()
dist = model(X)

samples = dist.sample(seed=1).numpy()


# %% sample
def get_samples_df(seed, model, x, y, targets):
    set_seed(seed)
    df_data = pd.DataFrame(y, columns=targets).assign(source="data").assign(mage=x)
    df_model = (
        pd.DataFrame(model(x).sample(seed=seed).numpy().squeeze(), columns=targets)
        .assign(source="model")
        .assign(mage=x)
    )
    df = pd.concat([df_data, df_model])

    return df


def plot_samples_grid(model, validation_data, N, targets, hue="source", **kwargs):
    x, y = validation_data
    df = get_samples_df(1, model, x, y, targets)

    g = plot_grid(df, vars=targets, hue=hue, **kwargs)
    g = g.add_legend()

    # g.figure.savefig(os.path.join(results_path, "samples.pdf"))


# plot_samples_grid(model, validation_data, 1000, targets, row="source")
df = get_samples_df(1, model, X, Y, targets)
# df["mage_combined"] = df.mage // 128
plot_grid(df.loc[df.source == "model"].sample(frac=0.2), vars=targets, hue="mage")
