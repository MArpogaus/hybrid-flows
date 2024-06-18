# A rich database available from Demographic and Health Surveys (DHS, https://dhsprogram.com/)
# provides nationally representative information about the health and nutritional status
# of populations in many of those countries. Here we use data from India that were
# collected in 1998.
#
# We used three indicators, stunting, wasting and underweight, as the response vector
# - stunting :: stunted growth, measured as an insufficient height with respect to the childs age
# - wasting and underweight :: refer to insufficient weight for height and insufficient weight for age

# %% import
import logging
import os

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from mctm.data.malnutrion import get_dataset
from mctm.models import DensityRegressionModel
from mctm.parameters import get_parameter_vector_fn
from mctm.utils.tensorflow import fit_distribution, set_seed
from mctm.utils.visualisation import get_figsize, setup_latex
from tensorflow import keras as K
from tensorflow_probability import distributions as tfd

# %% set log level
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)


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


def slice_loc_scale(parameters):
    loc = parameters[..., :dims]
    scale_tril = tfp.bijectors.FillScaleTriL()(parameters[..., dims:])
    return {"loc": loc, "scale_tril": scale_tril}


def get_samples_df(seed, model, x, y, targets):
    set_seed(seed)
    df_data = pd.DataFrame(y, columns=targets).assign(source="data").assign(cage=x)
    df_model = (
        pd.DataFrame(model(x).sample(seed=seed).numpy().squeeze(), columns=targets)
        .assign(source="model")
        .assign(cage=x)
    )
    df = pd.concat([df_data, df_model])

    return df


def ecdf(samples, x):
    ss = np.sort(samples)  # [..., None]
    cdf = np.searchsorted(ss, x, side="right") / float(ss.size)
    return cdf


def gen_pit_hist(ax, samples, measurements, **kwds):
    ecdf_samples = ecdf(samples, measurements.squeeze())

    pit_hist(ax, ecdf_samples, **kwds)


def pit_hist(ax, ecdf_samples, bins=20, title=None, **kwds):
    ax.hist(ecdf_samples.T, bins=bins, **kwds)

    ax.set_title(title if title else "Probability Integral Transform (PIT) Histograms")
    ax.set_xlabel("PIT=$F(y)$")
    ax.set_ylabel("Freqency")


def gen_reliablity_diagram(ax, samples, measurements, **kwds):
    # ss = np.sort(measurements.squeeze())
    ecdf_samples = ecdf(samples, measurements.squeeze())
    ecdf_measurements = ecdf(measurements, measurements.squeeze())

    reliablity_diagram(ax, ecdf_measurements, ecdf_samples, **kwds)


def reliablity_diagram(ax, ecdf_samples, ecdf_measurements, title=None, **kwds):
    ax.plot([0, 1], [0, 1], "k:")

    ax.plot(ecdf_measurements, ecdf_samples, **kwds)

    ax.set_title(title if title else "Reliability Diagram")
    ax.set_xlabel("Estimated Quantile")
    ax.set_ylabel("Observed Quantile")

    return ax


def validation_plot(samples, measurements, feature_name, covariates, **kwds):
    samples = np.sort(samples.squeeze())
    measurements = np.sort(measurements.squeeze())

    y = np.linspace(measurements.min(), measurements.max(), 1000)
    ecdf_samples_lin = ecdf(samples, y)
    ecdf_measurements_lin = ecdf(measurements, y)
    ecdf_samples_measurements = ecdf(samples, measurements)
    ecdf_measurements_measurements = ecdf(measurements, measurements)

    fig, ax = plt.subplots(1, 3, **kwds)

    ax[0].plot(y, ecdf_samples_lin.squeeze(), label="data")
    ax[0].plot(y, ecdf_measurements_lin.squeeze(), label="model")
    ax[0].set_title("Empirical CDF")
    ax[0].set_xlabel(feature_name)
    ax[0].set_ylabel(f"$F(y|{','.join(covariates)})$")
    ax[0].set_xscale("log")
    ax[0].legend(
        loc="upper left",
        fontsize=8,
        frameon=False,
    )

    reliablity_diagram(
        ax[1], ecdf_samples_measurements, ecdf_measurements_measurements
    )  # , s=2, alpha=0.1)
    pit_hist(ax[2], ecdf_samples_measurements)

    fig.tight_layout()

    return fig


# %% Setup Latex for plotting
setup_latex()

# %% paths
results_path = "./results/malnutrition_mctm"
figure_path = results_path + "/figures"
os.makedirs(figure_path, exist_ok=True)
img_ext = "png"

# %% Params
epochs = 100
seed = 1
covariates = ["cage"]
targets = ["stunting", "wasting", "underweight"]

dataset_kwargs = {
    "data_path": "datasets/malnutrition/india.raw",
    "covariates": covariates,
    "targets": targets,
    "stratify": True,
}
initial_learning_rate = 0.005
scheduler = K.optimizers.schedules.CosineDecay(
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
    "callbacks": [K.callbacks.LearningRateScheduler(scheduler)],
    "lr_patience": 50,
    "reduce_lr_on_plateau": False,
    "early_stopping": False,
    "verbose": True,
    "monitor": "val_loss",
}
preprocess_dataset = lambda data, _: {
    "x": data[0][0],
    "y": data[0][1],
    "validation_data": data[1],
}

# %% get dataset
set_seed(seed)
data, dims = get_dataset(
    **dataset_kwargs,
    scale=True,
    column_transformers=[
        ("passthrough", covariates),
    ],
)

# %% plot data
(unscaled_train_data, _, _), dims = get_dataset(**dataset_kwargs, scale=False)
unscaled_train_data_df = pd.DataFrame(
    np.concatenate([unscaled_train_data[1], unscaled_train_data[0]], -1),
    columns=targets + covariates,
)
fig = plot_grid(unscaled_train_data_df.sample(frac=0.2), vars=targets, hue="cage")
fig.savefig(figure_path + f"/malnutrition_data.{img_ext}")

# %% prepare data
preprocessed = preprocess_dataset(data, None)
X, Y, validation_data = preprocessed.values()

# %% marginal model
# TODO: Stimmt die Modellspezifikation in Klein et al. 2022?
marginal_transformations = [
    {
        "bijector_name": "bernstein_poly",
        "parameters_shape": [dims, 6],
        "parameters_fn": "parameter_vector",
        "parameters_fn_kwargs": {"dtype": "float32"},
        "parameters_constraint_fn": "mctm.activations.get_thetas_constrain_fn",
        "parameters_constraint_fn_kwargs": {
            "low": -3,
            "high": 4,
            "bounds": "smooth",
            "allow_flexible_bounds": True,
        },
        "extrapolation": True,
    },
    {
        "bijector_name": "shift",
        "parameters_shape": [dims],
        "parameters_fn": "bernstein_polynomial",  # "parameter_vector",
        "parameters_fn_kwargs": {
            "dtype": "float",
            "polynomial_order": 3,
            "conditional_event_shape": 1,
            "low": 0,
            "high": 35,
        },
        # "parameter_fn": "parameter_vector_or_simple_network",
        # "parameter_fn_kwargs": {
        #     "conditional": True,
        #     "conditional_event_shape": (1),
        #     "hidden_units": [16, 16],
        #     "activation": "relu",
        #     "batch_norm": False,
        #     "dropout": False,
        # },
        # "parameters_constraint_fn": tf.squeeze,
    },
    # {
    #     "bijector_name": "Scale_Matvec_Linear_Operator",
    #     "parameters_shape": [np.sum(np.arange(dims + 1))],
    #     "parameters_fn": get_bernstein_polynomial_parameter_lambda,  # "parameter_vector",
    #     "parameters_fn_kwargs": {
    #         "dtype": "float",
    #         "polynomial_order": 3,
    #         "conditional_event_shape": 1,
    #         "low": 0,
    #         "high": 35,
    #     },
    #     "parameters_constraint_fn": lambda x: tf.linalg.LinearOperatorLowerTriangular(
    #         tfb.FillScaleTriL(diag_shift=1e-5)
    #     ),
    # },
]

mctm_model = DensityRegressionModel(
    distribution="normalizing_flow",
    transformations=marginal_transformations,
    # base_distribution_kwargs={"dims": 0},
    base_distribution_kwargs={
        "distribution_name": "tfd.MultivariateNormalTriL",
        "parameters_shape": [dims + np.sum(np.arange(dims + 1))],
        "parameters_fn": "bernstein_polynomial",  # "parameter_vector",
        "parameters_fn_kwargs": {
            "dtype": "float",
            "polynomial_order": 3,
            "conditional_event_shape": 1,
            "low": 0,
            "high": 35,
        },
        "parameters_constraint_fn": slice_loc_scale,
    },
)

# %% distribution
joint_dist = mctm_model(tf.ones((1, 3)))
# tfd.Independent(dist, 2)
joint_dist

# %% log like
joint_dist.log_prob(tf.ones((2, 2, 3)))

# %% trainable parameters
mctm_model.trainable_parameters

# %% bijector
joint_dist.bijector  # .bijector.bijectors

# %% parameter function
params = mctm_model.parameter_fn(tf.ones((2, 1)))
params

# %% fit model
hist = fit_distribution(
    model=mctm_model,
    seed=seed,
    results_path=results_path,
    loss=nll_loss,
    **preprocessed,
    **fit_kwargs,
)

# %% Learning curve
pd.DataFrame(hist.history).plot()

# %% params after fit
t = np.linspace(0, max(X), 200, dtype="float32")
pv = mctm_model.parameter_fn(t[..., None])[0][1].numpy().squeeze()
pv
fig, axs = plt.subplots(1, len(targets), sharex=True, sharey=True)
for (i, c), label in zip(enumerate(targets), targets):
    axs[i].plot(t, pv[:, i])
    axs[i].set_xlabel("cage")
    axs[i].set_title(label)
axs[0].set_xticks((t.min(), t.max()))
fig.tight_layout(w_pad=-0.1)
fig.savefig(figure_path + f"/malnutrition_params.{img_ext}")


# %% rank correlation
ages = unscaled_train_data_df.cage.unique()
ages = np.sort(ages)
joint_dist = mctm_model(tf.convert_to_tensor(ages, dtype=mctm_model.dtype)[..., None])

cov = joint_dist.distribution.covariance().numpy()
std = np.sqrt(tf.linalg.diag_part(cov))
cor = cov / tf.matmul(std[..., None], std[..., None], transpose_b=True)

fig, axs = plt.subplots(1, len(targets), figsize=get_figsize("thesis"))

for ax, (a, b) in zip(
    axs.T,
    zip(["stunting", "stunting", "wasting"], ["wasting", "underweight", "underweight"]),
):
    i, j = targets.index(a), targets.index(b)
    # ax.set_aspect(1)
    rho = cor[:, i, j]
    rho_s = 6 / np.pi * np.arcsin(rho / 2)
    ax.plot(ages, rho_s)
    ax.set_title(f"$\\rho^S_{{{a},{b}}}$")
    ax.set_box_aspect(1)
    ax.set_xticks(ages[0:-1:8])

fig.tight_layout()
fig.savefig(figure_path + f"rank_correlation.{img_ext}")

# %% marginal cdf and pdf
palette = "icefire"
palette = "rocket_r"
palette = "mako_r"
ages = [1, 3, 6, 9, 12, 24]
# ages = unscaled_train_data_df.cage.unique()
colors = sns.color_palette(palette, as_cmap=True)(np.linspace(0, 1, len(ages))).tolist()
joint_dist = mctm_model(tf.convert_to_tensor(ages, dtype=mctm_model.dtype)[..., None])
marginal_dist = tfd.TransformedDistribution(
    distribution=tfd.Normal(0, 1), bijector=joint_dist.bijector
)

y = np.linspace(0, 1, 100)[..., None, None]

cdf = marginal_dist.cdf(y).numpy()
pdf = marginal_dist.prob(y).numpy()
fig, axs = plt.subplots(
    2,
    len(targets),
    figsize=get_figsize("thesis"),
    sharey="row",
    sharex=True,
)

for i, c in enumerate(targets):
    axs[0, i].set_prop_cycle("color", colors)
    axs[0, i].plot(y.flatten(), cdf[..., i], label=ages, lw=0.5)
    axs[1, i].set_prop_cycle("color", colors)
    axs[1, i].plot(y.flatten(), pdf[..., i], label=ages, lw=0.5)
    axs[1, i].set_xlabel(f"y={c}")
    if i == 0:
        axs[0, i].legend(
            ages,
            title="Age",
            # bbox_to_anchor=(1.05, 1),
            loc="right",
            fontsize=8,
            frameon=False,
        )

axs[0, 0].set_ylabel(r"$F(y|\text{age})$")
axs[1, 0].set_ylabel(r"$f(y|\text{age})$")

fig.tight_layout(w_pad=0)
fig.savefig(figure_path + f"/malnutrition_dist.{img_ext}")


# %% samples
# plot_samples_grid(model, validation_data, 1000, targets, row="source")
samples_df = get_samples_df(1, mctm_model, X, Y, targets)
# df["mage_combined"] = df.mage // 128
fig = plot_grid(
    samples_df.loc[samples_df.source == "model"].groupby("cage").sample(frac=0.2),
    vars=targets,
    hue="cage",
)
fig.savefig(figure_path + f"/mctm_samples.{img_ext}")


# %% validation plots
samples_df = get_samples_df(1, mctm_model, X, Y, targets)
cage = 3
column = targets[0]
measurements = samples_df[(samples_df.source == "data") & (samples_df.cage == cage)][
    column
]
samples = samples_df[(samples_df.source == "model") & (samples_df.cage == cage)][column]
print(len(samples))
fig = validation_plot(samples, measurements, feature_name=column, covariates=["cage"])
fig.savefig(figure_path + f"/validation_plots_{cage}.{img_ext}")
# %% reliability diagram (samples)
samples_df = get_samples_df(1, mctm_model, X, Y, targets)
column = targets[0]


def apply_ecdf(df):
    model_df = df.loc[df.source == "model", column]
    data_df = df.loc[df.source == "data", column]
    model_ecdf = ecdf(model_df.copy(), data_df.copy())
    data_ecdf = ecdf(data_df.copy(), data_df.copy())
    df.loc[df.source == "model", column + "_ecdf"] = model_ecdf
    df.loc[df.source == "data", column + "_ecdf"] = data_ecdf
    return df


samples_ecdf_df = (
    samples_df[["cage", "source", column]]
    .groupby("cage")
    .apply(apply_ecdf, include_groups=False)
    .reset_index()
    .sort_values("level_1")
    .drop(columns="level_1")
)

ecdf_df = pd.DataFrame(
    np.stack(
        [
            samples_ecdf_df.loc[
                samples_ecdf_df.source == "model", column + "_ecdf"
            ].values,
            samples_ecdf_df.loc[
                samples_ecdf_df.source == "data", column + "_ecdf"
            ].values,
            X.numpy().flatten(),
        ],
        1,
    ),
    columns=["model", "data", "cage"],
)

g = sns.lineplot(ecdf_df, x="model", y="data", hue="cage", estimator=None)
g.figure.savefig(figure_path + f"/reliability_plot_samples.{img_ext}")

# %% reliability diagram (CDF)
column = targets[0]

marginal_dist = tfd.TransformedDistribution(
    distribution=tfd.Normal(0, 1), bijector=mctm_model(X).bijector
)

cdf_model_df = pd.DataFrame(data=marginal_dist.cdf(Y), columns=targets).assign(cage=X)
ecdf_measurements = (
    pd.DataFrame(data=Y, columns=targets)
    .assign(cage=X)[["cage", column]]
    .groupby("cage")
    .apply(lambda x: x.apply(lambda x: ecdf(x, x), raw=True), include_groups=False)
    .reset_index()
    .sort_values("level_1")
    .drop(columns="level_1")
)

ecdf_df = pd.DataFrame(
    np.stack(
        [cdf_model_df[column].values, ecdf_measurements[column], X.numpy().flatten()], 1
    ),
    columns=["model", "data", "cage"],
)

g = sns.lineplot(ecdf_df, x="model", y="data", hue="cage", estimator=None)
g.figure.savefig(figure_path + f"/reliability_plot_cdf.{img_ext}")
