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
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

# %% set log level
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)


# %% functions
def nll_loss(y, dist):
    return -dist.log_prob(y)


def nll_loss(y, dist):
    marginal_dist = tfd.Independent(
        tfd.TransformedDistribution(
            distribution=tfd.Normal(0, 1),
            bijector=tfb.Invert(tfb.Chain(dist.bijector.bijector.bijectors[1:])),
        ),
        1,
    )

    return -dist.log_prob(y) - marginal_dist.log_prob(y)


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
    return cdf.astype(x.dtype)


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
    # ax[0].set_xscale("log")
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


# %% NLL
class MeanNegativeLogLikelihood(K.metrics.Mean):
    """Custom metric for mean negative log likelihood."""

    def __init__(
        self, name: str = "mean_negative_log_likelihood", **kwargs: dict
    ) -> None:
        """Initialize Keras metric for negative logarithmic likelihood.

        Parameters
        ----------
        name : str, optional
            Name of the metric instance, by default "mean_negative_log_likelihood".
        **kwargs : dict
            Additional keyword arguments for the base class.

        """
        super().__init__(name=name, **kwargs)

    def update_state(
        self, y_true: tf.Tensor, dist: object, sample_weight: tf.Tensor = None
    ) -> None:
        """Update the metric state with the true labels and the distribution.

        Parameters
        ----------
        y_true : tf.Tensor
            The ground truth values.
        dist : object
            The distribution object that provides the log probability method.
        sample_weight : tf.Tensor, optional
            Optional weighting of each example, by default None.

        """
        log_probs = -dist.log_prob(y_true)
        super().update_state(log_probs, sample_weight)


# %% Setup Latex for plotting
setup_latex()

# %% paths
results_path = "./results/malnutrition_hmbnf"
figure_path = results_path + "/figures"
os.makedirs(figure_path, exist_ok=True)
img_ext = "png"

# %% params
epochs = 200
batch_size = 128
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
mk_ds = lambda data: tf.data.Dataset.from_tensor_slices((data[0], data[1])).batch(
    batch_size, drop_remainder=True
)
preprocess_dataset = lambda data, _: {
    "x": mk_ds(data[0]),
    "validation_data": mk_ds(data[1]),
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
(train_data, _, _) = data
train_data_df = pd.DataFrame(
    np.concatenate([train_data[1], train_data[0]], -1),
    columns=targets + covariates,
)
fig = plot_grid(
    train_data_df.groupby(covariates).sample(frac=0.2),
    vars=targets,
    hue="cage",
)
fig.set(xlim=(-5, 5))
fig.savefig(figure_path + f"/malnutrition_data.{img_ext}", bbox_inches="tight")

# %% prepare data
X, Y = data[0][0], data[0][1]

# %% model
# TODO: Stimmt die Modellspezifikation in Klein et al. 2022?
nbins = 32
domain = (-4, 4)
x_domain = np.min(X), np.max(X)
bounds = (-4, 4)
bijectors = [
    {
        "bijector": "BernsteinBijector",
        "bijector_kwargs": {"extrapolation": False, "domain": domain},
        "parameters_fn": "parameter_vector",
        "parameters_fn_kwargs": {
            "parameter_shape": [
                dims,
            ],
            "dtype": "float32",
        },
        "parameters_constraint_fn": "bernstein_flow.activations.get_thetas_constrain_fn",
        "parameters_constraint_fn_kwargs": {
            "bounds": bounds,
            "smooth_bounds": False,
            "allow_flexible_bounds": False,
        },
    },
    {
        "bijector": "Shift",
        "parameters_fn": "bernstein_polynomial",  # "parameter_vector",
        "parameters_fn_kwargs": {
            "parameter_shape": [dims],
            "dtype": "float",
            "polynomial_order": 6,
            "conditional_event_shape": 1,
            "domain": x_domain,
        },
    },
    # {
    #     "bijector": "MaskedAutoregressiveFlow",
    #     "bijector": "BernsteinBijector",
    #     "parameters_fn": "masked_autoregressive_network",
    #     "parameters_fn_kwargs": {
    #         "parameter_shape": [dims, 32],
    #         "activation": "relu",
    #         "hidden_units": [128, 128],
    #         "conditional": True,
    #         "conditional_event_shape": 1,
    #     },
    #     "parameters_constraint_fn": "mctm.activations.get_thetas_constrain_fn",
    #     "parameters_constraint_fn_kwargs": {
    #         "low": -3,
    #         "high": 3,
    #         "bounds": "smooth",
    #         "allow_flexible_bounds": True,
    #     },
    # },
    # {
    #     "bijector": "MaskedAutoregressiveFlow",
    #     "bijector_kwargs": {
    #         "bijector": "RationalQuadraticSpline",
    #         "bijector_kwargs": {
    #             "range_min": -4,
    #         },
    #     },
    #     "parameters_fn": "masked_autoregressive_network",
    #     "parameters_fn_kwargs": {
    #         "parameter_shape": [dims, 32 * 3 - 1],
    #         "activation": "relu",
    #         "hidden_units": [128, 128],
    #         "conditional": True,
    #         "conditional_event_shape": 1,
    #     },
    #     "parameters_constraint_fn": "mctm.activations.get_spline_param_constrain_fn",
    #     "parameters_constraint_fn_kwargs": {
    #         "interval_width": 8,
    #         "min_slope": 0.001,
    #         "min_bin_width": 0.001,
    #         "nbins": 32,
    #     },
    # },
    {
        "bijector": "RealNVP",
        "bijector_kwargs": {
            "bijector": "MaskedAutoregressiveFlow",
            "bijector_kwargs": {
                "bijector": "RationalQuadraticSpline",
                "bijector_kwargs": {
                    "range_min": bounds[0],
                },
            },
            "num_masked": 1,
        },
        "parameters_fn": "masked_autoregressive_network_with_additive_conditioner",
        "parameters_fn_kwargs": {
            "input_shape": (1,),
            "parameter_shape": [dims - 1, nbins * 3 - 1],
            "made_kwargs": {
                "activation": "relu",
                "hidden_units": [32] * 2,
                "conditional": True,
                "conditional_event_shape": 1,
            },
            "x0_kwargs": {
                "activation": "relu",
                "hidden_units": [32] * 2,
                "batch_norm": False,
                "dropout": False,
                "conditional": True,
                "conditional_event_shape": 1,
            },
        },
        "parameters_constraint_fn": "mctm.activations.get_spline_param_constrain_fn",
        "parameters_constraint_fn_kwargs": {
            "interval_width": bounds[1] - bounds[0],
            "min_slope": 0.001,
            "min_bin_width": 0.001,
            "nbins": nbins,
        },
    },
    # {
    #     "bijector": "RealNVP",
    #     "bijector_kwargs": {
    #         "bijector": "MaskedAutoregressiveFlow",
    #         "bijector_kwargs": {
    #             "bijector": "bernstein_flow.bijectors.BernsteinPolynomial",
    #             "bijector_kwargs": {"extrapolation": False, "domain": domain},
    #         },
    #         "num_masked": 1,
    #     },
    #     "parameters_fn": "masked_autoregressive_network_with_additive_conditioner",
    #     "parameters_fn_kwargs": {
    #         "input_shape": (1,),
    #         "parameter_shape": [dims - 1, nbins],
    #         "made_kwargs": {
    #             "activation": "relu",
    #             "hidden_units": [32] * 2,
    #             "conditional": True,
    #             "conditional_event_shape": 1,
    #         },
    #         "x0_kwargs": {
    #             "activation": "relu",
    #             "hidden_units": [32] * 2,
    #             "batch_norm": False,
    #             "dropout": False,
    #             "conditional": True,
    #             "conditional_event_shape": 1,
    #         },
    #     },
    #     "parameters_constraint_fn": "bernstein_flow.activations.get_thetas_constrain_fn",
    #     "parameters_constraint_fn_kwargs": {
    #         "bounds": bounds,
    #         "allow_flexible_bounds": False,
    #         "smooth_bounds": False,
    #     },
    # },
]

hybrid_model = DensityRegressionModel(
    distribution="normalizing_flow",
    bijectors=bijectors,
    base_distribution_kwargs={"dims": dims},
)

# %% distribution
joint_dist = hybrid_model(tf.ones((5, 1)))
# tfd.Independent(dist, 2)
joint_dist

# %% log like
joint_dist.log_prob(tf.ones((5, 3)))

# %% trainable parameters
hybrid_model.trainable_parameters

# %% bijector
joint_dist.bijector  # .bijector.bijectors

# %% parameter function
params = hybrid_model.parameter_fn(tf.ones((2, 1)))
params

# %% fit model
hist = fit_distribution(
    model=hybrid_model,
    seed=seed,
    results_path=results_path,
    loss=nll_loss,
    # compile_kwargs={"run_eagerly": True},
    compile_kwargs={
        "jit_compile": True,
        "metrics": MeanNegativeLogLikelihood(name="nll"),
    },
    **preprocess_dataset(data, None),
    **fit_kwargs,
)

# %% Learning curve
pd.DataFrame(hist.history).plot()

# %% params after fit
t = np.linspace(0, max(X), 200, dtype="float32")
pv = hybrid_model.parameter_fn(t[..., None])[1].numpy().squeeze()
pv
fig, axs = plt.subplots(1, len(targets), sharex=True, sharey=True)
for (i, c), label in zip(enumerate(targets), targets):
    axs[i].plot(t, pv[:, i])
    axs[i].set_xlabel("cage")
    axs[i].set_title(label)
    axs[0].set_xticks((t.min(), t.max()))
    fig.tight_layout(w_pad=-0.1)
    fig.savefig(figure_path + f"/malnutrition_params.{img_ext}", bbox_inches="tight")


# %% marginal cdf and pdf
palette = "icefire"
palette = "rocket_r"
palette = "mako_r"
ages = [1, 3, 6, 9, 12, 24]
# ages = unscaled_train_data_df.cage.unique()
colors = sns.color_palette(palette, as_cmap=True)(np.linspace(0, 1, len(ages))).tolist()
joint_dist = hybrid_model(
    tf.convert_to_tensor(ages, dtype=hybrid_model.dtype)[..., None]
)
marginal_dist = tfd.TransformedDistribution(
    distribution=tfd.Normal(0, 1),
    bijector=tfb.Invert(tfb.Chain(joint_dist.bijector.bijector.bijectors[1:])),
)

y = np.linspace(-4, 4, 100)[..., None, None]

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
fig.savefig(figure_path + f"/malnutrition_dist.{img_ext}", bbox_inches="tight")


# %% samples
# plot_samples_grid(model, validation_data, 1000, targets, row="source")
samples_df = get_samples_df(1, hybrid_model, X, Y, targets)
# df["mage_combined"] = df.mage // 128
fig = plot_grid(
    samples_df.loc[samples_df.source == "model"].groupby("cage").sample(frac=0.2),
    vars=targets,
    hue="cage",
)
fig.set(xlim=(-4, 4))
fig.savefig(figure_path + f"/mctm_samples.{img_ext}", bbox_inches="tight")

# %% validation plots
samples_df = get_samples_df(1, hybrid_model, X, Y, targets)
cage = 30
column = targets[0]
measurements = samples_df[(samples_df.source == "data") & (samples_df.cage == cage)][
    column
]
samples = samples_df[(samples_df.source == "model") & (samples_df.cage == cage)][column]
print(len(samples))
fig = validation_plot(samples, measurements, feature_name=column, covariates=["cage"])
fig.savefig(figure_path + f"/validation_plots_{cage}.{img_ext}", bbox_inches="tight")

# %% reliability diagram (samples)
samples_df = get_samples_df(1, hybrid_model, X, Y, targets)
fig, axs = plt.subplots(1, len(targets), sharey=True, sharex=True)

for column, ax in zip(targets, axs):

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

    sns.lineplot(ecdf_df, x="model", y="data", hue="cage", estimator=None, ax=ax)
    ax.set_title(column)
fig.tight_layout(pad=0)
fig.savefig(
    figure_path + f"/reliability_plot_samples_{column}.{img_ext}", bbox_inches="tight"
)

# %% reliability diagram (cdf)
samples_df = get_samples_df(1, hybrid_model, X, Y, targets)
fig, axs = plt.subplots(1, len(targets), sharey=True, sharex=True)

for column, ax in zip(targets, axs):

    def apply_ecdf(df):
        data_df = df.loc[df.source == "data", column]
        marginal_dist = tfd.TransformedDistribution(
            distribution=tfd.Normal(0, 1),
            bijector=tfb.Invert(
                tfb.Chain(
                    hybrid_model(
                        df.cage.unique()[..., None]
                    ).bijector.bijector.bijectors[-2:]
                )
            ),
        )
        model_cdf = marginal_dist.cdf(data_df.copy().values[..., None]).numpy()[
            :, targets.index(column)
        ]
        data_ecdf = ecdf(data_df.copy(), data_df.copy())
        df.loc[df.source == "model", column + "_ecdf"] = model_cdf
        df.loc[df.source == "data", column + "_ecdf"] = data_ecdf
        return df

    samples_ecdf_df = (
        samples_df[["cage", "source", column]]
        .groupby("cage")
        .apply(apply_ecdf)
        .reset_index(0, drop=True)
        .sort_index()
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

    sns.lineplot(ecdf_df, x="model", y="data", hue="cage", estimator=None, ax=ax)
fig.tight_layout(pad=0)
fig.savefig(
    figure_path + f"/reliability_plot_samples_{column}.{img_ext}", bbox_inches="tight"
)


# %% reliability
reliability_df = get_samples_df(1, hybrid_model, X, Y, targets).pivot(columns="source")
reliability_df.columns = reliability_df.columns.map("{0[1]}_{0[0]}".format)
reliability_df = reliability_df.drop(columns="model_cage").rename(
    columns={"data_cage": "cage"}
)


def apply_cdf(df):
    data_cols = ["data_" + c for c in targets]
    model_cols = ["model_" + c for c in targets]
    measurements = df.loc[:, data_cols].values
    samples = df.loc[:, model_cols].values
    marginal_dist = tfd.TransformedDistribution(
        distribution=tfd.Normal(0, 1),
        bijector=tfb.Invert(
            tfb.Chain(
                hybrid_model(df.cage.unique()[..., None]).bijector.bijector.bijectors[
                    -2:
                ]
            )
        ),
    )
    model_cdf = marginal_dist.cdf(measurements).numpy()
    data_ecdf = np.stack(list(map(lambda x: ecdf(x, x), measurements.T)), 1)
    samples_ecdf = np.stack(list(map(lambda x: ecdf(x, x), samples.T)), 1)
    cdf_columns = ["cdf_" + c for c in targets]
    data_ecdf_columns = ["ecdf_data_" + c for c in targets]
    samples_ecdf_columns = ["ecdf_model_" + c for c in targets]
    df.loc[:, cdf_columns] = model_cdf
    df.loc[:, data_ecdf_columns] = data_ecdf
    df.loc[:, samples_ecdf_columns] = samples_ecdf
    return df


reliability_df = (
    reliability_df.groupby("cage")
    .apply(apply_cdf, include_groups=True)
    .reset_index(drop=True)
)

# Binning the predicted probabilities
# Create bins for the predicted probabilities
bins = np.linspace(0, 1, num=11)  # 10 equally spaced bins from 0 to 1
for column in targets:
    reliability_df.loc[:, "cdf_binned_" + column] = reliability_df.loc[
        :, "cdf_" + column
    ].apply(pd.cut, by_row=False, bins=bins, include_lowest=True)
for column in targets:
    reliability_df.loc[:, "ecdf_binned_" + column] = reliability_df.loc[
        :, "ecdf_model_" + column
    ].apply(pd.cut, by_row=False, bins=bins, include_lowest=True)


# %%
columnwidth = 400

fig, axs = plt.subplots(
    2,
    dims,
    sharey="row",
    sharex=True,
    # figsize=get_figsize(columnwidth, fraction=0.9),
)

common_errorbar_kwargs = dict(
    markersize=0.2,
    marker="o",
    # capsize=1,
    color="C0",
    linewidth=0.5,
)

# Iterate over groups for different kinds
for i, column in enumerate(targets):
    # Extract categories and corresponding mean ECDF values
    cdf_bin_col = "cdf_binned_" + column
    ecdf_bin_col = "ecdf_binned_" + column
    ecdf_data_col = "ecdf_data_" + column
    ecdf_model_col = "ecdf_model_" + column
    predicted_bins = reliability_df[cdf_bin_col].cat.categories.astype(str)
    grpd_data = reliability_df.groupby(cdf_bin_col)[ecdf_data_col]
    observed_freqs = grpd_data.mean()

    quantiles = grpd_data.quantile([0.25, 0.975]).unstack()

    pi = (quantiles - observed_freqs.values[..., None]).T.abs()

    axs[0][i].errorbar(
        predicted_bins, observed_freqs, yerr=pi, **common_errorbar_kwargs
    )
    axs[0][i].set_box_aspect(1)

    grpd_data = reliability_df.groupby(ecdf_bin_col)[ecdf_model_col]
    observed_freqs = grpd_data.mean()

    quantiles = grpd_data.quantile([0.25, 0.975]).unstack()

    pi = (quantiles - observed_freqs.values[..., None]).T.abs()

    axs[1][i].errorbar(
        predicted_bins, observed_freqs, yerr=pi, **common_errorbar_kwargs
    )
    axs[1][i].set_box_aspect(1)

    xticks = [0, len(predicted_bins) - 1]
    axs[1][i].set_xticks(xticks, predicted_bins[xticks])

    # Set labels and titles
    axs[0][i].set_title(
        f"{column.upper()}",
    )
    if i == 0:
        axs[0][i].set_ylabel("Observed relative\nfrequencies")
        axs[1][i].set_ylabel("Observed relative\nfrequencies")
    axs[1][i].set_xlabel("Predicted probabilities\n(binned)")

    # Add diagonal line
    axs[0][i].plot(
        [predicted_bins[0], predicted_bins[-1]],
        [0, 1],
        linestyle=":",
        linewidth=0.5,
        color="gray",
    )
    axs[1][i].plot(
        [predicted_bins[0], predicted_bins[-1]],
        [0, 1],
        linestyle=":",
        linewidth=0.5,
        color="gray",
    )

# axs[-1].set_xlabel("relative frequencies\n (samples)")

# Final adjustments
sns.despine()
# fig.autofmt_xdate()
fig.tight_layout(pad=0)
fig.savefig(figure_path + "/reliability_diagram.pdf", dpi=300, bbox_inches="tight")
