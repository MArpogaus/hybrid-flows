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

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from bernstein_flow.bijectors import BernsteinBijector
from matplotlib import pyplot as plt
from mctm.data.malnutrion import get_dataset
from mctm.models import DensityRegressionModel
from mctm.parameters import get_parameter_vector_fn
from mctm.utils.tensorflow import fit_distribution, set_seed
from mctm.utils.visualisation import get_figsize, setup_latex
from tensorflow import keras as K

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


def get_bernstein_polynomial_parameter_lambda(
    parameter_shape, polynomial_order, conditional_event_shape, high, low, dtype
):
    parameter_shape = (
        [conditional_event_shape]
        + parameter_shape
        + [
            polynomial_order + 1,
        ]
    )
    _, parameter_vector = get_parameter_vector_fn(
        parameter_shape=parameter_shape,
        dtype=dtype,  #  initializer=tf.ones
    )

    def get_parameter_lambda(conditional_input=None, **kwargs):
        b_poly = BernsteinBijector(parameter_vector)
        y = b_poly((conditional_input[..., None] - low) / (high - low))
        return tf.reduce_sum(y, 1)

    return get_parameter_lambda, parameter_vector


def slice_loc_scale(parameters):
    loc = parameters[..., :dims]
    scale_tril = tfp.bijectors.FillScaleTriL()(parameters[..., dims:])
    return {"loc": loc, "scale_tril": scale_tril}


# %% Setup Latex for plotting
setup_latex()

# %% Params
epochs = 100
seed = 1
covariates = ["cage"]
targets = ["stunting", "wasting", "underweight"]

dataset_kwargs = {
    "data_path": "datasets/malnutrition/india.raw",
    "covariates": covariates,
    "targets": targets,
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
preprocess_dataset = lambda data, model: {
    "x": data[0][0],
    "y": data[0][1],
    "validation_data": data[1],
}
results_path = "./results/malnutrition_new_nf_model"

# %% get dataset
set_seed(seed)
data, dims = get_dataset(
    **dataset_kwargs,
    scale=True,
    column_transformers=[
        ("passthrough", covariates),
    ],
)
data

# %% init Model
model = DensityRegressionModel(
    distribution="normalizing_flow",
    transformations=[
        {
            "bijector_name": "bernstein_poly",
            "parameters_shape": [dims, 8],
            "parameters_fn": "parameter_vector",
            "parameters_fn_kwargs": {"dtype": "float32"},
            "parameters_constraint_fn": "mctm.activations.get_thetas_constrain_fn",
            "parameters_constraint_fn_kwargs": {
                "low": -3,
            },
            "extrapolation": True,
        },
        {
            "bijector_name": "shift",
            "parameters_shape": [dims],
            "parameters_fn": get_bernstein_polynomial_parameter_lambda,  # "parameter_vector",
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
    ],
    base_distribution_kwargs={"dims": 0},
    # base_distribution_kwargs={
    #     "distribution_name": "tfd.MultivariateNormalTriL",
    #     "parameters_shape": [dims + np.sum(np.arange(dims + 1))],
    #     "parameters_fn": "parameter_vector_or_simple_network",
    #     "parameters_fn_kwargs": {
    #         # "input_shape": 3,
    #         # "hidden_units": [2] * 4,
    #         # "activation": "relu",
    #         # "batch_norm": False,
    #         # "dropout": False,
    #         "conditional": False,
    #         # "conditional_event_shape": (2),
    #     },
    #     # "parameters_fn": get_bernstein_polynomial_parameter_lambda,  # "parameter_vector",
    #     # "parameters_fn_kwargs": {
    #     #     "dtype": "float",
    #     #     "polynomial_order": 3,
    #     #     "conditional_event_shape": 1,
    #     #     "low": 0,
    #     #     "high": 35,
    #     # },
    #     "parameters_constraint_fn": slice_loc_scale,
    # },
)

# %% prepare data
preprocessed = preprocess_dataset(data, model)
X, Y, validation_data = preprocessed.values()

# %% distribution
dist = model(tf.ones((1, 1)))
# tfd.Independent(dist, 2)
dist

# %% log like
dist.log_prob(tf.ones((1, 3)))

# %% trainable parameters
model.trainable_parameters

# %% bijector
dist.bijector  # .bijector.bijectors

# %% parameter function
params = model.parameter_fn(tf.ones((2, 1)))
params

# %% fit model
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
t = np.linspace(0, max(X), 200, dtype="float32")
pv = model.parameter_fn(t[..., None])[1].numpy().squeeze()
pv
fig, axs = plt.subplots(1, len(targets), sharex=True, sharey=True)
for (i, c), label in zip(enumerate(targets), targets):
    axs[i].plot(t, pv[:, i])
    axs[i].set_xlabel("cage")
    axs[i].set_title(label)
axs[0].set_xticks((t.min(), t.max()))
fig.tight_layout(w_pad=-0.1)
fig.savefig("./org/gfx/malnutrition_params.png")

# %% marginal cdf and pdf
palette = "icefire"
palette = "rocket_r"
# palette = "mako_r"
ages = [1, 3, 6, 9, 12, 24, 35]
# ages = unscaled_train_data_df.cage.unique()
colors = sns.color_palette(palette, as_cmap=True)(np.linspace(0, 1, len(ages))).tolist()
dists = model(tf.convert_to_tensor(ages, dtype=model.dtype)[..., None])

y = np.linspace(0, 1, 100)[..., None, None]

cdf = dists.cdf(y)
pdf = dists.prob(y)
fig, axs = plt.subplots(
    2,
    len(targets),
    figsize=get_figsize("thesis"),
    sharey="row",
    sharex=True,
)

for i, c in enumerate(targets):
    axs[0, i].set_prop_cycle("color", colors)
    axs[0, i].plot(y.flatten(), cdf[..., i])
    axs[1, i].set_prop_cycle("color", colors)
    axs[1, i].plot(y.flatten(), pdf[..., i])
    axs[1, i].set_xlabel(f"y={c}")

axs[0, 0].set_ylabel(r"$F(y|\text{age})$")
axs[1, 0].set_ylabel(r"$f(y|\text{age})$")

fig.tight_layout(w_pad=0)
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
unscaled_train_data_df = get_samples_df(1, model, X, Y, targets)
# df["mage_combined"] = df.mage // 128
plot_grid(
    unscaled_train_data_df.loc[unscaled_train_data_df.source == "model"].sample(
        frac=0.2
    ),
    vars=targets,
    hue="mage",
)
