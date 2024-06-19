# %% imports

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as K
from matplotlib import pyplot as plt
from mctm.data.sklearn_datasets import get_dataset
from mctm.models import DensityRegressionModel
from mctm.utils.tensorflow import fit_distribution, set_seed
from mctm.utils.visualisation import plot_2d_data
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd


# logging.basicConfig(level=logging.DEBUG)
# %% functions
def nll_loss(y, dist):
    marginal_dist = tfd.Independent(
        tfd.TransformedDistribution(
            distribution=tfd.Normal(0, 1),
            bijector=tfb.Invert(tfb.Chain(dist.bijector.bijector.bijectors[1:])),
        ),
        1,
    )

    return -dist.log_prob(y) - marginal_dist.log_prob(y)


# %% data
set_seed(1)
data, dims = get_dataset("moons", n_samples=2**16, scale=(0.01, 0.99), noise=0.05)
plot_2d_data(*data)

# %% model
model = DensityRegressionModel(
    distribution="normalizing_flow",
    bijectors=[
        {
            "bijector": "BernsteinBijector",
            "bijector_kwargs": {
                "extrapolation": True,
            },
            "parameters_fn": "parameter_vector",
            "parameters_fn_kwargs": {"parameter_shape": [dims, 50], "dtype": "float32"},
            # "parameters_fn": "bernstein_polynomial",  # "parameter_vector",
            # "parameters_fn_kwargs": {
            #     "parameter_shape": [dims, 25],
            #     "dtype": "float",
            #     "polynomial_order": 3,
            #     "conditional_event_shape": 1,
            #     "low": 0,
            #     "high": 35,
            # },
            # "parameter_fn": "parameter_vector_or_simple_network",
            # "parameter_fn_kwargs": {
            #     # "input_shape": 3,
            #     # "hidden_units": [2] * 4,
            #     # "activation": "relu",
            #     # "batch_norm": False,
            #     # "dropout": False,
            #     "conditional": True,
            #     "conditional_event_shape": (1),
            # },
            "parameters_constraint_fn": "mctm.activations.get_thetas_constrain_fn",
            "parameters_constraint_fn_kwargs": {
                "low": -4,
                "high": 4,
                "bounds": "identity",
                "allow_flexible_bounds": False,
            },
        },
        # {
        #     "bijector": "Shift",
        #     "parameters_fn": "bernstein_polynomial",  # "parameter_vector",
        #     "parameters_fn_kwargs": {
        #         "parameter_shape": [dims],
        #         "dtype": "float",
        #         "polynomial_order": 3,
        #         "conditional_event_shape": 1,
        #         "low": 0,
        #         "high": 35,
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
        #         "hidden_units": [16, 16],
        #         # "conditional": True,
        #         # "conditional_event_shape": 1,
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
                "bijector": "RationalQuadraticSpline",
                "bijector_kwargs": {
                    "range_min": -4,
                },
                "num_masked": 1,
            },
            "parameters_fn": "fully_connected_network",
            "parameters_fn_kwargs": {
                "parameter_shape": [1, 32 * 3 - 1],
                "activation": "relu",
                "hidden_units": [16, 16],
                "input_shape": 1,
                "batch_norm": False,
                "dropout": False,
                # "conditional": True,
                # "conditional_event_shape": 1,
            },
            "parameters_constraint_fn": "mctm.activations.get_spline_param_constrain_fn",
            "parameters_constraint_fn_kwargs": {
                "interval_width": 8,
                "min_slope": 0.001,
                "min_bin_width": 0.001,
                "nbins": 32,
            },
        },
    ],
    base_distribution_kwargs={"dims": dims},
)

# %% dist
dist = model(None)
dist.bijector.bijector.bijectors

# %% params
results_path = "./results/test_nf"
epochs = 100
seed = 1
covariates = ["cage"]
targets = ["stunting", "wasting", "underweight"]
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
preprocessed = {
    "x": tf.convert_to_tensor(data[1][..., None], dtype=tf.float32),
    "y": tf.convert_to_tensor(data[0], dtype=tf.float32),
}

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

# %% samples
x, y = preprocessed.values()

dist = model(x)
# tfd.Independent(dist, 2)
dist

samples = dist.sample(len(y))

df = pd.concat(
    (
        pd.DataFrame(samples, columns=["$x_1$", "$x_2$"]).assign(source="model"),
        pd.DataFrame(preprocessed["y"], columns=["$x_1$", "$x_2$"]).assign(
            source="data"
        ),
    )
)

sns.jointplot(
    df.groupby("source").sample(5000),
    x="$x_1$",
    y="$x_2$",
    hue="source",
    alpha=0.5,
)

# %% plot trafos
joint_dist = model(x)
marginal_dist = tfd.TransformedDistribution(
    distribution=tfd.Normal(0, 1),
    bijector=tfb.Invert(tfb.Chain(joint_dist.bijector.bijector.bijectors[1:])),
)

maf_bijector = joint_dist.bijector.bijector.bijectors[0]

x, y = preprocessed.values()
z = joint_dist.bijector.inverse(y)
z1 = marginal_dist.bijector.inverse(y)
z2 = maf_bijector(z1)
pit = marginal_dist.cdf(y)

df = pd.DataFrame(
    columns=[
        "$y1$",
        "$y2$",
        "$z_{2,1}$",
        "$z_{2,2}$",
        "$z_{1,1}$",
        "$z_{1,2}$",
        "$z_{1}$",
        "$z_{2}$",
        "$F_1(y_1)$",
        "$F_2(y_2)$",
        "$x$",
    ],
    data=np.concatenate([y, z2, z1, z, pit, x], -1),
)
g = sns.JointGrid(data=df, x="$y1$", y="$y2$", height=2)
g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
g.plot_marginals(sns.kdeplot)
data_figure = g.figure

g = sns.JointGrid(data=df, x="$z_{1,1}$", y="$z_{1,2}$", height=2)
g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
g.plot_marginals(sns.kdeplot)
normalized_data_figure = g.figure

g = sns.jointplot(df, x="$F_1(y_1)$", y="$F_2(y_2)$", height=2, s=4, alpha=0.5)
pit_figure = g.figure

g = sns.JointGrid(data=df, x="$z_{2,1}$", y="$z_{2,2}$", height=2)
g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
g.plot_marginals(sns.kdeplot)
decorelated_data_figure = g.figure

g = sns.JointGrid(data=df, x="$z_{1}$", y="$z_{2}$", height=2)
g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
g.plot_marginals(sns.kdeplot)
latent_dist_figure = g.figure

# %% plot coupula
joint_dist = model(x)
marginal_dist = tfd.Independent(
    tfd.TransformedDistribution(
        distribution=tfd.Normal(0, 1),
        bijector=tfb.Invert(tfb.Chain(joint_dist.bijector.bijector.bijectors[1:])),
    ),
    1,
)

n = 50
x = np.linspace(0, 1, n)
xx, yy = np.meshgrid(x, x)
grid = np.stack([xx.flatten(), yy.flatten()], -1)

p_y = joint_dist.prob(grid).numpy().reshape(-1, n)
p_z1 = marginal_dist.prob(grid).numpy().reshape(-1, n)

# c(y) = p_y(y) / p_z1(y)
c = p_y / p_z1
# c = np.where(p_z1 < 1e-4, 0, c)  # for numerical stability

fig = plt.figure(figsize=plt.figaspect(0.3))
ax = fig.add_subplot(131, projection="3d")
ax.plot_surface(
    xx,
    yy,
    p_y,
    cmap="plasma",
    linewidth=0,
    antialiased=False,
    alpha=0.5,
)
ax = fig.add_subplot(132, projection="3d")
ax.plot_surface(
    xx,
    yy,
    p_z1,
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
