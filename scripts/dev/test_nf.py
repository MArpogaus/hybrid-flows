# %% imports

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as K
from matplotlib import pyplot as plt
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


# %% sim data
from mctm.data.sklearn_datasets import get_dataset

set_seed(1)
data, dims = get_dataset("moons", n_samples=2**16, scale=(0.01, 0.99), noise=0.05)
plot_2d_data(*data)
moons_preprocessed = {
    "x": tf.convert_to_tensor(data[1][..., None], dtype=tf.float32),
    "y": tf.convert_to_tensor(data[0], dtype=tf.float32),
}
x_y_moons = moons_preprocessed.values()

# %% malnutrition data
from mctm.data.malnutrion import get_dataset

seed = 1
covariates = ["cage"]
targets = ["stunting", "wasting", "underweight"]

dataset_kwargs = {
    "data_path": "datasets/malnutrition/india.raw",
    "covariates": covariates,
    "targets": targets,
    "stratify": True,
}
set_seed(seed)
data, dims = get_dataset(
    **dataset_kwargs,
    scale=True,
    column_transformers=[
        ("passthrough", covariates),
    ],
)
india_preprocessed = {
    "x": tf.convert_to_tensor(data[0][0], dtype=tf.float32),
    "y": tf.convert_to_tensor(data[0][1][:, 1:], dtype=tf.float32),
    # "validation_data": data[1],
}
dims -= 1
x_y_india = india_preprocessed.values()
india_dataset = tf.data.Dataset.from_tensor_slices((data[0][0], data[0][1][:, 1:]))
india_dataset

# %% model
nbins = 16
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
                "parameter_shape": [1, nbins * 3 - 1],
                "activation": "relu",
                "hidden_units": [16, 16],
                "input_shape": (1,),
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
                "nbins": nbins,
            },
        },
    ],
    base_distribution_kwargs={"dims": dims},
)

# %% dist
dist = model(None)
dist

# %% params
results_path = "./results/test_nf"
epochs = 100
seed = 1
initial_learning_rate = 0.005
scheduler = K.optimizers.schedules.CosineDecay(
    initial_learning_rate,
    decay_steps=epochs,
    # end_learning_rate=0.00001,
    # power=1,
)
fit_kwargs = {
    "epochs": epochs,
    # "validation_split": 0.25,
    "batch_size": 128,
    "learning_rate": initial_learning_rate,
    "callbacks": [K.callbacks.LearningRateScheduler(scheduler)],
    "lr_patience": 50,
    "reduce_lr_on_plateau": False,
    "early_stopping": False,
    "verbose": True,
    "monitor": "val_loss",
}


model.compile(loss=nll_loss, optimizer="adam")  # , run_eagerly=True)
model.fit(
    **india_preprocessed,  # .batch(32, drop_remainder=True),
    validation_split=None,
)
model.make_train_function()(
    iter(
        tf.data.Dataset.from_tensor_slices(tuple(moons_preprocessed.values())).batch(32)
    )
)
model.train_step(india_dataset)


# %% data handler
from keras.src.engine import data_adapter

data_handler = data_adapter.get_data_handler(
    **india_preprocessed,
    model=model,
)
train_func = model.make_train_function(force=True)


def my_make_train_function(model):
    def step_function(model, iterator):
        """Runs a single training step."""

        def run_step(data):
            outputs = model.train_step(data)
            # Ensure counter is updated only if `train_step` succeeds.
            # with tf.control_dependencies(_minimum_control_deps(outputs)):
            #     model._train_counter.assign_add(1)
            return outputs

        # if self.jit_compile:
        #     run_step = tf.function(
        #         run_step, jit_compile=True, reduce_retracing=True
        #     )
        data = next(iterator)
        outputs = model.distribute_strategy.run(run_step, args=(data,))
        # outputs = reduce_per_replica(
        #     outputs,
        #     self.distribute_strategy,
        #     reduction=self.distribute_reduction_method,
        # )
        return outputs

    @tf.function(jit_compile=True)
    def train_func(iterator):
        """Runs a training execution with a single step."""
        # return step_function(model, iterator)
        for data in iterator:
            print(data[0].shape[0])
            model.train_step(data)

    return train_func


train_func = my_make_train_function(model)

# train_func = tf.function(
#     train_func, reduce_retracing=True
# )

for epoch, iterator in data_handler.enumerate_epochs():
    train_func(iterator)
# %%
l = list(iterator)
tf.function(l[-1][1], model(None))
model.train_step(l[-1])

x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(l[-1])
# Run forward pass.
with tf.GradientTape() as tape:
    y_pred = model(x, training=True)
    loss = model.compute_loss(x, y, y_pred, sample_weight)
model._validate_target_and_loss(y, loss)
# Run backwards pass.
model.optimizer.minimize(loss, model.trainable_variables, tape=tape)

# %% fit model
hist = fit_distribution(
    model=model,
    seed=seed,
    results_path=results_path,
    loss=nll_loss,
    # **india_preprocessed,
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
