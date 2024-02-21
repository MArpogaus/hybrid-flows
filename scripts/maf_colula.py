# %% imports
import os
from copy import deepcopy

import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from mctm.data.sklearn_datasets import get_dataset
from mctm.models import DensityRegressionModel, HybridDenistyRegressionModel
from mctm.utils.visualisation import plot_2d_data, plot_copula_function, plot_samples
from tensorflow_probability import distributions as tfd

# %% Globals
params_file_path = "../params.yaml"
results_path = "../results"
dataset = "moons"
stage_name = "unconditional_hybrid_pre_trained"
model_name = "masked_autoregressive_flow_first_dim_masked"
model_name = "coupling_flow"

# %% Set Seed
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)


# %% Utility functions
def load_model(dims, stage_name, dataset, model_name, results_path=None, **kwds):
    model_kwds = deepcopy(
        deepcopy(params)[f"{stage_name}_distributions"][model_name][dataset]
    )
    model_kwds["distribution_kwds"].update(**kwds)
    model_kwds.pop("fit_kwds")
    print(model_kwds)
    if "base_distribution" in model_kwds.keys():
        model_class = HybridDenistyRegressionModel
    else:
        model_class = DensityRegressionModel
    model = model_class(
        dims=dims,
        distribution=model_name,
        base_checkpoint_path_prefix=results_path,
        **model_kwds,
    )
    if results_path is not None:
        cp_path = os.path.join(
            results_path, "_".join((stage_name, model_name, dataset)), "mcp/weights"
        )

        if os.path.exists(cp_path + ".index"):
            print("Loading weights for ", model_name)
            model.load_weights(cp_path)
    return model


# %% Load Parameters
with open(params_file_path) as params_file:
    params = yaml.safe_load(params_file)

# %% Generate Data
# dataset = "circles"
(X, Y), dims = get_dataset(dataset, **params["datasets"][dataset])
plot_2d_data(X, Y)

# %% Load Model
model = load_model(
    dims,
    stage_name,
    dataset,
    model_name,
    results_path,
)

dist = model(None)
dist_z1 = dist.distribution
t_2 = dist.bijector
t_1 = dist_z1.bijector

# %% Plot Samples
fig = plot_samples(dist, X, seed=1)

# %% Plots Samples
# fig1, fig2, fig3 = plot_flow(dist, None, X, seed=1)
samples = dist.sample(1000).numpy()
sns.jointplot(x=samples[..., 0], y=samples[..., 1])

# %% Z Tranformation
# Normalize the Data
z_trafo_X = t_1.inverse(t_2.inverse(tf.convert_to_tensor(X, dtype=tf.float32)))
g = sns.jointplot(
    x=z_trafo_X[:, 0],
    y=z_trafo_X[:, 1],
    alpha=0.5,
    s=10,
)
g.plot_joint(sns.kdeplot, legend=False)

# %% U Tranformation
# Transform the Marginals to Uniform Distribution
dist2 = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=tf.zeros([dims]), scale=1), bijector=dist_z1.bijector
)
u_trafo_X = dist2.cdf(tf.convert_to_tensor(X, dtype=tf.float32))
g = sns.jointplot(
    x=u_trafo_X[:, 0],
    y=u_trafo_X[:, 1],
    alpha=0.5,
    s=10,
)
g.plot_joint(sns.kdeplot, legend=False)

# %% Copula Funktion
# wenn basis unabhänig gleichverteilt -> Basis Dichte == 1
# p_y = p_z2(h(y))*|det \nabla h(y)|
# c(y) = p_y(y) / p_z1(y)

# Contour Plot
fig = plot_copula_function(dist, Y, "contour", -0.1, 1.1, 100)

# %% Surface Plot
fig = plot_copula_function(dist, Y, "surface", -0.1, 1.1, 100)
