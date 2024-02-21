# %% imports
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pyvinecopulib as pv
import seaborn as sns
import tensorflow as tf
import yaml
from matplotlib import pyplot as plt
from mctm.data.sklearn_datasets import get_dataset
from mctm.models import DensityRegressionModel, HybridDenistyRegressionModel
from mctm.utils.visualisation import plot_2d_data, plot_samples
from tensorflow_probability import distributions as tfd

# %% Globals
params_file_path = "../params.yaml"
results_path = "../results"
dataset = "moons"
stage_name = "unconditional"

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
        get_model = HybridDenistyRegressionModel
    else:
        get_model = DensityRegressionModel
    model = get_model(dims=dims, distribution=model_name, **model_kwds)
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
model = load_model(dims, stage_name, dataset, "elementwise_flow", results_path)

dist = model(None)
# dist.distribution = (tfd.Normal(loc=tf.zeros([dims]), scale=1),)
bijector = dist.bijector

# %% Plot Samples
fig = plot_samples(dist, X, seed=1)

# %% Z Tranformation
# Normalize the Data
z_trafo_X = bijector.inverse(tf.convert_to_tensor(X, dtype=tf.float32))
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
    distribution=tfd.Normal(loc=tf.zeros([dims]), scale=1), bijector=bijector
)
u_trafo_X = dist2.cdf(tf.convert_to_tensor(X, dtype=tf.float32))
g = sns.jointplot(
    x=u_trafo_X[:, 0],
    y=u_trafo_X[:, 1],
    alpha=0.5,
    s=10,
)
g.plot_joint(sns.kdeplot, legend=False)

# %% Create DataFrame
df = pd.DataFrame(
    data=np.concatenate([Y[..., None], X, z_trafo_X, u_trafo_X], -1),
    columns=["y", "x1", "x2", "z_x1", "z_x2", "u_x1", "u_x2"],
)
df
df.to_csv("copula.csv")
# %%

bicop = pv.Bicop(u_trafo_X)
bicop

# %%
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
grid = np.array(np.meshgrid(x, y)).reshape(2, -1)

pdf = bicop.pdf(grid.T)

plt.contour(*np.meshgrid(x, y), pdf.reshape(100, -1))
plt.scatter(u_trafo_X[:, 0], u_trafo_X[:, 1])

# %% Transform to Uniform
# Rosenblatt Trafo
# https://vinecopulib.github.io/rvinecopulib/reference/rosenblatt.html
x_uni = bicop.hfunc1(u_trafo_X)
plt.scatter(u_trafo_X[:, 0], x_uni)

# %% Transform to Normal
x_norm = tfd.Normal(0, 1).quantile([u_trafo_X[:, 0], x_uni]).numpy()
plt.scatter(x_norm[0, :], x_norm[1, :])

# %%
# wenn basis unabhänig gleichverteilt -> Basis Dichte == 1
# p_y = p_z2(h(y))*|det \nabla h(y)|
# c(y) = p_y(y) / p_z1(y)
