# %% imports
import os

import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from matplotlib import pyplot as plt
from tensorflow_probability import distributions as tfd

from hybrid_flows.data.benchmark import get_data
from hybrid_flows.models import DensityRegressionModel, HybridDensityRegressionModel
from hybrid_flows.utils.pipeline import prepare_pipeline
from hybrid_flows.utils.visualisation import (
    get_figsize,
    setup_latex,
)


# %% functions
def ecdf(samples, x):
    """Empirical cumulative density function."""
    ss = np.sort(samples)  # [..., None]
    cdf = np.searchsorted(ss, x, side="right") / float(ss.size)
    return cdf.astype(x.dtype)


def get_quantile(data_or_dist, q, i=None):
    if isinstance(data_or_dist, tfd.Distribution):
        return data_or_dist.quantile(q)
    else:
        return np.quantile(data_or_dist[..., i], q)


def qq_plot(
    x,
    y,
    n_plots,
    n_cols=2,
    width=3.5,
    n_probs=200,
    xlim=(-4.5, 4.5),
    ylim=(-4.5, 4.5),
    xlabel="Estimated Quantile",
    ylabel="Observed Quantile",
    **kwargs,
):
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        sharey=True,
        sharex=True,
        figsize=(width, n_rows / n_cols * width),
        **kwargs,
    )
    axs = axs.flatten()  # Flatten the array for easier indexing

    q = np.linspace(0, 1, n_probs)
    for i in range(n_plots):
        ax = axs[i]
        x_quantiles = get_quantile(x, q, i)
        y_quantiles = get_quantile(y, q, i)

        ax.plot(xlim, ylim, "k:", linewidth=1)
        ax.plot(x_quantiles, y_quantiles)

        # ax.set_title(targets[i])
        ax.set_aspect("equal")
        ax.set(xlim=xlim, ylim=ylim)
        ax.text(
            0.1,
            0.9,
            i,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )

        if i % n_cols == 0:
            ax.set_ylabel(ylabel)

        ax.set_xlabel(xlabel)

    # Hide any unused subplots
    for j in range(n_plots, len(axs)):
        axs[j].axis("off")

    fig.tight_layout(w_pad=0)

    return fig


# %% globals
exp_path = "./experiments/"
if os.path.exists(exp_path):
    os.chdir(exp_path)
with open("./params.yaml") as f:
    params = yaml.safe_load(f)

dataset_type = "benchmark"
dataset_names = list(map(lambda l: l["name"], params[f"{dataset_type}_datasets"]))
dataset = "bsds300"
dataset = "minibone"
dataset = "hepmass"
dataset = "gas"
dataset = "power"
model_name = "unconditional_hybrid_masked_autoregressive_flow_quadratic_spline"
stage_name = f"train-{dataset_type}@dataset{dataset_names.index(dataset)}-{model_name}"
results_path = os.path.join("results/", dataset_type, dataset, "", model_name)
figure_path = os.path.join(results_path, "figures")
fig_ext = "pdf"

os.makedirs(figure_path, exist_ok=True)

# %% load params
params = prepare_pipeline(
    results_path=results_path,
    log_file=None,
    log_level="info",
    stage_name_or_params_file_path=stage_name,
)

# %% prepare plotting
figsize = get_figsize(params["textwidth"])
fig_width = figsize[1]
setup_latex(fontsize=10)


# %% load model
tf.config.set_visible_devices([], "GPU")

model_kwargs = params["model_kwargs"]
dataset_kwargs = params["dataset_kwargs"][dataset]
(train_data, validation_data, _), dims = get_data(dataset)
Y = validation_data

if "marginal_bijectors" in model_kwargs.keys():
    get_model = HybridDensityRegressionModel
else:
    get_model = DensityRegressionModel

model = get_model(dims=dims, **model_kwargs)
model.load_weights(os.path.join(results_path, "model_checkpoint.weights.h5"))

# %% Empirical Data Distribution
# sns.kdeplot(train_data, fill=False)
y_min, y_max = Y.min(0), Y.max(0)

print(
    yaml.safe_dump({"domain": np.stack([np.floor(y_min), np.ceil(y_max)], 0).tolist()})
)
# %% Q-Q marginal
dist = model.marginal_distribution(None)
flow = dist.bijector
normal_base = dist.distribution.distribution

w = flow.inverse(Y[:10000])

fig = qq_plot(
    w,
    normal_base,
    Y.shape[-1],
    n_cols=3,
    width=fig_width,
    xlabel="$W$ Quantile",
    ylabel="Normal Quantile",
)
# plt.savefig("qq_power_M_4048.pdf")

# %% Q-Q joint
joint_dist = model(None)
normal_base = dist.distribution.distribution

z = joint_dist.bijector.inverse(Y)

fig = qq_plot(
    z,
    normal_base,
    Y.shape[-1],
    n_cols=3,
    width=fig_width,
    xlabel="$Z$ Quantile",
    ylabel="Normal Quantile",
)
# %% bp
from bernstein_flow.bijectors import BernsteinPolynomial

domain = [[-1.0, -5.0, -1.0, -1.0, -1.0, -2.0], [10.0, 4.0, 14.0, 14.0, 3.0, 2.0]]
b_poly = BernsteinPolynomial([[0, 1] * 6], domain=domain)
b_poly([[2] * 6])

xbp = np.linspace(-6, 15, 200)
ybp = b_poly(xbp[..., None])

plt.plot(xbp, ybp)
plt.legend(np.transpose(domain))

# %%
b_poly.inverse(xbp[..., None])
# %%
sns.kdeplot(train_data[..., 2], fill=False)
# %%
yy = np.linspace(*domain, 200)
m_dist = tfd.TransformedDistribution(distribution=normal_base, bijector=dist.bijector)
probs = m_dist.log_prob(yy)
probs.shape

# %%
from scipy import stats

i = 2
kde_probs = stats.gaussian_kde(train_data[..., i])(yy[..., i])
kde_log_probs = np.log(kde_probs)

fig = plt.figure()
plt.plot(yy[..., i], kde_log_probs)
plt.plot(yy[..., i], probs[..., i])

# plt.savefig("log_probs_vs_kde_M_4048.pdf")
