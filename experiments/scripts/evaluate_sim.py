# %% Description ###############################################################
"""Train multivariate density estimation models on different datasets."""

# %% imports ###################################################################
import argparse
import logging
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from tensorflow_probability import distributions as tfd

from hybrid_flows.data import get_dataset
from hybrid_flows.models import DensityRegressionModel, HybridDensityRegressionModel
from hybrid_flows.utils.mlflow import (
    log_and_save_figure,
    log_cfg,
    start_run_with_exception_logging,
)
from hybrid_flows.utils.pipeline import prepare_pipeline
from hybrid_flows.utils.visualisation import plot_samples, setup_latex

# %% globals ###################################################################
__LOGGER__ = logging.getLogger(__name__)


# %% functions #################################################################
def pdf_contour_plot(model, n=200):
    """Plot density as contour plot."""
    ls = np.linspace(0, 1, n, dtype=np.float32)
    xx, yy = np.meshgrid(ls[..., None], ls[..., None])
    grid = np.stack([xx.flatten(), yy.flatten()], -1)

    try:
        joint_dist = model(None)
        p_y = joint_dist.prob(grid).numpy().reshape(-1, n)
    except (TypeError, ValueError):
        joint_dist_0 = model(tf.convert_to_tensor([0.0]))
        joint_dist_1 = model(tf.convert_to_tensor([1.0]))
        p_y_0 = joint_dist_0.prob(grid).numpy().reshape(-1, n)
        p_y_1 = joint_dist_1.prob(grid).numpy().reshape(-1, n)
        p_y = p_y_0 + p_y_1

    fig = plt.figure(figsize=plt.figaspect(1))
    plt.contourf(
        xx,
        yy,
        p_y,
        cmap="plasma",
    )
    plt.axis("off")
    return fig


def plot_joint_and_marginal_samples(model, x, *args, **kwargs):
    """Plot samples from model."""
    joint_dist = model(x)

    marginal_fig = None
    joint_fig = plot_samples(joint_dist, *args, **kwargs)
    if isinstance(model, HybridDensityRegressionModel):
        marginal_dist = model.marginal_distribution(x)

        marginal_fig = plot_samples(marginal_dist, *args, **kwargs)

    return joint_fig, marginal_fig


def plot_hybrid_model(model, x, y):
    """Plot transformed samples of hybrid models."""
    joint_dist = model.joint_distribution(x)
    marginal_dist = model.marginal_distribution(x)
    w = marginal_dist.bijector.inverse(y)
    z = joint_dist.bijector.inverse(y)
    # HACK: Sample dist dos not support cdf
    marginal_dist2 = tfd.TransformedDistribution(
        distribution=marginal_dist.distribution.distribution.distribution,
        bijector=marginal_dist.bijector,
    )
    pit = marginal_dist2.cdf(y)

    df = pd.DataFrame(
        columns=[
            "$y_1$",
            "$y_2$",
            "$w_{1}$",
            "$w_{2}$",
            "$z_{1}$",
            "$z_{2}$",
            "$F_1(y_1)$",
            "$F_2(y_2)$",
            "$x$",
        ],
        data=np.concatenate([y, w, z, pit, x[..., None]], -1),
    )

    # %% data
    g = sns.JointGrid(data=df, x="$y_1$", y="$y_2$", height=2)
    g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
    g.plot_marginals(sns.histplot)
    data_fig = g.figure

    # %% normalized data
    g = sns.JointGrid(data=df, x="$w_{1}$", y="$w_{2}$", height=2)
    g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
    g.plot_marginals(sns.histplot)
    w_fig = g.figure

    # %% decorrelated data
    g = sns.JointGrid(data=df, x="$z_{1}$", y="$z_{2}$", height=2)
    g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
    g.plot_marginals(sns.histplot)
    z_fig = g.figure

    # %% PIT
    g = sns.jointplot(df, x="$F_1(y_1)$", y="$F_2(y_2)$", height=2, s=4, alpha=0.5)
    pit_fig = g.figure

    return data_fig, w_fig, z_fig, pit_fig


def plot_density_3d(model, n=200):
    """Plot marginal, joint an copula density."""
    ls = np.linspace(0, 1, n, dtype=np.float32)
    xx, yy = np.meshgrid(ls[..., None], ls[..., None])
    grid = np.stack([xx.flatten(), yy.flatten()], -1)
    try:
        joint_dist = model.joint_distribution(None)
        marginal_dist = model.marginal_distribution(None)
        p_y = joint_dist.prob(grid).numpy()
        p_z1 = marginal_dist.prob(grid).numpy()
    except (TypeError, ValueError):
        joint_dist_0 = model.joint_distribution(tf.convert_to_tensor([0.0]))
        joint_dist_1 = model.joint_distribution(tf.convert_to_tensor([1.0]))
        marginal_dist_0 = model.marginal_distribution(tf.convert_to_tensor([0.0]))
        marginal_dist_1 = model.marginal_distribution(tf.convert_to_tensor([1.0]))
        p_y_0 = joint_dist_0.prob(grid).numpy()
        p_y_1 = joint_dist_1.prob(grid).numpy()
        p_y = (p_y_0 + p_y_1) / 2
        p_z1_0 = marginal_dist_0.prob(grid).numpy()
        p_z1_1 = marginal_dist_1.prob(grid).numpy()
        p_z1 = (p_z1_0 + p_z1_1) / 2

    c = p_y / p_z1
    c = np.where(p_z1 < 1e-4, 0, c)  # for numerical stability

    fig = plt.figure(figsize=plt.figaspect(0.32))
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
    ax.set_title("Joint Density")
    ax.set_zlabel("$f_Y(\mathbf{y}|x)$")
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
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
    ax.set_title("Marginal Density")
    ax.set_zlabel("$F_Y(\mathbf{y}|x)$???")
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")
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
    ax.set_title("Copula Density")
    ax.set_zlabel("$c_U(\mathbf{u}|x)$???")
    ax.set_xlabel("$u_1$")
    ax.set_ylabel("$u_2$")

    fig.tight_layout()

    return fig


def evaluate(
    dataset_name: str,
    dataset_type: str,
    model_name: str,
    results_path: str,
    params: dict,
    figure_format: str = "pdf",
    experiment_name: str = None,
) -> tuple:
    """Execute experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    dataset_type : str
        Type of the dataset.
    model_name : str
        Name of the modle to evaluate.
    results_path : str
        Destination for model checkpoints and logs.
    params : dict
        Dictionary containing experiment parameters.
    figure_format: str
        Data format to save figures to.
    experiment_name : str, optional
        Name of the MLFlow experiment.

    Returns
    -------
    tuple
        Experiment results: history, model, and preprocessed data.

    """
    if experiment_name is None:
        experiment_name = os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", "_".join((dataset_type, "evaluation"))
        )
    run_name = "_".join((model_name, dataset_name, "evaluation"))

    __LOGGER__.info(f"{tf.__version__=}\n{tfp.__version__=}")
    tf.config.set_visible_devices([], "GPU")

    model_kwargs = params["model_kwargs"]
    dataset_kwargs = params["dataset_kwargs"][dataset_name]
    figure_path = os.path.join(results_path, "eval_figures/")
    os.makedirs(figure_path, exist_ok=True)

    get_dataset_fn, get_dataset_kwargs, preprocess_dataset = get_dataset(
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        test_mode=False,
        test_data=True,
        **dataset_kwargs,
    )
    data, dims = get_dataset_fn(**get_dataset_kwargs)

    if "marginal_bijectors" in model_kwargs.keys():
        get_model = HybridDensityRegressionModel
    else:
        get_model = DensityRegressionModel

    mlflow.set_experiment(experiment_name)
    with start_run_with_exception_logging(run_name=run_name):
        log_cfg(
            dict(
                dataset_name=dataset_name, dataset_type=dataset_type, **deepcopy(params)
            )
        )

        model = get_model(dims=dims, **model_kwargs)
        model.load_weights(os.path.join(results_path, "model_checkpoint.weights.h5"))
        model.compile(loss=lambda y, dist: -dist.log_prob(y))

        preprocessed = preprocess_dataset(data, model)
        x, y = preprocessed.values()

        test_loss = model.evaluate(x, y, batch_size=2**10)
        __LOGGER__.info("test_loss: %.4f", test_loss)
        mlflow.log_metric("test_loss", test_loss)

        with open(
            os.path.join(results_path, "evaluation_metrics.yaml"), "w+"
        ) as results_file:
            yaml.dump(
                {"test_loss": test_loss},
                results_file,
            )

        setup_latex(fontsize=10)
        fig = pdf_contour_plot(model)
        log_and_save_figure(
            figure=fig,
            figure_path=figure_path,
            file_name="_".join((dataset_name, model_name, "contour")),
            file_format=figure_format,
            bbox_inches="tight",
            transparent=True,
        )

        setup_latex(fontsize=10)
        joint_fig, marginal_fig = plot_joint_and_marginal_samples(model, x=x, data=y)
        log_and_save_figure(
            figure=joint_fig,
            figure_path=figure_path,
            file_name="_".join((dataset_name, model_name, "joint_samples")),
            file_format=figure_format,
            bbox_inches="tight",
            transparent=True,
        )
        if marginal_fig is not None:
            log_and_save_figure(
                figure=marginal_fig,
                figure_path=figure_path,
                file_name="_".join((dataset_name, model_name, "marginal_samples")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
            )

        if isinstance(model, HybridDensityRegressionModel):
            setup_latex(fontsize=10)
            data_fig, w_fig, z_fig, pit_fig = plot_hybrid_model(model, x, y)
            log_and_save_figure(
                figure=data_fig,
                figure_path=figure_path,
                file_name=dataset_name,
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
            )
            log_and_save_figure(
                figure=w_fig,
                figure_path=figure_path,
                file_name="_".join((dataset_name, model_name, "w")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
            )
            log_and_save_figure(
                figure=z_fig,
                figure_path=figure_path,
                file_name="_".join((dataset_name, model_name, "z")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
            )
            log_and_save_figure(
                figure=pit_fig,
                figure_path=figure_path,
                file_name="_".join((dataset_name, model_name, "pit")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
            )

            setup_latex(fontsize=10)
            density_fig = plot_density_3d(model)
            log_and_save_figure(
                figure=density_fig,
                figure_path=figure_path,
                file_name="_".join((dataset_name, model_name, "densities_3d")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
            )


# %% if main ###################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--log-file",
        type=str,
        help="path for log file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="logging severity level",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name of model to evaluate.",
        required=True,
    )
    parser.add_argument(
        "--stage-name",
        type=str,
        help="name of dvc stage",
        required=True,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="name of MLFlow experiment",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--dataset-type",
        type=str,
        help="type of dataset",
        required=True,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="name of dataset",
        required=True,
    )
    parser.add_argument(
        "--results-path",
        type=str,
        help="destination for model checkpoints and logs.",
        required=True,
    )

    args = parser.parse_args()
    __LOGGER__.info("CLI arguments: %s", vars(args))

    results_path = os.path.join(
        args.results_path, args.dataset_type, args.dataset_name, args.model_name
    )

    params = prepare_pipeline(
        results_path=results_path,
        log_file=args.log_file,
        log_level=args.log_level,
        stage_name_or_params_file_path=args.stage_name,
    )

    evaluate(
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        model_name=args.model_name,
        results_path=results_path,
        params=params,
        experiment_name=args.experiment_name,
    )
