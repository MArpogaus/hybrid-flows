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
import tensorflow as tf
import tensorflow_probability as tfp
import yaml
from tensorflow_probability import distributions as tfd

from hybrid_flows.data import get_dataset, make_benchmark_dataset
from hybrid_flows.models import DensityRegressionModel, HybridDensityRegressionModel
from hybrid_flows.utils.mlflow import (
    log_and_save_figure,
    log_cfg,
    start_run_with_exception_logging,
)
from hybrid_flows.utils.pipeline import prepare_pipeline
from hybrid_flows.utils.visualisation import get_figsize, setup_latex

# %% globals ###################################################################
__LOGGER__ = logging.getLogger(__name__)


# %% functions #################################################################
def get_quantile(data_or_dist, q, i=None):
    """Get qunatile from either distribution or ecdf."""
    if isinstance(data_or_dist, tfd.Distribution):
        return data_or_dist.quantile(q)
    else:
        return np.quantile(data_or_dist[..., i], q)


def qq_plots(
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
    """Create Quantile-Quantile-Plot."""
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

        if i % n_cols == 0:
            ax.set_ylabel(ylabel)

        ax.set_xlabel(xlabel)

    # Hide any unused subplots
    for j in range(n_plots, len(axs)):
        axs[j].axis("off")

    fig.tight_layout(w_pad=0)

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
    figure_path = os.path.join(results_path, "eval_figures/")
    os.makedirs(figure_path, exist_ok=True)

    get_dataset_fn, get_dataset_kwargs, _ = get_dataset(
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        test_mode=False,
    )
    (train_data, validation_data, test_data), dims = get_dataset_fn(
        **get_dataset_kwargs
    )
    Y = validation_data

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
        model.compile(
            loss=lambda y, dist: -dist.log_prob(y), **params["compile_kwargs"]
        )

        train_loss = model.evaluate(
            make_benchmark_dataset(train_data, batch_size=2**10)
        )
        validation_loss = model.evaluate(
            make_benchmark_dataset(validation_data, batch_size=2**10)
        )
        test_loss = model.evaluate(make_benchmark_dataset(test_data, batch_size=2**10))

        __LOGGER__.info("train_loss: %.3f", train_loss)
        __LOGGER__.info("validation_loss: %.3f", validation_loss)
        __LOGGER__.info("test_loss: %.3f", test_loss)

        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("validation_loss", validation_loss)
        mlflow.log_metric("test_loss", test_loss)

        with open(
            os.path.join(results_path, "evaluation_metrics.yaml"), "w+"
        ) as results_file:
            yaml.dump(
                {
                    "validation_loss": validation_loss,
                    "test_loss": test_loss,
                    "train_loss": train_loss,
                },
                results_file,
            )

        setup_latex(fontsize=10)
        figsize = get_figsize(params["textwidth"])
        fig_width = figsize[1]

        joint_dist = model(None)
        normal_base = joint_dist.distribution.distribution

        z = joint_dist.bijector.inverse(Y)

        fig = qq_plots(
            z,
            normal_base,
            Y.shape[-1],
            n_cols=3,
            width=fig_width,
            xlabel="$Z$ Quantile",
            ylabel="Normal Quantile",
        )
        log_and_save_figure(
            figure=fig,
            figure_path=figure_path,
            file_name="_".join((dataset_name, model_name, "z", "qq")),
            file_format=figure_format,
            bbox_inches="tight",
            transparent=True,
        )

        if isinstance(model, HybridDensityRegressionModel):
            setup_latex(fontsize=10)
            marginal_dist = model.marginal_distribution(None)
            normal_base = marginal_dist.distribution.distribution

            w = marginal_dist.bijector.inverse(Y)

            fig = qq_plots(
                w,
                normal_base,
                Y.shape[-1],
                n_cols=3,
                width=fig_width,
                xlabel="$W$ Quantile",
                ylabel="Normal Quantile",
            )
            log_and_save_figure(
                figure=fig,
                figure_path=figure_path,
                file_name="_".join((dataset_name, model_name, "w", "qq")),
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
