# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : evaluate_sim.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-11-18 14:16:47 (Marcel Arpogaus)
# changed : 2024-11-28 17:46:04 (Marcel Arpogaus)

# %% License ###################################################################

# %% Description ###############################################################
"""Train multivariate density estimation models on different datasets."""

# %% imports ###################################################################
import argparse
import logging
import os
from copy import deepcopy
from shutil import which

import mlflow
import tensorflow as tf
import tensorflow_probability as tfp

from mctm.data.malnutrion import get_dataset
from mctm.models import DensityRegressionModel, HybridDensityRegressionModel
from mctm.utils.mlflow import (
    log_and_save_figure,
    log_cfg,
    start_run_with_exception_logging,
)
from mctm.utils.pipeline import prepare_pipeline
from mctm.utils.visualisation import (
    get_figsize,
    plot_malnutrition_data,
    plot_malnutrition_samples,
    setup_latex,
)

# %% globals ###################################################################
__LOGGER__ = logging.getLogger(__name__)


# %% functions #################################################################
def evaluate(
    dataset_name: str,
    dataset_type: str,
    experiment_name: str,
    run_name: str,
    results_path: str,
    params: dict,
    figure_format: str = "pdf",
) -> tuple:
    """Execute experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    dataset_type : str
        Type of the dataset.
    experiment_name : str
        Name of the MLFlow experiment.
    run_name : str
        Name of the MLFlow run.
    results_path : str
        Destination for model checkpoints and logs.
    params : dict
        Dictionary containing experiment parameters.

    Returns
    -------
    tuple
        Experiment results: history, model, and preprocessed data.

    """
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_name)
    mlflow.set_experiment(experiment_name)

    __LOGGER__.info(f"{tf.__version__=}\n{tfp.__version__=}")
    tf.config.set_visible_devices([], "GPU")

    figsize = get_figsize(params["textwidth"])
    fig_height = figsize[0]
    model_kwargs = params["model_kwargs"]
    dataset_kwargs = params["dataset_kwargs"][dataset_name]
    figure_path = os.path.join(results_path, "eval_figures")
    os.makedirs(figure_path, exist_ok=True)

    (train_data, validation_data, _), dims = get_dataset(**dataset_kwargs)

    if "marginal_bijectors" in model_kwargs.keys():
        get_model = HybridDensityRegressionModel
    else:
        get_model = DensityRegressionModel

    with start_run_with_exception_logging(run_name=run_name):
        log_cfg(
            dict(
                dataset_name=dataset_name, dataset_type=dataset_type, **deepcopy(params)
            )
        )

        model = get_model(dims=dims, **model_kwargs)
        model.load_weights(os.path.join(results_path, "model_checkpoint.weights.h5"))

        if which("latex"):
            __LOGGER__.info("Using latex backend for plotting")
            setup_latex(fontsize=10)
        fig = plot_malnutrition_data(
            validation_data,
            targets=dataset_kwargs["targets"],
            covariates=dataset_kwargs["covariates"],
            hue=dataset_kwargs["covariates"][0],
            seed=params["seed"],
            frac=0.8,
            height=fig_height / 3,
        )
        log_and_save_figure(
            figure=fig,
            figure_path=figure_path,
            file_name="data",
            file_format=figure_format,
            bbox_inches="tight",
            transparent=True,
        )

        fig = plot_malnutrition_samples(
            model=model,
            x=validation_data[0],
            y=validation_data[1],
            seed=params["seed"],
            targets=dataset_kwargs["targets"],
            height=fig_height / 3,
        )
        log_and_save_figure(
            figure=fig,
            figure_path=figure_path,
            file_name="samples",
            file_format=figure_format,
            bbox_inches="tight",
            transparent=True,
        )

        if isinstance(model, HybridDensityRegressionModel):
            pass


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
        "--experiment-name",
        type=str,
        help="MLFlow experiment name",
        required=True,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="MLFlow run name",
        required=True,
    )
    parser.add_argument(
        "--stage-name",
        type=str,
        help="name of dvc stage",
        required=True,
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

    params = prepare_pipeline(
        results_path=args.results_path,
        log_file=args.log_file,
        log_level=args.log_level,
        stage_name_or_params_file_path=args.stage_name,
    )

    evaluate(
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        results_path=args.results_path,
        params=params,
    )
