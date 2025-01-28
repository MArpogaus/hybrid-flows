# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : train.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-12-12 09:45:44 (Marcel Arpogaus)
# changed : 2025-01-28 17:29:00 (Marcel Arpogaus)

# %% License ###################################################################

# %% Description ###############################################################
"""Train multivariate density estimation models on different datasets."""

# %% imports ###################################################################
import argparse
import logging
import os
from functools import partial
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from mctm.data import get_dataset
from mctm.models import DensityRegressionModel, HybridDensityRegressionModel
from mctm.utils import str2bool
from mctm.utils.pipeline import pipeline, prepare_pipeline
from mctm.utils.visualisation import (
    get_figsize,
    plot_2d_data,
    plot_malnutrition_data,
    plot_malnutrition_samples,
    plot_samples,
    setup_latex,
)

# %% globals ###################################################################
__LOGGER__ = logging.getLogger(__name__)


# %% functions #################################################################
def sim_after_fit_hook(
    model: object,
    x: tf.Tensor,
    y: tf.Tensor,
    results_path: str,
    is_hybrid: bool,
    **kwargs,
) -> callable:
    """Provide after fit plot.

    Parameters
    ----------
    model : object
        The model that was trained.
    x : tf.Tensor
        Input data.
    y : tf.Tensor
        True values.
    results_path : str
        Path to save the result images.
    is_hybrid : bool
        Flag to indicate if hybrid model is used.
    **kwargs
        Additional arguments.

    Returns
    -------
    callable
        Hook function to be called after fitting the model.

    """
    if len(x) > 2000:
        indices = np.random.choice(len(x), size=2000, replace=False)
        x = x.numpy()[indices]
        y = y.numpy()[indices]

    dist = model(x)
    fig = plot_samples(dist, y, seed=1, **kwargs)
    fig.savefig(os.path.join(results_path, "samples.pdf"), bbox_inches="tight")


def malnutrition_after_fit_hook(
    model: object,
    x: tf.Tensor,
    validation_data: tuple,
    results_path: str,
    seed: int,
    targets: list,
    plot_samples_kwargs: Dict[str, Any] = {},
    plot_marginals_kwargs: Dict[str, Any] = {},
) -> None:
    """Provide after fit plot for malnutrition data.

    Parameters
    ----------
    model : object
        The model that was trained.
    x : tf.Tensor
        Input data.
    y : tf.Tensor
        True values.
    validation_data : tuple
        Validation data, containing features and targets.
    results_path : str
        Path to save the result images.
    N : int
        Number of samples to plot.
    seed : int
        Random seed for sampling.
    targets : list
        List of target variable names.
    plot_samples_kwargs : Dict[str, Any]
        Additional kwargs passed to `plot_malnutrition_samples`
    plot_marginals_kwargs : Dict[str, Any]
        Additional kwargs passed to `plot_marginal_cdf_and_pdf`

    """
    x, y = validation_data._input_dataset._tensors
    fig = plot_malnutrition_samples(
        model, x, y, seed, targets, frac=0.5, **plot_samples_kwargs
    )
    fig.savefig(os.path.join(results_path, "samples.pdf"), bbox_inches="tight")


def run(
    dataset_name: str,
    dataset_type: str,
    log_file: str,
    log_level: str,
    results_path: str,
    test_mode: bool,
    params: dict,
    model_name: str = None,
    experiment_name: str = None,
    run_name: str = None,
) -> tuple:
    """Execute experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    dataset_type : str
        Type of the dataset.
    log_file : str
        Path for log file.
    log_level : str
        Logging severity level.
    results_path : str
        Destination for model checkpoints and logs.
    test_mode : bool
        Flag to activate test-mode.
    params : dict
        Dictionary containing experiment parameters.
    model_name : str, optional
        Name of the model to train.
    experiment_name : str, optional
        Name of the MLFlow experiment.
    run_name : str, optional
        Name of the MLFlow run.

    Returns
    -------
    tuple
        Experiment results: history, model, and preprocessed data.

    """
    model_kwargs = params["model_kwargs"]
    fit_kwargs = params["fit_kwargs"]
    compile_kwargs = params["compile_kwargs"]
    dataset_kwargs = params["dataset_kwargs"][dataset_name]

    if "marginal_bijectors" in model_kwargs.keys():
        get_model = HybridDensityRegressionModel
    else:
        get_model = DensityRegressionModel

    get_dataset_fn, get_dataset_kwargs, preprocess_dataset = get_dataset(
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        test_mode=test_mode,
        fit_kwargs=fit_kwargs,
        **dataset_kwargs,
    )

    if dataset_type == "benchmark":
        plot_data = None
        after_fit_hook = False
    elif dataset_type == "malnutrition":
        figsize = get_figsize(params["textwidth"])
        fig_height = figsize[0]

        plot_data = partial(
            plot_malnutrition_data,
            targets=dataset_kwargs["targets"],
            covariates=dataset_kwargs["covariates"],
            seed=params["seed"],
            frac=0.2,
            hue="cage",
            height=fig_height / 3,
        )
        after_fit_hook = partial(
            malnutrition_after_fit_hook,
            seed=params["seed"],
            results_path=results_path,
            targets=dataset_kwargs["targets"],
            plot_samples_kwargs=dict(height=fig_height / 3),
            plot_marginals_kwargs=dict(figsize=figsize),
        )
    elif dataset_type == "sim":
        fig_height = get_figsize(params["textwidth"], fraction=0.5)[0]
        plot_data = partial(plot_2d_data, figsize=(fig_height, fig_height))
        after_fit_hook = partial(
            sim_after_fit_hook,
            results_path=results_path,
            is_hybrid=get_model == HybridDensityRegressionModel,
            height=fig_height,
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    if experiment_name is None:
        experiment_name = os.environ.get(
            "MLFLOW_EXPERIMENT_NAME", "_".join((dataset_type, "train"))
        )
    if run_name is None:
        run_name = "_".join((model_name, dataset_name, "train"))

    if test_mode:
        __LOGGER__.info("Running in test-mode")
        run_name += "_test"
        if isinstance(fit_kwargs, dict):
            fit_kwargs.update(epochs=2)
        elif isinstance(fit_kwargs, list):
            for fkw in fit_kwargs:
                fkw.update(epochs=2)

    setup_latex(fontsize=10)

    if os.environ.get("CI", False):
        if isinstance(fit_kwargs, dict):
            fit_kwargs.update(verbose=2)
        elif isinstance(fit_kwargs, list):
            for fkw in fit_kwargs:
                fkw.update(verbose=2)

    p_gpus = tf.config.list_physical_devices("GPU")
    for gpu in p_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # l_gpus = tf.config.list_logical_devices("GPU")
    # if len(l_gpus) > 0:
    #     strategy_scope = tf.distribute.MirroredStrategy(l_gpus).scope()
    # else:
    #     strategy_scope = nullcontext()

    # with strategy_scope:
    return pipeline(
        experiment_name=experiment_name,
        run_name=run_name,
        results_path=results_path,
        log_file=log_file,
        seed=params["seed"],
        get_dataset_fn=get_dataset_fn,
        dataset_kwargs=get_dataset_kwargs,
        get_model_fn=get_model,
        model_kwargs=model_kwargs,
        preprocess_dataset=preprocess_dataset,
        fit_kwargs=fit_kwargs,
        compile_kwargs=compile_kwargs,
        plot_data=plot_data,
        after_fit_hook=after_fit_hook,
        two_stage_training=params.get("two_stage_training", False),
    )


# %% main ######################################################################
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
        "--test-mode",
        default=False,
        type=str2bool,
        help="activate test-mode",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="MLFlow experiment name",
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

    results_path = os.path.join(
        args.results_path, args.dataset_type, args.dataset_name, args.model_name
    )

    params = prepare_pipeline(
        results_path=results_path,
        log_file=args.log_file,
        log_level=args.log_level,
        stage_name_or_params_file_path=args.stage_name,
    )

    run(
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        model_name=args.model_name,
        log_file=args.log_file,
        log_level=args.log_level,
        results_path=results_path,
        test_mode=args.test_mode,
        params=params,
    )
