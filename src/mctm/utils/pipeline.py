"""Pipeline."""
# IMPORT MODULES ###############################################################
import logging
import os
import sys
from typing import Any, Protocol

import dvc.api
import mlflow
import numpy as np
import yaml
from matplotlib.pyplot import Figure

from mctm.utils.mlflow import log_cfg, start_run_with_exception_logging
from mctm.utils.tensorflow import fit_distribution, set_seed

# MODULE GLOBAL OBJECTS ########################################################
__LOGGER__ = logging.getLogger(__name__)


# CLASS DEFINITIONS ############################################################

# Function signatures for pipeline callbacks


class getDataset(Protocol):
    """Callback."""

    def __call__(self) -> "tuple[Any, Any]":
        """Call."""


class getModel(Protocol):
    """Callback."""

    def __call__(self, dataset: "tuple[Any,Any]") -> Any:
        """Call."""


class doPlotData(Protocol):
    """Callback."""

    def __call__(self, X: Any, Y: Any) -> "Figure":
        """Call."""


class doPreprocessDataset(Protocol):
    """Callback."""

    def __call__(self, X: Any, Y: Any, model: Any) -> "dict":
        """Call."""


class doAfterFit(Protocol):
    """Callback."""

    def __call__(self, model: Any, x: Any, y: Any, **kwds: dict) -> None:
        """Call."""


# PUBLIC FUNCTIONS #############################################################
def pipeline(
    experiment_name: str,
    run_name: str,
    results_path: str,
    log_file: str,
    seed: int,
    get_dataset_fn: getDataset,
    dataset_kwds: dict,
    get_model_fn: getModel,
    model_kwds: dict,
    preprocess_dataset: doPreprocessDataset,
    fit_kwds: dict,
    plot_data: doPlotData,
    after_fit_hook: doAfterFit,
    **extra_params_to_log,
):
    """Pipeline.

    The function represents a high-level machine learning pipeline that can
    be used to perform the experiments.
    It includes various stages such as loading a dataset, creating a model,
    preprocessing the dataset,
    training the model, and logging experiment results.

    Notes:
     - get_dataset_fn is a callback because we have no common
       interface for how to generate a dataset
     - assumes models history has "loss" and "val_loss"

    :param str experiment_name: The name of the MLflow experiment to
                               log the results.
    :param str run_name: The name of the MLflow run.
    :param str results_path: The path where the results and artifacts
                            will be stored.
    :param str log_file: The path to a log file, or None.
    :param int seed: The random seed for reproducibility.
    :param callable get_dataset_fn: A function that loads and
                                   returns the dataset.
    :param dict dataset_kwds: Keyword arguments for the
                             dataset loading function.
    :param callable get_model_fn: A function that creates and returns the model.
    :param dict model_kwds: Keyword arguments for the model creation function.
    :param callable preprocess_dataset: A function for preprocessing
                                       the dataset.
    :param dict fit_kwds: Keyword arguments for the model fitting function.
    :param callable plot_data: A function for plotting the dataset.
    :param callable after_fit_hook: A function to be executed after
                                   model fitting.
    :param dict extra_params_to_log: Additional parameters to log
                                     to params.yaml.
    :return: A tuple containing the training history, model, and
             preprocessed dataset.
    :rtype: Tuple
    """
    call_args = dict(filter(lambda x: not callable(x[1]), vars().items()))
    set_seed(seed)
    data, dims = get_dataset_fn(**dataset_kwds)
    model = get_model_fn(dims=dims, **model_kwds)

    # prepare results directory
    os.makedirs(results_path, exist_ok=True)

    # Evaluate Model
    if experiment_name:
        mlflow.set_experiment(experiment_name)
        __LOGGER__.info("Logging to MLFlow Experiment: %s", experiment_name)
    with start_run_with_exception_logging(run_name=run_name):
        # Auto log all MLflow entities
        mlflow.autolog()
        mlflow.log_dict(call_args, "params.yaml")
        log_cfg(call_args)

        if plot_data:
            fig = plot_data(*data)
            fig.savefig(os.path.join(results_path, "dataset.pdf"))

        if preprocess_dataset:
            preprocessed = preprocess_dataset(data, model)
        else:
            preprocessed = {"x": data[0], "y": data[1]}

        hist = fit_distribution(
            model=model,
            seed=seed,
            results_path=results_path,
            **preprocessed,
            **fit_kwds,
        )

        if after_fit_hook:
            after_fit_hook(model, **preprocessed)
        min_idx = np.argmin(hist.history["val_loss"])
        min_loss = hist.history["loss"][min_idx]
        min_val_loss = hist.history["val_loss"][min_idx]
        epochs = len(hist.history["loss"])
        __LOGGER__.info("training finished after %s epochs.", epochs)
        __LOGGER__.info("best train loss: %s", min_loss)
        __LOGGER__.info("best validation loss: %s", min_val_loss)
        __LOGGER__.info("minimum reached after %s epochs", min_idx)

        mlflow.log_metric("best_epoch", min_idx)
        mlflow.log_metric("final_epoch", epochs)
        mlflow.log_metric("min_loss", min_loss)
        mlflow.log_metric("min_val_loss", min_val_loss)

        with open(os.path.join(results_path, "metrics.yaml"), "w+") as results_file:
            yaml.dump({"loss": min_loss, "val_loss": min_val_loss}, results_file)

        mlflow.log_artifacts(results_path)

        return hist, model, preprocessed


def prepare_pipeline(args):
    """Prepare the pipeline by configuring logging and loading parameters.

    It creates the output path and builds the parameters from the arguments.

    Expects:
     - args.results_path
     - args.log_file
     - args.log_level
     - args.stage_name

    :param args: Command-line arguments and parameters.
    :return: A dictionary containing loaded parameters.
    :rtype: dict
    """
    # configure logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        log_file = os.path.join(args.results_path, args.log_file)
        handlers += [
            logging.FileHandler(log_file),
        ]
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    __LOGGER__.info("CLI arguments: %s", vars(args))

    # load params
    params = dvc.api.params_show(stages=args.stage_name)
    __LOGGER__.info("DVC params: %s", params)

    return params
