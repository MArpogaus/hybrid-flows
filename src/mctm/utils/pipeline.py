# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : pipeline.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-10-29 13:22:38 (Marcel Arpogaus)
# changed : 2024-10-29 13:40:17 (Marcel Arpogaus)

# %% License ###################################################################

# %% Description ###############################################################
"""Pipeline."""

# %% imports ###################################################################
import io
import logging
import os
import sys
from copy import deepcopy
from pprint import pformat
from typing import Any, Dict, Optional, Protocol, Tuple, Union

import dvc.api
import mlflow
import numpy as np
import yaml
from matplotlib.pyplot import Figure

from mctm.utils import filter_recursive
from mctm.utils.mlflow import log_cfg, start_run_with_exception_logging
from mctm.utils.tensorflow import fit_distribution, get_learning_rate, set_seed

# %% globals ###################################################################
__LOGGER__ = logging.getLogger(__name__)


# %% classes ###################################################################
class GetDataset(Protocol):
    """Signature for dataset retrieval function."""

    def __call__(self, **kwargs: Any) -> Tuple[Any, Any]:
        """Retrieve dataset."""


class GetModel(Protocol):
    """Signature for model creation function."""

    def __call__(self, dims: Any, **kwargs: Any) -> Any:
        """Create and return model."""


class DoPlotData(Protocol):
    """Signature for data plotting function."""

    def __call__(self, X: Any, Y: Any) -> Figure:
        """Plot data."""


class DoPreprocessDataset(Protocol):
    """Signature for dataset preprocessing function."""

    def __call__(self, data: Tuple[Any, Any], model: Any) -> Dict[str, Any]:
        """Preprocess dataset."""


class DoAfterFit(Protocol):
    """Signature for post-fit hook function."""

    def __call__(self, model: Any, **kwargs: Dict[str, Any]) -> None:
        """Execute post-fit operations."""


def prepare_pipeline(
    results_path: str,
    log_file: Optional[str],
    log_level: str,
    stage_name_or_params_file_path: Union[str, io.IOBase],
) -> Dict:
    """Prepare the pipeline configuration and load parameters.

    Parameters
    ----------
    results_path : str
        Directory path for storing results.
    log_file : Optional[str]
        File path for logging, or None.
    log_level : str
        Level of logging.
    stage_name_or_params_file_path : Union[str, io.IOBase]
        Path to the parameters file or a file-like object.

    Returns
    -------
    Dict
        Loaded parameters as a dictionary.

    """
    os.makedirs(results_path, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file = os.path.join(results_path, log_file)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    if isinstance(stage_name_or_params_file_path, io.IOBase):
        with stage_name_or_params_file_path as param_file:
            params = yaml.safe_load(param_file)
    else:
        params = dvc.api.params_show(stages=stage_name_or_params_file_path)

    __LOGGER__.info("params: %s", pformat(params))

    return params


def pipeline(
    experiment_name: str,
    run_name: str,
    results_path: str,
    log_file: Optional[str],
    seed: int,
    get_dataset_fn: GetDataset,
    dataset_kwargs: Dict[str, Any],
    get_model_fn: GetModel,
    model_kwargs: Dict[str, Any],
    preprocess_dataset: Optional[DoPreprocessDataset],
    fit_kwargs: Dict[str, Any],
    compile_kwargs: Dict[str, Any],
    plot_data: Optional[DoPlotData],
    after_fit_hook: Optional[DoAfterFit],
    **extra_params_to_log: Any,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """High-level machine learning pipeline for conducting experiments.

    This pipeline includes stages like dataset loading, model creation,
    dataset preprocessing, and model training with logging of results.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name for logging.
    run_name : str
        Name of the MLflow run.
    results_path : str
        Path for storing results and artifacts.
    log_file : Optional[str]
        Path to log file or None.
    seed : int
        Seed for random operations to ensure reproducibility.
    get_dataset_fn : GetDataset
        Callback function to load the dataset.
    dataset_kwargs : Dict[str, Any]
        Arguments for the dataset loading function.
    get_model_fn : GetModel
        Callback function to create the model.
    model_kwargs : Dict[str, Any]
        Arguments for the model creation function.
    preprocess_dataset : Optional[DoPreprocessDataset]
        Callback function for dataset preprocessing.
    fit_kwargs : Dict[str, Any]
        Arguments for the model fitting function.
    compile_kwargs : Dict[str, Any]
        Arguments for model compilation.
    plot_data : Optional[DoPlotData]
        Callback function for data plotting.
    after_fit_hook : Optional[DoAfterFit]
        Callback function executed after model fitting.
    extra_params_to_log: Dict[str, Any]
        Additional parameters to log with MLFlow.

    Returns
    -------
    Tuple[Dict[str, Any], Any, Dict[str, Any]]
        A tuple containing the training history, model, and preprocessed dataset.

    """
    learning_rate, lr_scheduler = get_learning_rate(fit_kwargs)
    fit_kwargs.update(learning_rate=learning_rate)
    call_args = filter_recursive(
        lambda x: not callable(x) and not isinstance(x, type),
        deepcopy(vars()),
    )

    # Drop Callback functions from MLFlow logging
    call_args["fit_kwargs"].pop("callbacks", None)

    set_seed(seed)
    data, dims = get_dataset_fn(**dataset_kwargs)
    model = get_model_fn(dims=dims, **model_kwargs)

    if experiment_name:
        mlflow.set_experiment(experiment_name)
        __LOGGER__.info("Logging to MLFlow Experiment: %s", experiment_name)
    with start_run_with_exception_logging(run_name=run_name):
        mlflow.autolog(log_models=False)
        mlflow.log_dict(call_args, "params.yaml")
        log_cfg(call_args)

        if plot_data:
            fig = plot_data(*data)
            fig.savefig(os.path.join(results_path, "dataset.pdf"))

        preprocessed = (
            preprocess_dataset(data, model)
            if preprocess_dataset
            else {"x": data[0], "y": data[1]}
        )
        fit_kwargs.update(preprocessed)

        hist = fit_distribution(
            model=model,
            seed=seed,
            results_path=results_path,
            compile_kwargs=compile_kwargs,
            **fit_kwargs,
        )

        if after_fit_hook:
            after_fit_hook(model, **preprocessed)
        min_idx = np.argmin(hist.history["val_loss"])
        min_loss = hist.history["loss"][min_idx]
        min_val_loss = hist.history["val_loss"][min_idx]
        epochs = len(hist.history["loss"])
        __LOGGER__.info("Training completed after %s epochs.", epochs)
        __LOGGER__.info("Best train loss: %s", min_loss)
        __LOGGER__.info("Best validation loss: %s", min_val_loss)
        __LOGGER__.info("Minimum loss reached after %s epochs.", min_idx)

        mlflow.log_metric("best_epoch", min_idx)
        mlflow.log_metric("final_epoch", epochs)
        mlflow.log_metric("min_loss", min_loss)
        mlflow.log_metric("min_val_loss", min_val_loss)

        with open(os.path.join(results_path, "metrics.yaml"), "w+") as results_file:
            yaml.dump({"loss": min_loss, "val_loss": min_val_loss}, results_file)

        mlflow.log_artifacts(results_path)

        return hist, model, preprocessed
