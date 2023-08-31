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


class getDataset(Protocol):
    def __call__(self) -> "tuple[Any, Any]":
        pass


class getModel(Protocol):
    def __call__(self, dataset: "tuple[Any,Any]") -> Any:
        pass


class doPlotData(Protocol):
    def __call__(self, X: Any, Y: Any) -> "Figure":
        pass


class doPlotSamples(Protocol):
    def __call__(self, X: Any, X_: Any) -> "Figure":
        pass


class doPreprocessDataset(Protocol):
    def __call__(self, X: Any, Y: Any, model: Any) -> "dict":
        pass


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
    plot_samples: doPlotSamples,
):
    """
    get_dataset is callback because we have no common
    interface for how to generate a dataset (?!)
    assumes models history has "loss" and "val_loss"
    log_file is optional and can be none.
    params: params from params.yaml to be logged
    plot_data is optional
    plot_samples is optional
    """
    call_args = dict(filter(lambda x: not callable(x[1]), vars().items()))
    set_seed(seed)
    X, Y = get_dataset_fn(**dataset_kwds)
    model = get_model_fn(dims=X.shape[-1], **model_kwds)

    # Evaluate Model
    mlflow.set_experiment(experiment_name)
    logging.info(f"Logging to MLFlow Experiment: {experiment_name}")
    # setup_latex(fontsize=10)
    with start_run_with_exception_logging(run_name=run_name):
        # Auto log all MLflow entities
        mlflow.autolog()
        mlflow.log_dict(call_args, "params.yaml")
        log_cfg(call_args)
        if plot_data:
            fig = plot_data(X, Y)
            mlflow.log_figure(fig, "dataset.svg")

        if preprocess_dataset:
            preprocessed = preprocess_dataset(X, Y, model)
        else:
            preprocessed = {"x": X, "y": Y}

        hist = fit_distribution(
            model=model,
            seed=seed,
            results_path=results_path,
            **preprocessed,
            **fit_kwds,
        )

        if plot_samples:
            fig = plot_samples(
                model(preprocessed["x"]), preprocessed["y"].numpy(), seed=1
            )
            mlflow.log_figure(fig, "samples.svg")

        min_idx = np.argmin(hist.history["val_loss"])
        min_loss = hist.history["loss"][min_idx]
        min_val_loss = hist.history["val_loss"][min_idx]
        epochs = len(hist.history["loss"])
        logging.info(f"training finished after {epochs} epochs.")
        logging.info(f"best train loss: {min_loss}")
        logging.info(f"best validation loss: {min_val_loss}")
        logging.info(f"minimum reached after {min_idx} epochs")

        mlflow.log_metric("best_epoch", min_idx)
        mlflow.log_metric("final_epoch", epochs)
        mlflow.log_metric("min_loss", min_loss)
        mlflow.log_metric("min_val_loss", min_val_loss)

        with open(os.path.join(results_path, "metrics.yaml"), "w+") as results_file:
            yaml.dump({"loss": min_loss, "val_loss": min_val_loss}, results_file)

        if log_file:
            mlflow.log_artifact(log_file)


def prepare_pipeline(args):
    # prepare results directory
    os.makedirs(args.results_path, exist_ok=True)

    # configure logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers += [
            logging.FileHandler(args.log_file),
        ]
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    logging.debug(f"meta params: {vars(args)}")

    # load params
    params = dvc.api.params_show(
        stages=f"{args.stage_name}@{args.distribution}-{args.dataset}"
    )
    logging.info(params)

    return params
