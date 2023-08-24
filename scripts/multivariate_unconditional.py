# IMPORT PACKAGES #############################################################

import argparse
import logging
import os
import pathlib
import sys

import dvc.api
import mlflow
import numpy as np
import yaml

from mctm.data.sklearn_datasets import get_dataset
from mctm.models import UnconditionalModel
from mctm.utils.mlflow import log_cfg, start_run_with_exception_logging
from mctm.utils.tensorflow import fit_distribution, set_seed
from mctm.utils.visualisation import plot_2d_data, plot_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--experiment-name", type=str, help="MLFlow experiment name")
    parser.add_argument("--log-file", type=str, help="path for log file")
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="logging severaty level"
    )
    parser.add_argument("--test-mode", action="store_true", help="activate test-mode")
    parser.add_argument(
        "stage_name",
        type=str,
        help="name of dvstage ",
    )
    parser.add_argument(
        "results_path",
        type=pathlib.Path,
        help="destination for model checkpoints and logs.",
    )
    args = parser.parse_args()

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

    logging.debug(vars(args))

    # prepare results directory
    os.makedirs(args.results_path, exist_ok=True)

    # load params
    params = dvc.api.params_show(stages=args.stage_name)
    distribution = list(params["distributions"].keys())[0]
    params = {
        **params["common"],
        "distribution": distribution,
        **params["distributions"][distribution],
    }
    logging.info(params)

    # ensure reproducibility
    set_seed(params["seed"])

    # generate data
    X, Y = get_dataset(params["dataset"], **params["data_kwds"])

    dims = X.shape[-1]

    # Evaluate Model
    experiment_name = args.experiment_name + ("_test" if args.test_mode else "")
    mlflow.set_experiment(experiment_name)
    logging.info(f"Logging to MLFlow Experiment: {experiment_name}")

    with start_run_with_exception_logging(
        run_name=params["distribution"] + "_training"
    ):
        # Auto log all MLflow entities
        mlflow.autolog()
        mlflow.set_tag("stage", "training")
        mlflow.log_dict(params, "params.yaml")
        log_cfg(params)
        mlflow.log_params(vars(args))
        fig = plot_2d_data(X, Y)
        mlflow.log_figure(fig, "dataset.svg")

        model = UnconditionalModel(
            dims=dims,
            distribution=params["distribution"],
            distribution_kwds=params["distribution_kwds"],
            parameter_kwds=params.get("parameter_kwds", {}),
        )
        hist = fit_distribution(
            model=model,
            seed=params["seed"],
            # unused but required
            x=X,
            y=X,
            results_path=args.results_path,
            verbose=True,
            monitor="val_loss",
            **params["fit_kwds"],
        )

        fig = plot_samples(model(X), X, seed=1)
        mlflow.log_figure(fig, "samples.svg")

        min_idx = np.argmin(hist.history["val_loss"])
        min_loss = hist.history["loss"][min_idx]
        min_val_loss = hist.history["val_loss"][min_idx]
        logging.info(f"training finished after {len(hist.history['loss'])} epochs.")
        logging.info(f"train loss: {min_loss}")
        logging.info(f"validation loss: {min_val_loss}")

        if not args.test_mode:
            with open(
                os.path.join(args.results_path, "metrics.yaml"), "w+"
            ) as results_file:
                yaml.dump({"loss": min_loss, "val_loss": min_val_loss}, results_file)

        if args.log_file:
            mlflow.log_artifact(args.log_file)
