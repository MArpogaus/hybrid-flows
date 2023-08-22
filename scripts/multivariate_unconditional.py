# IMPORT PACKAGES #############################################################

import argparse
import logging
import pathlib
import sys

import dvc.api
import mlflow
import tensorflow as tf

from mctm import distributions
from mctm.data.sklearn_datasets import get_dataset
from mctm.mlfow import log_cfg, start_run_with_exception_logging
from mctm.utils import fit_distribution, set_seed
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

    distribution_lambda, trainable_parameters = getattr(
        distributions, "get_" + params["distribution"]
    )(dims=dims, **params["distribution_kwds"])

    class UnconditionalModel(tf.keras.Model):
        def __init__(self, distribution_lambda, trainable_parameters, **kwds):
            super().__init__(**kwds)
            self.distribution_lambda = distribution_lambda
            self.distribution_parameters = trainable_parameters

        def call(self, *_):
            return self.distribution_lambda(self.distribution_parameters)

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

        model = UnconditionalModel(distribution_lambda, trainable_parameters)
        hist = fit_distribution(
            model,
            seed=params["seed"],
            # unused but required
            x=X,
            y=X,
            **params["fit_kwds"],
        )

        fig = plot_samples(model(X), X, seed=1)
        mlflow.log_figure(fig, "samples.svg")

        if args.log_file:
            mlflow.log_artifact(args.log_file)
