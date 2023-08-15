# IMPORT PACKAGES #############################################################

import argparse
import logging
import pathlib
import sys
import time

import dvc.api
import mlflow
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

from mctm import distributions
from mctm.data.sklearn_datasets import get_dataset
from mctm.mlfow import log_cfg, start_run_with_exception_logging
from mctm.utils import fit_distribution, set_seed


def plot_2d_data(X, Y):
    label = Y.astype(bool)
    X1, X2 = X[..., 0], X[..., 1]
    fig = plt.figure()
    plt.scatter(X1[label], X1[label], s=10, color="blue")
    plt.scatter(X1[~label], X2[~label], s=10, color="red")
    plt.legend(["label: 1", "label: 0"])
    return fig


def plot_samples(dist, data, seed=1):
    N = data.shape[0]
    # Use the fitted distribution.
    start = time.time()
    samples = dist.sample(N, seed=seed)
    end = time.time()
    print(f"sampling took {end-start} seconds.")

    df1 = pd.DataFrame(columns=["x1", "x2"], data=data)
    df1 = df1.assign(source="data")

    df2 = pd.DataFrame(columns=["x1", "x2"], data=samples.numpy())
    df2 = df2.assign(source="model")

    df = pd.concat([df1, df2])

    # sns.jointplot(data=df, x='x1', y='x2', hue='source', kind='kde')
    g = sns.jointplot(
        data=df,
        x="x1",
        y="x2",
        hue="source",
        alpha=0.5,
        xlim=(data[..., 0].min() - 0.1, data[..., 0].max() + 0.1),
        ylim=(data[..., 1].min() - 0.1, data[..., 1].max() + 0.1),
    )
    g.plot_joint(sns.kdeplot)
    # g.plot_marginals(sns.rugplot, height=-.15)
    return g.figure


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
    params = dvc.api.params_show(stages=args.stage_name)["params"]
    params = {k: v for d in params.values() for k, v in d.items()}
    logging.info(params)

    # ensure reproducibility
    set_seed(params["seed"])

    # generate data
    X, Y = get_dataset(params["dataset"], **params["data_kwds"])

    dims = X.shape[-1]

    get_distribution = getattr(distributions, "get_" + params["distribution"])
    distribution_lambda, parameters_shape = get_distribution(
        dims=dims, **params["distribution_kwds"]
    )

    trainable_parameters = tf.Variable(
        tf.random.normal(parameters_shape, dtype=tf.float32), trainable=True
    )

    class Model(tf.keras.Model):
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

    with mlflow.start_run(run_name=params["dataset"]):
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

            model = Model(distribution_lambda, trainable_parameters)
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
