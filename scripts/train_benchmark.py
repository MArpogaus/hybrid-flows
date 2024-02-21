"""Train benchmark."""
# IMPORT PACKAGES #############################################################
import argparse
import os

import tensorflow as tf
from mctm.data.benchmark import get_dataset
from mctm.models import DensityRegressionModel, HybridDenistyRegressionModel
from mctm.utils import str2bool
from mctm.utils.pipeline import pipeline, prepare_pipeline
from tensorflow_probability import distributions as tfd


def get_lr_schedule(**kwds):
    """Lr schedule."""
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(**kwds)
    return lr_decayed_fn


def run(
    experiment_name,
    results_path,
    log_file,
    log_level,
    dataset,
    stage_name,
    distribution,
    params,
    test_mode,
):
    """Experiment exec.

    params should be as defined in params.yaml
    """

    # define dataset, model and fit parameters
    stage = stage_name.split("@")[0]
    dataset_kwds = params["benchmark_datasets"][dataset]
    model_kwds = params[stage + "_distributions"][distribution][dataset]
    fit_kwds = model_kwds.pop("fit_kwds")

    model_kwds.update(
        distribution=distribution,
    )
    model_kwds["distribution_kwds"].update(dataset_kwds)

    if "base_distribution" in model_kwds.keys():
        get_model = HybridDenistyRegressionModel
        if not model_kwds["base_checkpoint_path"]:
            fit_kwds.update(
                loss=lambda y, dist: -dist.log_prob(y)
                - tfd.Independent(dist.distribution).log_prob(y)
            )
        else:
            model_kwds.update(base_checkpoint_path_prefix=results_path.split("/", 1)[0])
    else:
        get_model = DensityRegressionModel

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_name)
    run_name = "_".join((stage, distribution))

    # test mode
    if test_mode:
        run_name += "_test"
        fit_kwds.update(epochs=1)

    # don't show progress bar if running from CI
    if os.environ.get("CI", False):
        fit_kwds.update(verbose=2)

    # cosine_decay setup if relevant
    extra_params_to_log = {}
    if fit_kwds["learning_rate"] == "cosine_decay":
        cosine_decay_kwds = fit_kwds.pop("cosine_decay_kwds")
        fit_kwds["learning_rate"] = get_lr_schedule(**cosine_decay_kwds)
        extra_params_to_log = cosine_decay_kwds

    # execute experiment
    history, model, preprocessed = pipeline(
        experiment_name=experiment_name,
        run_name=run_name,
        results_path=results_path,
        log_file=log_file,
        seed=params["seed"],
        get_dataset_fn=get_dataset,
        dataset_kwds={"dataset_name": dataset},
        get_model_fn=get_model,
        model_kwds=model_kwds,
        preprocess_dataset=lambda data, model: {
            "x": tf.ones_like(data[0], dtype=model.dtype),
            "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
            "validation_data": (
                tf.ones_like(data[1], dtype=model.dtype),
                tf.convert_to_tensor(data[1], dtype=model.dtype),
            ),
        },
        fit_kwds=fit_kwds,
        plot_data=None,
        after_fit_hook=None,
        **extra_params_to_log,
    )

    return history, model, preprocessed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--log-file", type=str, help="path for log file")
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="logging severaty level"
    )
    parser.add_argument(
        "--test-mode", default=False, type=str2bool, help="activate test-mode"
    )
    parser.add_argument(
        "--experiment-name", type=str, help="MLFlow experiment name", required=True
    )
    parser.add_argument(
        "stage_name",
        type=str,
        help="name of dvstage",
    )
    parser.add_argument(
        "distribution",
        type=str,
        help="name of distribution",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="name of dataset",
    )
    parser.add_argument(
        "results_path",
        type=str,
        help="destination for model checkpoints and logs.",
    )
    args = parser.parse_args()

    # load params
    params = prepare_pipeline(args)

    run(
        args.experiment_name,
        args.results_path,
        args.log_file,
        args.log_level,
        args.dataset,
        args.stage_name,
        args.distribution,
        params,
        args.test_mode,
    )
