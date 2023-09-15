# IMPORT PACKAGES #############################################################
import argparse

import tensorflow as tf

from mctm.data.benchmark import get_dataset
from mctm.models import DensityRegressionModel
from mctm.utils import str2bool
from mctm.utils.pipeline import pipeline, prepare_pipeline


def main(args):
    # --- prepare for execution ---

    params = prepare_pipeline(args)

    dataset = args.dataset
    distribution = args.distribution
    stage = args.stage_name.split("@")[0]
    distribution_params = params[stage + "_distributions"][distribution][dataset]
    distribution_kwds = distribution_params["distribution_kwds"]
    fit_kwds = distribution_params["fit_kwds"]
    parameter_kwds = distribution_params["parameter_kwds"]
    dataset_kwds = params["benchmark_datasets"][dataset]

    distribution_kwds.update(dataset_kwds)

    experiment_name = args.experiment_name
    run_name = "_".join((stage, distribution))

    if args.test_mode:
        experiment_name += "_test"
        fit_kwds.update(epochs=1)

    # --- actually execute training ---

    pipeline(
        experiment_name=experiment_name,
        run_name=run_name,
        results_path=args.results_path,
        log_file=args.log_file,
        seed=params["seed"],
        get_dataset_fn=get_dataset,
        dataset_kwds={"dataset_name": dataset},
        get_model_fn=DensityRegressionModel,
        model_kwds=dict(
            distribution=distribution,
            distribution_kwds=distribution_kwds,
            parameter_kwds=parameter_kwds,
        ),
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
    )


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

    main(args)
