# IMPORT PACKAGES #############################################################
import argparse

import tensorflow as tf

from mctm.data.sklearn_datasets import get_dataset
from mctm.models import HybridDenistyRegressionModel
from mctm.utils.pipeline import pipeline, prepare_pipeline
from mctm.utils.visualisation import plot_2d_data, plot_samples


def main(args):
    # --- prepare for execution ---

    params = prepare_pipeline(args)

    results_path = args.results_path
    dataset = args.dataset
    dataset_kwds = params["datasets"][dataset]
    distribution = args.distribution
    distribution_params = params[args.stage_name + "_distributions"][distribution]
    freeze_base_model = distribution_params["freeze_base_model"]
    base_checkpoint_path = (
        f'results/{distribution_params["base_checkpoint_path"]}_{dataset}/mcp/weights'
        if distribution_params["base_checkpoint_path"]
        else False
    )
    base_distribution = distribution_params["base_distribution"]
    base_distribution_kwds = distribution_params["base_distribution_kwds"]
    base_parameter_kwds = distribution_params["base_parameter_kwds"]
    distribution_kwds = distribution_params["distribution_kwds"]
    fit_kwds = distribution_params["fit_kwds"]
    parameter_kwds = distribution_params["parameter_kwds"]

    experiment_name = args.experiment_name
    run_name = "_".join((args.stage_name, distribution))

    if args.test_mode:
        experiment_name += "_test"
        fit_kwds.update(epochs=1)

    # --- actually execute training ---

    pipeline(
        experiment_name=experiment_name,
        run_name=run_name,
        results_path=results_path,
        log_file=args.log_file,
        seed=params["seed"],
        get_dataset_fn=get_dataset,
        dataset_kwds={"dataset_name": dataset, **dataset_kwds},
        get_model_fn=HybridDenistyRegressionModel,
        model_kwds=dict(
            distribution=distribution,
            distribution_kwds=distribution_kwds,
            parameter_kwds=parameter_kwds,
            base_distribution=base_distribution,
            base_distribution_kwds=base_distribution_kwds,
            base_parameter_kwds=base_parameter_kwds,
            freeze_base_model=freeze_base_model,
            base_checkpoint_path=base_checkpoint_path,
        ),
        preprocess_dataset=lambda X, Y, model: {
            "x": tf.convert_to_tensor(Y[..., None], dtype=model.dtype),
            "y": tf.convert_to_tensor(X, dtype=model.dtype),
        },
        fit_kwds=fit_kwds,
        plot_data=plot_2d_data,
        plot_samples=plot_samples,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--log-file", type=str, help="path for log file")
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="logging severaty level"
    )
    parser.add_argument("--test-mode", action="store_true", help="activate test-mode")
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
