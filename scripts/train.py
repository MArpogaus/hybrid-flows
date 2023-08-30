# IMPORT PACKAGES #############################################################

import argparse
import pathlib

from mctm.data.sklearn_datasets import get_dataset
from mctm.models import DensityRegressionModel
from mctm.utils.pipeline import pipeline, prepare_pipeline
from mctm.utils.visualisation import plot_2d_data, plot_samples


def main(args):
    # --- prepare for execution ---

    params = prepare_pipeline(args)

    # --- actually execute training ---

    pipeline(
        args.experiment_name,
        params["distribution"],
        args.results_path,
        args.log_file,
        args.test_mode,
        params["seed"],
        get_dataset=lambda: get_dataset(args.dataset, **params["data_kwds"]),
        get_model=lambda DS: DensityRegressionModel(
            dims=DS[0].shape[-1],
            distribution=params["distribution"],
            distribution_kwds=params["distribution_kwds"],
            parameter_kwds=params.get("parameter_kwds", {}),
        ),
        fit_kwds=params["fit_kwds"],
        params=params,
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
        type=pathlib.Path,
        help="destination for model checkpoints and logs.",
    )
    args = parser.parse_args()

    main(args)
