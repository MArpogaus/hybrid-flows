# IMPORT PACKAGES #############################################################
import argparse
import logging
import os
from functools import partial
from shutil import which

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from mctm.data.sklearn_datasets import get_dataset
from mctm.models import DensityRegressionModel, HybridDenistyRegressionModel
from mctm.utils import str2bool
from mctm.utils.pipeline import pipeline, prepare_pipeline
from mctm.utils.visualisation import (
    get_figsize,
    plot_2d_data,
    plot_flow,
    plot_samples,
    setup_latex,
)

__LOGGER__ = logging.getLogger(__name__)


def get_after_fit_hook(results_path, is_hybrid, **kwds):
    def plot_after_fit(model, x, y):
        fig = plot_samples(model(x), y.numpy(), seed=1, **kwds)
        fig.savefig(os.path.join(results_path, "samples.pdf"))
        if is_hybrid:
            fig1, fig2, fig3 = plot_flow(model(x), x, y, seed=1, **kwds)
            fig1.savefig(os.path.join(results_path, "data.pdf"))
            fig2.savefig(os.path.join(results_path, "z1.pdf"))
            fig3.savefig(os.path.join(results_path, "z2.pdf"))

    return plot_after_fit


def main(args):
    # prepare for execution:
    # - read cli arguments
    # - configure logging
    # - load dvc params
    params = prepare_pipeline(args)

    dataset = args.dataset
    dataset_kwds = params["datasets"][dataset]
    distribution = args.distribution
    stage = args.stage_name.split("@")[0]
    distribution_params = params[stage + "_distributions"][distribution][dataset]
    distribution_kwds = distribution_params["distribution_kwds"]
    fit_kwds = distribution_params["fit_kwds"]
    parameter_kwds = distribution_params["parameter_kwds"]

    model_kwds = dict(
        distribution=distribution,
        distribution_kwds=distribution_kwds,
        parameter_kwds=parameter_kwds,
    )

    if "base_distribution" in distribution_params.keys():
        get_model = HybridDenistyRegressionModel
        model_kwds.update(
            freeze_base_model=distribution_params["freeze_base_model"],
            base_checkpoint_path=(
                f'results/{distribution_params["base_checkpoint_path"]}_{dataset}/mcp/weights'  # noqa: E501
                if distribution_params["base_checkpoint_path"]
                else False
            ),
            base_distribution=distribution_params["base_distribution"],
            base_distribution_kwds=distribution_params["base_distribution_kwds"],
            base_parameter_kwds=distribution_params["base_parameter_kwds"],
        )
        if not distribution_params["base_checkpoint_path"]:
            fit_kwds.update(
                loss=lambda y, dist: -dist.log_prob(y)
                - tfd.Independent(dist.distribution).log_prob(y)
            )
    else:
        get_model = DensityRegressionModel

    experiment_name = args.experiment_name
    run_name = "_".join((stage, distribution))

    if args.test_mode:
        __LOGGER__.info("Running in test-mode")
        experiment_name += "_test"
        fit_kwds.update(epochs=1)

    # configure mpl to use latex
    if which("latex"):
        __LOGGER__.info("Using latex backend for plotting")
        setup_latex(fontsize=10)

    fig_width = get_figsize(params["textwidth"], fraction=0.5)[0]

    # actually execute training
    pipeline(
        experiment_name=experiment_name,
        run_name=run_name,
        results_path=args.results_path,
        log_file=args.log_file,
        seed=params["seed"],
        get_dataset_fn=get_dataset,
        dataset_kwds={"dataset_name": dataset, **dataset_kwds},
        get_model_fn=get_model,
        model_kwds=model_kwds,
        preprocess_dataset=lambda data, model: {
            "x": tf.convert_to_tensor(data[1][..., None], dtype=model.dtype),
            "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
        },
        fit_kwds=fit_kwds,
        plot_data=partial(plot_2d_data, figsize=(fig_width, fig_width)),
        after_fit_hook=get_after_fit_hook(
            results_path=args.results_path,
            is_hybrid=get_model == HybridDenistyRegressionModel,
            height=fig_width,
        ),
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
