"""Train."""
# IMPORT PACKAGES #############################################################
import argparse
import logging
import os
from functools import partial
from shutil import which

import numpy as np
import tensorflow as tf
from mctm.data.benchmark import get_dataset as get_benchmark_dataset
from mctm.data.sklearn_datasets import get_dataset as get_train_dataset
from mctm.models import DensityRegressionModel, HybridDenistyRegressionModel
from mctm.utils import str2bool
from mctm.utils.pipeline import pipeline, prepare_pipeline
from mctm.utils.visualisation import (
    get_figsize,
    plot_2d_data,
    plot_copula_function,
    plot_flow,
    plot_samples,
    setup_latex,
)
from tensorflow_probability import distributions as tfd

__LOGGER__ = logging.getLogger(__name__)
# tf.debugging.experimental.enable_dump_debug_info(
#     "tfdbg2_logdir", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1
# )


def get_after_fit_hook(results_path, is_hybrid, **kwds):
    """Provide after fit plot."""

    def plot_after_fit(model, x, y):
        # Generating random indices
        indices = np.random.choice(len(x), size=2000, replace=False)

        # Selecting elements based on the indices
        x = x.numpy()[indices]
        y = y.numpy()[indices]

        fig = plot_samples(model(x), y, seed=1, **kwds)
        fig.savefig(os.path.join(results_path, "samples.pdf"))
        if is_hybrid:
            fig1, fig2, fig3 = plot_flow(model(x), x, y, seed=1, **kwds)
            fig1.savefig(os.path.join(results_path, "data.pdf"))
            fig2.savefig(os.path.join(results_path, "z1.pdf"))
            fig3.savefig(os.path.join(results_path, "z2.pdf"))

            # Plot Copula
            # Contour Plot
            c_fig = plot_copula_function(model(x), y, "contour", -0.1, 1.1, 200)
            c_fig.savefig(os.path.join(results_path, "copula_contour.pdf"))

            # Surface Plot
            c_fig = plot_copula_function(model(x), y, "surface", -0.1, 1.1, 200)
            c_fig.savefig(os.path.join(results_path, "copula_surface.pdf"))

    return plot_after_fit


def get_lr_schedule(decay, **kwds):
    """Lr schedule.

    decay: function like tf.keras.optimizers.schedules.CosineDecay
    kwds: kewyowrds that get passed into decay function.
    """
    lr_decayed_fn = decay(**kwds)
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

    IS_BENCHMARK = "benchmark" in stage_name

    stage = stage_name.split("@")[0]
    dataset_kwds = (
        params["datasets"][dataset]
        if not IS_BENCHMARK
        else params["benchmark_datasets"][dataset]
    )
    model_kwds = params[stage + "_distributions"][distribution][dataset]
    fit_kwds = model_kwds.pop("fit_kwds")

    model_kwds.update(
        distribution=distribution,
    )
    if IS_BENCHMARK:
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

    # test mode config
    if test_mode:
        __LOGGER__.info("Running in test-mode")
        run_name += "_test"
        fit_kwds.update(epochs=1)

    # configure mpl to use latex
    if which("latex"):
        __LOGGER__.info("Using latex backend for plotting")
        setup_latex(fontsize=10)

    # don't show progress bar if running from CI
    if os.environ.get("CI", False):
        fit_kwds.update(verbose=2)
    if not IS_BENCHMARK:
        fig_width = get_figsize(params["textwidth"], fraction=0.5)[0]

    extra_params_to_log = {}
    if IS_BENCHMARK:
        # cosine_decay setup if relevant
        if isinstance(fit_kwds["learning_rate"], str):
            schedule_key = fit_kwds["learning_rate"]
            __LOGGER__.info(f"{schedule_key=}")
            S = tf.keras.optimizers.schedules
            schedulers = {
                "cosine_decay": S.CosineDecay,
                "exponential_decay": S.ExponentialDecay,
                "polynomial_decay": S.PolynomialDecay,
            }
            decay_kwds = fit_kwds.pop(schedule_key + "_kwds")
            fit_kwds["learning_rate"] = get_lr_schedule(
                schedulers[schedule_key], **decay_kwds
            )
            extra_params_to_log = decay_kwds
            for key in schedulers.keys():
                fit_kwds.pop(key + "_kwds", None)

    # actually execute training
    history, model, preprocessed = pipeline(
        experiment_name=experiment_name,
        run_name=run_name,
        results_path=results_path,
        log_file=log_file,
        seed=params["seed"],
        get_dataset_fn=get_benchmark_dataset if IS_BENCHMARK else get_train_dataset,
        dataset_kwds={"dataset_name": dataset, **dataset_kwds}
        if not IS_BENCHMARK
        else {"dataset_name": dataset},
        get_model_fn=get_model,
        model_kwds=model_kwds,
        preprocess_dataset=lambda data, model: {
            "x": tf.convert_to_tensor(data[1][..., None], dtype=model.dtype),
            "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
        }
        if not IS_BENCHMARK
        else {
            "x": tf.ones_like(data[0], dtype=model.dtype),
            "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
            "validation_data": (
                tf.ones_like(data[1], dtype=model.dtype),
                tf.convert_to_tensor(data[1], dtype=model.dtype),
            ),
        },
        fit_kwds=fit_kwds,
        plot_data=partial(plot_2d_data, figsize=(fig_width, fig_width))
        if not IS_BENCHMARK
        else None,
        after_fit_hook=get_after_fit_hook(
            results_path=results_path,
            is_hybrid=get_model == HybridDenistyRegressionModel,
            height=fig_width,
        )
        if not IS_BENCHMARK
        else None,
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
