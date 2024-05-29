"""Train."""

# IMPORT PACKAGES #############################################################
import argparse
import logging
import os
from functools import partial
from shutil import which

import mctm.scheduler
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as K
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

__LOGGER__ = logging.getLogger(__name__)


def plot_trafos(joint_dist, x, y):
    marginal_dist = joint_dist.distribution.distribution
    z2 = joint_dist.bijector.inverse(y)
    z1 = marginal_dist.bijector.inverse(z2)
    pit = marginal_dist.cdf(y)

    df = pd.DataFrame(
        columns=[
            "$y1$",
            "$y2$",
            "$z_{2,1}$",
            "$z_{2,2}$",
            "$z_{1,1}$",
            "$z_{1,2}$",
            "$F_1(y_1)$",
            "$F_2(y_2)$",
            "$x$",
        ],
        data=np.concatenate([y, z2, z1, pit, x], -1),
    )
    g = sns.JointGrid(data=df, x="$y1$", y="$y2$", height=2)
    g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
    g.plot_marginals(sns.kdeplot)
    data_figure = g.figure

    # normalized data
    g = sns.JointGrid(data=df, x="$z_{1,1}$", y="$z_{1,2}$", height=2)
    g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
    g.plot_marginals(sns.kdeplot)
    normalized_data_figure = g.figure

    # PIT
    g = sns.jointplot(df, x="$F_1(y_1)$", y="$F_2(y_2)$", height=2, s=4, alpha=0.5)
    pit_figure = g.figure

    # decorrelated data
    g = sns.JointGrid(data=df, x="$z_{2,1}$", y="$z_{2,2}$", height=2)
    g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
    g.plot_marginals(sns.kdeplot)
    decorelated_data_figure = g.figure

    # normalized data
    g = sns.JointGrid(data=df, x="$z_{1,1}$", y="$z_{1,2}$", height=2)
    g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
    g.plot_marginals(sns.kdeplot)
    latent_dist_figure = g.figure

    return (
        data_figure,
        normalized_data_figure,
        pit_figure,
        decorelated_data_figure,
        latent_dist_figure,
    )


def get_after_fit_hook(results_path, is_hybrid, **kwargs):
    """Provide after fit plot."""

    def plot_after_fit(model, x, y):
        # Generating random indices
        if len(x) > 2000:
            indices = np.random.choice(len(x), size=2000, replace=False)

            # Selecting elements based on the indices
            x = x.numpy()[indices]
            y = y.numpy()[indices]

        fig = plot_samples(model(x), y, seed=1, **kwargs)
        fig.savefig(os.path.join(results_path, "samples.pdf"))
        if is_hybrid:
            fig1, fig2, fig3 = plot_flow(model(x), x, y, seed=1, **kwargs)
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

            # plot trafos
            (
                data_figure,
                normalized_data_figure,
                pit_figure,
                decorelated_data_figure,
                latent_dist_figure,
            ) = plot_trafos(model(x), x, y)

            # save the above figures as pdf
            data_figure.savefig(os.path.join(results_path, "data_transformation.pdf"))
            normalized_data_figure.savefig(
                os.path.join(results_path, "normalized_data_transformation.pdf")
            )
            pit_figure.savefig(
                os.path.join(results_path, "pit_data_transformation.pdf")
            )
            decorelated_data_figure.savefig(
                os.path.join(results_path, "decorelated_data_transformation.pdf")
            )
            latent_dist_figure.savefig(
                os.path.join(results_path, "latent_distribution.pdf")
            )

    return plot_after_fit


def get_learning_rate(fit_kwargs):
    """Lr schedule.

    decay: name of a scheduler in `K.optimizers.schedules` (i.e. CosineDecay)
    kwargs: kewyowrds that get passed into decay function.
    """
    if isinstance(fit_kwargs["learning_rate"], dict):
        scheduler_name = fit_kwargs["learning_rate"]["scheduler_name"]
        schduler_class_name = "".join(map(str.title, scheduler_name.split("_")))
        scheduler_kwargs = fit_kwargs["learning_rate"]["scheduler_kwargs"]
        __LOGGER__.info(f"{scheduler_name=}({scheduler_kwargs})")
        scheduler = getattr(
            mctm.scheduler,
            schduler_class_name,
            getattr(K.optimizers.schedules, schduler_class_name, None),
        )(**scheduler_kwargs)

        fit_kwargs["callbacks"] = [K.callbacks.LearningRateScheduler(scheduler)]
        return scheduler_kwargs["initial_learning_rate"], fit_kwargs["learning_rate"]
    else:
        return fit_kwargs["learning_rate"], {}


class MeanNegativeLogLiekelihood(K.metrics.Mean):
    def __init__(self, name="mean_negative_log_likelihood", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, dist, sample_weight):
        log_probs = -dist.log_prob(y_true)
        super().update_state(log_probs, sample_weight)


def run(
    dataset_name,
    dataset_type,
    experiment_name,
    run_name,
    log_file,
    log_level,
    results_path,
    test_mode,
    params,
):
    """Experiment exec.

    params should be as defined in params.yaml
    """
    model_kwargs = params["model_kwargs"]
    fit_kwargs = params["fit_kwargs"]
    compile_kwargs = params["compile_kwargs"]
    dataset_kwargs = params["dataset_kwargs"][dataset_name]

    learning_rate, extra_params_to_log = get_learning_rate(fit_kwargs)
    fit_kwargs["learning_rate"] = learning_rate

    if "base_distribution" in model_kwargs.keys():
        get_model = HybridDenistyRegressionModel
        if not model_kwargs.get("base_checkpoint_path", False):
            fit_kwargs.update(
                loss=lambda y, dist: -dist.log_prob(y) - dist.distribution.log_prob(y),
            )
            compile_kwargs.update(
                metrics=[MeanNegativeLogLiekelihood(name="nll")],
            )
        else:
            model_kwargs.update(
                base_checkpoint_path_prefix=results_path.split("/", 1)[0]
            )
    else:
        get_model = DensityRegressionModel

    if dataset_type == "benchmark":
        get_dataset_fn = get_benchmark_dataset
        get_dataset_kwargs = {"dataset_name": dataset_name, "test_mode": test_mode}
        model_kwargs["distribution_kwargs"].update(
            **dataset_kwargs,
        )

        def preprocess_dataset(data, model):
            return {
                "x": tf.ones_like(data[0], dtype=model.dtype),
                "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
                "validation_data": (
                    tf.ones_like(data[1], dtype=model.dtype),
                    tf.convert_to_tensor(data[1], dtype=model.dtype),
                ),
            }

        plot_data = None
        after_fit_hook = False
    else:
        get_dataset_fn = get_train_dataset
        get_dataset_kwargs = {
            **dataset_kwargs,
            "dataset_name": dataset_name,
            "test_mode": test_mode,
        }

        def preprocess_dataset(data, model):
            return {
                "x": tf.convert_to_tensor(data[1][..., None], dtype=model.dtype),
                "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
            }

        fig_width = get_figsize(params["textwidth"], fraction=0.5)[0]

        plot_data = partial(plot_2d_data, figsize=(fig_width, fig_width))
        after_fit_hook = get_after_fit_hook(
            results_path=results_path,
            is_hybrid=get_model == HybridDenistyRegressionModel,
            height=fig_width,
        )

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_name)

    # test mode config
    if test_mode:
        __LOGGER__.info("Running in test-mode")
        run_name += "_test"
        fit_kwargs.update(epochs=2)

    # configure mpl to use latex
    if which("latex"):
        __LOGGER__.info("Using latex backend for plotting")
        setup_latex(fontsize=10)

    # don't show progress bar if running from CI
    if os.environ.get("CI", False):
        fit_kwargs.update(verbose=2)

    # actually execute training
    history, model, preprocessed = pipeline(
        experiment_name=experiment_name,
        run_name=run_name,
        results_path=results_path,
        log_file=log_file,
        seed=params["seed"],
        get_dataset_fn=get_dataset_fn,
        dataset_kwargs=get_dataset_kwargs,
        get_model_fn=get_model,
        model_kwargs=model_kwargs,
        preprocess_dataset=preprocess_dataset,
        fit_kwargs=fit_kwargs,
        compile_kwargs=compile_kwargs,
        plot_data=plot_data,
        after_fit_hook=after_fit_hook,
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
    parser.add_argument("--run-name", type=str, help="MLFlow run name", required=True)
    parser.add_argument(
        "stage_name",
        type=str,
        help="name of dvc stage",
    )
    parser.add_argument(
        "dataset_type",
        type=str,
        help="type of dataset",
    )
    parser.add_argument(
        "dataset_name",
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
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        log_file=args.log_file,
        log_level=args.log_level,
        results_path=args.results_path,
        test_mode=args.test_mode,
        params=params,
    )
