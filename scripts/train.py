"""Train multivariate density estimation models on different datasets."""

import argparse
import logging
import os
from functools import partial
from shutil import which
from typing import Any, Dict

import mctm.scheduler
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as K
from mctm.data.benchmark import get_dataset as get_benchmark_dataset
from mctm.data.malnutrion import get_dataset as get_malnutrition_dataset
from mctm.data.sklearn_datasets import get_dataset as get_sim_dataset
from mctm.models import DensityRegressionModel, HybridDenistyRegressionModel
from mctm.utils import str2bool
from mctm.utils.pipeline import pipeline, prepare_pipeline
from mctm.utils.visualisation import (
    get_figsize,
    plot_2d_data,
    plot_copula_function,
    plot_flow,
    plot_malnutrition_data,
    plot_malnutrition_samples,
    plot_marginal_cdf_and_pdf,
    plot_samples,
    setup_latex,
)
from tensorflow_probability import distributions as tfd

__LOGGER__ = logging.getLogger(__name__)


def plot_trafos(joint_dist: tfd.Distribution, x: np.ndarray, y: np.ndarray) -> tuple:
    """Plot transformations of the joint distribution.

    Parameters
    ----------
    joint_dist : tfp.Distribution
        Joint distribution to be plotted.
    x : np.ndarray
        Input data.
    y : np.ndarray
        True values.

    Returns
    -------
    tuple
        A tuple of figures.

    """
    marginal_dist = joint_dist.distribution.distribution
    z2 = joint_dist.bijector.inverse(y)
    z1 = marginal_dist.bijector.inverse(y)
    z = marginal_dist.bijector.inverse(z2)
    pit = marginal_dist.cdf(y)

    df = pd.DataFrame(
        columns=[
            "$y1$",
            "$y2$",
            "$z_{2,1}$",
            "$z_{2,2}$",
            "$z_{1,1}$",
            "$z_{1,2}$",
            "$z_{1}$",
            "$z_{2}$",
            "$F_1(y_1)$",
            "$F_2(y_2)$",
            "$x$",
        ],
        data=np.concatenate([y, z2, z1, z, pit, x], -1),
    )

    g = sns.JointGrid(data=df, x="$y1$", y="$y2$", height=2)
    g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
    g.plot_marginals(sns.kdeplot)
    data_figure = g.figure

    g = sns.JointGrid(data=df, x="$z_{1,1}$", y="$z_{1,2}$", height=2)
    g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
    g.plot_marginals(sns.kdeplot)
    normalized_data_figure = g.figure

    g = sns.jointplot(df, x="$F_1(y_1)$", y="$F_2(y_2)$", height=2, s=4, alpha=0.5)
    pit_figure = g.figure

    g = sns.JointGrid(data=df, x="$z_{2,1}$", y="$z_{2,2}$", height=2)
    g.plot_joint(sns.scatterplot, s=4, alpha=0.5)
    g.plot_marginals(sns.kdeplot)
    decorelated_data_figure = g.figure

    g = sns.JointGrid(data=df, x="$z_{1}$", y="$z_{2}$", height=2)
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


def sim_after_fit_hook(
    model: object,
    x: tf.Tensor,
    y: tf.Tensor,
    results_path: str,
    is_hybrid: bool,
    **kwargs,
) -> callable:
    """Provide after fit plot.

    Parameters
    ----------
    model : object
        The model that was trained.
    x : tf.Tensor
        Input data.
    y : tf.Tensor
        True values.
    results_path : str
        Path to save the result images.
    is_hybrid : bool
        Flag to indicate if hybrid model is used.
    **kwargs
        Additional arguments.

    Returns
    -------
    callable
        Hook function to be called after fitting the model.

    """
    if len(x) > 2000:
        indices = np.random.choice(len(x), size=2000, replace=False)
        x = x.numpy()[indices]
        y = y.numpy()[indices]

    fig = plot_samples(model(x), y, seed=1, **kwargs)
    fig.savefig(os.path.join(results_path, "samples.pdf"), bbox_inches="tight")

    if is_hybrid:
        fig1, fig2, fig3 = plot_flow(model(x), x, y, seed=1, **kwargs)
        fig1.savefig(os.path.join(results_path, "data.pdf"), bbox_inches="tight")
        fig2.savefig(os.path.join(results_path, "z1.pdf"), bbox_inches="tight")
        fig3.savefig(os.path.join(results_path, "z2.pdf"), bbox_inches="tight")

        c_fig = plot_copula_function(model(x), y, "contour", -0.1, 1.1, 200)
        c_fig.savefig(
            os.path.join(results_path, "copula_contour.pdf"), bbox_inches="tight"
        )

        c_fig = plot_copula_function(model(x), y, "surface", -0.1, 1.1, 200)
        c_fig.savefig(
            os.path.join(results_path, "copula_surface.pdf"), bbox_inches="tight"
        )

        (
            data_figure,
            normalized_data_figure,
            pit_figure,
            decorelated_data_figure,
            latent_dist_figure,
        ) = plot_trafos(model(x), x, y)

        data_figure.savefig(os.path.join(results_path, "data_scatter.pdf"))
        normalized_data_figure.savefig(
            os.path.join(results_path, "normalized_data.pdf"), bbox_inches="tight"
        )
        pit_figure.savefig(os.path.join(results_path, "pit.pdf"))
        decorelated_data_figure.savefig(
            os.path.join(results_path, "decorelated_data.pdf"), bbox_inches="tight"
        )
        latent_dist_figure.savefig(
            os.path.join(results_path, "latent_distribution.pdf"), bbox_inches="tight"
        )


def malnutrition_after_fit_hook(
    model: object,
    x: tf.Tensor,
    validation_data: tuple,
    results_path: str,
    seed: int,
    targets: list,
    plot_samples_kwargs: Dict[str, Any] = {},
    plot_marginals_kwargs: Dict[str, Any] = {},
) -> None:
    """Provide after fit plot for malnutrition data.

    Parameters
    ----------
    model : object
        The model that was trained.
    x : tf.Tensor
        Input data.
    y : tf.Tensor
        True values.
    validation_data : tuple
        Validation data, containing features and targets.
    results_path : str
        Path to save the result images.
    N : int
        Number of samples to plot.
    seed : int
        Random seed for sampling.
    targets : list
        List of target variable names.
    plot_samples_kwargs : Dict[str, Any]
        Additional kwargs passed to `plot_malnutrition_samples`
    plot_marginals_kwargs : Dict[str, Any]
        Additional kwargs passed to `plot_marginal_cdf_and_pdf`

    """
    x, y = validation_data._input_dataset._tensors
    fig = plot_malnutrition_samples(model, x, y, seed, targets, **plot_samples_kwargs)
    fig.savefig(os.path.join(results_path, "samples.pdf"), bbox_inches="tight")

    fig = plot_marginal_cdf_and_pdf(
        model=model,
        covariates=[1, 3, 6, 9, 12, 24],
        target_names=targets,
        **plot_marginals_kwargs,
    )
    fig.savefig(
        os.path.join(results_path, "malnutrition_dist.pdf"), bbox_inches="tight"
    )


def get_learning_rate(fit_kwargs: dict) -> tuple:
    """Get learning rate scheduler.

    Parameters
    ----------
    fit_kwargs : dict
        Dictionary containing fit parameters.

    Returns
    -------
    tuple
        Initial learning rate and learning rate scheduler.

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


class MeanNegativeLogLikelihood(K.metrics.Mean):
    """Custom metric for mean negative log likelihood."""

    def __init__(
        self, name: str = "mean_negative_log_likelihood", **kwargs: dict
    ) -> None:
        """Initialize Keras metric for negative logarithmic likelihood.

        Parameters
        ----------
        name : str, optional
            Name of the metric instance, by default "mean_negative_log_likelihood".
        **kwargs : dict
            Additional keyword arguments for the base class.

        """
        super().__init__(name=name, **kwargs)

    def update_state(
        self, y_true: tf.Tensor, dist: object, sample_weight: tf.Tensor = None
    ) -> None:
        """Update the metric state with the true labels and the distribution.

        Parameters
        ----------
        y_true : tf.Tensor
            The ground truth values.
        dist : object
            The distribution object that provides the log probability method.
        sample_weight : tf.Tensor, optional
            Optional weighting of each example, by default None.

        """
        log_probs = -dist.log_prob(y_true)
        super().update_state(log_probs, sample_weight)


def run(
    dataset_name: str,
    dataset_type: str,
    experiment_name: str,
    run_name: str,
    log_file: str,
    log_level: str,
    results_path: str,
    test_mode: bool,
    params: dict,
) -> tuple:
    """Execute experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    dataset_type : str
        Type of the dataset.
    experiment_name : str
        Name of the MLFlow experiment.
    run_name : str
        Name of the MLFlow run.
    log_file : str
        Path for log file.
    log_level : str
        Logging severity level.
    results_path : str
        Destination for model checkpoints and logs.
    test_mode : bool
        Flag to activate test-mode.
    params : dict
        Dictionary containing experiment parameters.

    Returns
    -------
    tuple
        Experiment results: history, model, and preprocessed data.

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
                metrics=[MeanNegativeLogLikelihood(name="nll")],
            )
        else:
            model_kwargs.update(
                base_checkpoint_path_prefix=results_path.split("/", 1)[0]
            )
    else:
        get_model = DensityRegressionModel

    if dataset_type == "benchmark":
        get_dataset_fn = get_benchmark_dataset
        get_dataset_kwargs = {
            "dataset_name": dataset_name,
            "test_mode": test_mode,
        }

        def preprocess_dataset(data, model) -> dict:
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
    elif dataset_type == "malnutrition":
        get_dataset_fn = get_malnutrition_dataset
        get_dataset_kwargs = {
            **dataset_kwargs,
            "test_mode": test_mode,
        }
        batch_size = fit_kwargs.pop("batch_size")

        def mk_ds(data):
            return tf.data.Dataset.from_tensor_slices((data[0], data[1])).batch(
                batch_size, drop_remainder=True
            )

        def preprocess_dataset(data, model) -> dict:
            return {
                "x": mk_ds(data[0]),
                "validation_data": mk_ds(data[1]),
            }

        figsize = get_figsize(params["textwidth"])
        fig_height = figsize[0]

        plot_data = partial(
            plot_malnutrition_data,
            targets=dataset_kwargs["targets"],
            covariates=dataset_kwargs["covariates"],
            seed=params["seed"],
            frac=0.2,
            hue="cage",
            height=fig_height / 3,
        )
        after_fit_hook = partial(
            malnutrition_after_fit_hook,
            seed=params["seed"],
            results_path=args.results_path,
            targets=dataset_kwargs["targets"],
            plot_samples_kwargs=dict(height=fig_height / 3),
            plot_marginals_kwargs=dict(figsize=figsize),
        )

    elif dataset_type == "sim":
        get_dataset_fn = get_sim_dataset
        get_dataset_kwargs = {
            **dataset_kwargs,
            "dataset_name": dataset_name,
            "test_mode": test_mode,
        }

        def preprocess_dataset(data, model) -> dict:
            return {
                "x": tf.convert_to_tensor(data[1][..., None], dtype=model.dtype),
                "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
            }

        fig_height = get_figsize(params["textwidth"], fraction=0.5)[0]
        plot_data = partial(plot_2d_data, figsize=(fig_height, fig_height))
        after_fit_hook = partial(
            sim_after_fit_hook,
            results_path=results_path,
            is_hybrid=get_model == HybridDenistyRegressionModel,
            height=fig_height,
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_name)

    if test_mode:
        __LOGGER__.info("Running in test-mode")
        run_name += "_test"
        fit_kwargs.update(epochs=2)

    if which("latex"):
        __LOGGER__.info("Using latex backend for plotting")
        setup_latex(fontsize=10)

    if os.environ.get("CI", False):
        fit_kwargs.update(verbose=2)

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
    parser.add_argument(
        "--log-file",
        type=str,
        help="path for log file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="logging severity level",
    )
    parser.add_argument(
        "--test-mode",
        default=False,
        type=str2bool,
        help="activate test-mode",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="MLFlow experiment name",
        required=True,
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="MLFlow run name",
        required=True,
    )
    parser.add_argument(
        "--stage-name",
        type=str,
        help="name of dvc stage",
        required=True,
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        help="type of dataset",
        required=True,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="name of dataset",
        required=True,
    )
    parser.add_argument(
        "--results-path",
        type=str,
        help="destination for model checkpoints and logs.",
        required=True,
    )

    args = parser.parse_args()
    __LOGGER__.info("CLI arguments: %s", vars(args))

    params = prepare_pipeline(
        results_path=args.results_path,
        log_file=args.log_file,
        log_level=args.log_level,
        stage_name_or_params_file_path=args.stage_name,
    )

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
