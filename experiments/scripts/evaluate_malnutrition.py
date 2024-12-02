# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : evaluate_sim.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-11-18 14:16:47 (Marcel Arpogaus)
# changed : 2024-12-02 16:13:45 (Marcel Arpogaus)

import seaborn as sns
from tensorflow_probability import distributions as tfd

# %% License ###################################################################
# %% Description ###############################################################
"""Train multivariate density estimation models on different datasets."""

# %% imports ###################################################################
import argparse
import logging
import os
from copy import deepcopy
from shutil import which

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

from mctm.data.malnutrion import get_dataset
from mctm.models import DensityRegressionModel, HybridDensityRegressionModel
from mctm.utils.mlflow import (
    log_and_save_figure,
    log_cfg,
    start_run_with_exception_logging,
)
from mctm.utils.pipeline import prepare_pipeline
from mctm.utils.visualisation import (
    _get_malnutrition_samples_df,
    get_figsize,
    plot_malnutrition_data,
    plot_malnutrition_samples,
    setup_latex,
)

# %% globals ###################################################################
__LOGGER__ = logging.getLogger(__name__)


# %% functions #################################################################
def plot_params(model, x, targets, **kwargs):
    t = np.linspace(min(x), max(x), 200, dtype="float32")
    pv = (
        model.marginal_transformation_parameters_fn(t[..., None])[1][-1]["parameters"]
        .numpy()
        .squeeze()
    )
    fig, axs = plt.subplots(1, len(targets), sharex=True, sharey=True, **kwargs)
    for (i, c), label in zip(enumerate(targets), targets):
        axs[i].plot(t, pv[:, i])
        axs[i].set_xlabel("cage")
        axs[i].set_title(label)
    axs[0].set_xticks((t.min(), t.max()))
    fig.tight_layout(w_pad=-0.1)
    return fig


def plot_rank_corr(model, x, targets, **kwargs):
    # %% rank correlation
    ages = np.unique(x)
    ages = np.sort(ages)
    joint_dist = model(tf.convert_to_tensor(ages, dtype=model.dtype)[..., None])

    lambdas = joint_dist.bijector.bijectors[-1].bijector.parameters["scale"].to_dense()

    # cov = joint_dist.distribution.covariance().numpy()
    cov = tf.linalg.inv(lambdas) @ tf.linalg.inv(tf.transpose(lambdas, perm=[0, 2, 1]))
    std = np.sqrt(tf.linalg.diag_part(cov))
    cor = cov / tf.matmul(std[..., None], std[..., None], transpose_b=True)

    fig, axs = plt.subplots(1, len(targets), **kwargs)

    for ax, (a, b) in zip(
        axs.T,
        zip(
            ["stunting", "stunting", "wasting"],
            ["wasting", "underweight", "underweight"],
        ),
    ):
        i, j = targets.index(a), targets.index(b)
        # ax.set_aspect(1)
        rho = cor[:, i, j]
        rho_s = 6 / np.pi * np.arcsin(rho / 2)
        ax.plot(ages, rho_s)
        ax.set_title(f"$\\rho^S_{{{a},{b}}}$")
        ax.set_box_aspect(1)
        ax.set_xticks(ages[0:-1:8])

    fig.tight_layout()
    return fig


def plot_marginal_distribution(model, ages, targets, palette="mako_r", **kwargs):
    # %% marginal cdf and pdf
    # palette = "icefire"
    # palette = "rocket_r"
    # palette = "mako_r"
    # ages = unscaled_train_data_df.cage.unique()
    colors = sns.color_palette(palette, as_cmap=True)(
        np.linspace(0, 1, len(ages))
    ).tolist()
    marginal_dist = model.marginal_distribution(
        tf.convert_to_tensor(ages, dtype=model.dtype)[..., None]
    )
    marginal_dist = tfd.TransformedDistribution(
        distribution=marginal_dist.distribution.distribution.distribution,
        bijector=marginal_dist.bijector,
    )

    y = np.linspace(-4, 4, 100)[..., None, None]

    cdf = marginal_dist.cdf(y).numpy()
    pdf = marginal_dist.prob(y).numpy()
    fig, axs = plt.subplots(2, len(targets), sharey="row", sharex=True, **kwargs)

    for i, c in enumerate(targets):
        axs[0, i].set_prop_cycle("color", colors)
        axs[0, i].plot(y.flatten(), cdf[..., i], label=ages, lw=0.5)
        axs[1, i].set_prop_cycle("color", colors)
        axs[1, i].plot(y.flatten(), pdf[..., i], label=ages, lw=0.5)
        axs[1, i].set_xlabel(f"y={c}")
        if i == 0:
            axs[0, i].legend(
                ages,
                title="Age",
                # bbox_to_anchor=(1.05, 1),
                loc="right",
                fontsize=8,
                frameon=False,
            )

    axs[0, 0].set_ylabel(r"$F(y|\text{age})$")
    axs[1, 0].set_ylabel(r"$f(y|\text{age})$")

    fig.tight_layout(w_pad=0)
    return fig


def ecdf(samples, x):
    ss = np.sort(samples)  # [..., None]
    cdf = np.searchsorted(ss, x, side="right") / float(ss.size)
    return cdf.astype(x.dtype)


def plot_reliability_diagram(model, dims, x, y, targets, **kwargs):
    reliability_df = _get_malnutrition_samples_df(model, x, y, 1, targets).pivot(
        columns="source"
    )
    reliability_df.columns = reliability_df.columns.map("{0[1]}_{0[0]}".format)
    reliability_df = reliability_df.drop(columns="model_cage").rename(
        columns={"data_cage": "cage"}
    )

    def apply_cdf(df):
        data_cols = ["data_" + c for c in targets]
        model_cols = ["model_" + c for c in targets]
        measurements = df.loc[:, data_cols].values
        samples = df.loc[:, model_cols].values
        marginal_dist = model.marginal_distribution(df.cage.unique()[..., None])
        marginal_dist = tfd.TransformedDistribution(
            distribution=marginal_dist.distribution.distribution,
            bijector=marginal_dist.bijector,
        )
        model_cdf = marginal_dist.cdf(measurements).numpy()
        data_ecdf = np.stack(list(map(lambda x: ecdf(x, x), measurements.T)), 1)
        samples_ecdf = np.stack(list(map(lambda x: ecdf(x, x), samples.T)), 1)
        cdf_columns = ["cdf_" + c for c in targets]
        data_ecdf_columns = ["ecdf_data_" + c for c in targets]
        samples_ecdf_columns = ["ecdf_model_" + c for c in targets]
        df.loc[:, cdf_columns] = model_cdf
        df.loc[:, data_ecdf_columns] = data_ecdf
        df.loc[:, samples_ecdf_columns] = samples_ecdf
        return df

    reliability_df = (
        reliability_df.groupby("cage")
        .apply(apply_cdf, include_groups=True)
        .reset_index(drop=True)
    )

    # Binning the predicted probabilities
    # Create bins for the predicted probabilities
    bins = np.linspace(0, 1, num=11)  # 10 equally spaced bins from 0 to 1
    for column in targets:
        reliability_df.loc[:, "cdf_binned_" + column] = reliability_df.loc[
            :, "cdf_" + column
        ].apply(pd.cut, by_row=False, bins=bins, include_lowest=True)
    for column in targets:
        reliability_df.loc[:, "ecdf_binned_" + column] = reliability_df.loc[
            :, "ecdf_model_" + column
        ].apply(pd.cut, by_row=False, bins=bins, include_lowest=True)

    fig, axs = plt.subplots(
        2,
        dims,
        sharey="row",
        sharex=True,
    )

    common_errorbar_kwargs = dict(
        markersize=0.2,
        marker="o",
        # capsize=1,
        color="C0",
        linewidth=0.5,
    )

    # Iterate over groups for different kinds
    for i, column in enumerate(targets):
        # Extract categories and corresponding mean ECDF values
        cdf_bin_col = "cdf_binned_" + column
        ecdf_bin_col = "ecdf_binned_" + column
        ecdf_data_col = "ecdf_data_" + column
        ecdf_model_col = "ecdf_model_" + column
        predicted_bins = reliability_df[cdf_bin_col].cat.categories.astype(str)
        grpd_data = reliability_df.groupby(cdf_bin_col)[ecdf_data_col]
        observed_freqs = grpd_data.mean()

        quantiles = grpd_data.quantile([0.25, 0.975]).unstack()

        pi = (quantiles - observed_freqs.values[..., None]).T.abs()

        axs[0][i].errorbar(
            predicted_bins, observed_freqs, yerr=pi, **common_errorbar_kwargs
        )
        axs[0][i].set_box_aspect(1)

        grpd_data = reliability_df.groupby(ecdf_bin_col)[ecdf_model_col]
        observed_freqs = grpd_data.mean()

        quantiles = grpd_data.quantile([0.25, 0.975]).unstack()

        pi = (quantiles - observed_freqs.values[..., None]).T.abs()

        axs[1][i].errorbar(
            predicted_bins, observed_freqs, yerr=pi, **common_errorbar_kwargs
        )
        axs[1][i].set_box_aspect(1)

        xticks = [0, len(predicted_bins) - 1]
        axs[1][i].set_xticks(xticks, predicted_bins[xticks])

        # Set labels and titles
        axs[0][i].set_title(
            f"{column.upper()}",
        )
        if i == 0:
            axs[0][i].set_ylabel("Observed relative\nfrequencies (marginal)")
            axs[1][i].set_ylabel("Observed relative\nfrequencies (joint)")
        axs[1][i].set_xlabel("Predicted probabilities\n(binned)")

        # Add diagonal line
        axs[0][i].plot(
            [predicted_bins[0], predicted_bins[-1]],
            [0, 1],
            linestyle=":",
            linewidth=0.5,
            color="gray",
        )
        axs[1][i].plot(
            [predicted_bins[0], predicted_bins[-1]],
            [0, 1],
            linestyle=":",
            linewidth=0.5,
            color="gray",
        )

    # Final adjustments
    sns.despine()

    return fig


def evaluate(
    dataset_name: str,
    dataset_type: str,
    experiment_name: str,
    run_name: str,
    results_path: str,
    params: dict,
    figure_format: str = "pdf",
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
    results_path : str
        Destination for model checkpoints and logs.
    params : dict
        Dictionary containing experiment parameters.

    Returns
    -------
    tuple
        Experiment results: history, model, and preprocessed data.

    """
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_name)
    mlflow.set_experiment(experiment_name)

    __LOGGER__.info(f"{tf.__version__=}\n{tfp.__version__=}")
    tf.config.set_visible_devices([], "GPU")

    figsize = get_figsize(params["textwidth"])
    fig_height = figsize[0]
    model_kwargs = params["model_kwargs"]
    dataset_kwargs = params["dataset_kwargs"][dataset_name]
    figure_path = os.path.join(results_path, "eval_figures")
    os.makedirs(figure_path, exist_ok=True)

    (train_data, validation_data, _), dims = get_dataset(**dataset_kwargs)

    if "marginal_bijectors" in model_kwargs.keys():
        get_model = HybridDensityRegressionModel
    else:
        get_model = DensityRegressionModel

    with start_run_with_exception_logging(run_name=run_name):
        log_cfg(
            dict(
                dataset_name=dataset_name, dataset_type=dataset_type, **deepcopy(params)
            )
        )

        model = get_model(dims=dims, **model_kwargs)
        model.load_weights(os.path.join(results_path, "model_checkpoint.weights.h5"))

        if which("latex"):
            __LOGGER__.info("Using latex backend for plotting")
            setup_latex(fontsize=10)
        fig = plot_malnutrition_data(
            validation_data,
            targets=dataset_kwargs["targets"],
            covariates=dataset_kwargs["covariates"],
            hue=dataset_kwargs["covariates"][0],
            seed=params["seed"],
            frac=1,
            height=fig_height / 3,
        )
        log_and_save_figure(
            figure=fig,
            figure_path=figure_path,
            file_name="_".join((dataset_type, "data")),
            file_format=figure_format,
            bbox_inches="tight",
            transparent=True,
        )

        fig = plot_malnutrition_samples(
            model=model,
            x=validation_data[0],
            y=validation_data[1],
            seed=params["seed"],
            targets=dataset_kwargs["targets"],
            height=fig_height / 3,
        )
        log_and_save_figure(
            figure=fig,
            figure_path=figure_path,
            file_name="_".join((dataset_type, "samples")),
            file_format=figure_format,
            bbox_inches="tight",
            transparent=True,
        )

        if isinstance(model, HybridDensityRegressionModel):
            marginal_bijectors = model_kwargs["marginal_bijectors"]
            joint_bijectors = model_kwargs["joint_bijectors"]
            if (
                len(marginal_bijectors) == 2
                and marginal_bijectors[-1]["bijector"] == "Shift"
            ):
                fig = plot_params(
                    model=model,
                    x=validation_data[0],
                    targets=dataset_kwargs["targets"],
                    figsize=figsize,
                )
                log_and_save_figure(
                    figure=fig,
                    figure_path=figure_path,
                    file_name="_".join((dataset_type, "params")),
                    file_format=figure_format,
                    bbox_inches="tight",
                    transparent=True,
                )
            if (
                len(joint_bijectors) == 1
                and joint_bijectors[0]["bijector"] == "ScaleMatvecLinearOperator"
            ):
                fig = plot_rank_corr(
                    model=model,
                    x=validation_data[0],
                    targets=dataset_kwargs["targets"],
                    figsize=figsize,
                )
                log_and_save_figure(
                    figure=fig,
                    figure_path=figure_path,
                    file_name="_".join((dataset_type, "rank_corr")),
                    file_format=figure_format,
                    bbox_inches="tight",
                    transparent=True,
                )

            fig = plot_marginal_distribution(
                model=model,
                ages=[1, 3, 6, 9, 12, 24],
                targets=dataset_kwargs["targets"],
                figsize=figsize,
            )
            log_and_save_figure(
                figure=fig,
                figure_path=figure_path,
                file_name="_".join((dataset_type, "distribution")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
            )
            fig = plot_reliability_diagram(
                model=model,
                dims=dims,
                x=validation_data[0],
                y=validation_data[1],
                targets=dataset_kwargs["targets"],
                figsize=figsize,
            )
            log_and_save_figure(
                figure=fig,
                figure_path=figure_path,
                file_name="_".join((dataset_type, "reliability_diagram")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
            )


# %% if main ###################################################################
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

    evaluate(
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        results_path=args.results_path,
        params=params,
    )
