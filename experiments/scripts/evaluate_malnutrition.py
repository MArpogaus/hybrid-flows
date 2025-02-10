# %% Description ###############################################################
"""Train multivariate density estimation models on different datasets."""

# %% imports ###################################################################
import argparse
import logging
import os
from copy import deepcopy

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from hybrid_flows.data import get_dataset, make_malnutrition_dataset
from hybrid_flows.models import DensityRegressionModel, HybridDensityRegressionModel
from hybrid_flows.utils.mlflow import (
    log_and_save_figure,
    log_cfg,
    start_run_with_exception_logging,
)
from hybrid_flows.utils.pipeline import prepare_pipeline
from hybrid_flows.utils.visualisation import (
    _get_malnutrition_samples_df,
    get_figsize,
    plot_malnutrition_data,
    plot_malnutrition_samples,
    setup_latex,
)
from matplotlib import pyplot as plt
from tensorflow_probability import distributions as tfd

# %% globals ###################################################################
__LOGGER__ = logging.getLogger(__name__)


# %% functions #################################################################
def plot_params(model, x, targets, **kwargs):
    """Plot marginal parameter functions."""
    t = np.linspace(min(x), max(x), 200)
    pv = (
        model.marginal_transformation_parameters_fn(
            tf.convert_to_tensor(t, dtype=model.dtype)
        )[1][-1]["parameters"]
        .numpy()
        .squeeze()
    )
    fig, axs = plt.subplots(1, len(targets), sharex=True, sharey=True, **kwargs)
    for (i, c), label in zip(enumerate(targets), targets):
        axs[i].plot(t, -pv[:, i])
        axs[i].set_xlabel("cage")
        axs[i].set_title(label.upper())
    axs[0].set_ylabel(r"$\beta_x$")
    axs[0].set_xticks((t.min(), t.max()))
    fig.tight_layout(w_pad=-0.1)
    return fig


def plot_rank_corr(model, x, targets, **kwargs):
    """Plot rank correlation from linear dependency matrix."""
    # %% rank correlation
    ages = np.unique(x)
    ages = np.sort(ages)
    joint_dist = model(tf.convert_to_tensor(ages, dtype=model.dtype))

    lambdas = joint_dist.bijector.bijectors[-1].bijector.parameters["scale"].to_dense()

    # cov = joint_dist.distribution.covariance().numpy()
    cov = tf.linalg.inv(lambdas) @ tf.linalg.inv(tf.transpose(lambdas, perm=[0, 2, 1]))
    std = np.sqrt(tf.linalg.diag_part(cov))
    # corr(X_i,X_j) = cov(X_i, X_j) / (std(X_i), std(X_j))
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
        ax.set_xlabel("cage")
        ax.set_box_aspect(1)
        ax.set_xticks(ages[0:-1:8])

    fig.tight_layout()
    return fig


def plot_marginal_distribution(model, ages, targets, palette="mako_r", **kwargs):
    """Plot marginal cdf and pdf."""
    # palette = "icefire"
    # palette = "rocket_r"
    # palette = "mako_r"
    # ages = unscaled_train_data_df.cage.unique()
    colors = sns.color_palette(palette, as_cmap=True)(
        np.linspace(0, 1, len(ages))
    ).tolist()
    marginal_dist = model.marginal_distribution(
        tf.convert_to_tensor(ages, dtype=model.dtype)
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

    axs[0, 0].set_ylabel(r"$F(y|\text{cage})$")
    axs[1, 0].set_ylabel(r"$f(y|\text{cage})$")

    fig.tight_layout(w_pad=0)
    return fig


def ecdf(samples, x):
    """Empirical cumulative density function."""
    ss = np.sort(samples)  # [..., None]
    cdf = np.searchsorted(ss, x, side="right") / float(ss.size)
    return cdf.astype(x.dtype)


def plot_reliability_diagram(model, dims, x, y, targets, **kwargs):
    """Plot reliability diagram."""
    reliability_df = _get_malnutrition_samples_df(model, x, y, 1, targets).pivot(
        columns="source"
    )
    reliability_df.columns = reliability_df.columns.map("{0[1]}_{0[0]}".format)
    reliability_df = reliability_df.drop(columns="model_cage").rename(
        columns={"data_cage": "cage"}
    )

    def apply_cdf(df):
        data_cols = ["data_" + c for c in targets]
        measurements = df.loc[:, data_cols].values
        marginal_dist = model.marginal_distribution(
            tf.convert_to_tensor(df.cage.unique(), dtype=model.dtype)
        )
        marginal_dist = tfd.TransformedDistribution(
            distribution=marginal_dist.distribution.distribution.distribution,
            bijector=marginal_dist.bijector,
        )
        model_cdf = marginal_dist.cdf(measurements).numpy()
        data_ecdf = np.stack(list(map(lambda x: ecdf(x, x), measurements.T)), 1)
        cdf_columns = ["cdf_" + c for c in targets]
        data_ecdf_columns = ["ecdf_data_" + c for c in targets]
        df.loc[:, cdf_columns] = model_cdf
        df.loc[:, data_ecdf_columns] = data_ecdf
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

    fig, axs = plt.subplots(1, dims, sharey="row", sharex=True, **kwargs)

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
        ecdf_data_col = "ecdf_data_" + column
        predicted_bins = reliability_df[cdf_bin_col].cat.categories.astype(str)
        grpd_data = reliability_df.groupby(cdf_bin_col)[ecdf_data_col]
        observed_freqs = grpd_data.mean()

        quantiles = grpd_data.quantile([0.25, 0.975]).unstack()

        pi = (quantiles - observed_freqs.values[..., None]).T.abs()

        axs[i].errorbar(
            predicted_bins, observed_freqs, yerr=pi, **common_errorbar_kwargs
        )
        axs[i].set_box_aspect(1)

        xticks = [0, len(predicted_bins) - 1]
        axs[i].set_xticks(xticks, predicted_bins[xticks])
        axs[i].set_xlabel("Predicted probabilities\n(binned)")

        # Set labels and titles
        axs[i].set_title(column.upper())
        if i == 0:
            axs[i].set_ylabel("Observed relative\nfrequencies (marginal)")

        # Add diagonal line
        axs[i].plot(
            [predicted_bins[0], predicted_bins[-1]],
            [0, 1],
            linestyle=":",
            linewidth=0.5,
            color="gray",
        )

    # Final adjustments
    fig.tight_layout()
    sns.despine()

    return fig


def get_quantile(data_or_dist, q, i=None):
    """Get qunatile from either distribution or ecdf."""
    if isinstance(data_or_dist, tfd.Distribution):
        return data_or_dist.quantile(q).numpy()
    else:
        return np.quantile(data_or_dist[..., i], q)


def qq_plot(
    x,
    y,
    targets,
    n_probs=200,
    xlabel="Estimated Quantile",
    ylabel="Observed Quantile",
    low_lim=-5,
    high_lim=5,
    eps=1e-6,
    **kwargs,
):
    """Create a Qunatile-Quantile-Plot."""
    fig, axs = plt.subplots(1, len(targets), sharey=True, **kwargs)
    q = np.linspace(eps, 1 - eps, n_probs)
    for i, ax in enumerate(axs):
        x_quantiles = get_quantile(x, q, i)
        y_quantiles = get_quantile(y, q, i)

        ax.plot((low_lim, high_lim), (low_lim, high_lim), "k:", linewidth=1)
        ax.plot(x_quantiles, y_quantiles)

        ax.set_title(targets[i].upper())
        ax.set_xlabel(xlabel)
        ax.set_aspect("equal")
        ax.set(xlim=(low_lim, high_lim), ylim=(low_lim, high_lim))
        ax.set_xticks([])
        ax.set_yticks([])

    axs[0].set_ylabel(ylabel)

    fig.tight_layout(w_pad=0.2)

    return fig


def evaluate(
    dataset_name: str,
    dataset_type: str,
    model_name: str,
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
    model_name : str
        Name of the model to evaluate.
    results_path : str
        Destination for model checkpoints and logs.
    params : dict
        Dictionary containing experiment parameters.
    figure_format: str
        Data format to save figures to.

    Returns
    -------
    tuple
        Experiment results: history, model, and preprocessed data.

    """
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "_".join((dataset_type, "evaluation"))
    )
    run_name = "_".join((model_name, dataset_name, "evaluation"))

    __LOGGER__.info(f"{tf.__version__=}\n{tfp.__version__=}")
    tf.config.set_visible_devices([], "GPU")

    figsize = get_figsize(params["textwidth"])
    fig_height = figsize[0]
    figsize_half = get_figsize(params["textwidth"], fraction=0.7)
    model_kwargs = params["model_kwargs"]
    dataset_kwargs = params["dataset_kwargs"][dataset_name]
    figure_path = os.path.join(results_path, "eval_figures/")
    os.makedirs(figure_path, exist_ok=True)

    get_dataset_fn, get_dataset_kwargs, _ = get_dataset(
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        test_mode=False,
    )
    (train_data, validation_data, test_data), dims = get_dataset_fn(**dataset_kwargs)

    if "marginal_bijectors" in model_kwargs.keys():
        get_model = HybridDensityRegressionModel
    else:
        get_model = DensityRegressionModel

    mlflow.set_experiment(experiment_name)
    with start_run_with_exception_logging(run_name=run_name):
        log_cfg(
            dict(
                dataset_name=dataset_name, dataset_type=dataset_type, **deepcopy(params)
            )
        )

        model = get_model(dims=dims, **model_kwargs)
        model.load_weights(os.path.join(results_path, "model_checkpoint.weights.h5"))
        model.compile(
            loss=lambda y, dist: -dist.log_prob(y), **params["compile_kwargs"]
        )

        train_loss = model.evaluate(
            make_malnutrition_dataset(train_data, batch_size=2**10)
        )
        validation_loss = model.evaluate(
            make_malnutrition_dataset(validation_data, batch_size=2**10)
        )
        test_loss = model.evaluate(
            make_malnutrition_dataset(test_data, batch_size=2**10)
        )

        __LOGGER__.info("train_loss: %.3f", train_loss)
        __LOGGER__.info("validation_loss: %.3f", validation_loss)
        __LOGGER__.info("test_loss: %.3f", test_loss)

        mlflow.log_metric("train_loss", train_loss)
        mlflow.log_metric("validation_loss", validation_loss)
        mlflow.log_metric("test_loss", test_loss)

        setup_latex(fontsize=10)
        fig = plot_malnutrition_data(
            validation_data,
            targets=dataset_kwargs["targets"],
            covariates=dataset_kwargs["covariates"],
            hue=dataset_kwargs["covariates"][0],
            seed=params["seed"],
            frac=0.8,
            height=fig_height / 3,
        )
        log_and_save_figure(
            figure=fig,
            figure_path=figure_path,
            file_name="_".join((dataset_type, model_name, "data")),
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
            frac=0.8,
        )
        log_and_save_figure(
            figure=fig,
            figure_path=figure_path,
            file_name="_".join((dataset_type, model_name, "samples")),
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
                setup_latex(fontsize=10)
                fig = plot_params(
                    model=model,
                    x=validation_data[0],
                    targets=dataset_kwargs["targets"],
                    figsize=figsize_half,
                )
                log_and_save_figure(
                    figure=fig,
                    figure_path=figure_path,
                    file_name="_".join((dataset_type, model_name, "params")),
                    file_format=figure_format,
                    bbox_inches="tight",
                    transparent=True,
                    extract_data=True,
                )
            if (
                len(joint_bijectors) == 1
                and joint_bijectors[0]["bijector"] == "ScaleMatvecLinearOperator"
            ):
                setup_latex(fontsize=10)
                fig = plot_rank_corr(
                    model=model,
                    x=validation_data[0],
                    targets=dataset_kwargs["targets"],
                    figsize=figsize_half,
                )
                log_and_save_figure(
                    figure=fig,
                    figure_path=figure_path,
                    file_name="_".join((dataset_type, model_name, "rank_corr")),
                    file_format=figure_format,
                    bbox_inches="tight",
                    transparent=True,
                )

            setup_latex(fontsize=10)
            fig = plot_marginal_distribution(
                model=model,
                ages=[1, 3, 6, 9, 12, 24],
                targets=dataset_kwargs["targets"],
                figsize=figsize,
            )
            log_and_save_figure(
                figure=fig,
                figure_path=figure_path,
                file_name="_".join((dataset_type, model_name, "distribution")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
            )
            setup_latex(fontsize=10)
            fig = plot_reliability_diagram(
                model=model,
                dims=dims,
                x=validation_data[0],
                y=validation_data[1],
                targets=dataset_kwargs["targets"],
                figsize=figsize_half,
            )
            log_and_save_figure(
                figure=fig,
                figure_path=figure_path,
                file_name="_".join((dataset_type, model_name, "reliability_diagram")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
            )

            # Q-Q: data - samples
            X, Y = validation_data
            dist = model.joint_distribution(tf.squeeze(X))

            setup_latex(fontsize=10)
            fig = qq_plot(
                Y,
                dist.sample(seed=1),
                targets=dataset_kwargs["targets"],
                n_probs=200,
                xlabel="Observed Quantile",
                ylabel="Estimated Quantile",
                figsize=figsize_half,
            )
            log_and_save_figure(
                figure=fig,
                figure_path=figure_path,
                file_name="_".join((dataset_type, model_name, "qq_data_samples")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
                extract_data=True,
            )

            # Q-Q: normal - marginal
            X, Y = validation_data
            dist = model.marginal_distribution(X)
            flow = dist.bijector
            normal_base = dist.distribution.distribution.distribution

            normalized_data = flow.inverse(Y)

            setup_latex(fontsize=10)
            fig = qq_plot(
                normal_base,
                normalized_data,
                targets=dataset_kwargs["targets"],
                xlabel="Normal Quantile",
                ylabel="$W$ Quantile",
                figsize=figsize_half,
            )
            log_and_save_figure(
                figure=fig,
                figure_path=figure_path,
                file_name="_".join((dataset_type, model_name, "qq_w_base")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
                extract_data=True,
            )

            # Q-Q: normal - joint
            dist = model.joint_distribution(tf.squeeze(X))
            flow = dist.bijector
            normal_base = dist.distribution.distribution.distribution

            normalized_data = flow.inverse(Y)

            setup_latex(fontsize=10)
            fig = qq_plot(
                normal_base,
                normalized_data,
                targets=dataset_kwargs["targets"],
                xlabel="Normal Quantile",
                ylabel="$Z$ Quantile",
                figsize=figsize_half,
            )
            log_and_save_figure(
                figure=fig,
                figure_path=figure_path,
                file_name="_".join((dataset_type, model_name, "qq_z_base")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
                extract_data=True,
            )

            # Q-Q: w - z
            marginal_dist = model.marginal_distribution(tf.squeeze(X))
            joint_dist = model.joint_distribution(tf.squeeze(X))

            w = marginal_dist.bijector.inverse(Y)
            z = joint_dist.bijector.inverse(Y)

            setup_latex(fontsize=10)
            fig = qq_plot(
                w,
                z,
                targets=dataset_kwargs["targets"],
                xlabel="$W$ Quantile",
                ylabel="$Z$ Quantile",
                figsize=figsize_half,
            )
            log_and_save_figure(
                figure=fig,
                figure_path=figure_path,
                file_name="_".join((dataset_type, model_name, "qq_w_z")),
                file_format=figure_format,
                bbox_inches="tight",
                transparent=True,
                extract_data=True,
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
        "--model-name",
        type=str,
        help="model name",
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
    results_path = os.path.join(
        args.results_path, args.dataset_type, args.dataset_name, args.model_name
    )

    params = prepare_pipeline(
        results_path=results_path,
        log_file=args.log_file,
        log_level=args.log_level,
        stage_name_or_params_file_path=args.stage_name,
    )

    evaluate(
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        model_name=args.model_name,
        results_path=results_path,
        params=params,
    )
