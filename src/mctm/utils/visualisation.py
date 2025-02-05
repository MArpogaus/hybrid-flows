# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : visualisation.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-08-22 12:12:12 (Marcel Arpogaus)
# changed : 2025-02-05 22:25:36 (Marcel Arpogaus)


# %% License ###################################################################
# %% Description ###############################################################
"""Visualization utils."""

# %% imports ###################################################################
import logging
import time
from shutil import which
from typing import Any, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib.figure import Figure
from tensorflow_probability import distributions as tfd

from mctm.models import DensityRegressionModel

# %% globals
__LOGGER__ = logging.getLogger(__name__)


# %% private functions #########################################################
def _joint_kde_plot(data: pd.DataFrame, x: str, y: str, **kwargs: Any) -> Figure:
    """Generate a joint KDE plot of two variables from a dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing the variables to be plotted.
    x : str
        The name of the variable to plot on the x-axis.
    y : str
        The name of the variable to plot on the y-axis.
    kwargs
        Keyword arguments passed to `sns.jointplot`.

    Returns
    -------
    Figure
        The generated plot figure.

    """
    g = sns.jointplot(
        data=data,
        x=x,
        y=y,
        hue="source",
        alpha=0.5,
        s=10,
        xlim=(data[x].quantile(q=0.001) - 0.1, data[x].quantile(q=0.999) + 0.1),
        ylim=(data[y].quantile(q=0.001) - 0.1, data[y].quantile(q=0.999) + 0.1),
        **kwargs,
    )
    g.plot_joint(sns.kdeplot, legend=False)
    sns.move_legend(
        g.ax_joint,
        "upper left",
        bbox_to_anchor=(0.93, 1.25),
        title=None,
        frameon=False,
    )

    return g.figure


def _plot_grid(data: pd.DataFrame, **kwargs: Any) -> Figure:
    """Plot sns.PairGrid.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the data to plot.
    kwargs
        Additional arguments for sns.PairGrid.

    Returns
    -------
    Figure
        The figure object resulting from the PairGrid.

    """
    sns.set_theme(style="white")
    g = sns.PairGrid(data, diag_sharey=False, **kwargs)
    g.set(xlim=(-5, 5), ylim=(-5, 5))
    g.map_upper(sns.scatterplot, s=10, alpha=0.5)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)

    return g.figure


def _get_malnutrition_samples_df(
    model: DensityRegressionModel,
    x: tf.Tensor,
    y: np.ndarray,
    seed: int,
    targets: List[str],
) -> pd.DataFrame:
    """Generate a DataFrame with model samples and data for malnutrition analysis.

    Parameters
    ----------
    model : DensityRegressionModel
        The trained density regression model.
    x : tf.Tensor
        Input features for the model.
    y : np.ndarray
        Target values.
    seed : int
        Random seed for sampling.
    targets : list
        List of target variable names.

    Returns
    -------
    pd.DataFrame
        DataFrame containing model samples and data with source and cage information.

    """
    df_data = pd.DataFrame(y, columns=targets).assign(source="data").assign(cage=x)
    df_model = (
        pd.DataFrame(
            model(tf.squeeze(x)).sample(seed=seed).numpy().squeeze(), columns=targets
        )
        .assign(source="model")
        .assign(cage=x)
    )
    df = pd.concat([df_data, df_model])

    return df


# %% public functions ##########################################################
def get_figsize(
    width: Union[str, float], fraction: float = 1, subplots: Tuple[int, int] = (1, 1)
) -> Tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width : Union[str, float]
        Document width in points, or string of predefined document type.
    fraction : float, optional
        Fraction of the width which you wish the figure to occupy, by default 1.
    subplots : Tuple[int, int], optional
        The number of rows and columns of subplots, by default (1, 1).

    Returns
    -------
    Tuple[float, float]
        Dimensions of the figure in inches.

    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def setup_latex(fontsize: int = 10) -> None:
    """Set up LaTeX font and text rendering for matplotlib.

    Parameters
    ----------
    fontsize : int, optional
        Font size to use for labels and text in the plots, by default 10.

    """
    tex_fonts = {
        # Use LaTeX to write all text
        # https://stackoverflow.com/questions/43295837/latex-was-not-able-to-process-the-following-string-blp
        "text.usetex": True,
        "font.family": "serif",
        # for the align enivironment
        "text.latex.preamble": r"\usepackage{amsmath}",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": int(0.8 * fontsize),
        "xtick.labelsize": int(0.8 * fontsize),
        "ytick.labelsize": int(0.8 * fontsize),
    }

    if which("latex"):
        __LOGGER__.info("Using latex backend for plotting")
        plt.rcParams.update(tex_fonts)


def plot_2d_data(
    X: np.ndarray,
    Y: np.ndarray,
    plot_kwargs: dict = dict(s=5, alpha=0.7),
    **kwargs: Any,
) -> Figure:
    """Create a 2D scatter plot for binary labeled data.

    Parameters
    ----------
    X : np.ndarray
        Data points to plot.
    Y : np.ndarray
        Binary labels for the data points.
    plot_kwargs : dict, optional
        Keyword arguments passed to `plt.scatter`,
        by default dict(s=5, alpha=0.7).
    kwargs
        Keyword arguments passed to `plt.figure`.

    Returns
    -------
    Figure
        The generated plot figure.

    """
    label = Y.astype(bool)
    X1, X2 = X[..., 0], X[..., 1]
    fig = plt.figure(**kwargs)
    plt.scatter(X1[label], X2[label], color="blue", **plot_kwargs)
    plt.scatter(X1[~label], X2[~label], color="red", **plot_kwargs)
    plt.axis("equal")
    plt.legend(
        ["label: 1", "label: 0"],
        bbox_to_anchor=(0.5, 1.1),
        ncols=2,
        loc="upper center",
        frameon=False,
    )
    return fig


def plot_samples(
    dist: tfd.Distribution,
    data: np.ndarray,
    seed: int = 1,
    **kwargs: Any,
) -> Figure:
    """Create a joint KDE plot of data samples and a probability distribution.

    Parameters
    ----------
    dist : tfd.Distribution
        A TensorFlow probability distribution.
    data : np.ndarray
        Data samples to compare with the distribution.
    seed : int, optional
        Random seed for generating samples, by default 1.
    kwargs
        Keyword arguments passed to `_joint_kde_plot`.

    Returns
    -------
    Figure
        The generated plot figure.

    """
    columns = ["$y_1$", "$y_2$"]
    if len(dist.batch_shape) == 0:
        N = data.shape[0]
    else:
        N = []

    start = time.time()
    samples = dist.sample(N, seed=seed).numpy().squeeze()
    end = time.time()
    print(f"sampling took {end - start} seconds.")

    df1 = pd.DataFrame(columns=columns, data=data).assign(source="data")
    df2 = pd.DataFrame(columns=columns, data=samples).assign(source="model")
    df = pd.concat([df1, df2])

    return _joint_kde_plot(data=df, x=columns[0], y=columns[1], **kwargs)


def plot_flow(
    dist: tfp.bijectors.Bijector,
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 1,
    **kwargs: Any,
) -> Tuple[Figure, Figure, Figure]:
    """Create joint KDE plots to visualize data transformation through a flow.

    Parameters
    ----------
    dist : tfp.bijectors.Bijector
        A TensorFlow probability bijector representing the data transformation.
    x : np.ndarray
        Input data to the flow transformation.
    y : np.ndarray
        Transformed data after applying the flow.
    seed : int, optional
        Random seed for generating samples, by default 1.
    kwargs
        Keyword arguments passed to `_joint_kde_plot`.

    Returns
    -------
    Tuple[Figure, Figure, Figure]
        Figures of the joint KDE plots for:
            - Data transformation.
            - Independent components.
            - Base distribution.

    """
    base_dist = dist.distribution.distribution
    columns = ["$y_1$", "$y_2$", "$z_{1,1}$", "$z_{1,2}$", "$z_{2,1}$", "$z_{2,2}$"]
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

    # inverse flow
    z1 = dist.inverse(y_tensor)
    z2 = base_dist.bijector.inverse(z1)
    df_inv = pd.DataFrame(np.concatenate([y, z1, z2], 1), columns=columns).assign(
        label=x, source="data"
    )

    # forward flow (bnf in inverted)
    z2 = base_dist.distribution.sample(y.shape, seed=seed)
    z1 = base_dist.bijector.forward(z2)
    yy = dist.forward(z1)
    df_fwd = pd.DataFrame(
        np.concatenate([yy, z1, z2], 1),
        columns=columns,
    ).assign(label=x, source="model")
    df = pd.concat((df_inv, df_fwd))

    # plot joint
    fig1 = _joint_kde_plot(data=df, x=columns[0], y=columns[1], **kwargs)

    # plot independent
    fig2 = _joint_kde_plot(data=df, x=columns[2], y=columns[3], **kwargs)

    # plot base
    fig3 = _joint_kde_plot(data=df, x=columns[4], y=columns[5], **kwargs)

    return fig1, fig2, fig3


def plot_copula_function(
    dist: Any,
    y: np.ndarray,
    kind: str,
    x_min: float,
    x_max: float,
    n: int,
    **kwargs: Any,
) -> Figure:
    """Plot copula function either as surface or contour.

    Parameters
    ----------
    dist : Any
        Copula distribution from TensorFlow Probability.
    y : np.ndarray
        Data points for the Copula function.
    kind : str
        Type of plot to generate ('surface' or 'contour').
    x_min : float
        Minimum value for the x-axis.
    x_max : float
        Maximum value for the x-axis.
    n : int
        Number of points to generate between x_min and x_max.
    kwargs
        Keyword arguments passed to plot function.

    Returns
    -------
    Figure
        Generated Matplotlib figure.

    Raises
    ------
    ValueError
        If an unsupported plot kind is provided.

    """
    dist_z1 = dist.distribution

    x = np.linspace(x_min, x_max, n)
    xx, yy = np.meshgrid(x, x)
    grid = np.stack([xx.flatten(), yy.flatten()], -1)

    p_y = dist.prob(grid).numpy()
    p_z1 = dist_z1.prob(grid).numpy()

    # c(y) = p_y(y) / p_z1(y)
    c = p_y / p_z1
    c = np.where(p_z1 < 1e-4, 0, c)  # for numerical stability

    if kind == "surface":
        fig = plt.figure(figsize=plt.figaspect(0.3))
        ax = fig.add_subplot(131, projection="3d")
        ax.plot_surface(
            xx,
            yy,
            p_y.reshape(-1, n),
            cmap="plasma",
            linewidth=0,
            antialiased=False,
            alpha=0.5,
            **kwargs,
        )
        ax = fig.add_subplot(132, projection="3d")
        ax.plot_surface(
            xx,
            yy,
            p_z1.reshape(-1, n),
            cmap="plasma",
            linewidth=0,
            antialiased=False,
            alpha=0.5,
            **kwargs,
        )
        ax = fig.add_subplot(133, projection="3d")
        ax.plot_surface(
            xx,
            yy,
            c.reshape(-1, n),
            cmap="plasma",
            linewidth=0,
            antialiased=False,
            alpha=0.5,
            **kwargs,
        )
        return fig
    elif kind == "contour":
        fig, axs = plt.subplots(1, 3, figsize=plt.figaspect(0.3))
        axs[0].contourf(xx, yy, p_y.reshape(n, n), **kwargs)
        axs[1].contourf(xx, yy, p_z1.reshape(n, n), **kwargs)
        axs[2].contourf(xx, yy, c.reshape(n, n), **kwargs)
        return fig
    else:
        raise ValueError(f"Unsupported plot kind: {kind}")


def plot_malnutrition_data(
    *data: Tuple[Tuple[tf.Tensor, ...], ...],
    targets: List[str],
    covariates: List[str],
    seed: int,
    frac: float,
    **kwargs: Any,
) -> Figure:
    """Plot malnutrition data.

    Parameters
    ----------
    data : tuple
        Data containing the features and targets.
    targets : list
        List of target variable names.
    covariates : list
        List of covariate variable names.
    seed : int
        Random seed for sampling.
    frac : float
        Fraction of data to sample.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    Figure
        The resulting figure from the plot.

    """
    df = pd.DataFrame(
        np.concatenate([data[0][1], data[0][0]], -1),
        columns=targets + covariates,
    )
    return _plot_grid(
        df.groupby(covariates).sample(frac=frac, random_state=seed),
        vars=targets,
        **kwargs,
    )


def plot_malnutrition_samples(
    model: DensityRegressionModel,
    x: tf.Tensor,
    y: np.ndarray,
    seed: int,
    targets: List[str],
    frac: float,
    **kwargs: Any,
) -> Figure:
    """Generate and plot samples from a malnutrition model.

    Parameters
    ----------
    model : DensityRegressionModel
        The trained density regression model for malnutrition.
    x : tf.Tensor
        Input features for the model.
    y : np.ndarray
        Target values.
    seed : int
        Random seed for sampling.
    targets : list
        List of target variable names.
    frac : float
        Fraction of data to sample.
    **kwargs
        Additional plotting arguments passed to `_plot_grid`.

    Returns
    -------
    Figure
        The resulting figure from the plot.

    """
    df = _get_malnutrition_samples_df(model=model, x=x, y=y, seed=seed, targets=targets)
    fig = _plot_grid(
        df.loc[df.source == "model"]
        .groupby("cage")
        .sample(frac=frac, random_state=seed),
        vars=targets,
        hue="cage",
        **kwargs,
    )
    return fig


def plot_marginal_cdf_and_pdf(
    model: DensityRegressionModel,
    covariates: List[int],
    target_names: List[str],
    palette: str = "mako_r",
    y_range: Tuple[float, float] = (-4, 4),
    y_samples: int = 100,
    n_bijectors: int = 2,
    **kwargs: Any,
) -> Figure:
    """Plot the marginal CDF and PDF for a given model and dataset.

    Parameters
    ----------
    model : DensityRegressionModel
        The density regression model.
    covariates : List[int]
        A list of covariates to plot.
    target_names : List[str]
        A list of target names.
    palette : str, optional
        The color palette to use, by default "mako_r".
    y_range : Tuple[float, float], optional
        The range of y values to plot, by default (-4, 4).
    y_samples : int, optional
        The number of y samples to use, by default 100.
    n_bijectors : int, optional
        The number of bijectors to use, by default 2.
    **kwargs
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
    Figure
        The matplotlib figure object.

    """
    colors = sns.color_palette(palette, as_cmap=True)(
        np.linspace(0, 1, len(covariates))
    ).tolist()
    joint_dist = model(tf.convert_to_tensor(covariates, dtype=model.dtype)[..., None])
    marginal_dist = tfd.TransformedDistribution(
        distribution=tfd.Normal(0, 1),
        bijector=tfp.bijectors.Invert(
            tfp.bijectors.Chain(joint_dist.bijector.bijector.bijectors[-n_bijectors:])
        ),
    )

    y = np.linspace(y_range[0], y_range[1], y_samples)[..., None, None]

    cdf = marginal_dist.cdf(y).numpy()
    pdf = marginal_dist.prob(y).numpy()
    fig, axs = plt.subplots(2, len(target_names), sharey="row", sharex=True, **kwargs)

    for i, c in enumerate(target_names):
        axs[0, i].set_prop_cycle("color", colors)
        axs[0, i].plot(y.flatten(), cdf[..., i], label=covariates, lw=0.5)
        axs[1, i].set_prop_cycle("color", colors)
        axs[1, i].plot(y.flatten(), pdf[..., i], label=covariates, lw=0.5)
        axs[1, i].set_xlabel(f"y={c}")
        if i == 0:
            axs[0, i].legend(
                covariates,
                title="Age",
                loc="right",
                fontsize=8,
                frameon=False,
            )

    axs[0, 0].set_ylabel(r"$F(y|\text{age})$")
    axs[1, 0].set_ylabel(r"$f(y|\text{age})$")

    fig.tight_layout(w_pad=0)
    return fig
