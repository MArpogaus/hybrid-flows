"""Visualization utils."""
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : visualisation.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-08-22 13:19:25 (Marcel Arpogaus)
# changed : 2023-08-22 13:19:25 (Marcel Arpogaus)
# DESCRIPTION ##################################################################
# ...
# LICENSE ######################################################################
# ...
################################################################################
import time
from typing import Any, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp


def _joint_kde_plot(data: pd.DataFrame, x: str, y: str, **kwargs) -> plt.Figure:
    """Generate a joint KDE plot of two variables from a dataset.

    Parameters
    ----------
    data
        The dataset containing the variables to be plotted.
    x
        The name of the variable to plot on the x-axis.
    y
        The name of the variable to plot on the y-axis.
    kwargs
        Keyword arguments passed to `sns.jointplot`

    Returns
    -------
    plt.Figure
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


def get_figsize(
    width: Union[str, float], fraction: float = 1, subplots: Tuple[int, int] = (1, 1)
) -> Tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width
        Document width in points, or string of predefined document type.
    fraction
        Fraction of the width which you wish the figure to occupy.
    subplots
        The number of rows and columns of subplots.

    Returns
    -------
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
    fontsize
        Font size to use for labels and text in the plots.

    """
    tex_fonts = {
        # Use LaTeX to write all text
        # https://stackoverflow.com/questions/43295837/latex-was-not-able-to-process-the-following-string-blp
        "text.usetex": False,
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

    plt.rcParams.update(tex_fonts)


def plot_2d_data(
    X: np.ndarray, Y: np.ndarray, plot_kwargs: dict = dict(s=5, alpha=0.7), **kwargs
) -> plt.Figure:
    """Create a 2D scatter plot for binary labeled data.

    Parameters
    ----------
    X
        Data points to plot.
    Y
        Binary labels for the data points.
    plot_kwargs
        Keyword arguments passed to `plt.scatter`
    kwargs
        Keyword arguments passed to `plt.figure`

    Returns
    -------
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
    dist: tfp.distributions.Distribution, data: np.ndarray, seed: int = 1, **kwargs
) -> plt.Figure:
    """Create a joint KDE plot of data samples and a probability distribution.

    Parameters
    ----------
    dist
        A TensorFlow probability distribution.
    data
        Data samples to compare with the distribution.
    seed
        Random seed for generating samples.
    kwargs
        Keyword arguments passed to `_joint_kde_plot`

    Returns
    -------
        The generated plot figure.

    """
    columns = ["$y_1$", "$y_2$"]
    if len(dist.batch_shape) == 0 or dist.batch_shape[0] == data.shape[-1]:
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
    dist: tfp.bijectors.Bijector, x: np.ndarray, y: np.ndarray, seed: int = 1, **kwargs
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Create joint KDE plots to visualize data transformation through a flow.

    Parameters
    ----------
    dist
        A TensorFlow probability bijector representing the data transformation.
    x
        Input data to the flow transformation.
    y
        Transformed data after applying the flow.
    seed
        Random seed for generating samples.
    kwargs
        Keyword arguments passed to `_joint_kde_plot`

    Returns
    -------
        Figures of the joint KDE plots for data transformation.

    """
    base_dist = dist.distribution.distribution
    columns = ["$y_1$", "$y_2$", "$z_{1,1}$", "$z_{1,2}$", "$z_{2,1}$", "$z_{2,2}$"]
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # inverse flow
    z1 = dist.bijector.inverse(y)
    z2 = base_dist.bijector.inverse(z1)
    df_inv = pd.DataFrame(np.concatenate([y, z1, z2], 1), columns=columns).assign(
        label=x, source="data"
    )

    # forward flow (bnf in inverted)
    z2 = base_dist.distribution.sample(y.shape, seed=seed)
    z1 = base_dist.bijector.forward(z2)
    yy = dist.bijector.forward(z1)
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
    dist: Any, y: np.ndarray, kind: str, x_min: float, x_max: float, n: int, **kwargs
) -> plt.Figure:
    """Plot copula function either as surface or contour.

    Parameters
    ----------
    dist
        Copula distribution from TensorFlow Probability.
    y
        Data points for the Copula function.
    kind
        Type of plot to generate ('surface' or 'contour').
    x_min
        Minimum value for the x-axis.
    x_max
        Maximum value for the x-axis.
    n
        Number of points to generate between x_min and x_max.
    kwargs
        Keyword arguments passed to plot function.

    Returns
    -------
        Generated Matplotlib figure.

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
