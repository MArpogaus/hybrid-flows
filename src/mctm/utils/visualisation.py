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

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt


# PRIVATE FUNCTIONS ############################################################
def __joint_kde_plot__(data, x, y, **kwds):
    """
    Create a joint KDE (Kernel Density Estimation) plot of two variables.

    Generates a joint KDE plot of two variables from a given dataset.
    It visualizes the distribution of the variables in a two-dimensional
    space with contours.

    :param DataFrame data: The dataset containing the variables to be plotted.
    :param str x: The name of the variable to plot on the x-axis.
    :param str y: The name of the variable to plot on the y-axis.
    :param **kwds: Additional keyword arguments to customize the plot.
    :return: The generated plot figure.
    :rtype: Figure
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
        **kwds,
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


# PUBLIC FUNCTIONS #############################################################
def get_figsize(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    :param width: float or string
                  Document width in points, or string of predefined document type.
    :param fraction: float, optional
                     Fraction of the width which you wish the figure to occupy.
    :param subplots: array-like, optional
                     The number of rows and columns of subplots.
    :return: fig_dim: tuple
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

    return (fig_width_in, fig_height_in)


def setup_latex(fontsize=10):
    """
    Set up LaTeX font and text rendering for matplotlib.

    This function configures LaTeX font and text rendering settings
    for matplotlib plots.

    :param int fontsize: Font size to use for labels and text in the plots.
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


def plot_2d_data(X, Y, **kwds):
    """
    Create a 2D scatter plot for binary labeled data.

    This function creates a 2D scatter plot for binary labeled data points.
    It separates the data points by label and visualizes them in different
    colors.

    :param array X: Data points to plot.
    :param array Y: Binary labels for the data points.
    :param **kwds: Additional keyword arguments to customize the plot.
    :return: The generated plot figure.
    :rtype: Figure

    Example:
        fig = plot_2d_data(X, Y, s=20, alpha=0.6)
    """

    label = Y.astype(bool)
    X1, X2 = X[..., 0], X[..., 1]
    fig = plt.figure(**kwds)
    plt.scatter(X1[label], X2[label], s=10, color="blue")
    plt.scatter(X1[~label], X2[~label], s=10, color="red")
    plt.axis("equal")
    plt.legend(
        ["label: 1", "label: 0"],
        bbox_to_anchor=(0.5, 1.1),
        ncols=2,
        loc="upper center",
        frameon=False,
    )
    return fig


def plot_samples(dist, data, seed=1, **kwds):
    """
    Create a joint KDE plot of data samples and a probability distribution.

    This function generates a joint Kernel Density Estimation plot of data
    samples and a probability distribution. It visualizes the data and samples
    generated from the distribution.

    :param tfp.distributions.Distribution dist: A TensorFlow probability
                                                distribution.
    :param array data: Data samples to compare with the distribution.
    :param int seed: Random seed for generating samples, optional.
    :param **kwds: Additional keyword arguments to customize the plot.
    :return: The generated plot figure.
    :rtype: Figure
    """

    columns = ["$y_1$", "$y_2$"]
    if len(dist.batch_shape) == 0 or dist.batch_shape[0] == 1:
        N = data.shape[0]
    else:
        N = 1

    # Use the fitted distribution.
    start = time.time()
    samples = dist.sample(N, seed=seed).numpy().squeeze()
    end = time.time()
    print(f"sampling took {end-start} seconds.")

    df1 = pd.DataFrame(columns=columns, data=data)
    df1 = df1.assign(source="data")

    df2 = pd.DataFrame(columns=columns, data=samples)
    df2 = df2.assign(source="model")

    df = pd.concat([df1, df2])
    return __joint_kde_plot__(data=df, x=columns[0], y=columns[1], **kwds)


def plot_flow(dist, x, y, seed=1, **kwds):
    """
    Create joint KDE plots to visualize data transformation through a flow.

    This function generates joint Kernel Density Estimation plots to visualize
    data transformations through a flow model. It visualizes the input data,
    transformed data, and the transformed data's inverse.

    :param tfp.bijectors.Bijector dist: A TensorFlow probability bijector
                                         representing the data transformation.
    :param array x: Input data to the flow transformation.
    :param array y: Transformed data after applying the flow.
    :param int seed: Random seed for generating samples, optional.
    :param **kwds: Additional keyword arguments to customize the plots.
    :return: Figures of the joint KDE plots for data transformation.
    :rtype: tuple
    """

    columns = ["$y_1$", "$y_2$", "$z_{1,1}$", "$z_{1,2}$", "$z_{2,1}$", "$z_{2,2}$"]
    # forward flow (bnf in inverted)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    z1 = dist.bijector.inverse(y)
    z2 = dist.distribution.bijector.inverse(z1)
    df_inv = pd.DataFrame(np.concatenate([y, z1, z2], 1), columns=columns).assign(
        label=x, source="data"
    )

    # inverse flow
    z2 = dist.distribution.distribution.sample(y.shape[0], seed=seed)
    z1 = dist.distribution.bijector.forward(z2)
    yy = dist.bijector.forward(z1)
    df_fwd = pd.DataFrame(
        np.concatenate([yy, z1, z2], 1),
        columns=columns,
    ).assign(label=x, source="model")
    df = pd.concat((df_inv, df_fwd))

    # plot joint
    fig1 = __joint_kde_plot__(data=df, x=columns[0], y=columns[1], **kwds)

    # plot copula
    fig2 = __joint_kde_plot__(data=df, x=columns[2], y=columns[3], **kwds)

    # plot base
    fig3 = __joint_kde_plot__(data=df, x=columns[4], y=columns[5], **kwds)

    return fig1, fig2, fig3
