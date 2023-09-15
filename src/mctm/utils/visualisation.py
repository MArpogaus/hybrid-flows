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
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
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
    columns = ["$x_1$", "$x_2$"]
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
