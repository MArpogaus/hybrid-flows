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

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# PRIVATE FUNCTIONS ############################################################
def __set_size__(width, fraction=1, subplots=(1, 1)):
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


# PUBLIC FUNCTIONS #############################################################
def setup_latex(fontsize=10):
    tex_fonts = {
        # Use LaTeX to write all text
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

    plt.rcParams.update(tex_fonts)


def plot_2d_data(X, Y, **kwds):
    label = Y.astype(bool)
    X1, X2 = X[..., 0], X[..., 1]
    fig = plt.figure(**kwds)
    plt.scatter(X1[label], X2[label], s=10, color="blue")
    plt.scatter(X1[~label], X2[~label], s=10, color="red")
    plt.legend(["label: 1", "label: 0"])
    return fig


def plot_samples(dist, data, seed=1, **kwds):
    if len(dist.batch_shape) == 0 or dist.batch_shape[0] == 1:
        N = data.shape[0]
    else:
        N = 1

    # Use the fitted distribution.
    start = time.time()
    samples = dist.sample(N, seed=seed).numpy().squeeze()
    end = time.time()
    print(f"sampling took {end-start} seconds.")

    df1 = pd.DataFrame(columns=["x1", "x2"], data=data)
    df1 = df1.assign(source="data")

    df2 = pd.DataFrame(columns=["x1", "x2"], data=samples)
    df2 = df2.assign(source="model")

    df = pd.concat([df1, df2])

    # sns.jointplot(data=df, x='x1', y='x2', hue='source', kind='kde')
    g = sns.jointplot(
        data=df,
        x="x1",
        y="x2",
        hue="source",
        alpha=0.5,
        xlim=(data[..., 0].min() - 0.1, data[..., 0].max() + 0.1),
        ylim=(data[..., 1].min() - 0.1, data[..., 1].max() + 0.1),
        **kwds,
    )
    g.plot_joint(sns.kdeplot)
    # g.plot_marginals(sns.rugplot, height=-.15)
    return g.figure
