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


def plot_2d_data(X, Y):
    label = Y.astype(bool)
    X1, X2 = X[..., 0], X[..., 1]
    fig = plt.figure()
    plt.scatter(X1[label], X2[label], s=10, color="blue")
    plt.scatter(X1[~label], X2[~label], s=10, color="red")
    plt.legend(["label: 1", "label: 0"])
    return fig


def plot_samples(dist, data, seed=1):
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
    )
    g.plot_joint(sns.kdeplot)
    # g.plot_marginals(sns.rugplot, height=-.15)
    return g.figure
