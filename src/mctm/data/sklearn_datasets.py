# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : sklearn_datasets.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-08-14 16:01:19 (Marcel Arpogaus)
# changed : 2023-08-14 16:01:19 (Marcel Arpogaus)
# DESCRIPTION ##################################################################
# ...
# LICENSE ######################################################################
# ...
################################################################################
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler


def get_dataset(name, n_samples, scale, **kwds):
    X, Y = getattr(datasets, f"make_{name}")(n_samples=n_samples, **kwds)
    if scale:
        X = MinMaxScaler().fit_transform(X)
    return X, Y
