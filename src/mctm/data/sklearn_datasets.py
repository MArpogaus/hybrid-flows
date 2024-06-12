"""Provides access to sklearn datasets."""

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


def get_dataset(dataset_name, n_samples, scale, test_mode=False, **kwds):
    """Load data."""
    n_samples = 1000 if test_mode else n_samples
    X, Y = getattr(datasets, f"make_{dataset_name}")(n_samples=n_samples, **kwds)
    if scale:
        X = MinMaxScaler(feature_range=tuple(scale)).fit_transform(X)
    return (X, Y), X.shape[-1]
