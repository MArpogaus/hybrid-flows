# %% Description ###############################################################
"""Provides access to sklearn datasets."""

# %% imports ###################################################################
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler


def gen_data(
    dataset_name,
    n_samples,
    scale,
    random_state,
    test_data=False,
    test_mode=False,
    **kwds,
):
    """Load data."""
    if test_data:
        random_state += 1
    n_samples = 2**12 if test_mode else n_samples
    X, Y = getattr(datasets, f"make_{dataset_name}")(
        n_samples=n_samples, random_state=random_state, **kwds
    )
    if scale:
        X = MinMaxScaler(feature_range=tuple(scale)).fit_transform(X)
    return (X, Y), X.shape[-1]
