# %% Description ###############################################################
"""Defines functions to constrain NN outputs."""

# %% imports ###################################################################
from typing import Callable, Dict

import tensorflow as tf
from tensorflow_probability.python.internal import (
    dtype_util,
    tensor_util,
)


# %% private functions #########################################################
def _create_lambda_matrix(x: tf.Tensor) -> tf.Tensor:
    """Create a lower triangular lambda matrix from `x` with a diagonal of ones.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor containing elements to fill the lower triangular matrix.

    Returns
    -------
    tf.Tensor
        Resulting lambda matrix.

    """
    num_elements = x.shape[-1]
    n = tf.cast(
        tf.math.ceil(tf.math.sqrt(2.0 * tf.cast(num_elements, tf.float32))), tf.int32
    )
    diag_ones = tf.linalg.diag(tf.ones((n,), dtype=x.dtype))
    tri_lower_indices = tf.where(
        tf.linalg.band_part(tf.ones((n, n)), -1, 0) - tf.eye(n) > 0
    )
    lower_tri_elements = tf.scatter_nd(tri_lower_indices, x, shape=(n, n))
    mat = diag_ones + lower_tri_elements
    return mat


# %% functions #################################################################
def get_thetas_constrain_fn(
    low: float = -4.0,
    high: float = 4.0,
    bounds: str = "smooth",
    allow_flexible_bounds: bool = False,
    min_slope: float = 1e-5,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return function to constrain the Bernstein coefficients within specified bounds.

    Parameters
    ----------
    low : float, optional
        The lower bound for the first theta, by default -4.0.
    high : float, optional
        The upper bound for the last theta, by default 4.0.
    bounds : str, optional
        The type of bounds to apply, either "smooth", "identity", or others,
        by default "smooth".
    allow_flexible_bounds : bool, optional
        If True, allows dynamic adjustment of bounds based on input, by default False.
    min_slope : float, optional
        The minimum slope to ensure numeric stability, by default 1e-5.

    Returns
    -------
    Callable[[tf.Tensor], tf.Tensor]
        A function that takes a tensor and returns it constrained according to the
        specified options.

    """

    def constrain_fn(diff: tf.Tensor) -> tf.Tensor:
        dtype = dtype_util.common_dtype([diff], dtype_hint=tf.float32)
        diff = tensor_util.convert_nonref_to_tensor(diff, name="diff", dtype=dtype)
        if low is not None:
            low_theta = tensor_util.convert_nonref_to_tensor(
                low, name="low", dtype=dtype
            ) * tf.ones_like(diff[..., :1])
            if allow_flexible_bounds:
                low_theta -= tf.math.softplus(diff[..., :1])
                diff = diff[..., 1:]
        else:
            shift = tf.math.log(2.0) * tf.cast(diff.shape[-1], dtype) / 2
            low_theta = diff[..., :1] - shift
            diff = diff[..., 1:]

        if high is not None:
            high_theta = tensor_util.convert_nonref_to_tensor(
                high, name="high", dtype=dtype
            )
            if allow_flexible_bounds:
                high_theta += tf.math.softplus(diff[..., :1])
                diff = diff[..., 1:]

        if high is not None:
            diff_positive = tf.math.softmax(diff, axis=-1)
            diff_positive *= (
                high_theta - low_theta - (tf.cast(diff.shape[-1], dtype) * min_slope)
            )
        else:
            diff_positive = tf.math.softplus(diff)

        diff_positive += min_slope

        if bounds == "smooth":
            c = tf.concat(
                (
                    low_theta - diff_positive[..., :1],
                    diff_positive[..., :1],
                    diff_positive,
                    diff_positive[..., -1:],
                ),
                axis=-1,
            )
        elif bounds == "identity":
            order = diff_positive.shape[-1] + 4
            c = tf.concat(
                (
                    low_theta - 2 / order,
                    tf.ones_like(diff_positive[..., :2]) / order,
                    diff_positive,
                    tf.ones_like(diff_positive[..., :2]) / order,
                ),
                axis=-1,
            )
        else:
            c = tf.concat((low_theta, diff_positive), axis=-1)

        thetas_constrained = tf.cumsum(c, axis=-1, name="theta")
        return thetas_constrained

    return constrain_fn


def get_spline_param_constrain_fn(
    nbins: int, interval_width: float, min_bin_width: float, min_slope: float
) -> Callable[[tf.Tensor], Dict[str, tf.Tensor]]:
    """Create a function to constrain the parameters of a spline bijector.

    Parameters
    ----------
    nbins : int
        Number of bins in the spline.
    interval_width : float
        Total width of the interval to be covered by the bins.
    min_bin_width : float
        The minimum width that any bin can have.
    min_slope : float
        The minimum slope to ensure numeric stability.

    Returns
    -------
    Callable[[tf.Tensor], Dict[str, tf.Tensor]]
        A function that takes unconstrained parameters and returns them constrained
        within specified bounds, organized into a dictionary.

    """

    def _bin_positions(x: tf.Tensor) -> tf.Tensor:
        return (
            tf.math.softmax(x, axis=-1) * (interval_width - nbins * min_bin_width)
            + min_bin_width
        )

    def _slopes(x: tf.Tensor) -> tf.Tensor:
        return tf.math.softplus(x) + min_slope

    def constrain_fn(unconstrained_parameters: tf.Tensor) -> Dict[str, tf.Tensor]:
        bin_widths, bin_heights, knot_slopes = tf.split(
            unconstrained_parameters, [nbins, nbins, nbins - 1], axis=-1
        )

        return {
            "bin_widths": _bin_positions(bin_widths),
            "bin_heights": _bin_positions(bin_heights),
            "knot_slopes": _slopes(knot_slopes),
        }

    return constrain_fn


def lambda_parameters_constraint_fn(
    x: tf.Tensor,
) -> tf.linalg.LinearOperatorLowerTriangular:
    """Create lower triangular matrix with ones on the diagonal.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor containing the elements for the lower triangular matrix.

    Returns
    -------
    tf.linalg.LinearOperatorLowerTriangular
        Linear operator representing the constrained lower triangular matrix.

    """
    return tf.linalg.LinearOperatorLowerTriangular(
        tf.vectorized_map(_create_lambda_matrix, x)
    )
