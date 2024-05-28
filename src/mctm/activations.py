# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : __init__.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-03-10 15:39:04 (Marcel Arpogaus)
# changed : 2024-03-26 18:28:45 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# ...
###############################################################################
"""Defines functions to constrain NN outputs."""

from typing import Callable

import tensorflow as tf
from tensorflow_probability.python.internal import (
    dtype_util,
    tensor_util,
)


def get_thetas_constrain_fn(
    low: float = -4.0,
    high: float = 4.0,
    bounds: str = "smooth",
    allow_flexible_bounds: bool = False,
    min_slope: float = 1e-5,
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Create a function to constrain the Bernstein coeficents within specified bounds.

    Parameters
    ----------
    low
        The lower bound for the first theta, by default -4.0.
    high
        The upper bound for the last theta, by default 4.0.
    bounds
        The type of bounds to apply, either "smooth", "identity", or others,
        by default "smooth".
    allow_flexible_bounds
        If True, allows dynamic adjustment of bounds based on input, by default False.
    min_slope
        The minimum slope to ensure numeric stability, by default 1e-5.

    Returns
    -------
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
            c = tf.concat(
                (low_theta, diff_positive),
                axis=-1,
            )

        thetas_constrained = tf.cumsum(c, axis=-1, name="theta")
        return thetas_constrained

    return constrain_fn


def get_spline_param_constrain_fn(
    nbins: int, interval_width: float, min_bin_width: float, min_slope: float
) -> Callable[[tf.Tensor], dict]:
    """Create a function to constrain the parameters of a spline bijector.

    Parameters
    ----------
    nbins
        Number of bins in the spline.
    interval_width
        Total width of the interval to be covered by the bins.
    min_bin_width
        The minimum width that any bin can have.
    min_slope
        The minimum slope to ensure numeric stability.

    Returns
    -------
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

    def constrain_fn(unconstrained_parameters: tf.Tensor) -> dict:
        bin_widths, bin_heights, knot_slopes = tf.split(
            unconstrained_parameters, [nbins, nbins, nbins - 1], axis=-1
        )

        return {
            "bin_widths": _bin_positions(bin_widths),
            "bin_heights": _bin_positions(bin_heights),
            "knot_slopes": _slopes(knot_slopes),
        }

    return constrain_fn
