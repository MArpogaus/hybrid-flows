# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : __init__.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-03-10 15:39:04 (Marcel Arpogaus)
# changed : 2024-03-07 10:34:46 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# ...
###############################################################################
import tensorflow as tf
from tensorflow_probability.python.internal import (
    dtype_util,
    tensor_util,
)


def get_thetas_constrain_fn(
    low=-4,
    high=4,
    bounds="smooth",
    allow_flexible_bounds=False,
):
    # @tf.function
    def constrain_fn(diff):
        dtype = dtype_util.common_dtype([diff], dtype_hint=tf.float32)
        diff = tensor_util.convert_nonref_to_tensor(diff, name="diff", dtype=dtype)
        eps = dtype_util.eps(dtype)

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
                high_theta - low_theta - (tf.cast(diff.shape[-1], dtype) * eps)
            )
        else:
            diff_positive = tf.math.softplus(diff)

        diff_positive += eps

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
                (
                    low_theta,
                    diff_positive + eps,
                ),
                axis=-1,
            )

        thetas_constrained = tf.cumsum(c, axis=-1, name="theta")
        return thetas_constrained

    return constrain_fn


def get_spline_param_constrain_fn(nbins, interval_width, min_bin_width, min_slope):
    def _bin_positions(x):
        return (
            tf.math.softmax(x, axis=-1) * (interval_width - nbins * min_bin_width)
            + min_bin_width
        )

    def _slopes(x):
        return tf.math.softplus(x) + min_slope

    def constrain_fn(unconstrained_parameters):
        # shape: [dims, 3*nbins - 1 ]
        bin_widths, bin_heights, knot_slopes = tf.split(
            unconstrained_parameters, [nbins, nbins, nbins - 1], axis=-1
        )

        return dict(
            bin_widths=_bin_positions(bin_widths),
            bin_heights=_bin_positions(bin_heights),
            knot_slopes=_slopes(knot_slopes),
        )

    return constrain_fn
