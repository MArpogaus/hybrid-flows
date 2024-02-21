# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : __init__.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-03-10 15:39:04 (Marcel Arpogaus)
# changed : 2024-02-20 19:15:42 (Marcel Arpogaus)
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
    smooth_bounds=True,
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
            low_theta = diff[..., :1]
            diff = diff[..., 1:]

        if high is not None:
            high_theta = tensor_util.convert_nonref_to_tensor(
                high, name="high", dtype=dtype
            )

            if allow_flexible_bounds:
                high_theta += tf.math.softplus(diff[..., :1])
                diff = diff[..., 1:]

        if smooth_bounds:
            diff = tf.concat(
                (
                    diff[..., :1],
                    diff,
                    diff[..., -1:],
                ),
                axis=-1,
            )

        if high is not None:
            diff_positive = tf.math.softmax(diff, axis=-1)
            diff_positive *= (
                high_theta - low_theta - (tf.cast(diff.shape[-1], dtype) * eps)
            )
        else:
            diff_positive = tf.math.softplus(diff, axis=-1)

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
