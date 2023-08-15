# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : distributions.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-06-19 17:01:16 (Marcel Arpogaus)
# changed : 2023-06-21 17:18:39 (Marcel Arpogaus)
# DESCRIPTION ##################################################################
# ...
# LICENSE ######################################################################
# ...
################################################################################
# IMPORTS ######################################################################
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bernstein_flow.activations import get_thetas_constrain_fn
from bernstein_flow.bijectors import BernsteinBijectorLinearExtrapolate
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import prefer_static


# FUNCTIONS ####################################################################
def get_bernstein_flow(dims, order, **kwds):
    pv_shape = [dims, order]
    thetas_constrain_fn = get_thetas_constrain_fn(**kwds)

    def dist(pv):
        thetas = thetas_constrain_fn(pv)

        return tfd.TransformedDistribution(
            distribution=tfd.Sample(
                tfd.Normal(loc=0.0, scale=1.0), sample_shape=[dims]
            ),
            bijector=tfb.Invert(BernsteinBijectorLinearExtrapolate(thetas=thetas)),
        )

    return dist, pv_shape


def get_multivariate_bernstein_flow(dims, order, **kwds):
    pv_shape = [order * dims + np.sum(np.arange(dims + 1))]
    thetas_constrain_fn = get_thetas_constrain_fn(**kwds)

    def dist(pv):
        bs = prefer_static.shape(pv)[:-1]
        shape = tf.concat((bs, [dims, order]), 0)

        thetas = thetas_constrain_fn(tf.reshape(pv[..., : order * dims], shape))
        scale_tril = tfp.bijectors.FillScaleTriL()(pv[..., order * dims:])  # fmt: skip

        mv_normal = tfd.MultivariateNormalTriL(loc=0, scale_tril=scale_tril)

        return tfd.TransformedDistribution(
            distribution=mv_normal,
            bijector=tfb.Invert(BernsteinBijectorLinearExtrapolate(thetas=thetas)),
        )

    return dist, pv_shape


def get_multivariate_normal(dims):
    pv_shape = [dims + np.sum(np.arange(dims + 1))]

    def dist(pv):
        loc = pv[..., :dims]
        scale_tril = tfp.bijectors.FillScaleTriL()(pv[..., dims:])
        mv_normal = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
        return mv_normal

    return dist, pv_shape


def get_masked_autoregressive_bernstein_flow(
    dims,
    M,
    hidden_units,
    activation,
    base_distribution,
    conditional=False,
    conditional_event_shape=None,
    conditional_input_layers="all_layers",
    **kwds,
):
    def get_bijector_fn(network, thetas_constrain_fn, **kwds):
        def bijector_fn(y, *arg, **kwds):
            with tf.name_scope("bnf_made_bjector"):
                pvector = network(y, **kwds)  # todo: add conditionals
                thetas = thetas_constrain_fn(pvector)

                return tfb.Invert(BernsteinBijectorLinearExtrapolate(thetas=thetas))

        return bijector_fn

    made_net = tfb.AutoregressiveNetwork(
        params=M,
        hidden_units=hidden_units,
        event_shape=(dims,),
        activation=activation,
        conditional=conditional,
        conditional_event_shape=conditional_event_shape,
        conditional_input_layers=conditional_input_layers,
    )
    made_net.build((dims,))

    thetas_constrain_fn = get_thetas_constrain_fn(**kwds)

    bijector_fn = get_bijector_fn(
        network=made_net, thetas_constrain_fn=thetas_constrain_fn
    )

    distribution = tfd.TransformedDistribution(
        distribution=base_distribution,
        bijector=tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn),
    )

    return distribution, made_net.trainable_variables
