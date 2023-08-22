# LICENSE ######################################################################
# ...
################################################################################
# IMPORTS ######################################################################
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras as K


# Model DEFINITIONS #########################################################
def get_unconditional_model(distribution, extra_variables=None):
    class Model(tf.keras.Model):
        def __init__(self, **kwds):
            super().__init__(**kwds)
            self.distribution = distribution
            self.extra_variables = extra_variables

        def call(self, *_):
            return self.distribution

    return Model


def get_simple_fully_connected_network(
    input_shape, hidden_units, activation, batch_norm, output_shape
):
    inputs = K.Input(input_shape)
    if batch_norm:
        inputs = K.layers.BatchNormalization(name="batch_norm")(inputs)
    for i, h in enumerate(hidden_units):
        x = K.layers.Dense(h, activation=activation, name=f"hidden{i}")(inputs)
    pv = K.layers.Dense(tf.reduce_prod(output_shape), activation="linear", name="pv")(x)
    pv_reshaped = K.layers.Reshape(output_shape)(pv)
    return K.Model(inputs=inputs, outputs=pv_reshaped)


def get_parameter_model(
    input_shape, hidden_layers, activation, batch_norm, output_shape, dist_lambda
):
    inputs = K.Input(input_shape)
    if batch_norm:
        inputs = K.layers.BatchNormalization(name="batch_norm")(inputs)
    for i, h in enumerate(hidden_layers):
        x = K.layers.Dense(h, activation=activation, name=f"hidden{i}")(inputs)
    pv = K.layers.Dense(output_shape, activation="linear", name="pv")(x)
    dist = tfp.layers.DistributionLambda(dist_lambda)(pv)
    param_model = K.Model(inputs=inputs, outputs=dist)
    param_model.summary()
    return param_model
