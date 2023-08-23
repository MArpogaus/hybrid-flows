# LICENSE ######################################################################
# ...
################################################################################
# IMPORTS ######################################################################
import tensorflow as tf

from mctm import distributions


# MODEL DEFINITIONS #########################################################
class UnconditionalModel(tf.keras.Model):
    def __init__(self, dims, distribution, **kwds):
        super().__init__()
        self.distribution_lambda, self.distribution_parameters = getattr(
            distributions, "get_" + distribution
        )(dims=dims, **kwds)

    def call(self, *_):
        return self.distribution_lambda(self.distribution_parameters)
