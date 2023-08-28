# LICENSE ######################################################################
# ...
################################################################################
# IMPORTS ######################################################################
import tensorflow as tf

from mctm import distributions


# MODEL DEFINITIONS #########################################################
class DensityRegressionModel(tf.keras.Model):
    def __init__(self, dims, distribution, **kwds):
        super().__init__()
        (
            self.distribution_lambda,
            self.distribution_parameters_lambda,
            self.trainable_parameters,
        ) = getattr(distributions, "get_" + distribution)(dims=dims, **kwds)

    def call(self, *args, **kwds):
        return self.distribution_lambda(
            self.distribution_parameters_lambda(*args, **kwds)
        )
