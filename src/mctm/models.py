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


class HybridDenistyRegressionModel(DensityRegressionModel):
    def __init__(
        self,
        dims,
        distribution,
        distribution_kwds,
        parameter_kwds,
        base_distribution,
        base_distribution_kwds,
        base_parameter_kwds,
    ):
        super().__init__(
            dims,
            distribution=distribution,
            distribution_kwds={
                "base_distribution_lambda": self.get_base_distribution,
                **distribution_kwds,
            },
            parameter_kwds=parameter_kwds,
        )

        (
            self.base_distribution_lambda,
            self.base_distribution_parameters_lambda,
            self.base_trainable_parameters,
        ) = getattr(distributions, "get_" + base_distribution)(
            dims=dims,
            distribution_kwds=base_distribution_kwds,
            parameter_kwds=base_parameter_kwds,
        )

    def get_base_distribution(self, _):
        return self.base_distribution_lambda(self.base_distribution_parameters_lambda())
