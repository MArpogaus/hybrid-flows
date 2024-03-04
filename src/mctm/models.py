# LICENSE ######################################################################
# ...
################################################################################
# IMPORTS ######################################################################
"""Classes for density regression models.

The 'models' module defines classes for density regression models.

It includes classes for DensityRegressionModel and HybridDensityRegressionModel,
which are built on distribution from the 'distributions' module.

"""
import os

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from mctm import distributions


# MODEL DEFINITIONS #########################################################
class DensityRegressionModel(tf.keras.Model):
    """A class representing a Density Regression Model.

    :ivar callable distribution_lambda: A callable representing the
                                       distribution.
    :ivar callable distribution_parameters_lambda: A callable for distribution
                                                  parameters.
    :ivar list trainable_parameters: List of trainable parameters.

    :method call: Compute the distribution for given input arguments.
    """

    def __init__(self, dims, distribution, **kwds):
        """Initialize a DensityRegressionModel.

        :param int dims: The dimension of the model.
        :param str distribution: The type of distribution to use.
        :param **kwds: Additional keyword arguments.
        """
        super().__init__()
        (
            self.distribution_lambda,
            self.distribution_parameters_lambda,
            self.trainable_parameters,
        ) = getattr(distributions, "get_" + distribution)(dims=dims, **kwds)

    def call(self, inputs, **kwds):
        """Compute the distribution for the given input arguments.

        :param *args: Variable-length argument list.
        :param **kwds: Additional keyword arguments.
        :return: The computed distribution.
        :rtype: Distribution
        """
        return self.distribution_lambda(
            self.distribution_parameters_lambda(inputs, **kwds)
        )


class HybridDenistyRegressionModel(DensityRegressionModel):
    """A class representing a Hybrid Density Regression Model.

    :ivar DensityRegressionModel base_model: The base density regression model.

    :method get_base_distribution: Get the base distribution.
    """

    def __init__(
        self,
        dims,
        distribution,
        distribution_kwds,
        parameter_kwds,
        base_distribution,
        base_distribution_kwds,
        base_parameter_kwds,
        base_checkpoint_path,
        freeze_base_model,
        base_checkpoint_path_prefix="./",
    ):
        """Initialize a HybridDensityRegressionModel.

        :param int dims: The dimension of the model.
        :param str distribution: The type of distribution to use.
        :param dict distribution_kwds: Keyword arguments for the distribution.
        :param dict parameter_kwds: Keyword arguments for the parameters.
        :param str base_distribution: The type of base distribution.
        :param dict base_distribution_kwds: Keyword arguments for the
                                           base distribution.
        :param dict base_parameter_kwds: Keyword arguments for the base
                                        parameters.
        :param str base_checkpoint_path: The path to the base model's
                                        checkpoint.
        :param bool freeze_base_model: Whether to freeze the base model.
        """
        super().__init__(
            dims,
            distribution=distribution,
            distribution_kwds={
                "base_distribution_lambda": self.get_base_distribution,
                **distribution_kwds,
            },
            parameter_kwds=parameter_kwds,
        )

        self.base_model = DensityRegressionModel(
            dims=dims,
            distribution=base_distribution,
            distribution_kwds=base_distribution_kwds,
            parameter_kwds=base_parameter_kwds,
        )
        if base_checkpoint_path:
            self.base_model.load_weights(
                os.path.join(base_checkpoint_path_prefix, base_checkpoint_path)
            )
        if freeze_base_model:
            self.base_model.trainable = False

    def get_base_distribution(self, *args, **kwds):
        """Get the base distribution.

        :param *args: Variable-length argument list. (ignored)
        :param **kwds: Additional keyword arguments. (ignored)
        :return: The base distribution.
        :rtype: Distribution
        """
        return tfd.Independent(self.base_model(None), 1)
