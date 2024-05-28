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

import tensorflow.keras as K
from tensorflow_probability import distributions as tfd

from mctm import distributions, parameters


# MODEL DEFINITIONS #########################################################
class DensityRegressionModel(K.Model):
    """A class representing a Density Regression Model.

    :ivar callable distribution_lambda: A callable representing the
                                       distribution.
    :ivar callable distribution_parameters_lambda: A callable for distribution
                                                  parameters.
    :ivar list trainable_parameters: List of trainable parameters.

    :method call: Compute the distribution for given input arguments.
    """

    def __init__(self, dims, distribution, **kwargs):
        """Initialize a DensityRegressionModel.

        :param int dims: The dimension of the model.
        :param str distribution: The type of distribution to use.
        :param **kwargs: Additional keyword arguments.
        """
        get_parameter_fn = kwargs.get("get_parameter_fn", False)
        if isinstance(get_parameter_fn, str):
            kwargs["get_parameter_fn"] = getattr(
                parameters, f"get_{get_parameter_fn}_fn"
            )
        super().__init__()
        (
            self.distribuition_fn,
            self.parameter_fn,
            self.trainable_parameters,
        ) = getattr(distributions, "get_" + distribution)(dims=dims, **kwargs)

    def call(self, inputs, **kwargs):
        """Compute the distribution for the given input arguments.

        :param *args: Variable-length argument list.
        :param **kwargs: Additional keyword arguments.
        :return: The computed distribution.
        :rtype: Distribution
        """
        parameters = self.parameter_fn(inputs, **kwargs)
        return self.distribuition_fn(parameters)


class HybridDenistyRegressionModel(DensityRegressionModel):
    """A class representing a Hybrid Density Regression Model.

    :ivar DensityRegressionModel base_model: The base density regression model.

    :method get_base_distribution: Get the base distribution.
    """

    def __init__(
        self,
        dims,
        distribution,
        distribution_kwargs,
        parameter_kwargs,
        base_distribution,
        freeze_base_model,
        base_checkpoint_path=None,
        base_distribution_kwargs={},
        base_parameter_kwargs={},
        base_checkpoint_path_prefix="./",
        **kwargs,
    ):
        """Initialize a HybridDensityRegressionModel.

        :param int dims: The dimension of the model.
        :param str distribution: The type of distribution to use.
        :param dict distribution_kwargs: Keyword arguments for the distribution.
        :param dict parameter_kwargs: Keyword arguments for the parameters.
        :param str base_distribution: The type of base distribution.
        :param dict base_distribution_kwargs: Keyword arguments for the
                                           base distribution.
        :param dict base_parameter_kwargs: Keyword arguments for the base
                                        parameters.
        :param str base_checkpoint_path: The path to the base model's
                                        checkpoint.
        :param bool freeze_base_model: Whether to freeze the base model.
        """
        super().__init__(
            dims,
            distribution=distribution,
            distribution_kwargs={
                "get_base_distribution": self.get_base_distribution,
                **distribution_kwargs,
            },
            parameter_kwargs=parameter_kwargs,
            **kwargs,
        )

        if isinstance(base_distribution, DensityRegressionModel):
            self.base_model = base_distribution
        else:
            self.base_model = DensityRegressionModel(
                dims=dims,
                distribution=base_distribution,
                distribution_kwargs=base_distribution_kwargs,
                parameter_kwargs=base_parameter_kwargs,
            )
        if base_checkpoint_path:
            self.base_model.load_weights(
                os.path.join(base_checkpoint_path_prefix, base_checkpoint_path)
            )
        if freeze_base_model:
            self.base_model.trainable = False

    def get_base_distribution(self, *args, **kwargs):
        """Get the base distribution.

        :param *args: Variable-length argument list. (ignored)
        :param **kwargs: Additional keyword arguments. (ignored)
        :return: The base distribution.
        :rtype: Distribution
        """
        return tfd.Independent(self.base_model(None), 1)
