# LICENSE ######################################################################
# ...
################################################################################
# IMPORTS ######################################################################
"""
The 'models' module defines classes for density regression models.

It includes classes for DensityRegressionModel and HybridDensityRegressionModel,
which are built on distribution from the 'distributions' module.

"""

import tensorflow as tf

from mctm import distributions


# MODEL DEFINITIONS #########################################################
class DensityRegressionModel(tf.keras.Model):
    """
    A class representing a Density Regression Model.

    Attributes:
        distribution_lambda (callable): A callable representing the
                                        distribution.
        distribution_parameters_lambda (callable): A callable for distribution
                                                   parameters.
        trainable_parameters (list): List of trainable parameters.

    Methods:
        call(*args, **kwds): Compute the distribution for
                             given input arguments.

    """

    def __init__(self, dims, distribution, **kwds):
        """
        Initialize a DensityRegressionModel.

        Parameters:
            dims (int): The dimension of the model.
            distribution (str): The type of distribution to use.
            **kwds: Additional keyword arguments.

        """
        super().__init__()
        (
            self.distribution_lambda,
            self.distribution_parameters_lambda,
            self.trainable_parameters,
        ) = getattr(distributions, "get_" + distribution)(dims=dims, **kwds)

    def call(self, *args, **kwds):
        """
        Compute the distribution for the given input arguments.

        Parameters:
            *args: Variable-length argument list.
            **kwds: Additional keyword arguments.

        Returns:
            Distribution: The computed distribution.

        """
        return self.distribution_lambda(
            self.distribution_parameters_lambda(*args, **kwds)
        )


class HybridDenistyRegressionModel(DensityRegressionModel):
    """
    A class representing a Hybrid Density Regression Model.

    Attributes:
        base_model (DensityRegressionModel): The base density regression model.

    Methods:
        get_base_distribution(*args, **kwds): Get the base distribution.

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
    ):
        """
        Initialize a HybridDenistyRegressionModel.

        Parameters:
            dims (int): The dimension of the model.
            distribution (str): The type of distribution to use.
            distribution_kwds (dict): Keyword arguments for the distribution.
            parameter_kwds (dict): Keyword arguments for the parameters.
            base_distribution (str): The type of base distribution.
            base_distribution_kwds (dict): Keyword arguments for the
                                           base distribution.
            base_parameter_kwds (dict): Keyword arguments for the base
                                        parameters.
            base_checkpoint_path (str): The path to the base model's
                                        checkpoint.
            freeze_base_model (bool): Whether to freeze the base model.

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
            self.base_model.load_weights(base_checkpoint_path)
        if freeze_base_model:
            self.base_model.trainable = False

    def get_base_distribution(self, *args, **kwds):
        """
        Get the base distribution.

        Parameters:
            *args: Variable-length argument list. (ignored)
            **kwds: Additional keyword arguments. (ignored

        Returns:
            Distribution: The base distribution.

        """
        return self.base_model(None)
