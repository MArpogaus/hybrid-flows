# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : models.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-11-03 15:57:47 (Marcel Arpogaus)
# changed : 2024-11-11 18:30:56 (Marcel Arpogaus)

# %% License ###################################################################

# %% Description ###############################################################
"""Definition of TensorFlow Keras models for density regression."""

# %% imports ###################################################################
from typing import Any, Callable, Dict, List, Tuple

import tensorflow.keras as K
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from mctm import distributions


# %% classes ###################################################################
class DensityRegressionModel(K.Model):
    """Density Regression Model.

    Attributes
    ----------
    distribuition_fn : Callable
        Callable representing the distribution.
    parameter_fn : Callable
        Callable for distribution parameters.

    Methods
    -------
    call(inputs, **kwargs)
        Compute the distribution for given input arguments.

    """

    def __init__(self, distribution: str, **kwargs: Any) -> None:
        """Initialize DensityRegressionModel.

        Parameters
        ----------
        distribution : str
            Type of distribution to use.
        **kwargs : dict
            Additional keyword arguments for the distribution function.

        """
        super().__init__()
        (
            self.distribuition_fn,
            self.parameters_fn,
            trainable_variables,
            non_trainable_variables,
        ) = getattr(distributions, f"get_{distribution}")(**kwargs)
        self._trainable_weights.extend(trainable_variables)
        self._non_trainable_weights.extend(non_trainable_variables)

    def call(self, conditional_input: Any, **kwargs: Any) -> tfd.Distribution:
        """Compute distribution for given inputs.

        Parameters
        ----------
        conditional_input : Any
            Input data for distribution parameters.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tfd.Distribution
            Computed distribution.

        """
        parameters = self.parameters_fn(conditional_input, **kwargs)
        return self.distribuition_fn(parameters)


class HybridDensityRegressionModel(K.Model):
    """Hybrid Density Regression Model.

    Attributes
    ----------
    distribuition_fn : Callable
        Callable representing the transformed distribution.
    marginal_transformation_parametrization_fn : Callable
        Function for marginal transformation parametrization.
    marginal_transformation_parameters_fn : Callable
        Function for marginal transformation parameters.
    joint_transformation_parametrization_fn : Callable
        Function for joint transformation parametrization.
    joint_transformation_parameters_fn : Callable
        Function for joint transformation parameters.

    Methods
    -------
    get_flow_parametrization_fn()
        Return a function for flow transformations.
    parameter_fn(inputs, **kwargs)
        Compute parameters for marginal and joint transformations.
    call(inputs, **kwargs)
        Compute distribution for given inputs using the transformed distribution.

    """

    def __init__(
        self,
        marginal_bijectors: List[Dict[str, Any]],
        joint_bijectors: List[Dict[str, Any]],
        marginals_trainable: bool = True,
        joint_trainable: bool = True,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize HybridDensityRegressionModel.

        Parameters
        ----------
        marginal_bijectors: List[Dict[str, Any]]
            Bijector definitions for element-wise marginal transformations.
        joint_bijectors: List[Dict[str, Any]]
            Bijector definitions for multi-dimensional transformations used to
            de-correlate the data.
        marginals_trainable: bool, optional
            Train marginal flow variables if True. Default is `True`.
        joint_trainable: bool, optional
            Train joint flow variables if True. Default is `True`.
        **kwargs : Dict[str, Any]
            Additional keyword arguments for
           `distributions._get_transformed_distribution_fn`.

        """
        super().__init__()
        (
            self.marginal_transformation_parameters_fn,
            self.marginal_transformation_parametrization_fn,
            self.marginal_transformation_trainable_variables,
            self.marginal_transformation_non_trainable_variables,
        ) = distributions._get_flow_parametrization_fn(
            bijectors=marginal_bijectors,
            reverse_flow=False,
            inverse_flow=False,
            variables_name="marginal",
        )
        (
            self.joint_transformation_parameters_fn,
            self.joint_transformation_parametrization_fn,
            self.joint_transformation_trainable_variables,
            self.joint_transformation_non_trainable_variables,
        ) = distributions._get_flow_parametrization_fn(
            bijectors=joint_bijectors,
            reverse_flow=False,
            inverse_flow=False,
            variables_name="joint",
        )

        self.distribuition_fn = distributions._get_transformed_distribution_fn(
            self.get_flow_parametrization_fn(), **kwargs
        )

        self.marginals_trainable = marginals_trainable
        self.joint_trainable = joint_trainable

    def parameters_fn(self, conditional_input: Any, **kwargs: Any) -> Tuple:
        """Compute parameters for transformations.

        Parameters
        ----------
        conditional_input : Any
            Input data for transformation parameters.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Tuple
            Parameters for marginal and joint transformations.

        """
        input_shape, all_marginal_parameters = (
            self.marginal_transformation_parameters_fn(conditional_input, **kwargs)
        )
        _, all_joint_parameters = self.joint_transformation_parameters_fn(
            conditional_input, **kwargs
        )
        return input_shape, (all_marginal_parameters, all_joint_parameters)

    def get_flow_parametrization_fn(self) -> Callable:
        """Return function for flow transformations.

        Returns
        -------
        Callable
            Function that returns a Chain of bijectors.

        """

        def flow_parametrization_fn(all_parameters: list) -> tfb.Chain:
            marginal_parameters, joint_parameters = all_parameters
            bijectors_list = [
                self.marginal_transformation_parametrization_fn(marginal_parameters),
                self.joint_transformation_parametrization_fn(joint_parameters),
            ]

            return tfb.Chain(bijectors_list)

        return flow_parametrization_fn

    def call(self, inputs: Any, **kwargs: Any) -> tfd.Distribution:
        """Compute transformed distribution for given inputs.

        Parameters
        ----------
        inputs : Any
            Input data for distribution parameters.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tfd.Distribution
            Computed transformed distribution.

        """
        parameters = self.parameters_fn(inputs, **kwargs)
        return self.distribuition_fn(parameters)

    @property
    def marginals_trainable(self) -> bool:
        """Train marginal flow variables if True. Default is `True`."""
        return self._marginals_trainable

    @marginals_trainable.setter
    def marginals_trainable(self, value: bool) -> None:
        """Set attribute `marginals_trainable` to `value`."""
        self._marginals_trainable = value

    @property
    def joint_trainable(self) -> bool:
        """Train joint flow variables if True. Default is `True`."""
        return self._joint_trainable

    @joint_trainable.setter
    def joint_trainable(self, value: bool) -> None:
        """Set attribute `joint_trainable` to `value`."""
        self._joint_trainable = value

    @property
    def trainable_weights(self) -> List[Any]:
        """Return trainable weights to be tuned by the optimizer."""
        self._assert_weights_created()
        if not self._trainable:
            return []
        trainable_variables = []
        if self.marginals_trainable:
            trainable_variables += self.marginal_transformation_trainable_variables
        if self.joint_trainable:
            trainable_variables += self.joint_transformation_trainable_variables
        return self._dedup_weights(trainable_variables)

    @property
    def non_trainable_weights(self) -> List[Any]:
        """Return all non-trainable weights."""
        self._assert_weights_created()
        non_trainable_variables = (
            self.marginal_transformation_non_trainable_variables
            + self.joint_transformation_non_trainable_variables
        )

        if not self._trainable:
            non_trainable_variables += (
                self.marginal_transformation_trainable_variables
                + self.joint_transformation_trainable_variables
            )
        else:
            if not self.marginals_trainable:
                non_trainable_variables += (
                    self.marginal_transformation_trainable_variables
                )
            if not self.joint_trainable:
                non_trainable_variables += self.joint_transformation_trainable_variables

        return self._dedup_weights(non_trainable_variables)
