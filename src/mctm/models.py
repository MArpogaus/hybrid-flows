# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : models.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-11-03 15:57:47 (Marcel Arpogaus)
# changed : 2024-12-02 17:48:09 (Marcel Arpogaus)

# %% License ###################################################################
# %% Description ###############################################################
"""Definition of TensorFlow Keras models for density regression."""

# %% imports ###################################################################
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

import tensorflow as tf
import tensorflow.keras as K
from keras.src.saving import serialization_lib
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from mctm import distributions


# %% classes ###################################################################
class DensityRegressionBaseModel(ABC, K.Model):
    """Abstract base class for density regression models."""

    @abstractmethod
    def call(self, conditional_input: Any, **kwargs: Any) -> tfd.Distribution:
        """Compute distribution for given inputs."""

    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Abstract property holding the model configuration as a serializable dict."""

    def get_config(self) -> Dict[str, Any]:
        """Return dictionary containing all kwargs for serialization."""
        return serialization_lib.serialize_dict(self.config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Initialize model from serialized config dict."""
        model = cls(**serialization_lib.deserialize_keras_object(config))
        return model


class DensityRegressionModel(DensityRegressionBaseModel):
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
        self._config = deepcopy(kwargs)
        self._config.update(distribution=distribution)
        (
            self.distribuition_fn,
            self.parameters_fn,
            trainable_variables,
            non_trainable_variables,
        ) = getattr(distributions, f"get_{distribution}")(**kwargs)
        self._trainable_weights.extend(trainable_variables)
        self._non_trainable_weights.extend(non_trainable_variables)

    def call(self, conditional_input: tf.Tensor, **kwargs: Any) -> tfd.Distribution:
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

    @property
    def config(self) -> Dict[str, Any]:
        """Property holding the model configuration as a serializable dict."""
        return self._config


class HybridDensityRegressionModel(DensityRegressionBaseModel):
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
        dims: int,
        marginal_bijectors: List[Dict[str, Any]],
        joint_bijectors: List[Dict[str, Any]],
        marginals_trainable: bool = True,
        joint_trainable: bool = True,
        predict_marginals: bool = False,
        joint_flow_type: str = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize HybridDensityRegressionModel.

        Parameters
        ----------
        dims : int
            The dimension of the distribution.
        marginal_bijectors : List[Dict[str, Any]]
            Bijector definitions for element-wise marginal transformations.
        joint_bijectors : List[Dict[str, Any]]
            Bijector definitions for multi-dimensional transformations used to
            de-correlate the data.
        marginals_trainable : bool, optional
            Train marginal flow variables if True. Default is `True`.
        joint_trainable : bool, optional
            Train joint flow variables if True. Default is `True`.
        predict_marginals : bool, optional
            If `True` predict marginal else joint distribution. Default is `False`.
        joint_flow_type : str, optional
            Allows to specify special types of flows in a more compact notation.
            May be either "coupling_flow", "masked_autoregressive_flow" or
            "masked_autoregressive_flow_first_dim_masked". Default is `None`.
        **kwargs : Dict[str, Any]
            Additional keyword arguments for
           `distributions._get_transformed_distribution_fn`.

        """
        super().__init__()
        self._config = deepcopy(kwargs)
        self._config.update(
            marginal_bijectors=deepcopy(marginal_bijectors),
            joint_bijectors=deepcopy(joint_bijectors),
        )
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
        if joint_flow_type is not None:
            joint_bijectors = getattr(
                distributions, f"_get_{joint_flow_type}_bijector_def"
            )(dims=dims, **joint_bijectors)
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
            self.get_flow_parametrization_fn(), dims=dims, **kwargs
        )
        self.marginal_distribuition_fn = distributions._get_transformed_distribution_fn(
            self.marginal_transformation_parametrization_fn, dims=dims, **kwargs
        )

        self.marginals_trainable = marginals_trainable
        self.joint_trainable = joint_trainable
        self.predict_marginals = predict_marginals

    def parameters_fn(self, conditional_input: tf.Tensor, **kwargs: Any) -> Tuple:
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

    def marginal_distribution(
        self, conditional_input: tf.Tensor, **kwargs: Any
    ) -> tfd.Distribution:
        """Compute transformed marginal distribution for given inputs.

        Parameters
        ----------
        conditional_input : Any
            Input data for distribution parameters.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tfd.Distribution
            Computed transformed distribution.

        """
        marginal_parameters = self.marginal_transformation_parameters_fn(
            conditional_input, **kwargs
        )
        return self.marginal_distribuition_fn(marginal_parameters)

    def joint_distribution(
        self, conditional_input: tf.Tensor, **kwargs: Any
    ) -> tfd.Distribution:
        """Compute transformed distribution for given inputs.

        Parameters
        ----------
        conditional_input : Any
            Input data for distribution parameters.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tfd.Distribution
            Computed transformed distribution.

        """
        parameters = self.parameters_fn(conditional_input, **kwargs)
        return self.distribuition_fn(parameters)

    def copula_distribution(
        self, conditional_input: tf.Tensor, **kwargs: Any
    ) -> tfd.Distribution:
        """Compute transformed distribution for given inputs.

        Parameters
        ----------
        conditional_input : Any
            Input data for distribution parameters.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tfd.Distribution
            Computed transformed distribution.

        """
        raise NotImplementedError

    def call(self, conditional_input: tf.Tensor, **kwargs: Any) -> tfd.Distribution:
        """If `predict_marginals` is `True` predict marginal else joint distribution."""
        if self.predict_marginals:
            return self.marginal_distribution(conditional_input, **kwargs)
        else:
            return self.joint_distribution(conditional_input, **kwargs)

    @property
    def predict_marginals(self) -> bool:
        """If `True` predict marginal else joint distribution. Default is `False`."""
        return self._predict_marginals

    @predict_marginals.setter
    def predict_marginals(self, value: bool) -> None:
        """Set attribute `predict_marginals` to `value`."""
        self._predict_marginals = value

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

    @property
    def config(self) -> Dict[str, Any]:
        """Property holding the model configuration as a serializable dict."""
        return dict(
            marginals_trainable=self.marginals_trainable,
            joint_trainable=self.joint_trainable,
            predict_marginals=self.predict_marginals,
            **self._config,
        )
