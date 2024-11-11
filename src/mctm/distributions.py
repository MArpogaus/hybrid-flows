# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : distributions.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-10-03 12:48:17 (Marcel Arpogaus)
# changed : 2024-11-11 18:06:20 (Marcel Arpogaus)

# %% Description ###############################################################
"""Functions for probability distributions.

The 'distributions' module provides functions for defining and parametrizing
probability distributions. They get used in the 'models' module.

The module defines a list of private base functions that get used to compose
the final model in many cases.
"""

# %% imports ###################################################################
import logging
from copy import deepcopy
from functools import partial
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from bernstein_flow.bijectors import BernsteinPolynomial
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import prefer_static

from . import activations as activations_lib
from . import parameters as parameters_lib
from .utils import deepupdate, getattr_from_module
from .utils.decorators import (
    recurse_on_key,
    reduce_dict,
    skip_no_dict,
    vectorized_map_if_batched,
)

# %% globals ###################################################################
__LOGGER__ = logging.getLogger(__name__)

__BIJECTOR_KWARGS_KEY__ = "bijector_kwargs"
__BIJECTOR_NAME_KEY__ = "bijector"
__INVERT_BIJECTOR_KEY__ = "invert"
__NESTED_BIJECTOR_KEY__ = "nested_bijector"
__PARAMETERIZED_BY_PARENT_KEY__ = "parametrized_by_parent"
__PARAMETERS_CONSTRAINT_FN_KEY__ = "parameters_constraint_fn"
__PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__ = "parameters_constraint_fn_kwargs"
__PARAMETERS_FN_KEY__ = "parameters_fn"
__PARAMETERS_FN_KWARGS_KEY__ = "parameters_fn_kwargs"
__PARAMETERS_KEY__ = "parameters"
__PARAMETER_SLICE_SIZE_KEY__ = "parameters_slice_size"
__TRAINABLE_KEY__ = "trainable"
__ALL_KEYS__ = [
    __BIJECTOR_KWARGS_KEY__,
    __BIJECTOR_NAME_KEY__,
    __INVERT_BIJECTOR_KEY__,
    __NESTED_BIJECTOR_KEY__,
    __PARAMETERIZED_BY_PARENT_KEY__,
    __PARAMETERS_CONSTRAINT_FN_KEY__,
    __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__,
    __PARAMETERS_FN_KEY__,
    __PARAMETERS_FN_KWARGS_KEY__,
    __PARAMETERS_KEY__,
    __PARAMETER_SLICE_SIZE_KEY__,
    __TRAINABLE_KEY__,
]


# %% private functions #########################################################
def _get_multivariate_normal_fn(
    dims: int,
) -> Tuple[Callable[[tf.Tensor], tfd.Distribution], Tuple[int, ...]]:
    """Get function to parametrize Multivariate Normal distribution.

    Parameters
    ----------
    dims : int
        The dimension of the distribution.

    Returns
    -------
    dist : Callable[[tf.Tensor], tfd.Distribution]
        A function to parametrize the Multivariate Normal distribution.
    parameters_shape : Tuple[int, ...]
        The shape of the parameter vector.

    """
    parameters_shape = (dims + np.sum(np.arange(dims + 1)),)

    def dist(parameters: tf.Tensor) -> tfd.Distribution:
        loc = parameters[..., :dims]
        scale_tril = tfp.bijectors.FillScaleTriL()(parameters[..., dims:])
        mv_normal = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
        return mv_normal

    return dist, parameters_shape


def _get_trainable_distribution(
    dims: int,
    get_distribution_fn: Callable[..., Any],
    parameters_fn: Callable[..., Any],
    parameters_fn_kwargs: Dict[str, Any],
    **distribution_kwargs: Dict[str, Any],
) -> Tuple[
    Callable[[tf.Tensor], tfd.Distribution],
    Callable[[tf.Tensor], tf.Variable],
    List[tf.Variable],
    List[tf.Variable],
]:
    """Get functions and variables to fit a distribution.

    Parameters
    ----------
    dims : int
        The dimension of the distribution.
    get_distribution_fn
        A function to get the distribution lambda.
    parameters_fn
        A function to get the parameter lambda.
    parameters_fn_kwargs
        Keyword arguments for the parameters fn.
    distribution_kwargs
        Keyword arguments for the distribution.

    Returns
    -------
    distribution_fn : Callable[[tf.Tensor], tfd.Distribution]
        A function to parametrize the distribution
    parameter_fn : Callable[[tf.Tensor], tf.Variable]
        A function to obtain the parameters
    trainable_parameters : List[tf.Variable]
        List of trainable parameters
    non_trainable_parameters : List[tf.Variable]
        List of non-trainable parameters

    """
    distribution_fn, parameters_shape = get_distribution_fn(
        dims=dims, **distribution_kwargs
    )
    parameter_fn, trainable_variables = parameters_fn(
        parameters_shape, **parameters_fn_kwargs
    )
    return distribution_fn, parameter_fn, trainable_variables, []


def _get_base_distribution(
    dims: int = 0,
    distribution_name: str = "normal",
    **kwargs: Any,
) -> tfd.Distribution:
    """Get the default base distribution.

    Parameters
    ----------
    dims : int
        The dimension of the distribution.
    distribution_name : str
        The type of distribution (e.g., "normal", "lognormal", "uniform",
                                  "kumaraswamy").
    kwargs : Any
        Keyword arguments for the distribution.

    Returns
    -------
    dist : tfd.Distribution
        The default base distribution.

    """
    if distribution_name == "normal":
        default_kwargs = dict(loc=0.0, scale=1.0)
        default_kwargs.update(**kwargs)
        dist = tfd.Normal(**default_kwargs)
    elif distribution_name == "truncated_normal":
        default_kwargs = dict(loc=0.0, scale=1.0, low=-4, high=4)
        default_kwargs.update(**kwargs)
        dist = tfd.TruncatedNormal(**default_kwargs)
    elif distribution_name == "lognormal":
        default_kwargs = dict(loc=0.0, scale=1.0)
        default_kwargs.update(**kwargs)
        dist = tfd.LogNormal(**default_kwargs)
    elif distribution_name == "logistic":
        default_kwargs = dict(loc=0.0, scale=1.0)
        default_kwargs.update(**kwargs)
        dist = tfd.Logistic(**default_kwargs)
    else:
        dist = getattr_from_module(distribution_name)(**kwargs)

    if dims:
        dist = tfd.Sample(dist, sample_shape=[dims])

    return dist


def _get_parameters_constraint_fn(
    bijector_name: str,
    parameters_constraint_fn: Optional[Union[Callable, str]] = None,
    **parameters_constraint_fn_kwargs: Any,
) -> Optional[Callable]:
    """Get a parameters constraint function.

    Parameters
    ----------
    bijector_name : str
        The name of the bijector.
    parameters_constraint_fn : Optional[Union[Callable, str]], optional
        The parameters constraint function, can be a callable, a string
        referring to a function or None, by default None
    parameters_constraint_fn_kwargs : Any
        Keyword arguments to pass to the parameters constraint function.

    Returns
    -------
    Optional[Callable]
        The parameters constraint function, or `None` if not defined.

    """
    if parameters_constraint_fn is None:
        # infer from bijector name
        if bijector_name == BernsteinPolynomial.__name__:
            parameters_constraint_fn = activations_lib.get_thetas_constrain_fn
        elif bijector_name == tfb.RationalQuadraticSpline.__name__:
            parameters_constraint_fn = activations_lib.get_spline_param_constrain_fn
        else:
            __LOGGER__.debug(
                "No parameter constraint function defined for bijector: %s",
                bijector_name,
            )
            return None
    elif isinstance(parameters_constraint_fn, str):
        __LOGGER__.debug(
            "Importing parameter constraint function '%s'", parameters_constraint_fn
        )
        parameters_constraint_fn = getattr_from_module(parameters_constraint_fn)

    if len(parameters_constraint_fn_kwargs) > 0 or isinstance(
        parameters_constraint_fn, type
    ):
        return parameters_constraint_fn(**parameters_constraint_fn_kwargs)
    else:
        return parameters_constraint_fn


def _init_parameters_fn(
    bijectors: List[Dict[str, Any]], variables_name: str
) -> Tuple[List[Optional[Callable[..., Any]]], List[tf.Variable], List[tf.Variable]]:
    trainable_variables: List[tf.Variable] = []
    non_trainable_variables: List[tf.Variable] = []

    bj_count = 0

    def variable_renamer(next_creator, name, **kwargs):
        name = name if name is not None else ""
        name_seq = [f"bijector_{bj_count}", name]
        if variables_name is not None:
            name_seq = [variables_name] + name_seq
        name = "/".join(name_seq)
        var = next_creator(name=name, **kwargs)
        return var

    @recurse_on_key(__NESTED_BIJECTOR_KEY__)
    @reduce_dict(
        __PARAMETERS_KEY__,
        __ALL_KEYS__,
    )
    def process(
        parameters: Optional[Any] = None,
        parameters_fn: Optional[Union[Callable, str]] = None,
        parameters_fn_kwargs: Dict[str, Any] = {},
        trainable: bool = True,
        **kwargs: Any,
    ) -> Optional[Callable]:
        """Initialize a parameter function.

        This function initializes a parameter function based on the provided
        arguments. It handles cases where parameters are constant, provided as a
        callable function, or specified by a string that refers to a function
        in the `parameters_lib` module. Additionally, it manages the tracking
        of trainable and non-trainable variables.

        Parameters
        ----------
        parameters : Optional[Any], optional
            Constant parameters, by default None.
        parameters_fn : Optional[Union[Callable, str]], optional
            Callable parameter function or a string referring to a function in
            `parameters_lib`, by default None.
        parameters_fn_kwargs : Dict[str, Any], optional
            Keyword arguments to be passed to the parameter function, by default {}.
        trainable : bool, optional
            Whether the parameters are trainable, by default True.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Optional[Callable]
            The initialized parameter function, or `None` if no parameter
            initialization is performed.

        Raises
        ------
        ValueError
            If both `parameters` and `parameters_fn` are provided.

        """
        nonlocal bj_count
        bj_count += 1
        if parameters is not None and parameters_fn is not None:
            raise ValueError(
                "The arguments '%s' and '%s' are "
                "mutually exclusive. Only provide either one of them",
                __PARAMETERS_KEY__,
                __PARAMETERS_FN_KEY__,
            )

        variables: List[tf.Variable] = []

        # get parameter fn
        with tf.variable_creator_scope(variable_renamer):
            if parameters is not None:
                __LOGGER__.debug("Got constant parameters %s", str(parameters))

                if not isinstance(parameters, tf.Variable):
                    parameters = tf.Variable(parameters)

                def parameters_fn(*_, **__):
                    return parameters

                variables = [parameters]
            elif parameters_fn is not None:
                if callable(parameters_fn):
                    __LOGGER__.debug("Using provided callable as parameter function")
                    get_parameters_fn = parameters_fn
                else:
                    __LOGGER__.debug("Parameter function: %s", parameters_fn)
                    get_parameters_fn = getattr(
                        parameters_lib, f"get_{parameters_fn}_fn"
                    )

                __LOGGER__.debug("Initializing parametrization function")
                parameters_fn, variables = get_parameters_fn(
                    **parameters_fn_kwargs,
                )
            else:
                parameters_fn = None

        if variables:
            if trainable:
                trainable_variables.extend(variables)
            else:
                non_trainable_variables.extend(variables)

        return parameters_fn

    bijectors_parameters_fns = list(map(process, bijectors))

    return bijectors_parameters_fns, trainable_variables, non_trainable_variables


def _get_eval_parameter_fn(conditional_input, **kwargs):
    @recurse_on_key(__NESTED_BIJECTOR_KEY__)
    @reduce_dict(
        __PARAMETERS_KEY__,
        __ALL_KEYS__,
    )
    def eval_parameter_fn(parameters, **entry):
        if callable(parameters):
            val = parameters(conditional_input, **kwargs)
            if callable(val):
                val = vectorized_map_if_batched(val)
        else:
            val = parameters
        return val

    return eval_parameter_fn


def _get_bijector_class(bijector: Union[tfb.Bijector, type, str]) -> type:
    """Get a bijector class.

    Parameters
    ----------
    bijector : Union[tfb.Bijector, type, str]
        The bijector class, instance or name as string.

    Returns
    -------
    type
        The bijector class.

    Raises
    ------
    ValueError
        If the bijector class can not be inferred from the input.

    """
    if isinstance(bijector, type):
        bijector_cls = bijector
    elif bijector == BernsteinPolynomial.__name__:
        bijector_cls = BernsteinPolynomial
    elif "." not in bijector:
        bijector_cls = getattr_from_module("tfb." + bijector)
    else:
        bijector_cls = getattr_from_module(bijector)

    return bijector_cls


@skip_no_dict
def _init_bijector_from_dict(
    bijector_definition: Dict[str, Any],
) -> tfp.bijectors.Bijector:
    """Get a parameterized bijector instance.

    Parameters
    ----------
    bijector_definition : Dict[str, Any]
        Dictionary containing the bijector definition.

    Bijector Definition
    -------------------
    bijector : Union[tfb.Bijector, type, str]
    Bijector class, instance or name as string.
    parameters : Any
    Parameters to pass to the bijector.
    parameters_constraint_fn : Optional[Union[Callable, str]], optional
    Function to constrain the parameters, can be a callable, a string
    referring to a function or None, by default None
    invert : bool, optional
        If `True` the parameterized bijector gets inverted, by default False.
        bijector_kwargs : Dict[str, Any], optional
        Additional keyword arguments to pass to the bijector, by default {}.
        nested_bijector : Optional[List[Dict[str, Any]]], optional
        A list of dictionaries, each representing a nested bijector,
        by default None.

    Returns
    -------
    tfp.bijectors.Bijector
    The parameterized bijector.

    Raises
    ------
    ValueError
        If parameter slicing is required but not used in all nested bijectors.
        AssertionError
        If not all parameters from parent bijector are used.

    """
    bijector = bijector_definition[__BIJECTOR_NAME_KEY__]
    parameters = bijector_definition[__PARAMETERS_KEY__]
    parameters_constraint_fn = bijector_definition.get(
        __PARAMETERS_CONSTRAINT_FN_KEY__, None
    )
    parameters_constraint_fn_kwargs = bijector_definition.get(
        __PARAMETERS_CONSTRAINT_FN_KWARGS_KEY__, {}
    )
    invert = bijector_definition.get(__INVERT_BIJECTOR_KEY__, False)
    bijector_kwargs = bijector_definition.get(__BIJECTOR_KWARGS_KEY__, {})
    nested_bijector = bijector_definition.get(__NESTED_BIJECTOR_KEY__, None)
    bijector_cls = _get_bijector_class(bijector)
    __LOGGER__.debug("Initializing bijector: %s", bijector)
    parameters_constraint_fn = _get_parameters_constraint_fn(
        bijector_name=bijector,
        parameters_constraint_fn=parameters_constraint_fn,
        **parameters_constraint_fn_kwargs,
    )

    if callable(parameters):
        parameters_fn = parameters
    else:
        parameters_fn = None

    if parameters_constraint_fn is not None:
        __LOGGER__.debug("Parameter constrain function provided")
        if parameters_fn is not None:
            __LOGGER__.debug("Redefining constraint parameter_fn")

            def parameters_fn(*args, **kwargs):
                return parameters_constraint_fn(parameters(*args, **kwargs))
        else:
            parameters = parameters_constraint_fn(parameters)

    if nested_bijector is not None:
        __LOGGER__.debug("Detected nested bijector: %s", nested_bijector)
        if parameters_fn is not None:
            __LOGGER__.debug(
                "Parameters are callable: Passing bijector_fn to %s",
                bijector_cls.__name__,
            )
            bijector_fn = _get_bijector_fn(parameters_fn, nested_bijector)

            bijector = bijector_cls(bijector_fn=bijector_fn, **bijector_kwargs)
        else:
            bijector = bijector_cls(
                _init_nested_bijector_from_dict(
                    nested_bijector_definition=nested_bijector,
                    parent_parameters=parameters,
                ),
                **bijector_kwargs,
            )
    elif parameters is None:
        bijector = bijector_cls(**bijector_kwargs)
    elif isinstance(parameters, list):
        bijector = bijector_cls(*parameters, **bijector_kwargs)
    elif isinstance(parameters, dict):
        bijector = bijector_cls(**parameters, **bijector_kwargs)
    else:
        bijector = bijector_cls(parameters, **bijector_kwargs)

    if invert:
        return tfb.Invert(bijector)
    else:
        return bijector


def _get_bijector_fn(parameters_fn, nested_bijector):
    def bijector_fn(y: tf.Tensor, *_: Any, **kwargs: Any) -> tfb.Bijector:
        parent_parameters = parameters_fn(y, **kwargs)
        return _init_nested_bijector_from_dict(
            nested_bijector_definition=nested_bijector.copy(),
            parent_parameters=parent_parameters,
        )

    return bijector_fn


def get_nested_parameters_fn(
    nested_parameters_fn: Callable[tf.Tensor, tf.Tensor], parent_parameters: tf.Tensor
) -> Callable[tf.Tensor, tf.Tensor]:
    """Get the parameter function of nested bijectors.

    If the nested bijector has a parameter function of its own, we redefine it here
    and add the parent parameters to its result.

    Parameters
    ----------
    nested_parameters_fn : Callable[tf.Tensor, tf.Tensor]
        The parameters function of the nested bijector.
    parent_parameters : tf.Tensor
        The Parameters of the parent bijector.

    Returns
    -------
    Callable[tf.Tensor, tf.Tensor]
        The redefined parameter function, adding the parents parameters.


    """

    def parameters_fn(inputs: Any, **kwargs: Any) -> tf.Tensor:
        __LOGGER__.debug("Nested parameters fn: %s", nested_parameters_fn)
        __LOGGER__.debug("Nested parameters fn id: %s", id(nested_parameters_fn))
        return nested_parameters_fn(inputs, **kwargs) + parent_parameters

    return parameters_fn


def _init_nested_bijector_from_dict(
    nested_bijector_definition: List[Dict[str, Any]],
    parent_parameters: tf.Tensor,
) -> tfb.Bijector:
    initialized_nested_bijectors: List[tfb.Bijector] = []

    offset = 0
    for bj in nested_bijector_definition:
        if isinstance(bj, dict):
            nested_bijector_kwargs = bj.copy()
            __LOGGER__.debug("Nested bijector config: %s", nested_bijector_kwargs)
            if nested_bijector_kwargs.get(__PARAMETERIZED_BY_PARENT_KEY__, False):
                __LOGGER__.debug("Bijector is parameterized by parent")
                if __PARAMETER_SLICE_SIZE_KEY__ in nested_bijector_kwargs.keys():
                    __LOGGER__.debug("Parameters should be sliced")
                    parameters_slice_size = nested_bijector_kwargs[
                        __PARAMETER_SLICE_SIZE_KEY__
                    ]
                    if parameters_slice_size > 1:
                        processed_parent_parameters = parent_parameters[
                            ..., offset : offset + parameters_slice_size
                        ]
                    else:
                        processed_parent_parameters = parent_parameters[..., offset]

                    __LOGGER__.debug(
                        "Initializing bijector with parameters [%s:%s]",
                        offset,
                        offset + parameters_slice_size,
                    )
                    offset += parameters_slice_size
                elif offset == 0:
                    __LOGGER__.debug("Using whole parameter vector provided by parent")
                    processed_parent_parameters = parent_parameters
                else:
                    raise ValueError(
                        "Parameter slicing has to be used in all nested bijectors."
                    )

                nested_parameters = nested_bijector_kwargs.pop(__PARAMETERS_KEY__, None)

                __LOGGER__.debug("Nested parameters: %s", nested_parameters)
                __LOGGER__.debug("Nested parameters id: %s", id(nested_parameters))

                # TODO: allow custom reduction method here
                if callable(nested_parameters):
                    __LOGGER__.debug("Nested params are callable")
                    nested_bijector_kwargs[__PARAMETERS_KEY__] = (
                        get_nested_parameters_fn(nested_parameters, parent_parameters)
                    )
                elif nested_parameters is not None:
                    __LOGGER__.debug("Nested params are tensor: adding")
                    nested_bijector_kwargs[__PARAMETERS_KEY__] = (
                        nested_parameters + processed_parent_parameters
                    )
                else:
                    __LOGGER__.debug("Nested params are None: using parent")
                    nested_bijector_kwargs[__PARAMETERS_KEY__] = (
                        processed_parent_parameters
                    )

                __LOGGER__.debug(
                    "Nested bijector config after parameter handling: %s",
                    nested_bijector_kwargs,
                )
            bijector = _init_bijector_from_dict(
                nested_bijector_kwargs,
            )
        elif isinstance(bj, tfb.Bijector):
            __LOGGER__.debug("Nested bijector is already initialized: %s", bj)
            bijector = bj
        else:
            raise ValueError(
                "nested bijector has incompatible type: %s",
                nested_bijector_kwargs,
            )
        initialized_nested_bijectors.append(bijector)

    assert offset == 0 or (
        offset == parent_parameters.shape[-1]
    ), "Not all parameters from parent bijector used. Check your config!"
    __LOGGER__.debug(
        "Initialized nested bijectors: %s",
        initialized_nested_bijectors,
    )
    if len(initialized_nested_bijectors) > 1:
        return tfb.Chain(initialized_nested_bijectors)
    else:
        return initialized_nested_bijectors[0]


def _get_transformed_distribution_fn(
    flow_parametrization_fn: Callable[[tf.Tensor], tfb.Bijector],
    get_base_distribution: Callable[..., tfd.Distribution] = _get_base_distribution,
    **kwargs: Any,
) -> Callable[[tf.Tensor], tfd.TransformedDistribution]:
    """Get function to parametrize a transformed distribution.

    Parameters
    ----------
    flow_parametrization_fn : Callable[[tf.Tensor], tfb.Bijector]
        The flow parametrization function.
    get_base_distribution : Callable[..., tfd.Distribution]
        Function that returns base distribution if provided;
        otherwise, use default base distribution.
    **kwargs : Any
        Additional keyword parameters.

    Returns
    -------
    distribution_fn : Callable[[tf.Tensor], tfd.TransformedDistribution]
        The transformed distribution function.

    """

    def distribution_fn(parameters: tf.Tensor) -> tfd.TransformedDistribution:
        input_shape, parameters = parameters
        # TODO: implement parameterizeable base distribution
        # if base_parameters is not None:
        #     kwargs.update(base_parameters)
        #     __LOGGER__.debug("got parameters for base distribution.")

        __LOGGER__.debug("base distribution kwargs: %s", str(kwargs))
        base_distribution = get_base_distribution(**kwargs)
        if input_shape is not None:
            base_distribution = tfd.BatchBroadcast(
                base_distribution, with_shape=input_shape[:1]
            )

        bijector = flow_parametrization_fn(parameters)
        return tfd.TransformedDistribution(
            distribution=base_distribution,
            bijector=bijector,
        )

    return distribution_fn


def _get_num_masked(dims: int, layer: int) -> int:
    """Compute the number of masked dimensions.

    Parameters
    ----------
    dims : int
        The total number of dimensions.
    layer : int
        The layer number.

    Returns
    -------
    num_masked : int
        The number of masked dimensions.

    """
    num_masked = dims // 2
    if dims % 2 != 0:
        num_masked += layer % 2
    return num_masked


def _get_layer_overwrites(
    layer_overwrites: Dict[Union[int, str], Dict[str, Any]], layer: int, num_layers: int
) -> Dict[str, Any]:
    return layer_overwrites.get(layer, layer_overwrites.get(layer - num_layers, {}))


def _get_nested_bijector_def(num_layers, layer_overwrites, layer, **kwargs):
    return deepupdate(
        deepcopy(
            {
                __PARAMETERIZED_BY_PARENT_KEY__: True,
                **kwargs,
            }
        ),
        _get_layer_overwrites(layer_overwrites, layer, num_layers),
    )


@recurse_on_key(__NESTED_BIJECTOR_KEY__)
@skip_no_dict
def validate_bijector_definition(dict):
    """Check if Bijector definition contains all required and no unknonw keys.

    Parameters
    ----------
    dict : Dict[str, Any]
        Bijector definition


    """
    bj_err = f"Bijector definition invalid:\n{pformat(dict)}\n"
    assert __BIJECTOR_NAME_KEY__ in dict, (
        bj_err + f"Key '{__BIJECTOR_NAME_KEY__}' is required."
    )
    for k in dict:
        assert k in __ALL_KEYS__, bj_err + f"Unknown key: '{k}'"


# %% public functions ##########################################################
def get_normalizing_flow(
    bijectors: List[Dict[str, Any]],
    reverse_flow: bool = False,
    inverse_flow: bool = False,
    get_base_distribution: Callable[..., tfd.Distribution] = _get_base_distribution,
    variables_name: str = None,
    **kwargs,
) -> Tuple[
    Callable[[tf.Tensor], tfd.TransformedDistribution],
    Callable[[tf.Tensor], Union[list, Tuple[list, Any]]],
    List[tf.Variable],
    List[tf.Variable],
]:
    """Get functions and variables to parametrize transformed distributions.

    Parameters
    ----------
    bijectors : List[Dict[str, Any]]
        List of dictionaries describing bijective transformations.
    reverse_flow : bool, optional
        Reverse chain of bijectors, by default False.
    inverse_flow : bool, optional
        Invert flow to transform from the data to the base distribution,
        by default False.
    get_base_distribution : Callable, optional
        The base distribution lambda,
        by default `parameters_lib._get_base_distribution`.
    variables_name: str, optional
        A optional prefix, given to all initialized variables. Used to avoid
        duplicate names, when initializing multiple flows. By default: None

    **kwargs : Any
        Additional optional keyword arguments passed
        to `parameters_lib._get_transformed_distribution_fn`.

    Returns
    -------
    Tuple[Callable, Callable, List[tf.Variable], List[tf.Variable]]
        The parametrization function of the transformed distribution,
        the parameter function, a list of trainable parameters
        and a list of non-trainable parameters.

    """
    # check if bijector configuration is valid
    list(map(validate_bijector_definition, bijectors))

    (
        parameter_fn,
        flow_parametrization_fn,
        trainable_variables,
        non_trainable_variables,
    ) = _get_flow_parametrization_fn(
        bijectors=bijectors,
        inverse_flow=inverse_flow,
        reverse_flow=reverse_flow,
        variables_name=variables_name,
    )

    distribution_fn = _get_transformed_distribution_fn(
        flow_parametrization_fn,
        get_base_distribution=get_base_distribution,
        **kwargs,
    )

    return (
        distribution_fn,
        parameter_fn,
        trainable_variables,
        non_trainable_variables,
    )


def _get_flow_parametrization_fn(bijectors, inverse_flow, reverse_flow, variables_name):
    # initialize parameters functions
    bijectors_parameters_fns, trainable_variables, non_trainable_variables = (
        _init_parameters_fn(bijectors=bijectors, variables_name=variables_name)
    )

    def parameter_fn(
        conditional_input: Union[tf.Tensor, None], **kwargs: Any
    ) -> Union[list, Tuple[list, Any]]:
        if tf.is_tensor(conditional_input):
            input_shape = prefer_static.shape(conditional_input)
        else:
            input_shape = None
        eval_parameter_fn = _get_eval_parameter_fn(conditional_input, **kwargs)
        bijectors_parameters = list(map(eval_parameter_fn, bijectors_parameters_fns))
        return input_shape, bijectors_parameters

    def flow_parametrization_fn(all_parameters: list):
        bijectors_list = list(map(_init_bijector_from_dict, all_parameters))

        if reverse_flow:
            # The Chain bijector uses the reversed list in the forward call.
            # We change the direction here to get T = f₃ ∘ f₂ ∘ f₁.
            bijectors_list = list(reversed(bijectors_list))

        if len(bijectors_list) == 1:
            flow = bijectors_list[0]
        else:
            flow = tfb.Chain(bijectors_list)

        if inverse_flow:
            # If we invert the reversed flow we get
            # T = f₃⁻¹ ∘ f₂⁻¹ ∘ f₁⁻¹ and T⁻¹ = f₁ ∘ f₂ ∘ f₂.
            flow = tfb.Invert(flow)

        return flow

    return (
        parameter_fn,
        flow_parametrization_fn,
        trainable_variables,
        non_trainable_variables,
    )


def get_coupling_flow(
    dims: int,
    num_layers: int,
    num_parameters: int,
    num_masked: Union[int, None] = None,
    layer_overwrites: Dict[Union[int, str], Dict[str, Any]] = {},
    parameters_fn: Callable[..., Any] = parameters_lib.get_fully_connected_network_fn,
    parameters_fn_kwargs: Dict[str, Any] = {},
    get_base_distribution: Callable[..., tfd.Distribution] = _get_base_distribution,
    base_distribution_kwargs: Dict[str, Any] = {},
    nested_bijectors: Union[List[Union[Dict[str, Any], tfb.Bijector]], None] = None,
    **kwargs: Dict[str, Any],
) -> Tuple[
    Callable[[tf.Tensor], tfd.Distribution],
    Callable[[tf.Tensor], tf.Variable],
    List[tf.Variable],
    List[tf.Variable],
]:
    """Get functions and variables to parametrize a Coupling Flow distribution.

    Parameters
    ----------
    dims : int
        The dimension of the distribution.
    num_layers : int
        The number of layers in the flow.
    num_parameters : int
        The number of parameters in each layer.
    num_masked : Union[int, None], optional
        Number of dimensions to mask, by default None.
    layer_overwrites : Dict[Union[int, str], Dict[str, Any]], optional
        Layer specific overwrites for bijectors, by default {}.
    parameters_fn : Callable[..., Any], optional
        A function to get the parameter lambda,
        by default parameters_lib.get_fully_connected_network_fn.
    parameters_fn_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to `parameter_fn`,
        by default {}.
    get_base_distribution : Callable, optional
        The base distribution lambda,
        by default parameters_lib._get_base_distribution.
    base_distribution_kwargs : Dict[str, Any], optional
        Keyword arguments for the base distribution, by default {}.
    nested_bijectors: Union[List[Union[Dict, tfb.Bijector]], None] = None, optional
        In addition to provide nested bijector definition as kwargs, this
        argument can be used to provide multiple nested bijectors in a list.
    **kwargs : Dict[str, Any]
        Additional keyword arguments added to the nested bijector definition.

    Returns
    -------
    distribution_fn : Callable[[tf.Tensor], tfd.Distribution]
        A function to parametrize the distribution
    parameter_fn : Callable[[tf.Tensor], tf.Variable]
        A callable for parameter networks
    trainable_parameters : List[tf.Variable]
        A list of trainable parameters.
    non_trainable_parameters : List[tf.Variable]
        A list of non-trainable parameters.

    """
    bijectors = []

    for layer in range(num_layers):
        nm = num_masked if num_masked is not None else _get_num_masked(dims, layer)

        # RealNVP's nested bijector config is abstracted from the user.
        # In order took keep the overwrite mechanism intuitive we add it after the
        # overwrites have been applied.
        if nested_bijectors is None:
            nested_bijector_def = _get_nested_bijector_def(
                num_layers, layer_overwrites, layer, **kwargs
            )
        else:
            nested_bijector_def = [
                _get_nested_bijector_def(num_layers, layer_overwrites, layer, **d)
                for d in nested_bijectors
            ]
        realnvp_bijector_def = {
            __BIJECTOR_NAME_KEY__: "RealNVP",
            __BIJECTOR_KWARGS_KEY__: {
                "num_masked": nm,
            },
            __NESTED_BIJECTOR_KEY__: nested_bijector_def,
            __PARAMETERS_FN_KEY__: parameters_fn,
            __PARAMETERS_FN_KWARGS_KEY__: {
                "input_shape": (nm,),
                "parameter_shape": (dims - nm, num_parameters),
                "name": f"coupling_layer_{layer}",
                **parameters_fn_kwargs,
            },
        }
        bijectors.append(realnvp_bijector_def)

        permutation = list(range(nm, dims)) + list(range(nm))
        if num_layers % 2 != 0 and layer == (num_layers - 1):
            __LOGGER__.info(
                "uneven number of coupling layers -> skipping last permutation"
            )
        else:
            bijectors.append(tfb.Permute(permutation=permutation))

    __LOGGER__.info(pformat(bijectors))
    return get_normalizing_flow(
        dims=dims,
        bijectors=bijectors,
        reverse_flow=False,
        inverse_flow=False,
        get_base_distribution=get_base_distribution,
        **base_distribution_kwargs,
    )


def _get_masked_autoregressive_flow_bijector_def(
    dims: int,
    num_layers: int,
    num_parameters: int,
    layer_overwrites: Dict[Union[int, str], Dict[str, Any]] = {},
    parameters_fn: Callable[
        ..., Any
    ] = parameters_lib.get_masked_autoregressive_network_fn,
    parameters_fn_kwargs: Dict[str, Any] = {},
    maf_kwargs: Dict[str, Any] = {},
    nested_bijectors: Union[List[Union[Dict[str, Any], tfb.Bijector]], None] = None,
    **kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Get functions and variables to parametrize a Masked Autoregressive Flow (MAF).

    Parameters
    ----------
    dims : int
        The dimension of the distribution.
    num_layers : int
        The number of layers in the flow.
    num_parameters : int
        The number of parameters in each layer.
    layer_overwrites : Dict[Union[int, str], Dict[str, Any]], optional
        Layer specific overwrites for bijectors, by default {}.
    parameters_fn : Callable[..., Any], optional
        A function to get the parameter lambda,
        by default parameters_lib.get_masked_autoregressive_network_fn
    parameters_fn_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to `parameters_fn`,
        by default {}.
    maf_kwargs : Dict[str, Any]
        Additional keyword arguments added to the MAF bijector definition,
        by default {}.
    nested_bijectors: Union[List[Union[Dict, tfb.Bijector]], None] = None, optional
        In addition to provide nested bijector definition as kwargs, this
        argument can be used to provide multiple nested bijectors in a list.
    **kwargs : Dict[str, Any]
        Additional keyword arguments added to the nested bijector definition.


    Returns
    -------
    distribution_fn : Callable[[tf.Tensor], tfd.Distribution]
        A function to parametrize the distribution
    parameter_fn : Callable[[tf.Tensor], tf.Variable]
        A callable for parameter networks
    trainable_parameters : List[tf.Variable]
        A list of trainable parameters.
    non_trainable_parameters : List[tf.Variable]
        A list of non-trainable parameters.

    """
    bijectors = []

    for layer in range(num_layers):
        # The nested bijector config is abstracted from the user.
        # In order took keep the overwrite mechanism intuitive we add it after the
        # overwrites have been applied.

        if nested_bijectors is None:
            nested_bijector_def = _get_nested_bijector_def(
                num_layers, layer_overwrites, layer, **kwargs
            )
        else:
            nested_bijector_def = [
                _get_nested_bijector_def(num_layers, layer_overwrites, layer, **d)
                for d in nested_bijectors
            ]

        bijector_def = {
            __BIJECTOR_NAME_KEY__: "MaskedAutoregressiveFlow",
            __NESTED_BIJECTOR_KEY__: nested_bijector_def,
            __PARAMETERS_FN_KEY__: parameters_fn,
            __PARAMETERS_FN_KWARGS_KEY__: {
                "parameter_shape": (dims, num_parameters),
                **parameters_fn_kwargs,
                "name": f"maf_layer_{layer}",
            },
            **maf_kwargs.copy(),
        }
        bijectors.append(bijector_def)

    __LOGGER__.info(pformat(bijectors))

    return bijectors


def get_masked_autoregressive_flow(
    dims: int,
    get_base_distribution: Callable[..., tfd.Distribution] = _get_base_distribution,
    base_distribution_kwargs: Dict[str, Any] = {},
    **kwargs: Dict[str, Any],
) -> Tuple[
    Callable[[tf.Tensor], tfd.Distribution],
    Callable[[tf.Tensor], tf.Variable],
    List[tf.Variable],
    List[tf.Variable],
]:
    """Get functions and variables to parametrize a Masked Autoregressive Flow (MAF).

    Parameters
    ----------
    dims : int
        The dimension of the distribution.
    get_base_distribution : Callable, optional
        The base distribution lambda,
        by default parameters_lib._get_base_distribution.
    base_distribution_kwargs : Dict[str, Any], optional
        Keyword arguments for the base distribution, by default {}.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to
        `_get_masked_autoregressive_flow_bijector_def`.


    Returns
    -------
    distribution_fn : Callable[[tf.Tensor], tfd.Distribution]
        A function to parametrize the distribution
    parameter_fn : Callable[[tf.Tensor], tf.Variable]
        A callable for parameter networks
    trainable_parameters : List[tf.Variable]
        A list of trainable parameters.
    non_trainable_parameters : List[tf.Variable]
        A list of non-trainable parameters.

    """
    bijectors = _get_masked_autoregressive_flow_bijector_def(dims=dims, **kwargs)
    return get_normalizing_flow(
        dims=dims,
        bijectors=bijectors,
        reverse_flow=False,
        inverse_flow=False,
        get_base_distribution=get_base_distribution,
        **base_distribution_kwargs,
    )


def get_masked_autoregressive_flow_first_dim_masked(
    dims: int,
    num_parameters: int,
    x0_parameters_fn: Callable[
        ..., Any
    ] = parameters_lib.get_fully_connected_network_fn,
    x0_parameters_fn_kwargs: Dict[str, Any] = {},
    maf_parameters_fn: Callable[
        ..., Any
    ] = parameters_lib.get_masked_autoregressive_network_fn,
    maf_parameters_fn_kwargs: Dict[str, Any] = {},
    get_base_distribution: Callable[..., tfd.Distribution] = _get_base_distribution,
    base_distribution_kwargs: Dict[str, Any] = {},
    **kwargs: Dict[str, Any],
) -> Tuple[
    Callable[[tf.Tensor], tfd.Distribution],
    Callable[[tf.Tensor], tf.Variable],
    List[tf.Variable],
    List[tf.Variable],
]:
    """Get functions and variables to parametrize a MAF with the first dimension masked.

    Parameters
    ----------
    dims : int
        The dimension of the distribution.
    num_parameters : int
        The number of parameters in each layer.
    x0_parameters_fn : Callable[..., Any], optional
        A function to get the parameter lambda of the encapsulating RealNVP bjector,
        by default parameters_lib.get_fully_connected_network_fn.
    x0_parameters_fn_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to `x0_parameters_fn`,
        by default {}.
    maf_parameters_fn : Callable[..., Any], optional
        A function to get the parameter lambda of the MAF bjectors,
        by default parameters_lib.get_fully_connected_network_fn.
    maf_parameters_fn_kwargs : Dict[str, Any], optional
        Additional keyword arguments passed to `maf_parameters_fn`,
        by default {}.
    get_base_distribution : Callable, optional
        The base distribution lambda,
        by default parameters_lib._get_base_distribution.
    base_distribution_kwargs : Dict[str, Any], optional
        Keyword arguments for the base distribution, by default {}.
    **kwargs : Dict[str, Any]
        Additional keyword arguments passed to
        `_get_masked_autoregressive_flow_bijector_def`.


    Returns
    -------
    distribution_fn : Callable[[tf.Tensor], tfd.Distribution]
        A function to parametrize the distribution
    parameter_fn : Callable[[tf.Tensor], tf.Variable]
        A callable for parameter networks
    trainable_parameters : List[tf.Variable]
        A list of trainable parameters.
    non_trainable_parameters : List[tf.Variable]
        A list of non-trainable parameters.

    """
    nested_bijector_def = _get_masked_autoregressive_flow_bijector_def(
        dims=dims - 1,
        num_parameters=num_parameters,
        parameters_fn=maf_parameters_fn,
        parameters_fn_kwargs=maf_parameters_fn_kwargs,
        maf_kwargs={__PARAMETERIZED_BY_PARENT_KEY__: True},
        **kwargs,
    )
    bijector_def = {
        __BIJECTOR_NAME_KEY__: "RealNVP",
        __BIJECTOR_KWARGS_KEY__: {
            "num_masked": 1,
        },
        __NESTED_BIJECTOR_KEY__: nested_bijector_def,
        __PARAMETERS_FN_KEY__: x0_parameters_fn,
        __PARAMETERS_FN_KWARGS_KEY__: {
            "input_shape": (1,),
            "parameter_shape": (dims - 1, num_parameters),
            **x0_parameters_fn_kwargs,
        },
    }
    return get_normalizing_flow(
        dims=dims,
        bijectors=[bijector_def],
        reverse_flow=False,
        inverse_flow=False,
        get_base_distribution=get_base_distribution,
        **base_distribution_kwargs,
    )


get_multivariate_normal = partial(
    _get_trainable_distribution,
    get_distribution_fn=_get_multivariate_normal_fn,
    parameters_fn=parameters_lib.get_parameter_vector_or_simple_network_fn,
)
