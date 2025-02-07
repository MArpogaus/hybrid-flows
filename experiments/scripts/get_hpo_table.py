# %% imports
import re

import dvc.api
import numpy as np
import pandas as pd
import yaml
from tqdm.contrib.concurrent import process_map

from mctm.utils import flatten_dict

# %% load params
stages = (
    "train-sim@dataset0-conditional_coupling_flow_bernstein_poly",
    "train-sim@dataset0-conditional_coupling_flow_quadratic_spline",
    "train-sim@dataset0-conditional_masked_autoregressive_flow_bernstein_poly",
    "train-sim@dataset0-conditional_masked_autoregressive_flow_quadratic_spline",
    "train-sim@dataset0-conditional_multivariate_normal",
    "train-sim@dataset0-conditional_multivariate_transformation_model",
    "train-sim@dataset0-conditional_hybrid_coupling_flow_bernstein_poly",
    "train-sim@dataset0-conditional_hybrid_coupling_flow_quadratic_spline",
    "train-sim@dataset0-unconditional_coupling_flow_bernstein_poly",
    "train-sim@dataset0-unconditional_coupling_flow_quadratic_spline",
    "train-sim@dataset0-unconditional_masked_autoregressive_flow_bernstein_poly",
    "train-sim@dataset0-unconditional_masked_autoregressive_flow_quadratic_spline",
    "train-sim@dataset0-unconditional_multivariate_normal",
    "train-sim@dataset0-unconditional_multivariate_transformation_model",
    "train-sim@dataset0-unconditional_hybrid_coupling_flow_bernstein_poly",
    "train-sim@dataset0-unconditional_hybrid_coupling_flow_quadratic_spline",
    "train-sim@dataset1-conditional_coupling_flow_bernstein_poly",
    "train-sim@dataset1-conditional_coupling_flow_quadratic_spline",
    "train-sim@dataset1-conditional_masked_autoregressive_flow_bernstein_poly",
    "train-sim@dataset1-conditional_masked_autoregressive_flow_quadratic_spline",
    "train-sim@dataset1-conditional_multivariate_normal",
    "train-sim@dataset1-conditional_multivariate_transformation_model",
    "train-sim@dataset1-conditional_hybrid_coupling_flow_bernstein_poly",
    "train-sim@dataset1-conditional_hybrid_coupling_flow_quadratic_spline",
    "train-sim@dataset1-unconditional_coupling_flow_bernstein_poly",
    "train-sim@dataset1-unconditional_coupling_flow_quadratic_spline",
    "train-sim@dataset1-unconditional_masked_autoregressive_flow_bernstein_poly",
    "train-sim@dataset1-unconditional_masked_autoregressive_flow_quadratic_spline",
    "train-sim@dataset1-unconditional_multivariate_normal",
    "train-sim@dataset1-unconditional_multivariate_transformation_model",
    "train-sim@dataset1-unconditional_hybrid_coupling_flow_bernstein_poly",
    "train-sim@dataset1-unconditional_hybrid_coupling_flow_quadratic_spline",
    "train-benchmark@dataset0-unconditional_masked_autoregressive_flow_quadratic_spline",
    "train-benchmark@dataset0-unconditional_hybrid_masked_autoregressive_flow_quadratic_spline",
    "train-benchmark@dataset1-unconditional_masked_autoregressive_flow_quadratic_spline",
    "train-benchmark@dataset1-unconditional_hybrid_masked_autoregressive_flow_quadratic_spline",
    "train-benchmark@dataset2-unconditional_masked_autoregressive_flow_quadratic_spline",
    "train-benchmark@dataset2-unconditional_hybrid_masked_autoregressive_flow_quadratic_spline",
    "train-benchmark@dataset3-unconditional_masked_autoregressive_flow_quadratic_spline",
    "train-benchmark@dataset3-unconditional_hybrid_masked_autoregressive_flow_quadratic_spline",
    "train-benchmark@dataset4-unconditional_masked_autoregressive_flow_quadratic_spline",
    "train-benchmark@dataset4-unconditional_hybrid_masked_autoregressive_flow_quadratic_spline",
    "train-malnutrition@conditional_multivariate_transformation_model",
    "train-malnutrition@conditional_hybrid_masked_autoregressive_flow_bernstein_poly",
    "train-malnutrition@conditional_hybrid_masked_autoregressive_flow_quadratic_spline",
)


def process(stage):
    return flatten_dict(dvc.api.params_show(stages=stage))


params = list(process_map(process, stages))

# %% paramd desc
param_map = {
    "compile_kwargs.jit_compile": {
        "name": "JIT Compile",
        "desc": "Defines whether to compile the model using Just-In-Time compilation for performance",
    },
    "fit_kwargs.0.batch_size": {
        "name": "Marginal Batch Size",
        "desc": "Batch size for training the marginal distribution model",
    },
    "fit_kwargs.0.early_stopping": {
        "name": "Marginal Early Stopping",
        "desc": "Enables early stopping for the marginal model training",
    },
    "fit_kwargs.0.epochs": {
        "name": "Marginal Epochs",
        "desc": "Number of epochs for training the marginal distribution model",
    },
    "fit_kwargs.0.learning_rate": {
        "name": "Marginal Learning Rate",
        "desc": "Learning rate used to optimize the marginal distribution",
    },
    "fit_kwargs.0.monitor": {
        "name": "Marginal Monitor",
        "desc": "Metric to monitor during training for early stopping",
    },
    "fit_kwargs.0.reduce_lr_on_plateau": {
        "name": "Marginal Reduce LR on Plateau",
        "desc": "Whether to reduce learning rate on a plateau during training",
    },
    "fit_kwargs.0.validation_split": {
        "name": "Marginal Validation Split",
        "desc": "Fraction of training data to be used as validation data for the marginal model",
    },
    "fit_kwargs.0.verbose": {
        "name": "Marginal Verbosity",
        "desc": "Verbosity mode for the marginal model training",
    },
    "fit_kwargs.1.batch_size": {
        "name": "Joint Batch Size",
        "desc": "Batch size for training the joint distribution model",
    },
    "fit_kwargs.1.early_stopping": {
        "name": "Joint Early Stopping",
        "desc": "Enables early stopping for the joint model training",
    },
    "fit_kwargs.1.epochs": {
        "name": "Joint Epochs",
        "desc": "Maximum number of epochs for training the joint distribution model",
    },
    "fit_kwargs.1.learning_rate.scheduler_kwargs.decay_steps": {
        "name": "Joint Learning Rate Decay Steps",
        "desc": "Total number of decay steps for the learning rate scheduler of the joint model",
    },
    "fit_kwargs.1.learning_rate.scheduler_kwargs.initial_learning_rate": {
        "name": "Joint Initial Learning Rate",
        "desc": "Initial learning rate for the learning rate scheduler for the joint model",
    },
    "fit_kwargs.1.learning_rate.scheduler_name": {
        "name": "Joint Learning Rate Scheduler",
        "desc": "Name of the learning rate scheduler for the joint model",
    },
    "fit_kwargs.1.monitor": {
        "name": "Joint Monitor",
        "desc": "Metric to monitor during training for early stopping",
        "drop": True,
    },
    "fit_kwargs.1.reduce_lr_on_plateau": {
        "name": "Joint Reduce LR on Plateau",
        "desc": "Whether to reduce learning rate on a plateau during training",
        "drop": True,
    },
    "fit_kwargs.1.validation_split": {
        "name": "Joint Validation Split",
        "desc": "Fraction of training data to be used as validation data for the joint model",
    },
    "fit_kwargs.1.verbose": {
        "name": "Joint Verbosity",
        "desc": "Verbosity mode for the joint model training",
        "drop": True,
    },
    "fit_kwargs.batch_size": {
        "name": "Batch Size",
        "desc": "Batch size used for training",
    },
    "fit_kwargs.early_stopping": {
        "name": "Early Stopping",
        "desc": "Enable early stopping",
    },
    "fit_kwargs.epochs": {
        "name": "Epochs",
        "desc": "Maximum number of epochs",
    },
    "fit_kwargs.learning_rate.scheduler_kwargs.decay_steps": {
        "name": "Learning Rate Decay Steps",
        "desc": "Steps after which to decay the learning rate with the scheduler",
    },
    "fit_kwargs.learning_rate.scheduler_kwargs.initial_learning_rate": {
        "name": "Initial Learning Rate",
        "desc": "Initial learning rate for the learning rate scheduler",
    },
    "fit_kwargs.learning_rate.scheduler_name": {
        "name": "Learning Rate Scheduler",
        "desc": "Name of the learning rate scheduler",
    },
    "fit_kwargs.monitor": {
        "name": "Monitor",
        "desc": "Metric to monitor during training for early stopping",
        "drop": True,
    },
    "fit_kwargs.reduce_lr_on_plateau": {
        "name": "Reduce LR on Plateau",
        "desc": "Reduce learning rate on a plateau during training",
        "drop": True,
    },
    "fit_kwargs.validation_split": {
        "name": "Validation Split",
        "desc": "Fraction of training data to be used as validation data",
    },
    "fit_kwargs.verbose": {
        "name": "Verbosity",
        "desc": "Enable verbosity mode during training",
        "drop": True,
    },
    "model_kwargs.bijector": {
        "name": "Bijector",
        "desc": "Specifies the bijector type to be used in the model",
    },
    "model_kwargs.bijector_kwargs.domain.0": {
        "name": "Bernstein Bijector Domain Min",
        "desc": "Lower bound of the Bernstein polynomial bijector domain",
    },
    "model_kwargs.bijector_kwargs.domain.1": {
        "name": "Bernstein Bijector Domain Max",
        "desc": "Upper bound of the Bernstein polynomial bijector domain",
    },
    "model_kwargs.bijector_kwargs.extrapolation": {
        "name": "Bernstein Bijector Extrapolation",
        "desc": "Defines whether to allow extrapolation in the bijector",
    },
    "model_kwargs.bijector_kwargs.range_min": {
        "name": "Spline Bijector Domain Min",
        "desc": "Lower bound of the quadratic spline bijector domain",
    },
    "model_kwargs.distribution": {
        "name": "Output Distribution",
        "desc": "Specifies the type of output distribution used in the model",
    },
    "model_kwargs.invert": {
        "name": "Invert Bijector",
        "desc": "Indicates if the bijector should be inverted",
    },
    "model_kwargs.joint_bijectors.0.bijector": {
        "name": "First Joint Bijector",
        "desc": "Specifies the first bijector type for the joint distribution",
    },
    "model_kwargs.joint_bijectors.0.bijector_kwargs.num_masked": {
        "name": "Masked Dimensions",
        "desc": "Number of deimensions mask in the Coupling Flow",
    },
    "model_kwargs.joint_bijectors.0.invert": {
        "name": "Invert First Joint Bijector",
        "desc": "Indicates if the first joint bijector should allow inversion",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.bijector": {
        "name": "First Nested Bijector",
        "desc": "Specifies the type of nested bijector used in the first joint bijector",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.bijector_kwargs.domain.0": {
        "name": "Nested Bernstein Bijector Domain Min",
        "desc": "Lower boundary for the nested Bernstein bijector domain",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.bijector_kwargs.domain.1": {
        "name": "Nested Bernstein Bijector Domain Max",
        "desc": "Upper boundary for the nested Bernstein bijector domain",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.bijector_kwargs.extrapolation": {
        "name": "Nested Bernstein Bijector Extrapolation",
        "desc": "Defines whether to allow extrapolation in the nested Bernstein bijector",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.bijector_kwargs.range_min": {
        "name": "Nested Spline Bijector Range Min",
        "desc": "Minimum range value for the nested bijector transformation",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.invert": {
        "name": "Invert Nested Bijector",
        "desc": "Specifies if the nested bijector should be invertible",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.parameters_constraint_fn": {
        "name": "Nested Bijector Parameter Constraint Function",
        "desc": "Constraint function applied to the parameters of the nested bijector",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.parameters_constraint_fn_kwargs.allow_flexible_bounds": {
        "name": "Allow Flexible Bounds for Nested Bernstein Bijector",
        "desc": "Defines if flexible bounds are allowed for the nested Bernstein bijector",
        "drop": True,
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.parameters_constraint_fn_kwargs.bounds": {
        "name": "Bounds constraints of Nested Bernstein Bijector",
        "desc": "Defining the type of constraints on the bounds of the nested Bernstein bijector",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.parameters_constraint_fn_kwargs.high": {
        "name": "Nested Bernstein Bijector Codomain Max",
        "desc": "Defines the upper bound for the corresponding parameters",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.parameters_constraint_fn_kwargs.interval_width": {
        "name": "Interval Width for Nested Spline Bijector",
        "desc": "Specifies the interval width for constraint parameters",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.parameters_constraint_fn_kwargs.low": {
        "name": "Nested Bernstein Bijector Codomain Min",
        "desc": "Defines the lower bound for the corresponding parameters",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.parameters_constraint_fn_kwargs.min_bin_width": {
        "name": "Minimum Bin Width for Nested Spline Bijector",
        "desc": "Defines the minimum bin width for the nested Spline bijector",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.parameters_constraint_fn_kwargs.min_slope": {
        "name": "Minimum Slope for Nested Spline Bijector",
        "desc": "Defines the minimum slope constraint for parameters",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.parameters_constraint_fn_kwargs.nbins": {
        "name": "Number of Bins for Nested Spline Bijector",
        "desc": "Defines the number of bins used by the nested bijector",
    },
    "model_kwargs.joint_bijectors.0.nested_bijector.parametrized_by_parent": {
        "name": "Parametrize Nested by Parent Bijector",
        "desc": "Indicates if the parameters of the nested bijector are influenced by the parent bijector",
        "drop": True,
    },
    "model_kwargs.joint_bijectors.0.parameters_constraint_fn": {
        "name": "Parameter Constraint Function for Joint Bijector",
        "desc": "Constraint function applied to the parameters of the joint bijector",
        "drop": True,
    },
    "model_kwargs.joint_bijectors.0.parameters_fn": {
        "name": "Parameters Function for Joint Bijector",
        "desc": "Specifies the function for calculating the parameters of the joint bijector",
        "drop": True,
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.activation": {
        "name": "Activation Function for Joint Bijector",
        "desc": "Activation function used in the neural network used to estimate the joint bijector parameters",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.batch_norm": {
        "name": "Batch Normalization for Joint Bijector",
        "desc": "Specifies if batch normalization should be applied in the joint bijector",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.conditional": {
        "name": "Is Conditional for Joint Bijector",
        "desc": "Indicates if the joint bijector is conditional",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.conditional_event_shape": {
        "name": "Joint Bijector Conditional Event Shape",
        "desc": "Shape of the conditional events for the joint bijector",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.domain.0": {
        "name": "Nested Bernstein Bijector Domain Min",
        "desc": "Lower boundary for the nested Bernstein bijector domain",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.domain.1": {
        "name": "Nested Bernstein Bijector Domain Max",
        "desc": "Upper boundary for the nested Bernstein bijector domain",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.dropout": {
        "name": "Dropout Rate for Joint Bijector",
        "desc": "Dropout rate to apply in the joint bijector neural network",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.dtype": {
        "name": "Data Type for Joint Bijector",
        "desc": "Data type of the parameters in the joint bijector",
        "drop": True,
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.extrapolation": {
        "name": "Extrapolation for Joint Bijector",
        "desc": "Defines whether to allow extrapolation in the joint bijector",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.hidden_units.0": {
        "name": "First Hidden Units for Joint Bijector",
        "desc": "Number of units in the first hidden layer of the joint bijector",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.hidden_units.1": {
        "name": "Second Hidden Units for Joint Bijector",
        "desc": "Number of units in the second hidden layer of the joint bijector",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.hidden_units.2": {
        "name": "Third Hidden Units for Joint Bijector",
        "desc": "Number of units in the third hidden layer of the joint bijector",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.input_shape.0": {
        "name": "Input Shape 0 for Joint Bijector",
        "desc": "Defines the input shape dimension 0 for the joint bijector",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.parameter_shape.0": {
        "name": "Parameter Shape 0 for Joint Bijector",
        "desc": "Defines the parameter shape dimension 0 for the joint bijector",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.parameter_shape.1": {
        "name": "Parameter Shape 1 for Joint Bijector",
        "desc": "Defines the parameter shape dimension 1 for the joint bijector",
    },
    "model_kwargs.joint_bijectors.0.parameters_fn_kwargs.polynomial_order": {
        "name": "Polynomial Order for Joint Bijector",
        "desc": "Defines the order of the polynomial for computations in the joint bijector",
    },
    "model_kwargs.joint_bijectors.bijector": {
        "name": "Joint Bijector",
        "desc": "Specifies the bijector type to be used in the joint model",
    },
    "model_kwargs.joint_bijectors.bijector_kwargs.domain.0": {
        "name": "Nested Bernstein Bijector Domain Min",
        "desc": "Lower boundary for the nested Bernstein bijector domain",
    },
    "model_kwargs.joint_bijectors.bijector_kwargs.domain.1": {
        "name": "Nested Bernstein Bijector Domain Max",
        "desc": "Upper boundary for the nested Bernstein bijector domain",
    },
    "model_kwargs.joint_bijectors.bijector_kwargs.extrapolation": {
        "name": "Joint Bijector Extrapolation",
        "desc": "Defines whether to allow extrapolation in the joint bijector",
    },
    "model_kwargs.joint_bijectors.bijector_kwargs.range_min": {
        "name": "Joint Bijector Range Min",
        "desc": "Minimum range value for the joint bijector transformation",
    },
    "model_kwargs.joint_bijectors.invert": {
        "name": "Invert Joint Bijector",
        "desc": "Indicates if the joint bijector should be invertible",
    },
    "model_kwargs.joint_bijectors.maf_parameters_fn_kwargs.activation": {
        "name": "Activation Function for MAF",
        "desc": "Activation function used in the neural network for MAF parameters",
    },
    "model_kwargs.joint_bijectors.maf_parameters_fn_kwargs.conditional": {
        "name": "Is Conditional for MAF",
        "desc": "Indicates if the MAF parameters are conditional",
    },
    "model_kwargs.joint_bijectors.maf_parameters_fn_kwargs.conditional_event_shape": {
        "name": "MAF Conditional Event Shape",
        "desc": "Shape of the conditional events for the MAF",
    },
    "model_kwargs.joint_bijectors.maf_parameters_fn_kwargs.dtype": {
        "name": "Data Type for MAF Parameters",
        "desc": "Data type of the parameters in the MAF",
        "drop": True,
    },
    "model_kwargs.joint_bijectors.maf_parameters_fn_kwargs.hidden_units.0": {
        "name": "First Hidden Units for MAF",
        "desc": "Number of units in the first hidden layer of the MAF",
    },
    "model_kwargs.joint_bijectors.maf_parameters_fn_kwargs.hidden_units.1": {
        "name": "Second Hidden Units for MAF",
        "desc": "Number of units in the second hidden layer of the MAF",
    },
    "model_kwargs.joint_bijectors.maf_parameters_fn_kwargs.hidden_units.2": {
        "name": "Third Hidden Units for MAF",
        "desc": "Number of units in the third hidden layer of the MAF",
    },
    "model_kwargs.joint_bijectors.num_layers": {
        "name": "Number of Flows",
        "desc": "Total number of layers in the joint bijectors",
    },
    "model_kwargs.joint_bijectors.num_parameters": {
        "name": "Number of Parameters for Joint Bijectors",
        "desc": "Total number of parameters in the joint bijectors",
    },
    "model_kwargs.joint_bijectors.parameters_constraint_fn_kwargs.allow_flexible_bounds": {
        "name": "Allow Flexible Bounds for Joint Bijector",
        "desc": "Defines if flexible bounds are allowed for the joint bijector",
        "drop": True,
    },
    "model_kwargs.joint_bijectors.parameters_constraint_fn_kwargs.bounds": {
        "name": "Bounds constraints for Joint Bijector",
        "desc": "Defining the type of constraints on the bounds of the joint bijector",
        "drop": True,
    },
    "model_kwargs.joint_bijectors.parameters_constraint_fn_kwargs.high": {
        "name": "Joint Bijector Codomain Max",
        "desc": "Defines the upper bound for the corresponding parameters",
    },
    "model_kwargs.joint_bijectors.parameters_constraint_fn_kwargs.interval_width": {
        "name": "Interval Width for Joint Bijector",
        "desc": "Specifies the interval width for constraint parameters",
    },
    "model_kwargs.joint_bijectors.parameters_constraint_fn_kwargs.low": {
        "name": "Joint Bijector Codomain Min",
        "desc": "Defines the lower bound for the corresponding parameters",
    },
    "model_kwargs.joint_bijectors.parameters_constraint_fn_kwargs.min_bin_width": {
        "name": "Minimum Bin Width for Joint Bijector",
        "desc": "Defines the minimum bin width for the joint bijector",
    },
    "model_kwargs.joint_bijectors.parameters_constraint_fn_kwargs.min_slope": {
        "name": "Minimum Slope for Joint Bijector",
        "desc": "Defines the minimum slope constraint for parameters",
    },
    "model_kwargs.joint_bijectors.parameters_constraint_fn_kwargs.nbins": {
        "name": "Number of Bins for Joint Bijector",
        "desc": "Defines the number of bins used by the joint bijector",
    },
    "model_kwargs.joint_bijectors.random_permutation_seed": {
        "name": "Random Permutation Seed",
        "desc": "Seed value for random permutation in bijectors",
    },
    "model_kwargs.joint_bijectors.use_invertible_linear_transformations": {
        "name": "Use Invertible Linear Transformations",
        "desc": "Specifies if invertible linear transformations should be used",
    },
    "model_kwargs.joint_bijectors.x0_parameters_fn_kwargs.activation": {
        "name": "Activation Function for X0 Parameters",
        "desc": "Activation function for x0 parameters in the joint bijector",
    },
    "model_kwargs.joint_bijectors.x0_parameters_fn_kwargs.batch_norm": {
        "name": "Batch Normalization for X0 Parameters",
        "desc": "Specifies if batch normalization should be applied to x0 parameters",
    },
    "model_kwargs.joint_bijectors.x0_parameters_fn_kwargs.conditional": {
        "name": "Is Conditional for X0 Parameters",
        "desc": "Indicates if the x0 parameters are conditional",
    },
    "model_kwargs.joint_bijectors.x0_parameters_fn_kwargs.conditional_event_shape": {
        "name": "X0 Parameters Conditional Event Shape",
        "desc": "Shape of the conditional events for x0 parameters",
    },
    "model_kwargs.joint_bijectors.x0_parameters_fn_kwargs.dropout": {
        "name": "Dropout Rate for X0 Parameters",
        "desc": "Dropout rate for the x0 parameters neural network",
    },
    "model_kwargs.joint_bijectors.x0_parameters_fn_kwargs.dtype": {
        "name": "Data Type for X0 Parameters",
        "desc": "Data type of the x0 parameters in the model",
        "drop": True,
    },
    "model_kwargs.joint_bijectors.x0_parameters_fn_kwargs.hidden_units.0": {
        "name": "First Hidden Units for X0 Network",
        "desc": "Number of units in the first hidden layer for x0 parameters",
    },
    "model_kwargs.joint_bijectors.x0_parameters_fn_kwargs.hidden_units.1": {
        "name": "Second Hidden Units for X0 Network",
        "desc": "Number of units in the second hidden layer for x0 parameters",
    },
    "model_kwargs.joint_bijectors.x0_parameters_fn_kwargs.hidden_units.2": {
        "name": "Third Hidden Units for X0 Network",
        "desc": "Number of units in the third hidden layer for x0 parameters",
    },
    "model_kwargs.joint_flow_type": {
        "name": "Type of Joint Flow",
        "desc": "Specifies the type of flow used for the model",
    },
    "model_kwargs.layer_overwrites.-1.parameters_constraint_fn_kwargs.high": {
        "name": "High Constraint for Layer Overwrite -1",
        "desc": "Upper constraint for parameters in layer overwrite -1",
    },
    "model_kwargs.layer_overwrites.-1.parameters_constraint_fn_kwargs.low": {
        "name": "Low Constraint for Layer Overwrite -1",
        "desc": "Lower constraint for parameters in layer overwrite -1",
    },
    "model_kwargs.layer_overwrites.-2.parameters_constraint_fn_kwargs.high": {
        "name": "High Constraint for Layer Overwrite -2",
        "desc": "Upper constraint for parameters in layer overwrite -2",
    },
    "model_kwargs.layer_overwrites.-2.parameters_constraint_fn_kwargs.low": {
        "name": "Low Constraint for Layer Overwrite -2",
        "desc": "Lower constraint for parameters in layer overwrite -2",
    },
    "model_kwargs.layer_overwrites.0.bijector_kwargs.domain.0": {
        "name": "Layer Overwrite 0 Domain Min",
        "desc": "Lower boundary for the domain in layer overwrite 0",
    },
    "model_kwargs.layer_overwrites.0.bijector_kwargs.domain.1": {
        "name": "Layer Overwrite 0 Domain Max",
        "desc": "Upper boundary for the domain in layer overwrite 0",
    },
    "model_kwargs.marginal_bijectors.0.bijector": {
        "name": "First Marginal Bijector",
        "desc": "Specifies the type of the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.bijector_kwargs.domain.0": {
        "name": "Marginal Bijector Domain Min",
        "desc": "Lower boundary for the domain of the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.bijector_kwargs.domain.1": {
        "name": "Marginal Bijector Domain Max",
        "desc": "Upper boundary for the domain of the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.bijector_kwargs.extrapolation": {
        "name": "Marginal Bijector Extrapolation",
        "desc": "Defines whether to allow extrapolation in the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.invert": {
        "name": "Invert First Marginal Bijector",
        "desc": "Indicates if the first marginal bijector should allow inversion",
    },
    "model_kwargs.marginal_bijectors.0.parameters_constraint_fn": {
        "name": "First Marginal Bijector Parameter Constraint Function",
        "desc": "Constraint function for the parameters of the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.parameters_constraint_fn_kwargs.allow_flexible_bounds": {
        "name": "Allow Flexible Bounds for First Marginal Bijector",
        "desc": "Defines if flexible bounds are allowed for the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.parameters_constraint_fn_kwargs.bounds": {
        "name": "Bounds constraints for First Marginal Bijector",
        "desc": "Defining the constraints on the bounds for the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.parameters_constraint_fn_kwargs.high": {
        "name": "First Marginal Bijector Codomain Max",
        "desc": "Defines the upper bound for parameters in the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.parameters_constraint_fn_kwargs.low": {
        "name": "First Marginal Bijector Codomain Min",
        "desc": "Defines the lower bound for parameters in the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.parameters_fn": {
        "name": "Parameters Function for First Marginal Bijector",
        "desc": "Function for calculating parameters of the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.parameters_fn_kwargs.conditional_event_shape": {
        "name": "First Marginal Bijector Conditional Event Shape",
        "desc": "Shape of conditional events for the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.parameters_fn_kwargs.dtype": {
        "name": "Data Type for First Marginal Bijector",
        "desc": "Data type of the parameters in the first marginal bijector",
        "drop": True,
    },
    "model_kwargs.marginal_bijectors.0.parameters_fn_kwargs.extrapolation": {
        "name": "Extrapolation for First Marginal Bijector",
        "desc": "Defines whether to allow extrapolation in the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.parameters_fn_kwargs.parameter_shape.0": {
        "name": "First Marginal Bijector Parameter Shape 0",
        "desc": "Defines the parameter shape dimension 0 for the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.parameters_fn_kwargs.parameter_shape.1": {
        "name": "First Marginal Bijector Parameter Shape 1",
        "desc": "Defines the parameter shape dimension 1 for the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.0.parameters_fn_kwargs.polynomial_order": {
        "name": "Polynomial Order for First Marginal Bijector",
        "desc": "Defines the order of the polynomial for computations in the first marginal bijector",
    },
    "model_kwargs.marginal_bijectors.1.bijector": {
        "name": "Second Marginal Bijector",
        "desc": "Specifies the type of second marginal bijector",
    },
    "model_kwargs.marginal_bijectors.1.invert": {
        "name": "Invert Second Marginal Bijector",
        "desc": "Indicates if the second marginal bijector should allow inversion",
    },
    "model_kwargs.marginal_bijectors.1.parameters_fn": {
        "name": "Parameters Function for Second Marginal Bijector",
        "desc": "Function for calculating parameters of the second marginal bijector",
    },
    "model_kwargs.marginal_bijectors.1.parameters_fn_kwargs.conditional_event_shape": {
        "name": "Second Marginal Bijector Conditional Event Shape",
        "desc": "Shape of the conditional events for the second marginal bijector",
    },
    "model_kwargs.marginal_bijectors.1.parameters_fn_kwargs.domain.0": {
        "name": "Second Marginal Bijector Domain Min",
        "desc": "Lower boundary for the domain of the second marginal bijector",
    },
    "model_kwargs.marginal_bijectors.1.parameters_fn_kwargs.domain.1": {
        "name": "Second Marginal Bijector Domain Max",
        "desc": "Upper boundary for the domain of the second marginal bijector",
    },
    "model_kwargs.marginal_bijectors.1.parameters_fn_kwargs.dtype": {
        "name": "Data Type for Second Marginal Bijector",
        "desc": "Data type of the parameters in the second marginal bijector",
        "drop": True,
    },
    "model_kwargs.marginal_bijectors.1.parameters_fn_kwargs.extrapolation": {
        "name": "Extrapolation for Second Marginal Bijector",
        "desc": "Defines whether to allow extrapolation in the second marginal bijector",
    },
    "model_kwargs.marginal_bijectors.1.parameters_fn_kwargs.parameter_shape.0": {
        "name": "Second Marginal Bijector Parameter Shape 0",
        "desc": "Defines the parameter shape dimension 0 for the second marginal bijector",
    },
    "model_kwargs.marginal_bijectors.1.parameters_fn_kwargs.polynomial_order": {
        "name": "Polynomial Order for Second Marginal Bijector",
        "desc": "Defines the order of the polynomial for computations in the second marginal bijector",
    },
    "model_kwargs.num_layers": {
        "name": "Number of Flows",
        "desc": "Total number of layers in the model",
    },
    "model_kwargs.num_parameters": {
        "name": "Number of Parameters",
        "desc": "Total number of parameters in the model",
    },
    "model_kwargs.parameters_constraint_fn_kwargs.allow_flexible_bounds": {
        "name": "Allow Flexible Bounds for Parameters",
        "desc": "Defines if flexible bounds are allowed for the parameters",
    },
    "model_kwargs.parameters_constraint_fn_kwargs.bounds": {
        "name": "Bounds constraints for Parameters",
        "desc": "Defining the type of constraints on the bounds for the parameters",
    },
    "model_kwargs.parameters_constraint_fn_kwargs.high": {
        "name": "Nested Bernstein Bijector Codomain Max",
        "desc": "Defines the upper bound for the corresponding parameters",
    },
    "model_kwargs.parameters_constraint_fn_kwargs.interval_width": {
        "name": "Interval Width for Nested Spline Bijector",
        "desc": "Specifies the interval width for constraint parameters",
    },
    "model_kwargs.parameters_constraint_fn_kwargs.low": {
        "name": "Nested Bernstein Bijector Codomain Min",
        "desc": "Defines the lower bound for the corresponding parameters",
    },
    "model_kwargs.parameters_constraint_fn_kwargs.min_bin_width": {
        "name": "Minimum Bin Width for Nested Spline Bijector",
        "desc": "Defines the minimum bin width for the nested Spline bijector",
    },
    "model_kwargs.parameters_constraint_fn_kwargs.min_slope": {
        "name": "Minimum Slope for Nested Spline Bijector",
        "desc": "Defines the minimum slope constraint for parameters",
    },
    "model_kwargs.parameters_constraint_fn_kwargs.nbins": {
        "name": "Number of Bins for Nested Spline Bijector",
        "desc": "Defines the number of bins used by the nested bijector",
    },
    "model_kwargs.random_permutation_seed": {
        "name": "Random Permutation Seed",
        "desc": "Seed for random permutations in the model",
        "drop": True,
    },
    "model_kwargs.use_invertible_linear_transformations": {
        "name": "Use Invertible Linear Transformations",
        "desc": "Specifies if invertible linear transformations should be employed",
    },
    "model_kwargs.joint_flow_type": {
        "name": "Type of Joint Flow",
        "desc": "Specifies the type of flow used in the model",
    },
    "two_stage_training": {
        "name": "Two-Stage Training",
        "desc": "Indicates if the model should be trained in two stages",
    },
    "model_kwargs.parameters_fn_kwargs.activation": {
        "name": "Activation Function",
        "desc": "Activation function of the used neural network",
    },
    "model_kwargs.parameters_fn_kwargs.batch_norm": {
        "name": "Batch Normalization",
        "desc": "Specifies if batch normalization should be enabled in the used neural network",
    },
    "model_kwargs.parameters_fn_kwargs.conditional": {
        "name": "Conditional",
        "desc": "Indicates if the parameter functions should get the covariates as input",
    },
    "model_kwargs.parameters_fn_kwargs.conditional_event_shape": {
        "name": "Conditional Event Shape",
        "desc": "Shape of the covariate vector",
    },
    "model_kwargs.parameters_fn_kwargs.dropout": {
        "name": "Dropout Rate",
        "desc": "Dropout rate for the neural network",
    },
    "model_kwargs.parameters_fn_kwargs.dtype": {
        "name": "Data Type for X0 Parameters",
        "desc": "Data type of the x0 parameters in the model",
        "drop": True,
    },
    "model_kwargs.parameters_fn_kwargs.hidden_units.0": {
        "name": "First Hidden Units",
        "desc": "Number of units in the first hidden layer of the neural network",
    },
    "model_kwargs.parameters_fn_kwargs.hidden_units.1": {
        "name": "Second Hidden Units",
        "desc": "Number of units in the second hidden layer of the neural network",
    },
    "model_kwargs.parameters_fn_kwargs.hidden_units.2": {
        "name": "Third Hidden Units",
        "desc": "Number of units in the third hidden layer of the neural network",
    },
}
# %%
with open("params/sim/dataset.yaml", "r") as f:
    dataset_kwargs = yaml.safe_load(f)
sim_dataset_names = list(dataset_kwargs["dataset_kwargs"].keys())

with open("params/benchmark/dataset.yaml", "r") as f:
    dataset_kwargs = yaml.safe_load(f)
benchmark_dataset_names = list(dataset_kwargs["dataset_kwargs"].keys())
benchmark_dataset_names, sim_dataset_names

# %% column names
p = re.compile(r"dataset(\d)")


def get_column_index(stage_name):
    if "sim" in stage_name:
        dataset_index = int(p.search(stage_name).group(1))
        dataset_name = sim_dataset_names[dataset_index]
        dataset_type = "sim"
    elif "benchmark" in stage_name:
        dataset_index = int(p.search(stage_name).group(1))
        dataset_name = benchmark_dataset_names[dataset_index]
        dataset_type = "benchmark"
    elif "malnutrition" in stage_name:
        dataset_name = "malnutrition"
        dataset_type = "malnutrition"
    else:
        raise ValueError("undefined dataset")

    conditional = False
    if "-conditional" in stage_name:
        conditional = True

    column_name = ""
    if "hybrid" in stage_name:
        column_name += "H"

    if "coupling_flow" in stage_name:
        column_name += "CF"
    elif "masked_autoregressive_flow" in stage_name:
        column_name += "MAF"
    elif "multivariate_normal" in stage_name:
        column_name += "MVN"
    elif "multivariate_transformation_model" in stage_name:
        column_name += "MCTM"

    if "bernstein_poly" in stage_name:
        column_name += " (B)"
    elif "quadratic_spline" in stage_name:
        column_name += " (S)"

    column_name = column_name if len(column_name) else stage_name

    return pd.MultiIndex.from_arrays(
        [[dataset_type], [column_name], [dataset_name], [conditional]],
        names=["Dataset Type", "Model", "Dataset Name", "Conditional"],
    )


# %% flatten params
dfs = [
    pd.DataFrame(
        data=p.values(),
        index=p.keys(),
        columns=get_column_index(s),
    )
    for p, s in zip(params, stages)
]
# drop params equal for all modesl
params_df = pd.concat(
    dfs,
    axis=1,
)

# %% to tex
tex = params_df.round(3).to_latex(na_rep="--", longtable=True, float_format="%.3f")
with open("hpos_table.tex", "w+") as f:
    f.writelines(tex)

# %% common params
ignore_params = (
    params_df.loc[
        (params_df.nunique(1) == 1) & ~np.any(params_df.isna().values, 1)
    ].iloc[:, -1]
).index.to_list()  # .dropna()

mask = params_df.index.str.contains("dataset_kwargs")
ignore_params += params_df.index[mask].to_list()
ignore_params += list(
    dict(filter(lambda x: x[1].get("drop", False), param_map.items())).keys()
)
ignore_params += ["log-level"]
ignore_params
# %% on tab per model
ds_types = params_df.columns.get_level_values(0).unique()


def get_params_name(x):
    return param_map.get(x, {"name": x})["name"]


used_params = []


def generate_hpo_tables(df, ds_type, agg_level, postfix=None):
    common_params = df

    common_params = (
        common_params.loc[
            (common_params.nunique(1) == 1) & ~np.any(common_params.isna().values, 1)
        ]
        .iloc[:, -1]
        .rename(agg_level)
        .drop(ignore_params, errors="ignore")
    )
    # common_params.index = common_params.index.map(lambda x: x.split(".", 1)[-1])
    # common_params.index = common_params.index.map(lambda x: x[2:] if x[1] == "." else x)

    model_params = df
    model_params = model_params.loc[
        (model_params.nunique(1) != 1) | np.any(model_params.isna().values, 1)
    ].drop(ignore_params, errors="ignore")

    # model_params.index = model_params.index.map(lambda x: x.split(".", 1)[-1])
    # model_params.index = model_params.index.map(lambda x: x[2:] if x[1] == "." else x)
    for name, tab in zip(("common", "model"), (common_params, model_params)):
        if ds_type == "benchmark":
            mask = tab.index.str.contains("domain.")
            tab = tab[~mask]
        tab = tab.dropna(how="all")
        used_params.extend(tab.index.to_list())
        tab.index = tab.index.map(get_params_name)
        tex = tab.round(3).to_latex(
            na_rep="--",
            # longtable=True,
            float_format="%.3f",
            escape=True,
            # column_format="p",
            multicolumn_format="c",
            # caption=f"Hyper parameters for {d} on simulated data"
        )
        file_name = f"hpos_{ds_type}_{name}"
        if postfix:
            file_name += "_" + postfix
        with open(f"{file_name}.tex", "w+") as f:
            f.writelines(tex)


for ds_type in ds_types:
    df = params_df[ds_type]
    all_models = df.columns.get_level_values(0).unique()
    print(ds_type, all_models)
    if ds_type == "malnutrition":
        df.columns = df.columns.get_level_values(0)
        generate_hpo_tables(df, ds_type, ds_type)
    else:
        for model in all_models:
            postfix = model.lower().split(" ")
            if len(postfix) > 1:
                postfix = postfix[0] + "_" + postfix[1][1]
            else:
                postfix = postfix[0]
            d = df[model]
            generate_hpo_tables(d, ds_type, model, postfix)

# %%
params_description_df = (
    pd.DataFrame(param_map).T.loc[used_params].reset_index().drop(columns="drop")
)
params_description_df.columns = ["Parameter", "Table", "Description"]
params_description_df = params_description_df.set_index("Table").round(3).stack()
tex = params_description_df.to_latex(
    na_rep="--",
    longtable=True,
    escape=True,
    float_format="%.3f",
    index_names=False,
    column_format=r"lp{0.6\linewidth}",
)
with open("hpos_description.tex", "w+") as f:
    tex = (
        tex.replace("& Parameter ", "")
        .replace("& Description ", "")
        .replace(r"\cline{1-3}", r"\cline{1-2}")
        .replace(" &  & 0", "Table Name & Parameter / Description")
        .replace(
            "\multicolumn{3}{r}{Continued on next page}",
            "\multicolumn{2}{r}{Continued on next page}",
        )
    )
    f.writelines(tex)
