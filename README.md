[![img](https://img.shields.io/github/contributors/MArpogaus/hybrid-flows.svg?style=flat-square)](https://github.com/MArpogaus/hybrid-flows/graphs/contributors) [![img](https://img.shields.io/github/forks/MArpogaus/hybrid-flows.svg?style=flat-square)](https://github.com/MArpogaus/hybrid-flows/network/members) [![img](https://img.shields.io/github/stars/MArpogaus/hybrid-flows.svg?style=flat-square)](https://github.com/MArpogaus/hybrid-flows/stargazers) [![img](https://img.shields.io/github/issues/MArpogaus/hybrid-flows.svg?style=flat-square)](https://github.com/MArpogaus/hybrid-flows/issues) [![img](https://img.shields.io/github/license/MArpogaus/hybrid-flows.svg?style=flat-square)](https://github.com/MArpogaus/hybrid-flows/blob/main/LICENSE) [![img](https://img.shields.io/github/actions/workflow/status/MArpogaus/hybrid-flows/test.yaml.svg?label=test&style=flat-square)](https://github.com/MArpogaus/hybrid-flows/actions/workflows/test.yaml) [![img](https://img.shields.io/github/actions/workflow/status/MArpogaus/hybrid-flows/release.yaml.svg?label=release&style=flat-square)](https://github.com/MArpogaus/hybrid-flows/actions/workflows/release.yaml) [![img](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg?logo=pre-commit&style=flat-square)](https://github.com/MArpogaus/hybrid-flows/blob/main/.pre-commit-config.yaml) [![img](https://img.shields.io/badge/arXiv-2505.14164-B31B1B.svg?style=flat-square)](https://arxiv.org/abs/2505.14164) [![img](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://linkedin.com/in/MArpogaus)


# Hybrid Bernstein Normalizing Flows for Flexible Multivariate Density Regression with Interpretable Marginals

This repository provides the code and parameters to reproduce the results presented in the paper:

> M. Arpogaus, T. Kneib, T. Nagler, und D. Rügamer, „Hybrid Bernstein Normalizing Flows for Flexible Multivariate Density Regression with Interpretable Marginals“, 20. Mai 2025, arXiv: arXiv:2505.14164. https://doi.org/10.48550/arXiv.2505.14164.

1.  [About The Project](#orged23a43)
2.  [Abstract](#org4dc22ef)
3.  [Getting Started](#orgd77ce9a)
    1.  [Prerequisites](#org8dbb0ca)
    2.  [Installation](#org808e2b3)
    3.  [Reproduce Experiments](#org0f924c8)
4.  [Usage of the Density Regression Models](#orgddf71c5)
    1.  [Unconditional Multivariate Normal](#org3457d11)
    2.  [Masked Autoregressive Flow](#org59896a3)
    3.  [Multivariate Conditional Transformation Model](#orgd549016)
    4.  [Conditional Hybrid Masked Autoregressive Flow using Quadratic Splines](#org3ea6b68)
5.  [Contributing](#org89a3567)
6.  [License](#org52c8cb8)
7.  [Contact](#org1e7f2af)
8.  [Acknowledgments](#org1641752)


<a id="orged23a43"></a>

## About The Project

This repository provides the implementation of Hybrid Masked Autoregressive Flows (HMAFs), a novel hybrid approach that combines the strengths of Multivariate Conditional Transformation Models (MCTMs) and autoregressive Normalizing Flows (NFs). HMAFs allow for flexible modeling of the dependency structure in multivariate data while retaining the interpretability of marginal distributions. The code includes experiments on simulated and real-world datasets, comparing HMAFs with MCTMs, MAFs, Coupling Flows and other benchmark models. A detailed description of all hyperparameters is available in the supplementary material of the paper.


<a id="org4dc22ef"></a>

## Abstract

Density regression models allow a comprehensive understanding of data by modeling the complete conditional probability distribution. While flexible estimation approaches such as normalizing flows (NF) work particularly well in multiple dimensions, interpreting the input-output relationship of such models is often difficult, due to the black-box character of deep learning models. In contrast, existing statistical methods for multivariate outcomes such as multivariate conditional transformation models (MCTM) are restricted in flexibility and are often not expressive enough to represent complex multivariate probability distributions. In this paper, we combine MCTM with state-of-the-art and autoregressive NF to leverage the transparency of MCTM for modeling interpretable feature effects on the marginal distributions in the first step and the flexibility of neural-network-based NF techniques to account for complex and non-linear relationships in the joint data distribution. We demonstrate our method's versatility in various numerical experiments and compare it with MCTM and other NF models on both simulated and real-world data.


<a id="orgd77ce9a"></a>

## Getting Started

To get a local copy up and running follow these simple example steps.


<a id="org8dbb0ca"></a>

### Prerequisites

Amon others, this project uses the following python packages:

-   [`dvc`](https://github.com/iterative/dvc)
-   [`mlflow`](https://github.com/mlflow/mlflow)
-   [`seaborn`](https://github.com/seaborn/seaborn)
-   [`tensorflow`](https://github.com/tensorflow/tensorflow)
-   [`tensorflow_probability`](https://github.com/tensorflow/probability)

And my implementation of Bernstein polynomial bijectors available from [this repro](https://github.com/MArpogaus/TensorFlow-Probability-Bernstein-Polynomial-Bijector).

All the dependencies are defined in the [`pyproject.toml`](pyproject.toml) file of this project and are automatically met when installed using pip or uv.


<a id="org808e2b3"></a>

### Installation

1.  Clone the repository

    ```shell
    git clone https://github.com/anonymized/hybrid-flows.git
    cd hybrid-flows
    ```

2.  Install the required packages

    I'd recommend [uv](https://github.com/astral-sh/uv) to setup this project in a [venv](https://docs.astral.sh/uv/pip/environments).

    ```shell
    uv venv --python 3.11
    source .venv/bin/activate
    uv sync --extra train
    ```

    Alternative can used the provided conda environment (`conda_env.yaml`) to install all dependencies:

    ```shell
    conda create -n hybrid_flows -f conda_env.yaml
    conda activate hybrid_flows
    ```

    **Note:** Additional to `train` other `optional-dependenies` are defined in the [`pyproject.toml`](pyproject.toml) file, for cuda-support, optuna, testing, etc.


<a id="org0f924c8"></a>

### Reproduce Experiments

The repository provides several experiments implemented as [DVC](https://dvc.org/) stages in the `experiments` directory. All configuration files for the different models can be found in the `params` directory. The experiments cover training and evaluation of several models on different datasets.

A detailed explanation of the model parameters can be found in the supplementary material of the paper.

All experiments can be executed by simply running the following command in the `experiments` directory after installing all required dependencies:

```shell
dvc repro
```

The individual stages are described in the following.

1.  Benchmark Datasets

    These stages train and evaluate models on five real-world benchmark datasets: BSDS300, GAS, HEPMASS, MINIBOONE, and POWER ([dataset configuration](experiments/params/benchmark/dataset.yaml)). The model configurations are defined in [experiments/params/benchmark](experiments/params/benchmark).

    -   `train-benchmark`: Trains two models ([dvc stage definition](experiments/dvc.yaml)) on the benchmark datasets:
        -   MAF with RQS ([bsds300](experiments/params/benchmark/bsds300/unconditional_masked_autoregressive_flow_quadratic_spline.yaml), [gas](experiments/params/benchmark/gas/unconditional_masked_autoregressive_flow_quadratic_spline.yaml), [hepmass](experiments/params/benchmark/hepmass/unconditional_masked_autoregressive_flow_quadratic_spline.yaml), [minibone](experiments/params/benchmark/miniboone/unconditional_masked_autoregressive_flow_quadratic_spline.yaml), [power](experiments/params/benchmark/power/unconditional_masked_autoregressive_flow_quadratic_spline.yaml))
        -   HMAF with RQS ([bsds300](experiments/params/benchmark/bsds300/unconditional_hybrid_masked_autoregressive_flow_quadratic_spline.yaml), [gas](experiments/params/benchmark/gas/unconditional_hybrid_masked_autoregressive_flow_quadratic_spline.yaml), [hepmass](experiments/params/benchmark/hepmass/unconditional_hybrid_masked_autoregressive_flow_quadratic_spline.yaml), [minibone](experiments/params/benchmark/miniboone/unconditional_hybrid_masked_autoregressive_flow_quadratic_spline.yaml), [power](experiments/params/benchmark/power/unconditional_hybrid_masked_autoregressive_flow_quadratic_spline.yaml))

    -   `eval-benchmark`: Evaluates the trained models. ([dvc stage definition](experiments/dvc.yaml)). Generates various diagnostic plots, such as Q-Q plots, for analysis and comparison. Evaluation metrics are logged in `evaluation_metrics.yaml`.

2.  Simulated Datasets

    The following stages train and evaluate different models on two simulated datasets: **moons** and **circles** ([dataset configuration](experiments/params/sim/dataset.yaml)). The hyper parameters are defined in [experiments/params/sim](experiments/params/sim).

    -   `train-sim`: Trains a range of models ([dvc stage definition](experiments/dvc.yaml)) on both simulated datasets. The model configurations are defined in [params/sim/moons](experiments/params/sim/moons) and [params/sim/circles](experiments/params/sim/circles). Each dataset has 16,384 data points, with 25 % reserved for validation. A binary feature, `x`, is included for conditional density estimation, indicating spatial location. The following models are trained:
        -   Multivariate Normal (MVN) ([unconditional](experiments/params/sim/circles/unconditional_multivariate_normal.yaml)/[conditional](experiments/params/sim/circles/conditional_multivariate_normal.yaml))
        -   Multivariate Conditional Transformation Model (MCTM) ([unconditional](experiments/params/sim/circles/unconditional_multivariate_transformation_model.yaml)/[conditional](experiments/params/sim/circles/conditional_multivariate_transformation_model.yaml))
        -   Coupling Flow (CF) with spline ([unconditional](experiments/params/sim/circles/unconditional_coupling_flow_quadratic_spline.yaml)/[conditional](experiments/params/sim/circles/conditional_coupling_flow_quadratic_spline.yaml)) and Bernstein polynomial ([unconditional](experiments/params/sim/circles/unconditional_coupling_flow_bernstein_poly.yaml)/[conditional](experiments/params/sim/circles/conditional_coupling_flow_bernstein_poly.yaml)) transformations.
        -   Masked Autoregressive Flow (MAF) with spline ([unconditional](experiments/params/sim/circles/unconditional_masked_autoregressive_flow_quadratic_spline.yaml)/[conditional](experiments/params/sim/circles/conditional_masked_autoregressive_flow_quadratic_spline.yaml)) and Bernstein polynomial ([unconditional](experiments/params/sim/circles/unconditional_masked_autoregressive_flow_bernstein_poly.yaml)/[conditional](experiments/params/sim/circles/conditional_masked_autoregressive_flow_bernstein_poly.yaml)) transformations.
        -   Hybrid Coupling Flow (HCF) with spline ([unconditional](experiments/params/sim/circles/unconditional_hybrid_coupling_flow_quadratic_spline.yaml)/[conditional](experiments/params/sim/circles/conditional_hybrid_coupling_flow_quadratic_spline.yaml)) and Bernstein polynomial ([unconditional](experiments/params/sim/circles/unconditional_hybrid_coupling_flow_bernstein_poly.yaml)/[conditional](experiments/params/sim/circles/conditional_hybrid_coupling_flow_bernstein_poly.yaml)) transformations.

    -   `eval-sim`: Evaluates the trained models ([dvc stage definition](experiments/dvc.yaml)) on the simulated data. Evaluation metrics and visualizations, such as contour plots, Q-Q plots and transformed data distributions, are generated. Metrics are logged in `evaluation_metrics.yaml`.

3.  Malnutrition Dataset

    These stages concern a real-world dataset on childhood malnutrition in India ([dataset configuration](experiments/params/malnutrition/dataset.yaml)). Model parameters can be found in [experiments/params/malnutrition](experiments/params/malnutrition).

    -   `train-malnutrition`: Trains three models ([dvc stage definition](experiments/dvc.yaml)) to estimate the joint distribution of anthropometric indices (stunting, wasting, underweight) conditional on the child’s age.
        -   MCTM ([model configuration](experiments/params/malnutrition/conditional_multivariate_transformation_model.yaml))
        -   HMAF with Bernstein polynomials ([model configuration](experiments/params/malnutrition/conditional_hybrid_masked_autoregressive_flow_bernstein_poly.yaml))
        -   HMAF with quadratic splines ([model configuration](experiments/params/malnutrition/conditional_hybrid_masked_autoregressive_flow_quadratic_spline.yaml))
    -   `eval-malnutrition`: Evaluates the models' performance ([dvc stage definition](experiments/dvc.yaml)) using reliability diagrams, Q-Q plots, analysis of marginal distributions and feature effects.


<a id="orgddf71c5"></a>

## Usage of the Density Regression Models

The `hybriod_flows` python packages implements TensorFlow models for density regression using various methods. Here are some basic examples to get you started. Please also review the tests defined in <test/test_models.py>, and the model parameters used for the experiments described above if you require more examples.


<a id="org3457d11"></a>

### Unconditional Multivariate Normal

```python
from hybrid_flows.models import DensityRegressionModel

# Define model parameters
model_parameters = {
    "distribution": "multivariate_normal",
    "parameters_fn_kwargs": {"conditional": False},
    "dims": 2,
}
# Initialize and compile the model
model = DensityRegressionModel(**model_parameters)
model.compile(optimizer="adam", loss=lambda y, p_y: -p_y.log_prob(y))

# load data
x, y = ...

# Fit the model (replace with your actual data)
model.fit(x=x, y=y, epochs=1)

# Access the distribution
dist = model(x)
```


<a id="org59896a3"></a>

### Masked Autoregressive Flow

```python
import tensorflow as tf
from hybrid_flows.models import DensityRegressionModel

# Define model parameters
DATA_DIMS=10
model_parameters = {
    "distribution": "masked_autoregressive_flow",
    "model_kwargs": {"parameters_fn_kwargs": {"conditional": False}},
    "dims": DATA_DIMS,
    "num_layers": 2,
    "num_parameters": 8,
    "nested_bijectors": [
        {
            "bijector": "Scale",
            "parameters_constraint_fn": tf.math.softplus,
            "parameters_slice_size": 1,
        },
        {
            "bijector": "Shift",
            "parameters_slice_size": 1,
        },
        {
            "parameters_constraint_fn_kwargs": {
                "allow_flexible_bounds": False,
                "bounds": "linear",
                "high": -4,
                "low": 4,
            },
            "bijector": "BernsteinPolynomial",
            "bijector_kwargs": {
                "domain": [0, 1],
                "extrapolation": False,
            },
            "invert": True,
            "parameters_slice_size": 6,
        },
    ],
    "parameters_fn_kwargs": {
        "hidden_units": [16] * 4,
        "activation": "relu",
        "conditional": True,
        "conditional_event_shape": DATA_DIMS,
    },
}
# Initialize and compile the model
model = DensityRegressionModel(**model_parameters)
model.compile(optimizer="adam", loss=lambda y, p_y: -p_y.log_prob(y))

# load data
x, y = ...

# Fit the model (replace with your actual data)
model.fit(x=x, y=y, epochs=1)

# Access the distribution
dist = model(x)
```


<a id="orgd549016"></a>

### Multivariate Conditional Transformation Model

```python
import tensorflow as tf

from hybrid_flows.models import HybridDensityRegressionModel

# Define model parameters
DATA_DIMS = 3
model_parameters = {
    "marginal_bijectors": [
        {
            "bijector": "BernsteinPolynomial",
            "bijector_kwargs": {
                "domain": (-4, 4),
                "extrapolation": True,
            },
            "parameters_fn": "parameter_vector",
            "parameters_fn_kwargs": {
                "parameter_shape": [DATA_DIMS, 10],
                "dtype": "float32",
            },
            "parameters_constraint_fn": "hybrid_flows.activations.get_thetas_constrain_fn",  # noqa: E501
            "parameters_constraint_fn_kwargs": {
                "low": -4,
                "high": 4,
                "bounds": "smooth",
                "allow_flexible_bounds": True,
            },
        },
        {
            "bijector": "Shift",
            "parameters_fn": "bernstein_polynomial",  # "parameter_vector",
            "parameters_fn_kwargs": {
                "parameter_shape": [DATA_DIMS],
                "dtype": "float",
                "polynomial_order": 6,
                "conditional_event_shape": DATA_DIMS,
                "extrapolation": True,
            },
        },
    ],
    "joint_bijectors": [
        {
            "bijector": "ScaleMatvecLinearOperator",
            "parameters_fn": "bernstein_polynomial",
            "parameters_fn_kwargs": {
                "parameter_shape": [sum(range(DATA_DIMS))],
                "dtype": "float",
                "polynomial_order": 6,
                "conditional_event_shape": DATA_DIMS,
                "domain": (-1, 1),
                "extrapolation": True,
                "initializer": tf.ones,
            },
            "parameters_constraint_fn": "hybrid_flows.activations.lambda_parameters_constraint_fn",  # noqa: E501
        }
    ],
    "dims": DATA_DIMS
}

# Initialize and compile the model
model = HybridDensityRegressionModel(**model_parameters)
model.compile(optimizer="adam", loss=lambda y, p_y: -p_y.log_prob(y))

# load data
x, y = ...

# Fit the model (replace with your actual data and parameters)
model.fit(x=x, y=y, epochs=1)

# Get the joint distribution
joint_dist = model(x)

# Get the marginal distributions
marginal_dist = model.marginal_distribution(x)
```


<a id="org3ea6b68"></a>

### Conditional Hybrid Masked Autoregressive Flow using Quadratic Splines

```python
import tensorflow as tf

from hybrid_flows.models import HybridDensityRegressionModel

# Define model parameters
DATA_DIMS = 3
model_parameters = {
    "marginal_bijectors": [
        {
            "bijector": "BernsteinPolynomial",
            "invert": True,
            "bijector_kwargs": {"domain": [0, 1], "extrapolation": False},
            "parameters_constraint_fn": "hybrid_flows.activations.get_thetas_constrain_fn",
            "parameters_constraint_fn_kwargs": {
                "allow_flexible_bounds": False,
                "bounds": "linear",
                "high": 5,
                "low": -5,
            },
            "parameters_fn": "bernstein_polynomial",
            "parameters_fn_kwargs": {
                "dtype": "float32",
                "extrapolation": True,
                "conditional_event_shape": 1,
                "polynomial_order": 1,
                "parameter_shape": [2, 300],
            },
        }
    ],
    "joint_bijectors": [
        {
            "bijector": "RealNVP",
            "bijector_kwargs": {"num_masked": 1},
            "nested_bijector": {
                "bijector": "RationalQuadraticSpline",
                "bijector_kwargs": {"range_min": -5},
                "parameters_constraint_fn_kwargs": {
                    "interval_width": 10,
                    "min_slope": 0.001,
                    "min_bin_width": 0.001,
                    "nbins": 32,
                },
                "parametrized_by_parent": True,
            },
            "parameters_fn": "fully_connected_network",
            "parameters_fn_kwargs": {
                "activation": "relu",
                "batch_norm": False,
                "dropout": False,
                "hidden_units": [128, 128, 128],
                "dtype": "float32",
                "input_shape": [1],
                "parameter_shape": [1, 95],
                "conditional": True,
                "conditional_event_shape": 1,
            },
        }
    ],
    "dims": DATA_DIMS,
}

# Initialize and compile the model
model = HybridDensityRegressionModel(**model_parameters)
model.compile(optimizer="adam", loss=lambda y, p_y: -p_y.log_prob(y))

# load data
x, y = ...

# Fit the model (replace with your actual data and parameters)
model.fit(x=x, y=y, epochs=1)

# Get the joint distribution
joint_dist = model(x)

# Get the marginal distributions
marginal_dist = model.marginal_distribution(x)
```


<a id="org89a3567"></a>

## Contributing

Any Contributions are greatly appreciated! If you have a question, an issue or would like to contribute, please read our [contributing guidelines](CONTRIBUTING.md).


<a id="org52c8cb8"></a>

## License

Distributed under the [Apache License 2.0](experiments/LICENSE)


<a id="org1e7f2af"></a>

## Contact

[Marcel Arpogaus](https://github.com/MArpogaus/) - [znepry.necbtnhf@tznvy.pbz](mailto:znepry.necbtnhf@tznvy.pbz) (encrypted with [ROT13](https://rot13.com/))

Project Link: <https://github.com/MArpogaus/hybrid-flows>


<a id="org1641752"></a>

## Acknowledgments

This research was funded by the Carl-Zeiss-Stiftung in the project ”DeepCarbPlanner” (grant no. P2021-08-007). We thank the [DACHS](https://wiki.bwhpc.de/e/DACHS) data analysis cluster, hosted at Hochschule Esslingen and co-funded by the MWK within the DFG’s ,,Großgeräte der Länder” program, for providing the computational resources necessary for this research.
