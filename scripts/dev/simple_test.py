# %% imports
from pprint import pprint

import dvc.api
import tensorflow as tf
from mctm.models import DensityRegressionModel
from tensorflow_probability import distributions as tfd

# %% params
stage_name = "unconditional@coupling_flow-moons"
# stage_name = "unconditional@elementwise_flow-moons"
stage, distribution_dataset = stage_name.split("@")

dims = 2

distribution, dataset = distribution_dataset.split("-")

params = dvc.api.params_show(stages=stage_name)

dataset_kwds = params["datasets"][dataset]
model_kwds = params[stage + "_distributions"][distribution][dataset]
fit_kwds = model_kwds.pop("fit_kwds")

pprint(model_kwds)

# %% params
model_kwds["distribution_kwds"].update(low=None, high=None)
# %% model
model = DensityRegressionModel(dims=dims, distribution=distribution, **model_kwds)
model.build([None, dims])
model.summary()
# %% log_prob
dist = model(None)
# dist.bijector.bijector.thetas
dist
# %% plot initial dist
y = tf.linspace(-2.0, 2.0, 200)
inp = tf.repeat(y[..., None], dims, axis=1)
inp.shape
p = dist.prob(inp)
pn = tfd.Normal(0, 1).prob(y)

# %%
from matplotlib import pyplot as plt

fig = plt.figure()
plt.plot(y, p)
plt.plot(y, pn)
plt.show()

# %%
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# A common choice for a normalizing flow is to use a Gaussian for the base
# distribution. (However, any continuous distribution would work.) E.g.,
nvp = tfd.TransformedDistribution(
    distribution=tfd.Normal(0.0, 1),
    bijector=tfb.RealNVP(
        num_masked=2,
        shift_and_log_scale_fn=tfb.real_nvp_default_template(hidden_layers=[512, 512]),
    ),
)

x = nvp.sample()
nvp.log_prob(x)
nvp.log_prob([0.0, 0.0, 0.0])
