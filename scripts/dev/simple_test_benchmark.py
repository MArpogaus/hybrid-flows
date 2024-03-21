# %% imports
from pprint import pprint

import dvc.api
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mctm.data.benchmark import get_dataset
from mctm.models import DensityRegressionModel

# %% params
stage_name = "unconditional_benchmark@masked_autoregressive_flow-power"
# stage_name = "unconditional@elementwise_flow-moons"
stage, distribution_dataset = stage_name.split("@")
distribution, dataset = distribution_dataset.split("-")
dims = 2

params = dvc.api.params_show(stages=stage_name)

dataset_kwds = params["benchmark_datasets"][dataset]
model_kwds = params[stage + "_distributions"][distribution][dataset]
model_kwds["distribution_kwds"].update(dataset_kwds)
model_kwds["distribution_kwds"].update(low=0, high=1)
fit_kwds = model_kwds.pop("fit_kwds")

pprint(model_kwds)

# %% dataset
(train_data, val_data, _), dims = get_dataset(dataset, test_mode=True)
train_data.min(), train_data.max()

# %% model
model = DensityRegressionModel(dims=dims, distribution=distribution, **model_kwds)
model.build([None, dims])
model.summary()

# %% log_prob
dist = model(None)
# dist.bijector.bijector.thetas
dist

# %% apply scale and shift bijectors
train_data_scaled = train_data
for bj in dist.bijector.bijectors[:2]:
    train_data_scaled = bj.inverse(train_data_scaled).numpy()
train_data_scaled.min(), train_data_scaled.max()

# %% apply manual scale and shift
train_data_scaled = (train_data + dataset_kwds["shift"]) * dataset_kwds["scale"]
train_data_scaled.min(), train_data_scaled.max()

# %% apply manual scale and shift chain
from tensorflow_probability import bijectors as tfb

chain = tfb.Chain([tfb.Shift(dataset_kwds["shift"]), tfb.Scale(dataset_kwds["scale"])])
chain = tfb.Chain(dist.bijector.bijectors[:2])
train_data_scaled = chain.inverse(train_data).numpy()
train_data_scaled.min(), train_data_scaled.max()

# %% full chain data
train_data_scaled = dist.bijector.inverse(train_data).numpy()
train_data_scaled.min(), train_data_scaled.max()

# %% MAF
maf = dist.bijector.bijectors[-1]
arnet = maf._bijector_fn.__closure__[1].cell_contents.func
y = tf.ones_like(train_data[0])
thetas = []
arnet_out = []
ys = []
for _ in range(dims):
    bj = maf._bijector_fn(y)
    y = bj.inverse(tf.identity(train_data[0]))
    ys.append(y)
    arnet_out.append(arnet(y))
    thetas.append(bj.bijector.thetas)
# %%
ts = np.stack(ys, 0)
plt.figure()
plt.plot(ts)
plt.show()

# %%
ts = np.stack(thetas, 0)
fig = plt.figure()
plt.plot(ts[:, 0, 5])
plt.show()

# %%
fig = plt.figure()
plt.plot(thetas[0].numpy().T)
plt.show()
