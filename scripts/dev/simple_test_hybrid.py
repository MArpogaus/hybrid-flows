# %% imports
from pprint import pprint

import dvc.api
import tensorflow as tf
from mctm.data.sklearn_datasets import get_dataset
from mctm.models import HybridDenistyRegressionModel
from mctm.utils.visualisation import plot_flow, plot_samples
from tensorflow import keras as K

# %% params
stage_name = "unconditional_hybrid@coupling_flow-moons"
stage_name = (
    "unconditional_hybrid_pre_trained@masked_autoregressive_flow_first_dim_masked-moons"
)
# stage_name = "unconditional@elementwise_flow-moons"
stage, distribution_dataset = stage_name.split("@")

dims = 2

distribution, dataset = distribution_dataset.split("-")

params = dvc.api.params_show(stages=stage_name)

dataset_kwds = params["datasets"][dataset]
model_kwds = params[stage + "_distributions"][distribution][dataset]
fit_kwds = model_kwds.pop("fit_kwds")
model_kwds["base_checkpoint_path"] = None
pprint(model_kwds)

# %% alter params
# model_kwds["distribution_kwds"]["num_layers"] = 1

# %% data
(y, x), dims = get_dataset("moons", n_samples=2000, scale=(0, 1))

# %% model
model = HybridDenistyRegressionModel(dims=dims, distribution=distribution, **model_kwds)
model.build([None, dims])
model.summary()


# %% train
# @tf.function
def loss(y, dist):
    return (-dist.log_prob(y),)


model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-3), loss=loss)
model.build(x.shape)
model.summary()
K.utils.plot_model(model, expand_nested=True, show_shapes=True)

model.fit(x=x, y=y)

# %% dbg tracing


@tf.function
def trace(y):
    dist = model(None)
    return dist.log_prob(y)


trace(y=tf.convert_to_tensor(y, dtype=model.dtype))

# %% log_prob
dist = model(None)
# dist.bijector.bijector.thetas
dist.log_prob(y)

# %% plots
fig = plot_samples(model(x), y, seed=1)

# %% plot flow
fig1, fig2, fig3 = plot_flow(model(x), x, y, seed=1)

# %% dbg
dist = model(x)
base_dist = dist.distribution.distribution

y = tf.convert_to_tensor(y, dtype=tf.float32)

# inverse flow
z1 = dist.bijector.inverse(y)
z2 = base_dist.bijector.inverse(z1)

z2 = base_dist.distribution.sample(y.shape, seed=1)
z1 = base_dist.bijector.forward(z2)
yy = dist.bijector.forward(z1)

# %% junk
input1 = K.Input(shape=(37,), name="input1")
input2 = K.Input(shape=(37,), name="input2")
x1 = K.layers.Dense(32, activation="relu")(input1)
x2 = K.layers.Dense(32, activation="relu")(input2)
outputs = K.layers.Add()([x1, x2])
model = K.Model(inputs=[input1, input2], outputs=outputs)
model.summary()
model(input1=1, input2=3)
