# %% imports
import tensorflow as tf
from mctm.nn import (
    build_fully_connected_autoregressive_net,
    build_fully_connected_autoregressive_res_net,
    build_fully_connected_net,
    build_fully_connected_res_net,
    build_masked_autoregressive_net,
    build_masked_autoregressive_res_net,
)
from tensorflow import keras as K

# %% fc
fc_net = build_fully_connected_net(
    input_shape=[2],
    output_shape=[2, 3],
    batch_norm=True,
    dropout=0.2,
    hidden_units=[100, 100],
    activation="relu",
)
K.utils.plot_model(
    fc_net,
    expand_nested=True,
    show_shapes=True,
    show_layer_activations=True,
)

# %% MADE
made_net = build_masked_autoregressive_net(
    input_shape=[2],
    output_shape=[2],
    hidden_units=[100, 100],
    activation="relu",
)
made_net.build([2])
K.utils.plot_model(
    made_net._network,
    expand_nested=True,
    show_shapes=True,
    show_layer_activations=True,
)

# %% fc ar net
fc_ar_net = build_fully_connected_autoregressive_net(
    input_shape=[3],
    output_shape=[2, 3],
    batch_norm=True,
    dropout=0.2,
    hidden_units=[100, 100],
    activation="relu",
)
K.utils.plot_model(
    fc_ar_net,
    expand_nested=True,
    show_shapes=True,
    show_layer_activations=True,
)

# %% fc res net
res_net = build_fully_connected_res_net(
    input_shape=[2],
    output_shape=[2, 3],
    res_blocks=3,
    res_block_units=50,
    batch_norm=True,
    dropout=0.2,
    hidden_units=[100, 100],
    activation="relu",
)
K.utils.plot_model(
    res_net,
    to_file="res_net.png",
    expand_nested=True,
    show_shapes=True,
    show_layer_activations=True,
)
# %% fc ar res net net
fc_ar_res_net = build_fully_connected_autoregressive_res_net(
    input_shape=[3],
    output_shape=[2, 3],
    res_blocks=3,
    res_block_units=50,
    batch_norm=True,
    dropout=0.2,
    hidden_units=[100, 100],
    activation="relu",
)
K.utils.plot_model(
    fc_ar_res_net,
    expand_nested=True,
    show_shapes=True,
    show_layer_activations=True,
)

# %% ResMADE
made_res_net = build_masked_autoregressive_res_net(
    input_shape=[3],
    output_shape=[3, 2],
    res_blocks=3,
    res_block_units=50,
    hidden_units=[100, 100],
    activation="relu",
)
K.utils.plot_model(
    made_res_net,
    expand_nested=True,
    show_shapes=True,
    show_layer_activations=True,
)


# %% testing
params = dict(parameter_shape=(3, 2), hidden_units=[100, 100], activation="relu")
x = tf.ones((100, 3))
model_made, _ = get_masked_autoregressive_network_fn(**params)

model_made()(x).shape

model_ar, _ = get_autoregressive_res_net_parameter_network_fn(
    **params,
    batch_norm=True,
    dropout=0.2,
    res_blocks=2,
    conditional=True,
    conditional_event_shape=[3],
)
model = model_ar()
model(x).shape
model.summary()
K.utils.plot_model(
    model,
    to_file="model.png",
    expand_nested=True,
    show_shapes=True,
    show_layer_activations=True,
    show_layer_names=True,
)

# %% masks
from matplotlib import pyplot as plt
from tensorflow_probability import bijectors as tfb

mdam = tfb.masked_autoregressive._make_dense_autoregressive_masks
masks = mdam(params=2, event_size=3, hidden_units=[5, 5])
fig, ax = plt.subplots(len(masks))
for a, m in zip(ax, masks):
    a.imshow(m, cmap="binary")
fig.tight_layout()
