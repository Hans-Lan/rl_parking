from typing import Dict, List, Optional, Sequence, Tuple

import gym.spaces
import numpy as np
import tensorflow as tf

from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType
from ray.rllib.models.tf.misc import normc_initializer


class AppendLogStd(tf.keras.layers.Layer):
    def __init__(self, num_outputs: int, init_log_std: float = -0.5):
        super(AppendLogStd, self).__init__()
        self.log_std = tf.Variable(
            initial_value=tf.constant(init_log_std, dtype=tf.float32, shape=(num_outputs,)),
            trainable=True, name="log_std"
        )

    def call(self, x):
        log_std = tf.broadcast_to(self.log_std, tf.shape(x))
        return tf.concat([x, log_std], axis=-1)

def create_fc_policy_branch(
    hiddens: List[int], activation,
    input_layer,
    no_final_linear: bool, num_outputs: int,
    free_log_std: bool, init_log_std: float):

    # Last hidden layer output (before logits outputs).
    last_layer = input_layer
    input_size = input_layer.shape[1]
    logits_out = None
    i = 1
    for size in hiddens[:-1]:
        last_layer = tf.keras.layers.Dense(
            size,
            name=f"fc_{i}",
            activation=activation,
            kernel_initializer=normc_initializer(1.0))(last_layer)
        i += 1

    if no_final_linear and num_outputs:
        logits_out = tf.keras.layers.Dense(
            num_outputs,
            name="fc_out",
            activation=activation,
            kernel_initializer=normc_initializer(1.0))(last_layer)
    else:
        if len(hiddens) > 0:
            last_layer = tf.keras.layers.Dense(
                hiddens[-1],
                name=f"fc_{i}",
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
        if num_outputs:
            logits_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(last_layer)
        else:
            num_outputs = ([input_size] + hiddens[-1:])[-1]
    if free_log_std and logits_out is not None:
        logits_out = AppendLogStd(num_outputs, init_log_std)(logits_out)
    return logits_out, last_layer, num_outputs

def create_fc_value_branch(
    hiddens: List[int], activation: str,
    input_layer, last_layer,
    vf_share_layers: bool = False, name: str = "value",
    output_activation: str = None):
    last_vf_layer = None
    if not vf_share_layers:
        last_vf_layer = input_layer
        i = 1
        for size in hiddens:
            last_vf_layer = tf.keras.layers.Dense(
                size,
                name=f"fc_{name}_{i}",
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_vf_layer)
            i += 1
    value_out = tf.keras.layers.Dense(
        1,
        name=f"{name}_out",
        activation=output_activation,
        kernel_initializer=normc_initializer(0.01))(
            last_vf_layer if last_vf_layer is not None else last_layer)
    return value_out

def extract_config_from_default(cls: type):
    import inspect
    from ray.rllib.models.catalog import MODEL_DEFAULTS
    sig = inspect.signature(cls.__init__)
    d = {
        k: MODEL_DEFAULTS[k]
        for k, p in sig.parameters.items()
        if p.kind == inspect.Parameter.KEYWORD_ONLY and k in MODEL_DEFAULTS
    }
    return d

class Keras_FullyConnectedNetwork(tf.keras.Model):
    """Generic fully connected network implemented in tf Keras."""

    def __init__(
            self,
            input_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: Optional[int] = None,
            *,
            name: str = "",
            fcnet_hiddens: Optional[Sequence[int]],
            fcnet_activation: Optional[str],
            post_fcnet_hiddens: Optional[Sequence[int]],
            post_fcnet_activation: Optional[str],
            no_final_linear: bool,
            vf_share_layers: bool,
            free_log_std: bool,
            init_log_std: Optional[float] = None,
            **kwargs,
    ):
        super().__init__(name=name)
        hiddens = list(fcnet_hiddens or ()) + \
            list(post_fcnet_hiddens or ())
        activation = fcnet_activation
        if not fcnet_hiddens:
            activation = post_fcnet_activation
        activation = get_activation_fn(activation)

        if free_log_std:
            assert num_outputs % 2 == 0 and init_log_std is not None
            num_outputs = num_outputs // 2

        # We are using obs_flat, so take the flattened shape as input.
        inputs = tf.keras.layers.Input(
            shape=(int(np.product(input_space.shape)), ), name="observations")

        logits_out, last_layer, self.num_outputs = create_fc_policy_branch(
            hiddens, activation, inputs, no_final_linear, num_outputs, free_log_std, init_log_std)
        policy_out = logits_out if logits_out is not None else last_layer
        value_out = create_fc_value_branch(hiddens, activation, inputs, last_layer, vf_share_layers, "value")

        self.base_model = tf.keras.Model(
            inputs, [policy_out, value_out])

    def call(self, input_dict: SampleBatch) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        model_out, value_out = self.base_model(input_dict[SampleBatch.OBS])
        extra_outs = {SampleBatch.VF_PREDS: tf.reshape(value_out, [-1])}
        return model_out, [], extra_outs
