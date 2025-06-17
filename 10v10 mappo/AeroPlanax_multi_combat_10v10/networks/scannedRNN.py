import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from flax.linen.initializers import constant, orthogonal
import functools
from typing import Sequence, NamedTuple, Tuple, Optional, Union, Any, Dict
from jax.tree_util import tree_map
import optax


def clip_gradients(grads, clip_value):
    """
    Proper gradient clipping function that clips gradients during backpropagation.
    """
    return jax.tree_map(lambda g: jnp.clip(g, -clip_value, clip_value), grads)


class ScannedRNN(nn.Module):
    """Scanned RNN with gradient clipping for stable training."""
    grad_clip_value: float = 0.5
    
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        
        # Use standard GRUCell
        cell = nn.GRUCell(features=ins.shape[1])
        new_rnn_state, y = cell(rnn_state, ins)
        
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

    def get_optimizer(self, learning_rate: float):
        """Returns an optimizer with gradient clipping."""
        return optax.chain(
            optax.clip_by_global_norm(self.grad_clip_value),
            optax.adam(learning_rate)
        )
