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


def clip_gradients(x, clip_value):
    """
    A simple function that clips gradients during backpropagation.
    Uses the straight-through estimator approach.
    """
    # During forward pass, this is identity
    # During backward pass, gradients will be clipped
    return x + jax.lax.stop_gradient(
        jnp.clip(x, -clip_value, clip_value) - x
    )


class ScannedRNN(nn.Module):
    """Scanned RNN with gradient clipping for stable training."""
    grad_clip_value: float = 1.0
    
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
        
        # Apply gradient clipping to the output
        if self.grad_clip_value > 0:
            # Clip gradients for the hidden state (new_rnn_state)
            new_rnn_state = tree_map(
                lambda t: clip_gradients(t, self.grad_clip_value), 
                new_rnn_state
            )
            # Clip gradients for the output (y)
            y = clip_gradients(y, self.grad_clip_value)
            
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))
