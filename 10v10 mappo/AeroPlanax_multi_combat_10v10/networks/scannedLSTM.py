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


class ScannedLSTM(nn.Module):
    """Scanned LSTM with gradient clipping for stable training."""
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
        lstm_state = carry
        ins, resets = x
        
        # Get dimensions from the input
        batch_size, hidden_size = ins.shape
        
        # Reset LSTM state where needed
        # LSTM state is a tuple of (c, h)
        c, h = lstm_state
        c = jnp.where(
            resets[:, np.newaxis],
            jnp.zeros_like(c),
            c,
        )
        h = jnp.where(
            resets[:, np.newaxis],
            jnp.zeros_like(h),
            h,
        )
        lstm_state = (c, h)
        
        # Use standard LSTMCell
        cell = nn.LSTMCell(features=hidden_size)
        new_lstm_state, y = cell(lstm_state, ins)
        
        # Apply gradient clipping to the output
        if self.grad_clip_value > 0:
            # Clip gradients for the hidden state (new_lstm_state)
            new_lstm_state = tree_map(
                lambda t: clip_gradients(t, self.grad_clip_value), 
                new_lstm_state
            )
            # Clip gradients for the output (y)
            y = clip_gradients(y, self.grad_clip_value)
            
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Initialize both the cell state (c) and hidden state (h)
        return (
            jnp.zeros((batch_size, hidden_size)),  # c (cell state)
            jnp.zeros((batch_size, hidden_size)),  # h (hidden state)
        ) 