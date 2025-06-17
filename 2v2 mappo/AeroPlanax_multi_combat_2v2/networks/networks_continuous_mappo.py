import functools
import os
from typing import Dict, Sequence, Tuple, Any
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import orbax.checkpoint as ocp

class ScannedRNN(nn.Module):
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
        rnn_state, actor_features = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            jnp.zeros_like(rnn_state),
            rnn_state,
        )
        new_rnn_state, y = nn.GRU(features=rnn_state.shape[-1])(rnn_state, ins)
        return (new_rnn_state, y), y


class ActorCriticRNN(nn.Module):
    action_dims: int  # For continuous, this is the number of action dimensions
    config: Dict

    @nn.compact
    def __call__(self, rnn_state, actor_features, x):
        """
        Applies the Actor and Critic models in the MAPPO
        
        Args:
            rnn_state: GRU cell state
            actor_features: Initial actor hidden state
            x: Tuple of (observations, dones)
        
        Returns:
            (new_rnn_state, new_actor_features), (action_distribution, value)
        """
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        obs, dones = x
        batch_size = obs.shape[0]
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = activation(embedding)
        
        # GRU layer
        (new_rnn_state, new_actor_features), embedding = ScannedRNN()(
            (rnn_state, actor_features), (embedding, dones)
        )
        
        # Actor network - outputs mean and log_std for continuous actions
        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        
        # Output action means (3 dimensions: pitch, heading, velocity)
        action_mean = nn.Dense(
            self.action_dims, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        
        # Output log standard deviations, one per action dimension
        # These are constrained between LOG_STD_MIN and LOG_STD_MAX for stability
        LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dims,))
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        action_std = jnp.exp(log_std)
        
        # Create a multivariate normal distribution
        action_distribution = distrax.MultivariateNormalDiag(
            loc=action_mean,
            scale_diag=action_std * jnp.ones_like(action_mean)
        )
        
        # Critic network
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return (new_rnn_state, new_actor_features), (action_distribution, jnp.squeeze(value, axis=-1))


def init_network_mappoRNN_continuous(config, action_dims=3):
    """
    Initialize actor-critic networks for continuous MAPPO with RNN
    
    Args:
        config: Configuration dictionary
        action_dims: Number of continuous action dimensions (default 3 for pitch, heading, velocity)
    
    Returns:
        (actor_network, critic_network), (actor_state, critic_state), start_epoch
    """
    def linear_schedule(count):
        if config["ANNEAL_LR"]:
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
            return config["LR"] * frac
        return config["LR"]
    
    # Initialize seeds
    rng = jax.random.PRNGKey(config["SEED"])
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    
    # Create and initialize the network
    network = ActorCriticRNN(action_dims=action_dims, config=config)
    
    dummy_obs = jnp.zeros((1, config["NUM_STEPS"], config["NUM_ENVS"], config["OBS_DIM"]))
    dummy_dones = jnp.zeros((1, config["NUM_STEPS"], config["NUM_ENVS"]))
    dummy_rnn_state = jnp.zeros((config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]))
    dummy_actor_features = jnp.zeros((config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]))
    dummy_x = (dummy_obs, dummy_dones)
    
    network_params = network.init(actor_key, dummy_rnn_state, dummy_actor_features, dummy_x)
    
    # Create the optimizers and training states
    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
    
    ac_train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    
    cr_train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    
    start_epoch = jnp.array(0)
    
    # Load from checkpoint if LOADDIR is provided and valid
    if "LOADDIR" in config and config["LOADDIR"] and os.path.exists(config["LOADDIR"]):
        try:
            print(f"Loading checkpoint from {config['LOADDIR']}...")
            state = {"params": ac_train_state.params, "tx": tx}
            ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
            checkpoint = ckptr.restore(config["LOADDIR"], args=ocp.args.StandardRestore(item=state))
            start_epoch = checkpoint["epoch"]
            ac_train_state = ac_train_state.replace(params=checkpoint["params"])
            cr_train_state = cr_train_state.replace(params=checkpoint["params"])
            print(f"Checkpoint loaded successfully, resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    
    return (network, network), (ac_train_state, cr_train_state), start_epoch 