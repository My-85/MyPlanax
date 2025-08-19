import os
import jax
import jax.numpy as jnp
import functools
import time
import datetime
import optax
import flax
import numpy as np
import tensorboardX
import distrax
from typing import Dict, Tuple, Callable, NamedTuple, Any
import orbax.checkpoint as ocp

from envs.aeroplanax import EnvState, EnvParams
from flax.training.train_state import TrainState
from envs.wrappers_mul import LogWrapper

MICRO_CONFIG = {
    "NUM_ENVS": 20,
    "NUM_STEPS": 50,
    "UPDATE_EPOCHS": 5,
    "NUM_MINIBATCHES": 5,
    "TOTAL_TIMESTEPS": 2000,
}
MINI_CONFIG = {
    "NUM_ENVS": 40,
    "NUM_STEPS": 80,
    "UPDATE_EPOCHS": 10,
    "NUM_MINIBATCHES": 5,
    "TOTAL_TIMESTEPS": 1e5,
}
MEDIUM_CONFIG = {
    "NUM_ENVS": 500,
    "NUM_STEPS": 100,
    "UPDATE_EPOCHS": 10,
    "NUM_MINIBATCHES": 5,
    "TOTAL_TIMESTEPS": 5e6,
}
HUGE_CONFIG = {
    "NUM_ENVS": 1000,
    "NUM_STEPS": 1000,
    "UPDATE_EPOCHS": 16,
    "NUM_MINIBATCHES": 5,
    "TOTAL_TIMESTEPS": 1e8,
}

RENDER_CONFIG = {
    "NUM_ENVS": 1,
    "NUM_STEPS": 100,
    "UPDATE_EPOCHS": 10,
    "NUM_MINIBATCHES": 5,
    "TOTAL_TIMESTEPS": 1e6,
}


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _update_continuous_step(
    num_envs: int, 
    num_steps: int, 
    num_agents: int,
    global_step: int,
    rng: jax.Array,
    hstate: Tuple[jax.Array, jax.Array],
    actor_state: TrainState,
    critic_state: TrainState,
    obs_batch: jax.Array,
    action_batch: jax.Array,
    logprob_batch: jax.Array,
    target_batch: jax.Array,
    advantage_batch: jax.Array,
    masks_batch: jax.Array,
    dones_batch: jax.Array
):
    """
    Update step for continuous MAPPO
    Args:
        num_envs: Number of environments
        num_steps: Number of steps per environment
        num_agents: Number of agents
        global_step: Global step counter
        rng: JAX PRNG key
        hstate: Hidden state tuple (rnn_state, actor_features)
        actor_state: Actor TrainState
        critic_state: Critic TrainState
        obs_batch: Batched observations
        action_batch: Batched actions
        logprob_batch: Batched log probabilities
        target_batch: Batched targets
        advantage_batch: Batched advantages
        masks_batch: Batched masks
        dones_batch: Batched dones

    Returns:
        Updated actor_state, critic_state, hstate, and metrics
    """
    rng, _rng = jax.random.split(rng)
    
    batch_size = num_envs * num_steps * num_agents
    
    def _update_epoch(update_state, _):
        actor_state, critic_state, hstate, _rng = update_state
        rng_actor, rng_critic = jax.random.split(_rng, 2)
        
        # For RNN-based MARL, sample trajectories
        permutation = jax.random.permutation(rng_actor, num_envs)
        # Need to keep time dimension in order
        batch = {
            "obs": obs_batch.reshape(num_steps, num_envs * num_agents, -1),
            "actions": action_batch.reshape(num_steps, num_envs * num_agents, -1),
            "old_logprobs": logprob_batch.reshape(num_steps, num_envs * num_agents),
            "targets": target_batch.reshape(num_steps, num_envs * num_agents),
            "advantages": advantage_batch.reshape(num_steps, num_envs * num_agents),
            "masks": masks_batch.reshape(num_steps, num_envs * num_agents),
            "dones": dones_batch.reshape(num_steps, num_envs * num_agents),
        }
        
        def _update_minibatch_continuous(actor_critic_state, minibatch):
            actor_state, critic_state, hstate = actor_critic_state
            rnn_state, actor_features = hstate
            
            # Extract minibatch data
            mb_obs = minibatch["obs"]
            mb_actions = minibatch["actions"]
            mb_old_logprobs = minibatch["old_logprobs"]
            mb_targets = minibatch["targets"]
            mb_advantages = minibatch["advantages"]
            mb_masks = minibatch["masks"]
            mb_dones = minibatch["dones"]
            
            # Create input for the actor-critic network
            network_input = (mb_obs, mb_dones)
            
            # Actor loss
            def actor_loss_fn(actor_params):
                # Forward pass through the actor network
                _, (pi, _) = actor_state.apply_fn(
                    {"params": actor_params},
                    rnn_state,
                    actor_features,
                    network_input
                )
                
                # Calculate log probabilities for the actions
                log_probs = pi.log_prob(mb_actions)
                
                # Calculate entropy for exploration
                entropy = pi.entropy().mean()
                
                # Calculate ratios for PPO
                ratio = jnp.exp(log_probs - mb_old_logprobs)
                
                # Clipped surrogate objective
                clipped_ratio = jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2)
                actor_loss1 = -ratio * mb_advantages
                actor_loss2 = -clipped_ratio * mb_advantages
                actor_loss = jnp.maximum(actor_loss1, actor_loss2).mean()
                
                # Add entropy term to encourage exploration
                return actor_loss - 0.001 * entropy, (log_probs, entropy)
            
            # Critic loss
            def critic_loss_fn(critic_params):
                # Forward pass through the critic network
                _, (_, values) = critic_state.apply_fn(
                    {"params": critic_params},
                    rnn_state,
                    actor_features,
                    network_input
                )
                
                # Value function loss
                value_pred_clipped = mb_targets + jnp.clip(
                    values - mb_targets, -0.2, 0.2
                )
                value_losses = jnp.square(values - mb_targets)
                value_losses_clipped = jnp.square(value_pred_clipped - mb_targets)
                critic_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                
                return critic_loss, (values,)
            
            # Calculate actor gradients and update actor
            (actor_loss, (log_probs, entropy)), actor_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(actor_state.params)
            actor_state = actor_state.apply_gradients(grads=actor_grads)
            
            # Calculate critic gradients and update critic
            (critic_loss, (values)), critic_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True
            )(critic_state.params)
            critic_state = critic_state.apply_gradients(grads=critic_grads)
            
            # Update hidden state
            hstate = (rnn_state, actor_features)
            
            # Calculate metrics
            metrics = {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy": entropy,
                "approx_kl": ((mb_old_logprobs - log_probs) ** 2).mean(),
            }
            
            return (actor_state, critic_state, hstate), metrics
        
        # Create minibatches and update
        microbatch_size = num_envs // 16  # This makes each batch have 16 environments
        
        # Prepare initial state and metrics
        metrics = []
        update_state = (actor_state, critic_state, hstate)
        
        # Process each microbatch
        for idx in range(0, num_envs, microbatch_size):
            end_idx = min(idx + microbatch_size, num_envs)
            mb_indices = permutation[idx:end_idx]
            
            # Extract data for the microbatch
            minibatch = {
                k: batch[k][:, mb_indices.reshape(-1) * num_agents:(mb_indices.reshape(-1) + 1) * num_agents]
                for k in batch.keys()
            }
            
            # Update on this microbatch
            update_state, minibatch_metrics = _update_minibatch_continuous(update_state, minibatch)
            metrics.append(minibatch_metrics)
        
        # Average metrics across all minibatches
        metrics_mean = {
            k: jnp.mean(jnp.stack([m[k] for m in metrics]))
            for k in metrics[0].keys()
        }
        
        return (update_state[0], update_state[1], update_state[2], rng_critic), metrics_mean
    
    # Run multiple epochs
    init_state = (actor_state, critic_state, hstate, _rng)
    (actor_state, critic_state, hstate, _), metrics = jax.lax.scan(
        _update_epoch, init_state, jnp.arange(16)  # 16 epochs
    )
    
    # Average metrics across epochs
    metrics = jax.tree_map(lambda x: x.mean(), metrics)
    
    return actor_state, critic_state, hstate, metrics


def make_train_mappo_continuous(
    config: Dict,
    env: LogWrapper,
    networks: Tuple,
    train_mode: bool = True,
    save_epochs: int = 50,
):
    """
    Create a continuous MAPPO training function for multi-agent environments
    
    Args:
        config: Configuration dictionary
        env: Environment wrapper
        networks: Tuple of (actor_network, critic_network)
        train_mode: Whether to train or evaluate
        save_epochs: How often to save checkpoints
        
    Returns:
        Training function
    """
    actor_network, critic_network = networks
    
    def train(
        rng: jax.Array,
        train_states: Tuple[TrainState, TrainState],
        start_epoch: jax.Array,
    ) -> Dict:
        """
        Training function for continuous MAPPO
        
        Args:
            rng: PRNG key
            train_states: Tuple of (actor_state, critic_state)
            start_epoch: Starting epoch
            
        Returns:
            Dictionary with training results
        """
        # Unpack train states
        actor_state, critic_state = train_states
        
        # Setup summaries writer
        if config["WANDB"]:
            env_name = config["GROUP"]
            run_name = f"{env_name}__{config['SEED']}__{int(time.time())}"
            summaries_dir = os.path.join(config["LOGDIR"], run_name)
            writer = tensorboardX.SummaryWriter(summaries_dir)
        else:
            writer = None
        
        # Environment setup
        num_envs = config["NUM_ENVS"]
        num_agents = config["NUM_AGENTS"]
        num_steps = config["NUM_STEPS"]
        
        # Initialize environment
        reset_rng, rng = jax.random.split(rng)
        reset_rng = jax.random.split(reset_rng, num_envs)
        obsdict, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env.default_params)
        
        # Initialize RNN state
        rnn_state = jnp.zeros((num_envs * num_agents, config["GRU_HIDDEN_DIM"]))
        actor_features = jnp.zeros((num_envs * num_agents, config["GRU_HIDDEN_DIM"]))
        hstate = (rnn_state, actor_features)
        
        # Convert observation dict to batch
        last_obs = jnp.stack([obsdict[key] for key in env.agents], axis=1)
        last_obs = last_obs.reshape(num_envs * num_agents, -1)
        
        # Initialize done flags
        last_done = jnp.zeros((num_envs * num_agents), dtype=jnp.bool_)
        
        # Metrics for evaluation
        returned_episode_returns = jnp.zeros((num_envs * num_agents,))
        returned_episode_lengths = jnp.zeros((num_envs * num_agents,))
        returned_episode = jnp.zeros((num_envs * num_agents,), dtype=jnp.bool_)
        success = jnp.zeros((num_envs * num_agents,), dtype=jnp.bool_)
        
        # Other metrics specific to the environment
        info_metrics = {}
        if hasattr(env, "info_metrics"):
            for k in env.info_metrics:
                info_metrics[k] = jnp.zeros((num_envs * num_agents,))
        
        # Training loop
        def _env_step(runner_state, unused):
            # Unpack runner state
            (
                train_states,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                episode_returns,
                episode_lengths,
                returned_episode_returns,
                returned_episode_lengths,
                returned_episode,
                success,
                info_metrics,
            ) = runner_state
            
            # Get actions from the policy
            actor_state, critic_state = train_states
            rnn_state, actor_features = hstate
            rng, step_rng = jax.random.split(rng)
            
            # Forward pass through the actor network
            ac_in = (
                last_obs[jnp.newaxis, :],
                last_done[jnp.newaxis, :],
            )
            (new_rnn_state, new_actor_features), (pi, value) = actor_network.apply(
                actor_state.params, rnn_state, actor_features, ac_in
            )
            
            # Sample actions from the policy
            action = pi.sample(seed=step_rng)
            log_prob = pi.log_prob(action)
            action = action.squeeze()
            
            # Update RNN state
            hstate = (new_rnn_state.squeeze(), new_actor_features.squeeze())
            
            # Reshape actions for the environment
            actiondict = {
                agent: action.reshape(num_envs, num_agents)[:, i]
                for i, agent in enumerate(env.agents)
            }
            
            # Step the environment
            step_rng = jax.random.split(step_rng, num_envs)
            obsdict, env_state, rewarddict, done_dict, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_rng, env_state, actiondict, env.default_params)
            
            # Process observations, rewards, and dones
            obs = jnp.stack([obsdict[key] for key in env.agents], axis=1)
            obs = obs.reshape(num_envs * num_agents, -1)
            
            rewards = jnp.stack([rewarddict[key] for key in env.agents], axis=1)
            rewards = rewards.reshape(num_envs * num_agents)
            
            dones = jnp.stack([done_dict[key] for key in env.agents], axis=1)
            dones = dones.reshape(num_envs * num_agents)
            
            # Update episode metrics
            masks = 1.0 - dones.astype(jnp.float32)
            episode_returns = episode_returns * masks + rewards
            episode_lengths = episode_lengths * masks + 1
            
            # Store returns and lengths for completed episodes
            returned_episode_returns = jnp.where(
                dones, episode_returns, returned_episode_returns
            )
            returned_episode_lengths = jnp.where(
                dones, episode_lengths, returned_episode_lengths
            )
            returned_episode = returned_episode | dones
            success = success | (dones & info["success"].reshape(num_envs * num_agents))
            
            # Update environment-specific metrics
            for k in info_metrics.keys():
                info_metrics[k] = jnp.where(
                    dones,
                    info[k].reshape(num_envs * num_agents),
                    info_metrics[k],
                )
            
            # Reset episode metrics for completed episodes
            episode_returns = episode_returns * masks
            episode_lengths = episode_lengths * masks
            
            # Collect transition data
            transition = {
                "obs": last_obs,
                "action": action,
                "reward": rewards,
                "done": dones,
                "value": value.squeeze(),
                "log_prob": log_prob,
                "mask": masks,
            }
            
            # Update runner state
            new_runner_state = (
                (actor_state, critic_state),
                env_state,
                obs,
                dones,
                hstate,
                rng,
                episode_returns,
                episode_lengths,
                returned_episode_returns,
                returned_episode_lengths,
                returned_episode,
                success,
                info_metrics,
            )
            
            return new_runner_state, transition
        
        # Run multiple environment steps to collect transitions
        runner_state = (
            (actor_state, critic_state),
            env_state,
            last_obs,
            last_done,
            hstate,
            rng,
            jnp.zeros((num_envs * num_agents,)),
            jnp.zeros((num_envs * num_agents,)),
            returned_episode_returns,
            returned_episode_lengths,
            returned_episode,
            success,
            info_metrics,
        )
        
        # Collect transitions
        runner_state, transitions = jax.lax.scan(
            _env_step, runner_state, None, num_steps
        )
        
        # Unpack final runner state
        (
            train_states,
            env_state,
            last_obs,
            last_done,
            hstate,
            rng,
            episode_returns,
            episode_lengths,
            returned_episode_returns,
            returned_episode_lengths,
            returned_episode,
            success,
            info_metrics,
        ) = runner_state
        
        # Compute GAE advantages and returns
        actor_state, critic_state = train_states
        
        # Get final value estimate
        rnn_state, actor_features = hstate
        ac_in = (
            last_obs[jnp.newaxis, :],
            last_done[jnp.newaxis, :],
        )
        _, (_, last_val) = actor_network.apply(
            actor_state.params, rnn_state, actor_features, ac_in
        )
        last_val = last_val.squeeze()
        
        def _calculate_gae(
            traj_batch, last_val, gamma=0.99, gae_lambda=0.95
        ):
            # Get rewards and values from trajectory
            rewards = traj_batch["reward"]  # [T, B]
            values = traj_batch["value"]  # [T, B]
            masks = traj_batch["mask"]  # [T, B]
            
            # Append last value to values array
            all_values = jnp.concatenate([values, last_val[jnp.newaxis, ...]])
            
            # Calculate advantages using GAE
            advantages = jnp.zeros_like(rewards)
            gae = jnp.zeros_like(rewards[0])
            
            def _gae_step(gae, timestep):
                # Unpack timestep data
                reward = rewards[timestep]
                mask = masks[timestep]
                value = all_values[timestep]
                next_value = all_values[timestep + 1]
                
                # Calculate TD error and GAE
                delta = reward + gamma * next_value * mask - value
                new_gae = delta + gamma * gae_lambda * mask * gae
                
                return new_gae, new_gae
            
            # Reverse scan to compute advantages backwards
            _, advantages = jax.lax.scan(
                _gae_step,
                gae,
                jnp.arange(num_steps - 1, -1, -1),
                reverse=True,
            )
            
            # Calculate returns as advantages + values
            returns = advantages + values
            
            return advantages, returns
        
        # Compute GAE advantages and returns
        advantages, targets = _calculate_gae(
            transitions, last_val, config["GAMMA"], config["GAE_LAMBDA"]
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update networks if in training mode
        if train_mode:
            # Prepare batched data for updates
            actor_state, critic_state = train_states
            
            # Reshape transitions for updating
            b_obs = transitions["obs"]
            b_actions = transitions["action"]
            b_logprobs = transitions["log_prob"]
            b_advantages = advantages
            b_targets = targets
            b_masks = transitions["mask"]
            b_dones = transitions["done"]
            
            # Update networks
            actor_state, critic_state, hstate, update_metrics = _update_continuous_step(
                num_envs,
                num_steps,
                num_agents,
                start_epoch,
                rng,
                hstate,
                actor_state,
                critic_state,
                b_obs,
                b_actions,
                b_logprobs,
                b_targets,
                b_advantages,
                b_masks,
                b_dones,
            )
            
            # Update train states
            train_states = (actor_state, critic_state)
        
        # Metrics for logging
        metrics = {
            "returned_episode_returns": returned_episode_returns,
            "returned_episode_lengths": returned_episode_lengths,
            "returned_episode": returned_episode,
            "success": success,
            "update_steps": start_epoch + 1,
        }
        
        # Add environment-specific metrics
        for k in info_metrics.keys():
            metrics[k] = info_metrics[k]
        
        # Add update metrics
        if train_mode:
            metrics["loss"] = update_metrics
        
        # Update epoch counter
        update_steps = start_epoch + 1
        
        # Write summaries if wandb is enabled
        if config["WANDB"] and writer is not None and train_mode:
            env.train_callback(metrics, writer, train_mode)
        
        # Save checkpoint
        if train_mode and (update_steps % save_epochs == 0 or update_steps == config["NUM_UPDATES"]):
            save_train_mappo_continuous((actor_state, critic_state), update_steps, config["SAVEDIR"])
        
        # Return runner state and metrics
        runner_state = (
            train_states,
            env_state,
            last_obs,
            last_done,
            hstate,
            rng,
        )
        
        return {"runner_state": (runner_state, update_steps), "metrics": metrics}
    
    return train


def save_train_mappo_continuous(train_state, epoch, save_path):
    """
    Save training state to checkpoint
    
    Args:
        train_state: Tuple of (actor_state, critic_state)
        epoch: Current epoch
        save_path: Path to save checkpoint
    """
    actor_state, critic_state = train_state
    checkpoint = {
        "params": actor_state.params,
        "opt_state": actor_state.opt_state,
        "epoch": epoch,
    }
    
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Save checkpoint
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    save_args = ocp.args.StandardSave(save_path)
    ckptr.save(save_args, checkpoint)
    
    print(f"Saved checkpoint at epoch {epoch}") 