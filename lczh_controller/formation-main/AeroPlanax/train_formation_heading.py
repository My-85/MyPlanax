import os
import functools
import time
import datetime
from typing import Dict, NamedTuple, Sequence, Tuple, Any
import numpy as np
from tqdm import tqdm
import distrax
import optax
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import orbax.checkpoint
from tensorboardX import SummaryWriter

import chex
from gymnax.environments import spaces

from envs.aeroplanax_formation_heading import FormationHeadingTaskParams, AeroPlanaxFormationHeadingEnv


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        obs = x
        
        # Actor network (decentralized)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = activation(embedding)

        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"]*2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = activation(embedding)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_throttle_mean = nn.Dense(
            self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_elevator_mean = nn.Dense(
            self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_aileron_mean = nn.Dense(
            self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_rudder_mean = nn.Dense(
            self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi_throttle = distrax.Categorical(logits=actor_throttle_mean)
        pi_elevator = distrax.Categorical(logits=actor_elevator_mean)
        pi_aileron = distrax.Categorical(logits=actor_aileron_mean)
        pi_rudder = distrax.Categorical(logits=actor_rudder_mean)

        # Critic network (centralized)
        # For the critic, we need to process observations from this agent independently,
        # not try to reshape all agents' observations
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        critic = activation(critic)
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Dict


def batchify(x: Dict, num_actors: int, agent_names: Sequence[str]) -> jnp.ndarray:
    """Convert dict keyed by agent names to a batch of values stacked along axis 0."""
    return jnp.array([x[agent] for agent in agent_names])


def unbatchify(x: jnp.ndarray, agent_names: Sequence[str]) -> Dict:
    """Convert batch of values stacked along axis 0 to dict keyed by agent names."""
    return {agent: x[i] for i, agent in enumerate(agent_names)}


def merge_agent_actions(control_actions: Dict, agent_names: Sequence[str]) -> Dict:
    """
    Combine actions from different control dimensions into per-agent action dictionaries.
    Now JAX-compatible.
    
    Args:
        control_actions: Dictionary with keys 'throttle', 'elevator', etc., each containing a batch of actions
        agent_names: List of agent names
    
    Returns:
        Dictionary mapping agent names to their action dictionaries
    """
    # Create a dictionary that maps agent names to empty dictionaries
    result = {}
    
    # For each agent
    for i, agent in enumerate(agent_names):
        agent_actions = {}
        # For each control dimension
        for control_key, control_values in control_actions.items():
            if i < len(control_values):
                # Just store the raw JAX array - don't convert to int
                agent_actions[control_key] = control_values[i]
        result[agent] = agent_actions
    
    return result


def format_actions_for_env(control_actions, num_envs, agents):
    """
    Format control actions into the structure expected by the environment.
    Now compatible with JAX tracing.
    
    Args:
        control_actions: Dict with keys like 'throttle', 'elevator', etc.
        num_envs: Number of environments
        agents: List of agent names
    
    Returns:
        Properly formatted actions for the environment
    """
    action_env = []
    
    for env_idx in range(num_envs):
        env_actions = {}
        for agent_idx, agent in enumerate(agents):
            agent_actions = {}
            for control_key, control_values in control_actions.items():
                # Keep the JAX value as-is - don't try to convert to Python int
                agent_actions[control_key] = control_values[env_idx, agent_idx]
            env_actions[agent] = agent_actions
        action_env.append(env_actions)
    
    return action_env


def make_train(config):
    env_params = FormationHeadingTaskParams(
        num_allies=config["NUM_ACTORS"],
        formation_type=config["FORMATION_TYPE"],
        team_spacing=config["TEAM_SPACING"],
        safe_distance=config["SAFE_DISTANCE"],
        formation_reward_weight=config["FORMATION_REWARD_WEIGHT"],
        heading_reward_weight=config["HEADING_REWARD_WEIGHT"],
    )
    env = AeroPlanaxFormationHeadingEnv(env_params)
    config["NUM_ACTIONS"] = 4  # throttle, elevator, aileron, rudder
    
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic([31, 41, 41, 41], config=config)
        rng, _rng = jax.random.split(rng)
        
        # Get a sample observation to determine shape
        sample_reset_rng = jax.random.PRNGKey(0)
        sample_obs, sample_state = env.reset(sample_reset_rng, env_params)
        print(f"Sample obs shape: {tree_map(lambda x: x.shape, sample_obs)}")
        print(f"Sample obs type: {type(sample_obs)}")
        
        # Get observation space shape
        obs_shape = env.observation_spaces[env.agents[0]].shape[0]
        print(f"Observation space shape: {obs_shape}")
        
        # Print agent information
        print(f"Agents: {env.agents}")
        print(f"Num agents: {len(env.agents)}")
        
        # Debug: Print the action space
        print(f"Action space: {env.action_spaces[env.agents[0]]}")
        
        # Init network parameters with proper shapes - one set for each agent
        init_x = jnp.zeros((1, obs_shape))  # Single agent observation for init
        
        # Debug output for initialization
        print(f"init_x shape: {init_x.shape}")
        
        network_params = network.init(_rng, init_x)
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
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        
        # Convert multi-agent obs to batched format for each agent
        # We'll handle individual agent observations separately
        batched_obs = jax.vmap(batchify, in_axes=(0, None, None))(obsv, config["NUM_ACTORS"], env.agents)
        
        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            train_state, env_state, last_obs, rng = runner_state
            
            # Generate batched action for each agent
            def _env_step(step_runner_state, unused):
                train_state, env_state, last_obs, rng = step_runner_state
                rng, _rng = jax.random.split(rng)
                
                # Last_obs shape: [num_envs, num_agents, obs_dim]
                num_envs, num_agents, obs_dim = last_obs.shape
                
                # Reshape to process with individual agent networks
                flat_obs = last_obs.reshape(num_envs * num_agents, obs_dim)
                
                # Select action for each agent
                pi, value = network.apply(train_state.params, flat_obs)
                
                # Reshape value back to [num_envs, num_agents]
                value = value.reshape(num_envs, num_agents)
                
                # Sample actions for each control dimension
                rng, _rng = jax.random.split(rng)
                action0 = pi[0].sample(seed=_rng).reshape(num_envs, num_agents)
                log_prob0 = pi[0].log_prob(action0.reshape(-1)).reshape(num_envs, num_agents)
                
                rng, _rng = jax.random.split(rng)
                action1 = pi[1].sample(seed=_rng).reshape(num_envs, num_agents)
                log_prob1 = pi[1].log_prob(action1.reshape(-1)).reshape(num_envs, num_agents)
                
                rng, _rng = jax.random.split(rng)
                action2 = pi[2].sample(seed=_rng).reshape(num_envs, num_agents)
                log_prob2 = pi[2].log_prob(action2.reshape(-1)).reshape(num_envs, num_agents)
                
                rng, _rng = jax.random.split(rng)
                action3 = pi[3].sample(seed=_rng).reshape(num_envs, num_agents)
                log_prob3 = pi[3].log_prob(action3.reshape(-1)).reshape(num_envs, num_agents)
                
                # Stack actions for storing in the trajectory
                stacked_actions = jnp.stack([action0, action1, action2, action3], axis=-1)
                
                # Sum the log probs from each action dimension
                log_probs = log_prob0 + log_prob1 + log_prob2 + log_prob3
                
                # Create action dictionary with control dimensions
                control_actions = {
                    "throttle": action0,
                    "elevator": action1,
                    "aileron": action2,
                    "rudder": action3,
                }
                
                # Use the safe formatting function
                action_env = format_actions_for_env(
                    control_actions, config["NUM_ENVS"], env.agents
                )
                
                # Step environment using normal Python loop instead of vmap
                # This avoids JAX tracer issues with the environment step
                rng, _rng = jax.random.split(rng)
                step_rng = jax.random.split(_rng, config["NUM_ENVS"])
                
                # For the multi-environment case, env_state is a single object, not a list
                # We need to step each environment separately
                obsv_list = []
                reward_list = []
                done_list = []
                info_list = []
                
                # First step the environment once to get state structure
                try:
                    env_obsv, env_state_new, env_reward, env_done, env_info = env.step(
                        step_rng[0], env_state, action_env[0], env_params
                    )
                    obsv_list.append(env_obsv)
                    reward_list.append(env_reward)
                    done_list.append(env_done)
                    info_list.append(env_info)
                    
                    # Now handle the rest of the environments
                    for i in range(1, config["NUM_ENVS"]):
                        if config["DEBUG"] and i == 1:
                            print(f"Step environment {i}")
                        
                        env_obsv, _, env_reward, env_done, env_info = env.step(
                            step_rng[i], env_state, action_env[i], env_params
                        )
                        obsv_list.append(env_obsv)
                        reward_list.append(env_reward)
                        done_list.append(env_done)
                        info_list.append(env_info)
                        
                except Exception as e:
                    if config["DEBUG"]:
                        print(f"Error in env step: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Use the first environment state for all
                    env_state_new = env_state
                    
                    # Create default observations and rewards
                    for i in range(config["NUM_ENVS"]):
                        obsv_list.append({agent: jnp.zeros(env.observation_spaces[agent].shape) for agent in env.agents})
                        reward_list.append({agent: 0.0 for agent in env.agents})
                        done_list.append({agent: False for agent in env.agents})
                        info_list.append({})
                
                # Process done flags - convert from dict format to boolean array
                done_array = jnp.zeros(config["NUM_ENVS"], dtype=bool)
                for i in range(config["NUM_ENVS"]):
                    if isinstance(done_list[i], dict):
                        # Check if any agent in this environment is done
                        any_done = False
                        for agent in env.agents:
                            if agent in done_list[i] and done_list[i][agent]:
                                any_done = True
                        done_array = done_array.at[i].set(any_done)
                    elif isinstance(done_list[i], bool):
                        done_array = done_array.at[i].set(done_list[i])
                
                # Combine with environment done flag (if available)
                if hasattr(env_state_new, 'done') and env_state_new.done is not None:
                    # If done is a scalar, expand it
                    if not hasattr(env_state_new.done, "__len__") or len(env_state_new.done) == 1:
                        done_array = jnp.logical_or(done_array, jnp.repeat(env_state_new.done, config["NUM_ENVS"]))
                    else:
                        done_array = jnp.logical_or(done_array, env_state_new.done)
                
                # Reset environments that are done
                # Since we can't maintain separate env states, we'll just reset and use the
                # most recent state for all environments
                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
                
                for i in range(config["NUM_ENVS"]):
                    if done_array[i]:
                        reset_obs, reset_state = env.reset(reset_rng[i], env_params)
                        obsv_list[i] = reset_obs
                        # Use the reset_state for the next iteration
                        env_state_new = reset_state
                
                # Convert observations to batched format for the network
                batched_obs = []
                for i in range(config["NUM_ENVS"]):
                    agent_batch = []
                    for agent in env.agents:
                        agent_batch.append(obsv_list[i][agent])
                    batched_obs.append(jnp.array(agent_batch))
                batched_obs = jnp.array(batched_obs)
                
                # Convert rewards to batched format
                batched_rewards = []
                for i in range(config["NUM_ENVS"]):
                    agent_rewards = []
                    for agent in env.agents:
                        agent_rewards.append(reward_list[i].get(agent, 0.0))
                    batched_rewards.append(jnp.array(agent_rewards))
                batched_rewards = jnp.array(batched_rewards)
                
                transition = Transition(
                    done=done_array,
                    action=stacked_actions,
                    value=value,
                    reward=batched_rewards,
                    log_prob=log_probs,
                    obs=last_obs,
                    info=info_list,
                )
                runner_state = (train_state, env_state_new, batched_obs, rng)
                return runner_state, transition

            # Collect trajectories
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE RETURNS
            train_state, env_state, last_obs, rng = runner_state
            
            # last_obs is already in the correct format [num_envs, num_agents, obs_dim]
            num_envs, num_agents, obs_dim = last_obs.shape
            flat_obs = last_obs.reshape(num_envs * num_agents, obs_dim)
            _, last_val_flat = network.apply(train_state.params, flat_obs)
            last_val = last_val_flat.reshape(num_envs, num_agents)
            
            # Bootstrap from last state value - handle done flags for multiple agents
            # Add a dimension to done to broadcast correctly across agents
            returns = jax.lax.scan(
                lambda carry, transition: (
                    transition.reward
                    + config["GAMMA"] * carry * (1 - transition.done[:, None]),
                    None
                ),
                last_val,
                traj_batch,
                reverse=True,
            )[0]
            
            # Compute advantages
            advantages = returns - traj_batch.value
            
            update_state = (
                train_state,
                traj_batch,
                advantages,
                returns,
                rng,
            )
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # Reshape observations for network
                        batch_size, num_agents, obs_dim = traj_batch.obs.shape
                        flat_obs = traj_batch.obs.reshape(batch_size * num_agents, obs_dim)
                        
                        # RERUN NETWORK with flattened observations
                        pi, value = network.apply(params, flat_obs)
                        
                        # Reshape value back to [batch, agents]
                        value = value.reshape(batch_size, num_agents)
                        
                        # Reshape for action log prob calculation
                        action_shape = traj_batch.action.shape
                        flat_actions = traj_batch.action.reshape(batch_size * num_agents, action_shape[-1])
                        
                        # Calculate log probs for each action dimension
                        log_prob0 = pi[0].log_prob(flat_actions[:, 0])
                        log_prob1 = pi[1].log_prob(flat_actions[:, 1])
                        log_prob2 = pi[2].log_prob(flat_actions[:, 2])
                        log_prob3 = pi[3].log_prob(flat_actions[:, 3])
                        
                        # Sum log probs across action dimensions and reshape to [batch, agents]
                        log_prob_sum = log_prob0 + log_prob1 + log_prob2 + log_prob3
                        log_prob = log_prob_sum.reshape(batch_size, num_agents)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        
                        # Normalize advantages at the batch level
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        
                        # Calculate entropy for all action distributions
                        entropy = (pi[0].entropy().mean() + pi[1].entropy().mean() + 
                                 pi[2].entropy().mean() + pi[3].entropy().mean()) / 4.0

                        # Debug metrics
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch = (
                    traj_batch,
                    advantages,
                    targets,
                )
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = tree_map(lambda x: x[permutation] if hasattr(x, "__getitem__") else x, batch)
                
                batch_size = config["NUM_ENVS"] // config["NUM_MINIBATCHES"]
                batch_indices = jnp.arange(config["NUM_ENVS"]).reshape(
                    config["NUM_MINIBATCHES"], batch_size
                )
                
                # Update in minibatches
                minibatches = []
                for indices in batch_indices:
                    minibatch = tree_map(lambda x: x[indices] if hasattr(x, "__getitem__") else x, batch)
                    minibatches.append(minibatch)
                
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            # Run multiple epochs of updates
            update_state, losses = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            
            train_state = update_state[0]
            runner_state = (train_state, env_state, last_obs, rng)
            
            return (runner_state, update_steps + 1), losses

        runner_state = (train_state, env_state, batched_obs, rng)
        return jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )

    return train


from flax.training import train_state
class TrainState(train_state.TrainState):
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='formation_heading_ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--formation_type', type=int, default=0)  # 0: wedge, 1: line, 2: diamond
    parser.add_argument('--num_actors', type=int, default=3)
    parser.add_argument('--team_spacing', type=float, default=15000.0)
    parser.add_argument('--safe_distance', type=float, default=3000.0)
    parser.add_argument('--formation_reward_weight', type=float, default=1.0)
    parser.add_argument('--heading_reward_weight', type=float, default=0.5)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    config = {
        # Environment
        "NUM_ENVS": 64,
        "NUM_STEPS": 128,
        "FORMATION_TYPE": args.formation_type,
        "NUM_ACTORS": args.num_actors,
        "TEAM_SPACING": args.team_spacing,
        "SAFE_DISTANCE": args.safe_distance,
        "FORMATION_REWARD_WEIGHT": args.formation_reward_weight,
        "HEADING_REWARD_WEIGHT": args.heading_reward_weight,
        
        # PPO
        "GAMMA": 0.99,
        "LR": 3e-4,
        "GAE_LAMBDA": 0.95,
        "NUM_UPDATES": 1000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        
        # Network
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,  # Still used as dimension size in some layers
        
        # Debug settings
        "DEBUG": args.debug,
    }
    
    # Setup logging
    exp_name = f"{args.exp_name}_actors{args.num_actors}_form{args.formation_type}_seed{args.seed}"
    run_dir = os.path.join(
        "results",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + exp_name,
    )
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
        0,
    )

    # Setup checkpoint saving
    def save_checkpoint(params, path):
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpointer.save(path, params)

    # Train with error handling
    print(f"Running training with config: {config}")
    rng = jax.random.PRNGKey(args.seed)
    train_fn = make_train(config)
    
    try:
        (runner_state, update_steps), (metrics, loss_info) = train_fn(rng)
        
        # Save final model
        save_checkpoint(runner_state[0].params, os.path.join(run_dir, "final_checkpoint"))
        print("Training complete!")
        
    except Exception as e:
        import traceback
        print(f"\nError during training: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        
        print("\n\nTroubleshooting suggestions:")
        print("1. Try running with --debug flag for more verbose output")
        print("2. Check if the number of agents (--num_actors) matches the environment expectations")
        print("3. Verify that the formation type is valid (0: wedge, 1: line, 2: diamond)")
        print("4. If there are shape mismatches, modify the observation or action handling functions accordingly")
        print("5. For more detailed analysis, add print statements in the _env_step and _update_step functions") 