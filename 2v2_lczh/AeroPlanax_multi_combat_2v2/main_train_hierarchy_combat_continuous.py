import os
# Configure JAX to avoid using all GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import wandb
import jax.numpy as jnp
from datetime import datetime
from pathlib import Path
from envs.wrappers_mul import LogWrapper
from envs.aeroplanax_combat_hierarchy_continuous import (
    AeroPlanaxContinuousHierarchicalCombatEnv as Env,
    HierarchicalCombatTaskParams as TaskParams
)

# Import continuous MAPPO implementations
from networks import init_network_mappoRNN_continuous
from maketrains import (
    make_train_mappo_continuous as make_train,
    save_train_mappo_continuous as save_train,
    RENDER_CONFIG,
    MICRO_CONFIG,
    MINI_CONFIG,
    MEDIUM_CONFIG,
    HUGE_CONFIG
)

# Create environment parameters
env_params = TaskParams(
    num_allies=5,
    num_enemies=5,
    formation_type=0,     # 0: wedge formation
    action_type=1,        # Now converted to continuous in our implementation
    observation_type=0,
    sim_freq=50,
    agent_interaction_steps=10,
    max_altitude=20000,
    min_altitude=1000,
    max_vt=340,
    min_vt=100,
    team_spacing=600,
    safe_distance=100,
    posture_reward_scale=1.0,
    use_baseline=True,    # Use baseline for enemy control
    noise_features=10,
    top_k_ego_obs=1,
    top_k_enm_obs=2
)

# Configuration setup
str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
config = {
    "LR": 3e-5,           # Learning rate for continuous control
    "FC_DIM_SIZE": 256,   # Hidden layer size
    "GRU_HIDDEN_DIM": 256, # RNN hidden state size
    "UPDATE_EPOCHS": 16,
    "NUM_MINIBATCHES": 5,
    "GAMMA": 0.99,        # Discount factor
    "GAE_LAMBDA": 0.95,   # GAE lambda parameter
    "CLIP_EPS": 0.2,      # PPO clipping parameter
    "ENT_COEF": 1e-3,     # Entropy coefficient
    "VF_COEF": 1,         # Value function coefficient
    "MAX_GRAD_NORM": 2,   # Gradient clipping
    "RNN_GRAD_CLIP_VALUE": 1.0,  # Internal gradient clipping for RNN
    "ACTIVATION": "relu", # Activation function
    "ANNEAL_LR": False,   # Whether to anneal learning rate
    "DEBUG": True,
    "NUM_ENVS": 1000,     # Number of parallel environments
    "NUM_STEPS": 1000,    # Number of steps per environment per update
    "TOTAL_TIMESTEPS": 1e8, # Total timesteps for training
    "SEED": 42,
    "NOISE_SEED": 42,
    "GROUP": "combat_hierarchy_continuous",
    "FOR_LOOP_EPOCHS": 10,
    "WANDB": True,
    "TRAIN": True,
    "WANDB_API_KEY" : "4c0cc04699296bed768adea4824fbaecea35dc59", # Replace with your API key
    "OUTPUTDIR": "results/" + "combat_hierarchy_continuous_" + str_date_time,
    "LOGDIR": "results/" + "combat_hierarchy_continuous_" + str_date_time + "/logs",
    "SAVEDIR": "results/" + "combat_hierarchy_continuous_" + str_date_time + "/checkpoints",
}

# Calculate number of updates
config["NUM_UPDATES"] = (
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)

# Initialize random number generators
rng = jax.random.PRNGKey(config["SEED"])
if "NOISE_SEED" in config.keys():
    _noise_rng = jax.random.PRNGKey(config["NOISE_SEED"])
else:
    rng, _noise_rng = jax.random.split(rng)

# Create environment and wrap it with logger
env = Env(env_params)
env = LogWrapper(env, rng=_noise_rng)

# Get environment information for config
config = config | env.get_env_information_for_config()

# Setup wandb if enabled
if config["WANDB"]:
    if config["WANDB_API_KEY"] == "4c0cc04699296bed768adea4824fbaecea35dc59":
        # Replace with your own API key when running
        pass
    
    os.environ['WANDB_API_KEY'] = config["WANDB_API_KEY"]
    wandb.tensorboard.patch(root_logdir=config['LOGDIR'])
    wandb.init(
        project="AeroPlanax",
        config=config,
        name=f'lczh_2v2(lczh_version)_continuous_mappo_{config["SEED"]}',
        group=Env.__name__,
        notes=Env.__name__ + " with continuous MAPPO controller",
        reinit=True,
    )

# Create save directory
Path(config["SAVEDIR"]).mkdir(parents=True, exist_ok=True)

# Initialize network - for continuous actions, we use 3 dimensions (pitch, heading, velocity)
(actor_network, critic_network), (ac_train_state, cr_train_state), start_epoch = init_network_mappoRNN_continuous(config, action_dims=3)

# Create JIT-compiled training function
train_jit = jax.jit(make_train(
    config,
    env,
    (actor_network, critic_network),
    train_mode=config["TRAIN"],
))

# Training loop
for i in range(config["FOR_LOOP_EPOCHS"]):
    print(f"Starting training epoch {i+1}/{config['FOR_LOOP_EPOCHS']}")
    
    # Run training
    out = train_jit(rng, (ac_train_state, cr_train_state), start_epoch)
    
    # Extract results
    runner_state = out['runner_state'][0]
    (ac_train_state, cr_train_state) = runner_state[0]
    rng = runner_state[5]
    start_epoch = jnp.array(out['runner_state'][1])
    
    # Save model
    if config["TRAIN"]:
        save_train((ac_train_state, cr_train_state), start_epoch, config["SAVEDIR"])
        print(f"Saved model at epoch {start_epoch}")

# Clean up wandb
if config["WANDB"]:
    wandb.finish() 