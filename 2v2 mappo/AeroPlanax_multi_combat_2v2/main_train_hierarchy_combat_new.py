import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import wandb
import jax.numpy as jnp
from datetime import datetime
from pathlib import Path
from envs.wrappers_mul import LogWrapper, RewardNormWrapper
# from envs.aeroplanax_formation import (
#     AeroPlanaxFormationEnv as Env,
#     FormationTaskParams as TaskParams
# )
from envs.aeroplanax_combat_hierarchy_FSM import (
    AeroPlanaxHierarchicalCombatEnv as Env,
    HierarchicalCombatTaskParams as TaskParams
)

from maketrains import (
    make_train_mappo_discrete as make_train,
    save_train_mappo_discrete as save_train,

    RENDER_CONFIG,
    MICRO_CONFIG,
    MINI_CONFIG,
    MEDIUM_CONFIG,
    HUGE_CONFIG
)
from networks import (
    init_network_mappoRNN_discrete as init_network,
    # init_network_poolppo_discrete as init_network_poolppo,
    # init_network_ppoRNN_discrete as init_network,
)
# For the pitch-heading-velocity controller
PPO_DISCRETE_HIERARCHY_DEFAULT_DIMS = [30, 30, 30]  # Updated dimensions: 30 steps for pitch, 30 steps for heading, 30 steps for velocity
DEFUALT_DIMS = PPO_DISCRETE_HIERARCHY_DEFAULT_DIMS
# Formation environment uses 3 discretized continuous actions
# PPO_DISCRETE_FORMATION_DEFAULT_DIMS = [41, 41, 41]
# DEFUALT_DIMS = PPO_DISCRETE_FORMATION_DEFAULT_DIMS

# Create environment parameters with appropriate settings for combat task
env_params = TaskParams(
    num_allies=2,
    num_enemies=2,
    formation_type=0,  # 0: wedge formation
    action_type=1,     # Discrete action space
    observation_type=0,
    sim_freq=50,
    agent_interaction_steps=10,
    max_altitude=20000,
    min_altitude=1000,
    max_vt=340,
    min_vt=100,
    team_spacing=1000,
    safe_distance=100,
    posture_reward_scale=1.0,
    use_baseline=True,  # Use baseline for enemy control
    noise_features=10,
    top_k_ego_obs=1,
    top_k_enm_obs=2
)

str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
config = {
    "ACTOR_LR": 3e-4,  # Actor learning rate
    "CRITIC_LR": 3e-3,  # Critic learning rate (higher than actor)
    "FC_DIM_SIZE": 256,  # Increased network capacity
    "GRU_HIDDEN_DIM": 256,  # Increased network capacity
    "UPDATE_EPOCHS": 8,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.995,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 1e-3,  # 增加熵正则化系数以鼓励更多探索
    "VF_COEF": 1,
    "MAX_GRAD_NORM": 0.5,
    "RNN_GRAD_CLIP_VALUE": 0.5,  # Internal gradient clipping for RNN cells to prevent explosion
    "ACTIVATION": "relu",
    "ANNEAL_LR": False,
    "DEBUG": True,
    "NUM_ENVS": 500,
    "NUM_STEPS": 1500,
    "TOTAL_TIMESTEPS": 1e8,
    "SEED": 42,
    "NOISE_SEED": 42,
    "GROUP": "combat_hierarchy_pitch",
    "FOR_LOOP_EPOCHS": 15,
    "WANDB": True,
    "TRAIN": True,
    "WANDB_API_KEY" : "f4316927feb010654a1d429360c2f2c824e84387",
    "OUTPUTDIR": "results/" + "combat_hierarchy_pitch_" + str_date_time,
    "LOGDIR": "results/" + "combat_hierarchy_pitch_" + str_date_time + "/logs",
    "SAVEDIR": "results/" + "combat_hierarchy_pitch_" + str_date_time + "/checkpoints",
    "LOADDIR": "/home/lczh/Git Project/results/combat_hierarchy_pitch_2025-05-23-11-47/checkpoints/checkpoint_epoch_1094" ,
    
    # Reward normalization settings
    "REWARD_NORM": False,
    "REWARD_NORM_GAMMA": 0.99,
    "REWARD_NORM_EPSILON": 1e-8,
    "REWARD_NORM_CLIP": 10.0,
}

'''
NOTE:
RENDER/MICRO用于测试
MEDIUM已验证可以训练
'''
# config = config | MICRO_CONFIG
config["NUM_UPDATES"] = (
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)

rng = jax.random.PRNGKey(config["SEED"])
if "NOISE_SEED" in config.keys():
    _noise_rng = jax.random.PRNGKey(config["NOISE_SEED"])
else:
    rng, _noise_rng = jax.random.split(rng)

# Create environment and apply wrappers
env = Env(env_params)

# Apply reward normalization if enabled in config
if config.get("REWARD_NORM", True):
    env = RewardNormWrapper(
        env, 
        gamma=config.get("REWARD_NORM_GAMMA", 0.99),
        epsilon=config.get("REWARD_NORM_EPSILON", 1e-8),
        clip_reward=config.get("REWARD_NORM_CLIP", 10.0)
    )
    print(f"Applied reward normalization with clip={config.get('REWARD_NORM_CLIP', 10.0)}")

env = LogWrapper(env, rng=_noise_rng)

# NOTE:从wrappers_mul中取得obs_dim、num_agents等数据
config = config | env.get_env_information_for_config()

if config["WANDB"]:
    if config["WANDB_API_KEY"] == "my_wandb_api_key":
        raise ValueError("no wandb api key!")
    
    os.environ['WANDB_API_KEY'] = config["WANDB_API_KEY"]
    wandb.tensorboard.patch(root_logdir=config['LOGDIR'])
    wandb.init(
        project="AeroPlanax",
        config=config,
        name=f'pitch_controller_seed_{config["SEED"]}',
        group=Env.__name__,
        notes=Env.__name__ + " with pitch-heading-velocity controller",
        reinit=True,
    )

Path(config["SAVEDIR"]).mkdir(parents=True, exist_ok=True)

# INIT NETWORK
(actor_network, critic_network), (ac_train_state, cr_train_state), start_epoch = init_network(config, DEFUALT_DIMS)


train_jit = jax.jit(make_train(
    config,
    env,
    (actor_network, critic_network),
    train_mode=config["TRAIN"],
    # NOTE:启用频繁保存
    # save_epochs=1
))

# dont use for loop
for i in range(config["FOR_LOOP_EPOCHS"]):
    out = train_jit(rng, (ac_train_state, cr_train_state), start_epoch)
    # out : Dict
    # {
    #   'runner_state': (
    #                   (train_states, env_state, last_obs, last_done, hstates, rng),
    #                    update_steps{NOTE:epoch}
    #               ),
    #   'metric': metric{NOTE:DISABLED} 
    # }

    runner_state = out['runner_state'][0]
    
    (ac_train_state, cr_train_state) = runner_state[0]
    rng = runner_state[5]
    start_epoch = jnp.array(out['runner_state'][1])
    
    if config["TRAIN"]:
        save_train((ac_train_state, cr_train_state), start_epoch, config["SAVEDIR"])

if config["WANDB"]:
    wandb.finish()



# output_dir = config["OUTPUTDIR"]
# Path(output_dir).mkdir(parents=True, exist_ok=True)
# import matplotlib.pyplot as plt
# plt.plot(out["metric"]["returned_episode_returns"].mean(-1).reshape(-1))
# plt.xlabel("Update Step")
# plt.ylabel("Return")
# plt.savefig(output_dir + '/returned_episode_returns.png')
# plt.cla()
# plt.plot(out["metric"]["returned_episode_lengths"].mean(-1).reshape(-1))
# plt.xlabel("Update Step")
# plt.ylabel("Return")
# plt.savefig(output_dir + '/returned_episode_lengths.png')
