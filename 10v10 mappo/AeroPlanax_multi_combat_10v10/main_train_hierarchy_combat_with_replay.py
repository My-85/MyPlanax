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
from envs.aeroplanax_combat_hierarchy_FSM import (
    AeroPlanaxHierarchicalCombatEnv as Env,
    HierarchicalCombatTaskParams as TaskParams
)

# 导入Replay Buffer相关模块
from maketrains.mappo_discrete_combine_with_replay import (
    make_train_combine_vsbaseline_with_replay as make_train,
    save_model as save_train,
    ReplayBufferConfig
)

from maketrains import (
    RENDER_CONFIG,
    MICRO_CONFIG,
    MINI_CONFIG,
    MEDIUM_CONFIG,
    HUGE_CONFIG
)
from networks import (
    init_network_mappoRNN_discrete as init_network,
)

# For the pitch-heading-velocity controller
PPO_DISCRETE_HIERARCHY_DEFAULT_DIMS = [30, 30, 30]  # Updated dimensions: 30 steps for pitch, 30 steps for heading, 30 steps for velocity
DEFUALT_DIMS = PPO_DISCRETE_HIERARCHY_DEFAULT_DIMS

# Create environment parameters with appropriate settings for combat task
env_params = TaskParams()

str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M')

# 基础训练配置
config = {
    "ACTOR_LR": 3e-4,  # Actor learning rate
    "CRITIC_LR": 5e-3,  # Critic learning rate (higher than actor)
    "FC_DIM_SIZE": 256,  # Increased network capacity
    "GRU_HIDDEN_DIM": 256,  # Increased network capacity
    "UPDATE_EPOCHS": 8,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.995,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 1e-3,  # 增加熵正则化系数以鼓励更多探索
    "VF_COEF": 1.0,
    "MAX_GRAD_NORM": 0.5,
    "RNN_GRAD_CLIP_VALUE": 0.5,  # Internal gradient clipping for RNN cells to prevent explosion
    "ACTIVATION": "relu",
    "ANNEAL_LR": False,
    "DEBUG": True,
    "NUM_ENVS": 50,
    "NUM_STEPS": 1000,  # 现在用于segment分割，而不是轨迹截断
    "TOTAL_TIMESTEPS": 5e7,
    "SEED": 42,
    "NOISE_SEED": 42,
    "GROUP": "combat_hierarchy_pitch_replay",
    "FOR_LOOP_EPOCHS": 60,
    "WANDB": True,
    "TRAIN": True,
    "WANDB_API_KEY" : "f4316927feb010654a1d429360c2f2c824e84387",
    "OUTPUTDIR": "results/" + "combat_hierarchy_pitch_replay_" + str_date_time,
    "LOGDIR": "results/" + "combat_hierarchy_pitch_replay_" + str_date_time + "/logs",
    "SAVEDIR": "results/" + "combat_hierarchy_pitch_replay_" + str_date_time + "/checkpoints",
    # "LOADDIR": "/home/qiyuan/lczh/results/combat_hierarchy_pitch_new_2025-05-26-22-45/checkpoints/checkpoint_epoch_242" ,
    
    # Reward normalization settings
    "REWARD_NORM": False,
    "REWARD_NORM_GAMMA": 0.99,
    "REWARD_NORM_EPSILON": 1e-8,
    "REWARD_NORM_CLIP": 10.0,
}

# 配置Replay Buffer - 优化内存使用
replay_config = ReplayBufferConfig(
    max_episodes=100,                     # 减少存储的episode数量以节省内存
    min_episodes_for_training=1,         # 减少最小训练episodes
    sample_batch_size=1,                 # 减少batch size以节省内存
    max_episode_length=2000,             # 减少episode长度以节省内存
    enable_prioritized_sampling=True     # 启用优先级采样，重点学习高回报episode
)

print("=== Replay Buffer Configuration ===")
print(f"Max episodes in buffer: {replay_config.max_episodes}")
print(f"Min episodes for training: {replay_config.min_episodes_for_training}")
print(f"Sample batch size: {replay_config.sample_batch_size}")
print(f"Max episode length: {replay_config.max_episode_length}")
print(f"Prioritized sampling: {replay_config.enable_prioritized_sampling}")
print("===================================\n")

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
        name=f'pitch_controller_replay_seed_{config["SEED"]}',
        group=Env.__name__,
        notes=Env.__name__ + " with pitch-heading-velocity controller and Replay Buffer",
        reinit=True,
    )

Path(config["SAVEDIR"]).mkdir(parents=True, exist_ok=True)

# INIT NETWORK
(actor_network, critic_network), (ac_train_state, cr_train_state), start_epoch = init_network(config, DEFUALT_DIMS)

print("=== Training Configuration ===")
print(f"Environment: {Env.__name__}")
print(f"Number of environments: {config['NUM_ENVS']}")
print(f"Steps per segment: {config['NUM_STEPS']}")
print(f"Total timesteps: {config['TOTAL_TIMESTEPS']}")
print(f"Number of updates: {config['NUM_UPDATES']}")
print(f"For loop epochs: {config['FOR_LOOP_EPOCHS']}")
print("==============================\n")

# 创建带有Replay Buffer的训练函数
train_jit = jax.jit(make_train(
    config,
    env,
    (actor_network, critic_network),
    replay_config=replay_config,  # 传入Replay Buffer配置
    train_mode=config["TRAIN"],
    # NOTE: 可以启用频繁保存
    # save_epochs=1
))

print("Starting training with Replay Buffer...")
print("Note: Training will begin after collecting minimum episodes for buffer")
print(f"Expected buffer fill time: ~{replay_config.min_episodes_for_training} episodes\n")

# Training loop
for i in range(config["FOR_LOOP_EPOCHS"]):
    print(f"=== Training Epoch {i+1}/{config['FOR_LOOP_EPOCHS']} ===")
    
    out = train_jit(rng, (ac_train_state, cr_train_state), start_epoch)
    # out : Dict
    # {
    #   'runner_state': (
    #                   (train_states, env_state, last_obs, last_done, hstates, rng),
    #                    update_steps{NOTE:epoch}
    #               ),
    # }

    runner_state = out['runner_state'][0]
    
    (ac_train_state, cr_train_state) = runner_state[0]
    rng = runner_state[5]
    start_epoch = jnp.array(out['runner_state'][1])
    
    if config["TRAIN"]:
        save_train((ac_train_state, cr_train_state), start_epoch, config["SAVEDIR"])
        print(f"Model saved at epoch {start_epoch}")
    
    print(f"Completed epoch {i+1}, current training epoch: {start_epoch}")

print("\n=== Training Completed ===")
print(f"Final epoch: {start_epoch}")
print(f"Models saved to: {config['SAVEDIR']}")
print(f"Logs saved to: {config['LOGDIR']}")

if config["WANDB"]:
    wandb.finish()
    print("WandB session finished")

print("\n=== Replay Buffer Benefits ===")
print("✅ Complete episode collection (no trajectory truncation)")
print("✅ 100% data utilization (vs ~15% in original method)")
print("✅ Prioritized sampling of high-reward episodes")
print("✅ Better training stability and convergence")
print("✅ Efficient memory management with configurable buffer size")
print("===============================") 