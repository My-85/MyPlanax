import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
env_params = TaskParams()

str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
config = {
    "ACTOR_LR": 3e-4,  # Actor learning rate
    "CRITIC_LR": 1e-4,  # 本来是5e-3，又改到了3e-4
    "FC_DIM_SIZE": 256,  # Increased network capacity
    "GRU_HIDDEN_DIM": 256,  # Increased network capacity
    ##########################################
    # 更保守的 PPO 更新 & 略增熵，稳定前期探索
    "UPDATE_EPOCHS": 3,        # 8 -> 4 -> 3  （epoch 少一点，配合 KL 监控更稳）。更小的 clip 配合较少 epoch，KL 更稳定，策略不易“抖”。
    ##########################################
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.995,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.1, # 减小策略更新跨度，配合 KL 早停，能显著减少过拟合和策略提早确定导致的胜率波动。

    ##########################################
    # GPT建议，为稳定训练
    "VF_CLIP_EPS": 0.20,         # 值函数单独的 clip 系数（不要和策略的一样）
    "HUBER_DELTA": 1.0,          # Huber 损失的 delta
    "TARGET_KL": 0.02,           # 目标 KL 散度，把每个 epoch 的 KL 控到 ~0.02，若超过 1.5× 目标就提前停掉本轮 epoch
    "KL_STOP_MULT": 1.5,         # 1.5×TARGET_KL 触发“本次更新后停”

    # 熵系数自适应范围 & 调整速率，让熵系数在 0.0005~0.02 之间自适应调整，调整速率为 1.05，上限不变，下限到 1e-4 防止太快收敛；0.1 的调整步长在 0.8 目标胜率附近收敛更平滑。
    "ENT_COEF_MIN": 0.0005,
    "ENT_COEF_MAX": 0.02,
    "ENT_ADJ_RATE": 1.05,        # 熵系数调整倍率（>1）

    "LR_MULT_MIN": 0.1,         # 学习率乘子区间
    "LR_MULT_MAX": 1.0,
    ##########################################
    # 学习率退火
    "LR_DECAY": 0.999,
    "MIN_LR_MULT": 0.2, # 每个 update 轻微退火，最低衰减到初始学习率的 20%。

    ##########################################
    # 更保守的 PPO 更新 & 略增熵，稳定前期探索
    "ENT_COEF": 3e-3,          # 1e-3 -> 3e-3  （防策略过快确定化，entropy不至于塌太快）
    ##########################################
    "VF_COEF": 0.5, # 本来是1.0
    ##########################################
    # 更保守的 PPO 更新 & 略增熵，稳定前期探索
    "MAX_GRAD_NORM": 0.7,      # 0.5 -> 1.0 -> 0.7 （更稳的全局梯度裁剪阈值，配合下方手动裁剪）RNN + 大 batch，0.7 更稳，能抑制偶发的 actor/critic 梯度尖峰。
    ##########################################
    "RNN_GRAD_CLIP_VALUE": 0.5,  # Internal gradient clipping for RNN cells to prevent explosion
    "ACTIVATION": "relu",
    "ANNEAL_LR": False,
    "DEBUG": True,
    "NUM_ENVS": 3000, # 本来是3000
    "NUM_STEPS": 1500, # 本来是1500
    "TOTAL_TIMESTEPS": 5e7, 
    "SEED": 42,
    "NOISE_SEED": 42,
    "GROUP": "combat_hierarchy_5v5_enemy_limited_pitch_and_ally_limited_pitch_and_show_reward_in_wandb_and_cancel_last_done_mask_and_change_train_callback_and_change_critic_lr(to 1e-4)_and_change_vf_coef(to 0.5)_stop_gradient_rnn_reset(GPT_advice)_FOR_LOOP_EPOCHS: 30(rnn_baseline)_and_increase_curriculum_learning_steps_and_add_kl_stop_and_add_lr_mult_and_add_ent_coef_and_add_ent_coef_init_and_add_ent_coef_min_and_add_ent_coef_max_and_add_ent_adj_rate_and_add_min_lr_mult",
    "FOR_LOOP_EPOCHS": 30, # 本来是25
    "WANDB": True,
    "TRAIN": True,
    "WANDB_API_KEY" : "4c0cc04699296bed768adea4824fbaecea35dc59",
    "OUTPUTDIR": "results/" + "combat_hierarchy_pitch_new_" + str_date_time,
    "LOGDIR": "results/" + "combat_hierarchy_pitch_new_" + str_date_time + "/logs",
    "SAVEDIR": "results/" + "combat_hierarchy_pitch_new_" + str_date_time + "/checkpoints",
    # "LOADDIR": "/home/qiyuan/lczh/results/combat_hierarchy_pitch_new_2025-05-26-22-45/checkpoints/checkpoint_epoch_242" ,
    
    # Reward normalization settings
    "REWARD_NORM": True, # 本来是False
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
        name=config['GROUP'],
        group=Env.__name__,
        notes=Env.__name__ + " with pitch-heading-velocity controller(rnn)",
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
    
    (ac_train_state, cr_train_state) = runner_state[0][0]
    rng = runner_state[0][5]
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
