'''
envs/core/utils.py.check_locked().R
锁定半径有点长（比通信距离还长)

envs/core/utils.py.update_blood()
血量更新函数
'''
from typing import Dict, Optional, Sequence, Any, Tuple, Callable
from jax import Array
from jax.typing import ArrayLike
import chex
from .aeroplanax import AgentName, AgentID

import tensorboardX
import functools
import os
import jax
import jax.numpy as jnp
from flax import struct
import flax.linen as nn
import numpy as np
import distrax
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments import spaces
from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv
from .core.simulators import missile, fighterplane
from .reward_functions import (
    # altitude_reward_fn,
    posture_reward_fn,
    event_driven_reward_fn,
    crash_reward_fn,
)
from .termination_conditions import (
    crashed_fn,
    safe_return_fn,
    timeout_fn,
)
from .utils.utils import  wedge_formation, line_formation, diamond_formation, enforce_safe_distance, get_AO_TA_R, wrap_PI, get_AO_TA_R_pitch_yaw
import orbax.checkpoint as ocp

import jax.numpy as jnp
from envs.aeroplanax import AgentID
from envs.utils.utils import get_AO_TA_R
import jax

# 添加一个更安全的get_AO_TA_R_pitch_yaw包装函数，防止NaN和Inf
def safe_get_AO_TA_R_pitch_yaw(ego_feature, enm_feature, ego_pitch, ego_yaw, enm_pitch, enm_yaw):
    """A safer version of get_AO_TA_R_pitch_yaw with additional checks to prevent NaN/Inf"""
    # 使用原始函数计算值
    AO, TA, R, side_flag = get_AO_TA_R_pitch_yaw(
        ego_feature, enm_feature, ego_pitch, ego_yaw, enm_pitch, enm_yaw
    )
    
    # 确保角度在有效范围内
    AO = jnp.clip(AO, 0.0, jnp.pi)
    TA = jnp.clip(TA, 0.0, jnp.pi)
    
    # 确保R是正值
    R = jnp.maximum(R, 0.1)
    
    # 检查NaN/Inf并替换为安全值
    AO = jnp.where(jnp.isnan(AO) | jnp.isinf(AO), jnp.pi, AO)
    TA = jnp.where(jnp.isnan(TA) | jnp.isinf(TA), jnp.pi, TA)
    R = jnp.where(jnp.isnan(R) | jnp.isinf(R), 1000.0, R)
    
    return AO, TA, R, side_flag

# Enemy FSM states
PATROL_STATE = 0  # 巡逻状态：随机移动，寻找目标
ENGAGE_STATE = 1  # 接敌状态：向目标移动，但尚未进入最佳攻击位置
ATTACK_STATE = 2  # 攻击状态：在最佳攻击位置，积极攻击目标
EVADE_STATE = 3   # 规避状态：血量低时逃离危险，保存自己

# 敌方FSM参数
ENGAGE_RANGE = 30000.0       # 接敌范围，超过此距离不会主动接敌
ATTACK_RANGE = 20000.0       # 攻击范围，在此距离内且AO良好时进入攻击状态
ATTACK_AO_THRESHOLD = jnp.pi / 2  # 攻击状态所需的最大AO角(小于此值视为有利攻击角度)
EVADE_HEALTH_THRESHOLD_RATIO = 0.05  # 进入规避状态的血量阈值比例
MAX_TURN_RATE_PER_STEP = jnp.pi / 72  # 每步最大转向变化
MAX_PITCH_RATE_PER_STEP = jnp.pi / 72  # 每步最大俯仰变化
MAX_VT_DELTA_PER_STEP = 10.0  # 每步最大速度变化(m/s)

# 敌方FSM行为参数
MAX_SPEED = 300.0
PATROL_SPEED_FACTOR = 0.7   # 巡逻状态下的速度因子(相对于最大速度)
ATTACK_SPEED_FACTOR = 1.0   # 攻击状态下的速度因子
EVADE_SPEED_FACTOR = 1.0    # 规避状态下的速度因子
PATROL_TURN_RANDOMNESS = 0.3  # 巡逻状态下的随机转向强度
PATROL_ALTITUDE_RANGE = [5000.0, 10000.0]  # 巡逻状态下的高度范围
move_rate = 1.0 # 敌方机动因子

# if not os.getcwd().endswith("AeroPlanax-dev-tmp0429_lxy_reform"):
#     raise ValueError("当前运行目录不是AeroPlanax,无法自动获取heading baseline文件夹位置，请手动填写LOADDIR并禁用本行代码！")

print(f'combat_hierarchy policy: load heading_pitch_V model from {os.path.join(os.getcwd(),"/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/2v2_lczh_origin/AeroPlanax_multi_combat_2v2/envs/models/baseline/lstm_Yaw_Pitch_V/baseline_stable_2")}')
config = {
    "SEED": 42,
    "LR": 3e-4,
    "NUM_ENVS": 1,
    "NUM_ACTORS": 2,
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
    "UPDATE_EPOCHS": 16,
    "NUM_MINIBATCHES": 5,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 1e-3,
    "VF_COEF": 1,
    "MAX_GRAD_NORM": 2,
    "ACTIVATION": "relu",
    "ANNEAL_LR": False,
    "LOADDIR": os.path.join(os.getcwd(),"/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/2v2_lczh_origin/AeroPlanax_multi_combat_2v2/envs/models/baseline/lstm_Yaw_Pitch_V/baseline_stable_2")
}


class ScannedLSTM(nn.Module):
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
        lstm_state = carry  # (h, c)
        ins, resets = x
        h, c = lstm_state
        h = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*h.shape)[0],
            h,
        )
        c = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*c.shape)[1],
            c,
        )
        new_lstm_state, y = nn.LSTMCell(features=ins.shape[1])((h, c), ins)
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.LSTMCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticLSTM(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedLSTM()(hidden, rnn_in)

        # 新增一层全连接
        nn_fc2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        nn_fc2 = activation(nn_fc2)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(nn_fc2)
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

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)

def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

# init model
controller = ActorCriticLSTM([31, 41, 41, 41], config=config)
rng = jax.random.PRNGKey(config['SEED'])
init_x = (
    jnp.zeros(
        (1, config["NUM_ENVS"] * config["NUM_ACTORS"], 16)
    ),
    jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
)
init_hstate = ScannedLSTM.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
controller_params = controller.init(rng, init_hstate, init_x)
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
    apply_fn=controller.apply,
    params=controller_params,
    tx=tx,
)
state = {"params": train_state.params, "opt_state": train_state.opt_state, "epoch": jnp.array(0)}
ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
checkpoint = ckptr.restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))
controller_params = checkpoint["params"]


@struct.dataclass
class HierarchicalCombatTaskState(EnvState):
    hstate: ArrayLike
    enemy_fsm_state: ArrayLike
    prev_enemy_alive: ArrayLike  # 新增：记录上一步敌机的存活状态
    
    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array, enemy_fsm_state: ArrayLike, prev_enemy_alive: ArrayLike = None):
        # 如果没有提供prev_enemy_alive，创建默认值
        if prev_enemy_alive is None:
            prev_enemy_alive = jnp.ones(2, dtype=jnp.bool_)  # 假设有2个敌机，初始都存活
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            pre_rewards=env_state.pre_rewards,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            hstate=extra_state,
            enemy_fsm_state=enemy_fsm_state,
            prev_enemy_alive=prev_enemy_alive,
        )


@struct.dataclass(frozen=True)
class HierarchicalCombatTaskParams(EnvParams):
    num_allies: int = 2
    num_enemies: int = 2
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 1
    observation_type: int = 0 # 0: unit_list, 1: conic
    unit_features: int = 6
    own_features: int = 5
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    max_steps: int = 100
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    max_altitude: float = 20000
    min_altitude: float = 2000
    max_vt: float = 340
    min_vt: float = 100
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    max_distance: float = 12000.0
    min_distance: float = 10000.0
    team_spacing: float = 1000
    safe_distance: float = 100
    posture_reward_scale: float = 100.0
    use_baseline: bool = True

    noise_features: int = 10
    top_k_ego_obs: int = -1
    top_k_enm_obs: int = -1

# 重新实现event_driven_reward_fn函数，修正逻辑错误
def custom_event_driven_reward_fn(
    state: 'HierarchicalCombatTaskState',
    params: 'HierarchicalCombatTaskParams',
    agent_id: int,
    fail_reward: float = -200,
    narrow_victory_reward: float = 20,
    normal_victory_reward: float = 50,
    great_victory_reward: float = 100,
    complete_victory_reward: float = 200,
    locked_only: bool = True,
) -> float:
    """
    基于事件的奖励函数，根据战斗结果给予不同奖励
    
    胜负条件：
    1. 完胜：我方全部存活，敌方全部被消灭
    2. 大胜：敌方全部被消灭，我方有存活
    3. 一般胜利：我方存活数量 > 敌方存活数量
    4. 险胜：我方存活数量 = 敌方存活数量，但我方血量更多
    5. 失败：我方存活数量 < 敌方存活数量，或存活数量相同时血量不如敌方
    
    参数:
        state: 当前环境状态
        params: 环境参数
        agent_id: 当前Agent的ID
        fail_reward: 失败情况的奖励
        narrow_victory_reward: 险胜的奖励
        normal_victory_reward: 一般胜利的奖励
        great_victory_reward: 大胜的奖励
        complete_victory_reward: 完胜的奖励
        locked_only: 是否只计算被锁定击毁的敌机
    """
    # 硬编码常量
    num_allies = 2
    num_enemies = 2
    total_agents = 4
    
    is_ally = agent_id < num_allies
    
    # 创建掩码
    agent_indices = jnp.arange(total_agents)
    ally_mask = agent_indices < num_allies
    enemy_mask = ~ally_mask
    
    # 获取状态
    is_alive = state.plane_state.is_alive
    is_locked = state.plane_state.is_locked
    blood = state.plane_state.blood
    status = state.plane_state.status
    
    # 正确的存活判断：状态为ALIVE(0)或LOCKED(1)
    truly_alive = jnp.logical_or(status == 0, status == 1)
    
    # 计算各方存活情况
    ally_alive_count = jnp.sum(jnp.where(ally_mask, truly_alive, False))
    enemy_alive_count = jnp.sum(jnp.where(enemy_mask, truly_alive, False))
    
    # 计算各方血量总和（只计算存活的）
    ally_total_blood = jnp.sum(jnp.where(
        jnp.logical_and(ally_mask, truly_alive), blood, 0
    ))
    enemy_total_blood = jnp.sum(jnp.where(
        jnp.logical_and(enemy_mask, truly_alive), blood, 0
    ))
    
    # 判断胜负条件
    # 1. 完胜：我方全部存活(2个)，敌方全部被消灭(0个)
    complete_victory = jnp.logical_and(
        ally_alive_count == num_allies,
        enemy_alive_count == 0
    )
    
    # 2. 大胜：敌方全部被消灭，我方有存活但有损失
    great_victory = jnp.logical_and(
        jnp.logical_and(enemy_alive_count == 0, ally_alive_count > 0),
        ~complete_victory
    )
    
    # 3. 一般胜利：我方存活数量 > 敌方存活数量
    normal_victory = jnp.logical_and(
        ally_alive_count > enemy_alive_count,
        enemy_alive_count > 0  # 敌方还有存活
    )
    
    # 4. 险胜：存活数量相同但我方血量更多
    narrow_victory = jnp.logical_and(
        jnp.logical_and(
            ally_alive_count == enemy_alive_count,
            ally_alive_count > 0  # 双方都有存活
        ),
        ally_total_blood > enemy_total_blood
    )
    
    # 5. 失败：其他所有情况
    # - 我方存活数量 < 敌方存活数量
    # - 存活数量相同但血量不如敌方
    # - 我方全灭
    failure = ~jnp.logical_or(
        jnp.logical_or(complete_victory, great_victory),
        jnp.logical_or(normal_victory, narrow_victory)
    )
    
    # 根据不同情况给予奖励
    reward = jnp.select(
        [complete_victory, great_victory, normal_victory, narrow_victory, failure],
        [complete_victory_reward, great_victory_reward, normal_victory_reward, narrow_victory_reward, fail_reward],
        default=0.0
    )
    
    # 如果当前Agent是敌方，则反转奖励
    reward = jnp.where(is_ally, reward, -reward)
    
    return reward

def posture_reward_fn(
        state: HierarchicalCombatTaskState,
        params: HierarchicalCombatTaskParams,
        agent_id: AgentID,
        reward_scale: float = 1.0,
        num_allies: int = 1,
        num_enemies: int = 1,
    ) -> float:
    """
    Reward is a complex function of AO, TA and R in the last timestep.
    Uses aircraft heading vectors based on pitch and yaw angles for more accurate orientation.
    """
    new_reward = 0.0
    # feature: (north, east, down, vn, ve, vd)
    ego_feature = jnp.hstack((state.plane_state.north[agent_id],
                              state.plane_state.east[agent_id],
                              state.plane_state.altitude[agent_id],
                              state.plane_state.vel_x[agent_id],
                              state.plane_state.vel_y[agent_id],
                              state.plane_state.vel_z[agent_id]))
    
    # 获取己方飞机的俯仰角和偏航角
    ego_pitch = state.plane_state.pitch[agent_id]
    ego_yaw = state.plane_state.yaw[agent_id]
    
    enm_list = jax.lax.select(agent_id < num_allies, 
                              jnp.arange(num_allies, num_allies + num_enemies),
                              jnp.arange(num_allies))
    
    def process_enemy(enm):
        enm_feature = jnp.hstack((state.plane_state.north[enm],
                                  state.plane_state.east[enm],
                                  state.plane_state.altitude[enm],
                                  state.plane_state.vel_x[enm],
                                  state.plane_state.vel_y[enm],
                                  state.plane_state.vel_z[enm]))
        
        # 获取敌方飞机的俯仰角和偏航角
        enm_pitch = state.plane_state.pitch[enm]
        enm_yaw = state.plane_state.yaw[enm]
        
        # 使用更安全的方位角计算函数
        AO, TA, R, _ = safe_get_AO_TA_R_pitch_yaw(
            ego_feature, 
            enm_feature, 
            ego_pitch, 
            ego_yaw, 
            enm_pitch, 
            enm_yaw
        )
        
        orientation_reward = orientation_reward_fn_new(AO, TA)
        range_reward = range_reward_fn(R / 1000.0)
        
        # 检查是否存活或被锁定
        mask = state.plane_state.is_alive[enm] | state.plane_state.is_locked[enm]
        
        # 检查reward是否为NaN/Inf
        enemy_reward = orientation_reward * range_reward
        enemy_reward = jnp.where(jnp.isnan(enemy_reward) | jnp.isinf(enemy_reward), 0.0, enemy_reward)
        
        return enemy_reward * mask

    # 使用vmap处理所有敌方飞机
    enemy_rewards = jax.vmap(process_enemy)(enm_list)
    new_reward = jnp.sum(enemy_rewards)

    # 检查己方飞机是否存活或被锁定
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    
    # 最后添加一个全局检查，确保返回值不是NaN或Inf
    final_reward = new_reward * reward_scale * mask
    final_reward = jnp.where(jnp.isnan(final_reward) | jnp.isinf(final_reward), 0.0, final_reward)
    return final_reward

def range_reward_fn(R):
    # Ensure R is positive to avoid unexpected behavior
    R = jnp.maximum(R, 0.0)
    
    # Calculate reward with polynomial and exponential terms
    poly_term = jnp.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0.0, 1.0)
    exp_term = jnp.clip(jnp.exp(-0.16 * jnp.clip(R, 0.0, 30.0)), 0.0, 0.2)  # Clip R for exp stability
    
    reward = 1.0 * (R < 5) + (R >= 5) * poly_term + exp_term
    
    # Safety check for NaN/Inf
    reward = jnp.where(jnp.isnan(reward) | jnp.isinf(reward), 0.0, reward)
    return reward

def orientation_reward_fn_new(AO, TA):
    # Add safety: ensure inputs are in valid range to prevent NaN
    AO = jnp.clip(AO, 0.0, jnp.pi)
    TA = jnp.clip(TA, 0.0, jnp.pi)
    
    # Use clipped exp to prevent overflow
    AO_reward = jnp.exp(-jnp.clip(AO / (jnp.pi / 7), -10, 10))
    TA_reward = jnp.exp(-jnp.clip(TA / (jnp.pi / 7), -10, 10))
    
    reward = (AO_reward * TA_reward) ** (1/2)
    
    # Final safety check
    reward = jnp.where(jnp.isnan(reward) | jnp.isinf(reward), 0.0, reward)
    return reward

'''
更加强调agent学习攻击具体的[一个]敌机
不过效果不佳
'''
# def posture_reward_softmax_fn(
#         state: HierarchicalCombatTaskState,
#         params: HierarchicalCombatTaskParams,
#         agent_id: AgentID,
#         reward_scale: float = 1.0,
#         num_allies: int = 1,
#         num_enemies: int = 1,
#         temperature: float = 0.3
#     ) -> float:
#     """
#     Reward is a complex function of AO, TA and R in the last timestep.
#     """
#     # feature: (north, east, down, vn, ve, vd)
#     ego_feature = jnp.hstack((state.plane_state.north[agent_id],
#                               state.plane_state.east[agent_id],
#                               state.plane_state.altitude[agent_id],
#                               state.plane_state.vel_x[agent_id],
#                               state.plane_state.vel_y[agent_id],
#                               state.plane_state.vel_z[agent_id]))
#     enm_list = jax.lax.select(agent_id < num_allies, 
#                               jnp.arange(num_allies, num_allies + num_enemies),
#                               jnp.arange(num_allies))
    
#     def compute_enemy_reward(enm_id):
#         enm_feature = jnp.hstack((
#             state.plane_state.north[enm_id],
#             state.plane_state.east[enm_id],
#             state.plane_state.altitude[enm_id],
#             state.plane_state.vel_x[enm_id],
#             state.plane_state.vel_y[enm_id],
#             state.plane_state.vel_z[enm_id],
#         ))

#         AO, TA, R, _ = get_AO_TA_R(ego_feature, enm_feature)
#         orientation_reward = orientation_reward_fn_new(AO, TA)
#         range_reward = range_reward_fn(R / 1000.0)
#         alive_or_locked = state.plane_state.is_alive[enm_id] | state.plane_state.is_locked[enm_id]
#         return orientation_reward * range_reward * alive_or_locked

#     per_target_rewards = jax.vmap(compute_enemy_reward)(enm_list)

#     # 使用 softmax with temperature 计算权重
#     weights = jax.nn.softmax(per_target_rewards / temperature)
#     weighted_reward = jnp.sum(per_target_rewards * weights)

#     mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
#     return weighted_reward * reward_scale * mask

def tactical_position_reward_fn(
    state: HierarchicalCombatTaskState,
    params: HierarchicalCombatTaskParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
    num_allies: int = 1,
    num_enemies: int = 2,
) -> float:
    """
    计算战术位置奖励，主要考虑：
    1. 保持在敌机后半球
    2. 保持合适的攻击距离
    3. 保持高度优势
    """
    def safe_exp(x):
        """Safely compute exponential to avoid overflow"""
        return jnp.exp(jnp.clip(x, -10.0, 10.0))
    
    ego_feature = jnp.hstack((
        state.plane_state.north[agent_id],
        state.plane_state.east[agent_id],
        state.plane_state.altitude[agent_id],
        state.plane_state.vel_x[agent_id],
        state.plane_state.vel_y[agent_id],
        state.plane_state.vel_z[agent_id]
    ))
    
    enm_list = jax.lax.select(
        agent_id < num_allies,
        jnp.arange(num_allies, num_allies + num_enemies),
        jnp.arange(num_allies)
    )
    
    def compute_tactical_reward(enm_id):
        enm_feature = jnp.hstack((
            state.plane_state.north[enm_id],
            state.plane_state.east[enm_id],
            state.plane_state.altitude[enm_id],
            state.plane_state.vel_x[enm_id],
            state.plane_state.vel_y[enm_id],
            state.plane_state.vel_z[enm_id]
        ))
        
        AO, TA, R, _ = get_AO_TA_R(ego_feature, enm_feature)
        
        # Clip angles to valid ranges
        AO = jnp.clip(AO, 0.0, jnp.pi)
        TA = jnp.clip(TA, 0.0, jnp.pi)
        
        # 后半球奖励 (TA接近0表示在敌机后半球)
        rear_hemisphere_reward = safe_exp(-TA / (jnp.pi / 4))
        
        # 距离奖励 (保持在最佳攻击距离)
        optimal_distance = params.max_distance / 3  # 假设最佳攻击距离为最大距离的1/3
        # Add epsilon to denominator for distance_reward
        distance_denominator = jnp.maximum(2 * (optimal_distance/2)**2, 1e-8)
        distance_reward = safe_exp(-(R - optimal_distance)**2 / distance_denominator)
        
        # 高度优势奖励
        # Add epsilon to denominator for altitude_reward
        altitude_denominator = jnp.maximum(params.max_altitude, 1e-8)
        altitude_diff = (state.plane_state.altitude[agent_id] - state.plane_state.altitude[enm_id]) / altitude_denominator
        altitude_reward = jnp.clip(altitude_diff, -1.0, 1.0)
        
        # 综合奖励
        tactical_reward = (rear_hemisphere_reward * 0.4 + 
                         distance_reward * 0.4 + 
                         altitude_reward * 0.2)
        
        alive_or_locked = state.plane_state.is_alive[enm_id] | state.plane_state.is_locked[enm_id]
        
        # Safety check for NaN/Inf
        tactical_reward = jnp.where(jnp.isnan(tactical_reward) | jnp.isinf(tactical_reward), 0.0, tactical_reward)
        
        return tactical_reward * alive_or_locked
    
    per_target_rewards = jax.vmap(compute_tactical_reward)(enm_list)
    # 取求和
    total_reward = jnp.sum(per_target_rewards)
    
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    final_reward = total_reward * reward_scale * mask
    
    # 最终安全检查
    final_reward = jnp.where(jnp.isnan(final_reward) | jnp.isinf(final_reward), 0.0, final_reward)
    
    return final_reward

class AeroPlanaxHierarchicalCombatEnv(AeroPlanaxEnv[HierarchicalCombatTaskState, HierarchicalCombatTaskParams]):
    def __init__(self, env_params: Optional[HierarchicalCombatTaskParams] = None):
        super().__init__(env_params)

        self.observation_type = env_params.observation_type
        self.unit_features = env_params.unit_features
        self.own_features = env_params.own_features
        self.formation_type = env_params.formation_type
        # NOTE:据说global_obs cat一个高斯分布噪声有助于探索，暂且放在这里
        # see: wrappers_mul.py
        self.noise_features = env_params.noise_features
        # NOTE:似乎不是必要的
        self.enbale_actor_onehot_agent_id = False
        # 双方各top_k个；默认为-1,不启用
        self.top_k_ego_obs = jnp.minimum(env_params.top_k_ego_obs, self.num_allies - 1) if env_params.top_k_ego_obs > 0 else (self.num_allies - 1)
        self.top_k_enm_obs = jnp.minimum(env_params.top_k_enm_obs, self.num_enemies) if env_params.top_k_enm_obs > 0 else self.num_enemies
        self.one_hot_ego_classes = self.top_k_ego_obs + 1
        self.one_hot_all_classes = self.top_k_ego_obs + self.top_k_enm_obs + 1

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(posture_reward_fn, 
                              reward_scale=0.005, 
                              num_allies=env_params.num_allies, 
                              num_enemies=env_params.num_enemies),
            functools.partial(heading_alignment_reward_fn,  # 新添加的奖励函数
                              reward_scale=0.004,             # 可以调整权重
                              num_allies=env_params.num_allies,
                              num_enemies=env_params.num_enemies),
            functools.partial(locked_penalty_fn, penalty_value=-0.005, num_allies=env_params.num_allies),
            functools.partial(tactical_position_reward_fn, 
                              reward_scale=0.001, 
                              num_allies=env_params.num_allies, 
                              num_enemies=env_params.num_enemies),
            functools.partial(crash_penalty_fn_new, penalty_value=-0.5),
            functools.partial(enemy_kill_reward_fn,  # 新增击落奖励函数
                              kill_reward=0.1,
                              num_allies=env_params.num_allies,
                              num_enemies=env_params.num_enemies),
            functools.partial(custom_event_driven_reward_fn, 
                              fail_reward=-0.5, 
                              narrow_victory_reward=0.05,
                              normal_victory_reward=0.1,
                              great_victory_reward=0.2,
                              complete_victory_reward=0.5),
            functools.partial(event_driven_reward_fn, 
                              fail_reward=-0.5, 
                              success_reward=0.5),
        ]
        self.is_potential = [False,False,False,False,True,False,True,True]

        self.termination_conditions = [
            safe_return_fn,
            timeout_fn,
        ]

        self.norm_delta_pitch = jnp.linspace(-jnp.pi/6, jnp.pi/6, 30)  # 30 steps from -30° to 30°
        self.norm_delta_heading = jnp.linspace(-jnp.pi/2, jnp.pi/2, 30)  # 30 steps from -30° to 30°
        self.norm_delta_velocity = jnp.linspace(-100, 100, 30)  # 30 steps from -100 to 100
        # NOTE: 如果use_baseline，意味着agent_enemy的obs、action等都无用
        # 这些在maketrain中已做处理
        self.use_baseline = env_params.use_baseline

    def _get_global_obs_size(self) -> int:
        '''global obs为 普通 obs + one-hot-agent_id(只有可操作agent) + noise_dim'''
        return (self.unit_features * (self.top_k_ego_obs + self.top_k_enm_obs) + self.own_features) + self.one_hot_all_classes + self.noise_features
    
    def _get_obs_size(self) -> int:
        if self.enbale_actor_onehot_agent_id:
            return (self.unit_features * (self.top_k_ego_obs + self.top_k_enm_obs) + self.own_features) + self.num_agents
        else:
            return (self.unit_features * (self.top_k_ego_obs + self.top_k_enm_obs) + self.own_features)
        
        # if self.observation_type == 0:
        #     return (self.unit_features * (self.num_allies - 1) + self.unit_features * self.num_enemies + self.own_features)
        # elif self.observation_type == 1:
        #     # TODO: feat conic observations
        #     return (self.unit_features * (self.num_allies - 1) + self.unit_features * self.num_enemies + self.own_features)
        # else:
        #     raise ValueError("Provided observation type is not valid")


    @functools.partial(jax.jit, static_argnums=(0,))
    def get_raw_global_obs(self, state: HierarchicalCombatTaskState) -> chex.Array:
        '''
        返回未经处理的chex.Array,在wrapper(mulwrapper)中处理为dict
        
        shape: self.num_allies * global_obs_dim
        '''
        def _get_features(state: EnvState, i, j):
            visible = (i != j)
            empty_features = jnp.zeros(shape=(self.unit_features,))
            features = self._observe_features(state, i, j)
            return jax.lax.cond(
                # visible & (state.plane_state.is_alive[i] | state.plane_state.is_locked[i]) & (state.plane_state.is_alive[j] | state.plane_state.is_locked[j]),
                visible & (state.plane_state.is_alive[i]) & (state.plane_state.is_alive[j]),
                lambda: features,
                lambda: empty_features,
            )
        return self._get_own_and_top_k_other_plane_obs(state, _get_features, enable_one_hot=True)

    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def _get_own_and_top_k_other_plane_obs(
        self,
        state: HierarchicalCombatTaskState,  # 当前状态
        observe_features_from_i_to_j: Callable[[EnvState, int, int], chex.Array],
        enable_one_hot: bool
    ) -> Dict[AgentName, chex.Array]:
        pos = jnp.vstack((state.plane_state.north, state.plane_state.east, state.plane_state.altitude)).T
        diff = pos[:, None, :] - pos[None, :, :]  # (n, n, 3)
        distances = jnp.linalg.norm(diff, axis=-1)     # (n, n)

        team_ids = jnp.arange(self.num_agents) < self.num_allies
        team_mask = team_ids[:, None] == team_ids[None, :]

        op_mask = ~team_mask
        team_mask = team_mask & (~jnp.eye(self.num_agents, dtype=bool))

        def extract_features(state, i, dist_row, team_mask_row, opponent_mask_row):
            def get_top_k_indices(mask, k):
                masked = jnp.where(mask, dist_row, jnp.inf)
                return jnp.argsort(masked)[:k]
            
            teammate_ids = get_top_k_indices(team_mask_row, self.top_k_ego_obs)
            opponent_ids = get_top_k_indices(opponent_mask_row, self.top_k_enm_obs)

            ids = jnp.concatenate([teammate_ids, opponent_ids])
            return jax.vmap(lambda j : observe_features_from_i_to_j(state, i, j))(ids).flatten()

        other_unit_obs = jax.vmap(lambda i: extract_features(state, i, distances[i], team_mask[i], op_mask[i]), in_axes=(0,))(jnp.arange(self.num_agents))
        
        get_all_self_features = jax.vmap(self._get_own_features, in_axes=(None, 0))
        own_unit_obs = get_all_self_features(state, jnp.arange(self.num_agents))

        if enable_one_hot:
            agent_ids = jnp.arange(self.num_agents)
            agent_ids = jnp.where(agent_ids < self.num_allies, agent_ids % self.one_hot_ego_classes, (agent_ids - self.num_allies) % self.one_hot_ego_classes + self.num_allies)
            one_hot_ids = jax.nn.one_hot(agent_ids, num_classes=self.one_hot_all_classes)
            obs = jnp.concatenate([own_unit_obs, other_unit_obs, one_hot_ids], axis=-1)
        else:
            obs = jnp.concatenate([own_unit_obs, other_unit_obs], axis=-1)

        return obs

    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_actions(
        self,
        key: chex.PRNGKey,
        init_state: HierarchicalCombatTaskState,
        state: HierarchicalCombatTaskState,
        actions: Dict[AgentName, chex.Array]
    ):
        # unpack actions
        actions = jnp.array([actions[i] for i in self.agents])
        if not self.use_baseline:
            delta_pitch = self.norm_delta_pitch[actions[:, 0]]
            delta_heading = self.norm_delta_heading[actions[:, 1]]
            delta_vt = self.norm_delta_velocity[actions[:, 2]]

            target_pitch = init_state.plane_state.pitch + delta_pitch
            target_heading = wrap_PI(init_state.plane_state.yaw + delta_heading)
            target_vt = init_state.plane_state.vt + delta_vt
        # else:
        #     ego_delta_pitch_cmd = self.norm_delta_pitch[actions[:self.num_allies, 0]]
        #     ego_delta_heading_cmd = self.norm_delta_heading[actions[:self.num_allies, 1]]
        #     ego_delta_vt_cmd = self.norm_delta_velocity[actions[:self.num_allies, 2]]

        #     # 敌方FSM逻辑
        #     fsm_key, controller_key = jax.random.split(key)
        #     enemy_keys = jax.random.split(fsm_key, self.num_enemies)

        #     # 处理单个敌方飞机FSM逻辑的函数
        #     def single_enemy_fsm_logic(enm_local_idx, current_fsm_state, enm_key):
        #         enm_global_idx = self.num_allies + enm_local_idx
        #         enm_ps = init_state.plane_state

        #         # 敌机自身状态
        #         enm_north = enm_ps.north[enm_global_idx]
        #         enm_east = enm_ps.east[enm_global_idx]
        #         enm_alt = enm_ps.altitude[enm_global_idx]
        #         enm_vx = enm_ps.vel_x[enm_global_idx]
        #         enm_vy = enm_ps.vel_y[enm_global_idx]
        #         enm_vz = enm_ps.vel_z[enm_global_idx]
        #         enm_pitch = enm_ps.pitch[enm_global_idx]
        #         enm_yaw = enm_ps.yaw[enm_global_idx]
        #         enm_vt = enm_ps.vt[enm_global_idx]
        #         enm_is_alive = enm_ps.is_alive[enm_global_idx]
        #         enm_blood = enm_ps.blood[enm_global_idx]
        #         enm_max_blood = 100.0
                
        #         enm_feature_6d = jnp.array([enm_north, enm_east, enm_alt, enm_vx, enm_vy, enm_vz])

        #         # 友方飞机状态
        #         ally_indices = jnp.arange(self.num_allies)
        #         ally_norths = enm_ps.north[ally_indices]
        #         ally_easts = enm_ps.east[ally_indices]
        #         ally_alts = enm_ps.altitude[ally_indices]
        #         ally_vxs = enm_ps.vel_x[ally_indices]
        #         ally_vys = enm_ps.vel_y[ally_indices]
        #         ally_vzs = enm_ps.vel_z[ally_indices]
        #         ally_vts = enm_ps.vt[ally_indices]
        #         ally_pitches = enm_ps.pitch[ally_indices]
        #         ally_is_alives = enm_ps.is_alive[ally_indices]
        #         ally_bloods = enm_ps.blood[ally_indices]
                
        #         # 计算对每个友方飞机的测量值
        #         def calculate_metrics_to_ally(ally_idx):
        #             ally_feature_6d = jnp.array([
        #                 ally_norths[ally_idx], ally_easts[ally_idx], ally_alts[ally_idx],
        #                 ally_vxs[ally_idx], ally_vys[ally_idx], ally_vzs[ally_idx]
        #             ])
        #             ao, ta, r, side_flag = get_AO_TA_R(enm_feature_6d, ally_feature_6d)
        #             return r, ao, ta, side_flag, ally_is_alives[ally_idx], ally_bloods[ally_idx]
                
        #         # 对所有友机计算指标
        #         ally_metrics = jax.vmap(calculate_metrics_to_ally)(jnp.arange(self.num_allies))
        #         # ally_metrics是一个包含6个数组的元组，不是一个2D数组
        #         # 直接解构元组而不是尝试对它进行索引
        #         distances, aos, tas, side_flags, is_alives, bloods = ally_metrics
                
        #         # 仅考虑存活的友机
        #         masked_distances = jnp.where(is_alives, distances, jnp.inf)
                
        #         # 找到最近的目标以及其指标
        #         any_ally_alive = jnp.any(is_alives)
        #         closest_ally_idx = jnp.where(any_ally_alive, jnp.argmin(masked_distances), 0)
        #         closest_ally_distance = masked_distances[closest_ally_idx]
        #         closest_ally_ao = aos[closest_ally_idx]
        #         closest_ally_ta = tas[closest_ally_idx]
        #         closest_ally_side_flag = side_flags[closest_ally_idx]
        #         closest_ally_is_alive = is_alives[closest_ally_idx]
        #         closest_ally_blood = bloods[closest_ally_idx]
                
        #         # 默认为无效值
        #         closest_ally_distance = jnp.where(closest_ally_is_alive, closest_ally_distance, 1e8)
        #         closest_ally_ao = jnp.where(closest_ally_is_alive, closest_ally_ao, jnp.pi)
                
        #         # --- FSM状态转换逻辑 ---
        #         # 首先计算各种状态条件
        #         is_alive_condition = enm_is_alive
        #         is_low_health = enm_is_alive & (enm_blood < (enm_max_blood * EVADE_HEALTH_THRESHOLD_RATIO))
        #         no_targets_alive = ~jnp.any(is_alives)

        #         # 攻击条件 - 最高优先级
        #         can_attack = is_alive_condition & (closest_ally_distance < ATTACK_RANGE) & closest_ally_is_alive & ~is_low_health

        #         # 接敌条件 - 第二优先级
        #         can_engage = is_alive_condition & (closest_ally_distance < ENGAGE_RANGE) & closest_ally_is_alive & ~is_low_health

        #         # 规避条件 - 第三优先级
        #         should_evade = is_alive_condition & is_low_health

        #         # 巡逻条件 - 最低优先级（默认状态）
        #         should_patrol = is_alive_condition & (no_targets_alive | (~can_attack & ~can_engage & ~should_evade))

        #         # 如果死亡，强制设为巡逻状态
        #         is_dead = ~is_alive_condition

        #         # 按照优先级顺序选择状态
        #         next_fsm_state = jnp.select(
        #             [can_attack, can_engage, should_evade, should_patrol, is_dead],
        #             [ATTACK_STATE, ENGAGE_STATE, EVADE_STATE, PATROL_STATE, PATROL_STATE],
        #             default=PATROL_STATE  # 默认为巡逻状态
        #         )
                
        #         # --- FSM行为逻辑 ---
                
        #         # 定义各状态下的行为函数
                
        #         # 巡逻行为: 随机转向并保持适中的速度和高度
        #         def patrol_action():
        #             # 随机改变航向，在 -MAX_TURN_RATE_PER_STEP/2 到 MAX_TURN_RATE_PER_STEP/2 之间
        #             key_turn, key_alt = jax.random.split(enm_key)
        #             random_turn = jax.random.uniform(key_turn, minval=-MAX_TURN_RATE_PER_STEP*PATROL_TURN_RANDOMNESS, 
        #                                             maxval=MAX_TURN_RATE_PER_STEP*PATROL_TURN_RANDOMNESS)
                    
        #             # 高度控制 - 尝试保持在巡逻高度范围内
        #             target_alt = jnp.mean(jnp.array(PATROL_ALTITUDE_RANGE))
        #             alt_diff = target_alt - enm_alt
        #             d_pitch = jnp.clip(alt_diff / 1000.0, -MAX_PITCH_RATE_PER_STEP/3, MAX_PITCH_RATE_PER_STEP/3)
                    
        #             # 速度控制 - 巡逻时保持较低速度以节省燃料
        #             target_vt = MAX_SPEED * PATROL_SPEED_FACTOR
        #             d_vt = jnp.clip(target_vt - enm_vt, -MAX_VT_DELTA_PER_STEP/2, MAX_VT_DELTA_PER_STEP/2)
                    
        #             return d_pitch, random_turn, d_vt
                
        #         # 接敌行为: 指向目标方向并调整高度准备攻击
        #         def engage_action():
        #             # 获取目标位置和当前位置
        #             target_north = ally_norths[closest_ally_idx]
        #             target_east = ally_easts[closest_ally_idx]
        #             target_alt = ally_alts[closest_ally_idx]
                    
        #             # 计算指向目标的航向
        #             delta_n = target_north - enm_north
        #             delta_e = target_east - enm_east
        #             desired_heading = jnp.arctan2(delta_e, delta_n)
        #             d_heading = wrap_PI(desired_heading - enm_yaw)
                    
        #             # 高度控制 - 尝试比目标略高一些(高度优势)
        #             # height_advantage = 100.0  # 500米高度优势
        #             # target_alt_with_advantage = target_alt + height_advantage
        #             alt_diff = target_alt - enm_alt
        #             # d_pitch = wrap_PI(jnp.clip(alt_diff / 1000.0, -MAX_PITCH_RATE_PER_STEP/2, MAX_PITCH_RATE_PER_STEP/2))
        #             # 计算指向预测位置的俯仰角
        #             horizontal_distance = jnp.sqrt(delta_n**2 + delta_e**2)  # 水平距离
        #             desired_pitch = jnp.arctan2(-alt_diff, horizontal_distance)  
        #             d_pitch = jnp.clip(desired_pitch - enm_pitch, -MAX_PITCH_RATE_PER_STEP, MAX_PITCH_RATE_PER_STEP)
                    
        #             # 速度控制 - 接敌时略微加速但不全速
        #             target_vt = MAX_SPEED * 0.7
        #             d_vt = jnp.clip(target_vt - enm_vt, -MAX_VT_DELTA_PER_STEP/2, MAX_VT_DELTA_PER_STEP/2)
                    
        #             return d_pitch, d_heading, d_vt
                
        #         # 攻击行为: 高速、激进地追求优势攻击位置
        #         def attack_action():
        #             # # 获取目标位置和当前位置
        #             # target_north = ally_norths[closest_ally_idx]
        #             # target_east = ally_easts[closest_ally_idx]
        #             # target_alt = ally_alts[closest_ally_idx]
        #             # target_vx = ally_vxs[closest_ally_idx]
        #             # target_vy = ally_vys[closest_ally_idx]
        #             # target_vz = ally_vzs[closest_ally_idx]
        #             # target_vt = ally_vts[closest_ally_idx]
                    
        #             # # 计算水平方向的向量和距离
        #             # delta_x, delta_y = target_north - enm_north, target_east - enm_east
        #             # R = jnp.linalg.norm(jnp.array([delta_x, delta_y]))
                    
        #             # # 计算敌机速度和速度在连线方向上的投影
        #             # enm_v = jnp.linalg.norm(jnp.array([enm_vx, enm_vy]))
        #             # proj_dist = delta_x * enm_vx + delta_y * enm_vy
                    
        #             # # 计算AO角和侧向标志
        #             # AO = jnp.arccos(jnp.clip(proj_dist / (R * enm_v + 1e-6), -1, 1))
        #             # side_flag = jnp.sign(enm_vx * delta_y - enm_vy * delta_x)
                    
        #             # # 计算航向变化
        #             # d_heading = AO * side_flag
        #             # d_heading = jnp.clip(d_heading, -MAX_TURN_RATE_PER_STEP, MAX_TURN_RATE_PER_STEP)
                    
        #             # # 计算俯仰角变化 - 使用类似的逻辑
        #             # delta_alt = target_alt - enm_alt
        #             # # 使用简单的几何计算得到所需俯仰角
        #             # desired_pitch = jnp.arctan2(delta_alt, R)
        #             # d_pitch = jnp.clip(desired_pitch - enm_pitch, -MAX_PITCH_RATE_PER_STEP, MAX_PITCH_RATE_PER_STEP)
                    
        #             # # 速度控制 - 调整到接近目标速度
        #             # d_vt = jnp.clip(target_vt - enm_vt, -MAX_VT_DELTA_PER_STEP, MAX_VT_DELTA_PER_STEP)
                                        
                    
        #             # delta pitch - enemies attempt to match allies' pitch plus a random factor
        #             # 获取目标位置和当前位置
        #             target_north = ally_norths[closest_ally_idx]
        #             target_east = ally_easts[closest_ally_idx]
        #             target_alt = ally_alts[closest_ally_idx]
        #             target_pitch = ally_pitches[closest_ally_idx]
        #             target_vx = ally_vxs[closest_ally_idx]
        #             target_vy = ally_vys[closest_ally_idx]
        #             target_vz = ally_vzs[closest_ally_idx]
        #             target_vt = ally_vts[closest_ally_idx]
        #             enm_delta_pitch = target_pitch - enm_pitch
        #             # delta heading

        #             ego_v = jnp.linalg.norm(jnp.array([target_vx, target_vy]))
        #             delta_x, delta_y = enm_east - target_east, enm_north - target_north
        #             R = jnp.linalg.norm(jnp.array([delta_x, delta_y]))
        #             proj_dist = delta_x * target_vx + delta_y * target_vy
        #             ego_AO = jnp.arccos(jnp.clip(proj_dist / (R * ego_v + 1e-6), -1, 1))
        #             # side_flag = jnp.sign(jnp.cross(jnp.vstack((ego_vx, ego_vy)), jnp.vstack((delta_x, delta_y))))
        #             side_flag = jnp.sign(target_vx * delta_y - target_vy * delta_x)
        #             enm_delta_heading = ego_AO * side_flag
        #             # delta velocity
        #             enm_delta_vt = target_vt - enm_vt
                    
        #             # NOTE:过大的值似乎容易导致飞机crash
        #             enm_delta_pitch = jnp.clip(enm_delta_pitch, -jnp.pi/10, jnp.pi/10)
        #             enm_delta_heading = jnp.clip(enm_delta_heading, -jnp.pi / 10, jnp.pi / 10)
        #             enm_delta_vt = jnp.clip(enm_delta_vt, -20, 20)
                    
        #             return enm_delta_pitch, enm_delta_heading, enm_delta_vt
                
        #         # 规避行为: 远离威胁，大幅度机动
        #         def evade_action():
        #             # 获取最近威胁的位置
        #             threat_north = ally_norths[closest_ally_idx]
        #             threat_east = ally_easts[closest_ally_idx]
                    
        #             # 计算远离威胁的航向(反方向)
        #             delta_n = threat_north - enm_north
        #             delta_e = threat_east - enm_east
        #             threat_heading = jnp.arctan2(delta_e, delta_n)
        #             evade_heading = wrap_PI(threat_heading + jnp.pi)  # 反方向
        #             d_heading = wrap_PI(evade_heading - enm_yaw)
                    
        #             # 高度控制 - 随机变化以增加难以预测性
        #             key_pitch, key_speed = jax.random.split(enm_key)
        #             random_pitch_factor = jax.random.uniform(key_pitch, minval=-1.0, maxval=1.0)
        #             d_pitch = random_pitch_factor * MAX_PITCH_RATE_PER_STEP
                    
        #             # 速度控制 - 全速逃离
        #             target_vt = MAX_SPEED * EVADE_SPEED_FACTOR
        #             d_vt = jnp.clip(target_vt - enm_vt, 0, MAX_VT_DELTA_PER_STEP)  # 只允许加速，不减速
                    
        #             return d_pitch, d_heading, d_vt
                
        #         # 根据当前FSM状态执行相应行为
        #         p_pitch, p_heading, p_vt = patrol_action()
        #         e_pitch, e_heading, e_vt = engage_action()
        #         a_pitch, a_heading, a_vt = attack_action()
        #         ev_pitch, ev_heading, ev_vt = evade_action()
                
        #         # 选择对应状态的行为输出
        #         d_pitch = jnp.select([next_fsm_state == PATROL_STATE,
        #                               next_fsm_state == ENGAGE_STATE,
        #                               next_fsm_state == ATTACK_STATE,
        #                               next_fsm_state == EVADE_STATE],
        #                              [p_pitch, e_pitch, a_pitch, ev_pitch],
        #                              default=0.0)
                
        #         d_heading = jnp.select([next_fsm_state == PATROL_STATE,
        #                                next_fsm_state == ENGAGE_STATE,
        #                                next_fsm_state == ATTACK_STATE,
        #                                next_fsm_state == EVADE_STATE],
        #                               [p_heading, e_heading, a_heading, ev_heading],
        #                               default=0.0)
                
        #         d_vt = jnp.select([next_fsm_state == PATROL_STATE,
        #                            next_fsm_state == ENGAGE_STATE,
        #                            next_fsm_state == ATTACK_STATE,
        #                            next_fsm_state == EVADE_STATE],
        #                           [p_vt, e_vt, a_vt, ev_vt],
        #                           default=0.0)
                
        #         # 最终限幅并对死亡飞机置零
        #         d_pitch = jnp.clip(d_pitch, -MAX_PITCH_RATE_PER_STEP, MAX_PITCH_RATE_PER_STEP)
        #         d_heading = jnp.clip(d_heading, -MAX_TURN_RATE_PER_STEP, MAX_TURN_RATE_PER_STEP)
        #         d_vt = jnp.clip(d_vt, -MAX_VT_DELTA_PER_STEP, MAX_VT_DELTA_PER_STEP)
                
        #         d_pitch = wrap_PI(jnp.where(enm_is_alive, d_pitch, 0.0))
        #         d_heading = wrap_PI(jnp.where(enm_is_alive, d_heading, 0.0))
        #         d_vt = jnp.where(enm_is_alive, d_vt, 0.0)
                
        #         return next_fsm_state, d_pitch, d_heading, d_vt

        #     # 对所有敌方飞机应用FSM逻辑
        #     vmapped_fsm_results = jax.vmap(
        #         single_enemy_fsm_logic, in_axes=(0, 0, 0)
        #     )(jnp.arange(self.num_enemies), state.enemy_fsm_state, enemy_keys)
            
        #     # 解包结果
        #     new_enemy_fsm_states = vmapped_fsm_results[0]
        #     enm_delta_pitch = vmapped_fsm_results[1]
        #     enm_delta_heading = vmapped_fsm_results[2]
        #     enm_delta_vt = vmapped_fsm_results[3]
            
        #     # 更新环境状态中的FSM状态
        #     state = state.replace(enemy_fsm_state=new_enemy_fsm_states)
            
        #     # 合并友方和敌方的动作
        #     delta_pitch = jnp.hstack((ego_delta_pitch_cmd, enm_delta_pitch))
        #     delta_heading = jnp.hstack((ego_delta_heading_cmd, enm_delta_heading))
        #     delta_vt = jnp.hstack((ego_delta_vt_cmd, enm_delta_vt))
            
        #     # 计算目标状态用于控制器
        #     target_pitch = wrap_PI(init_state.plane_state.pitch + delta_pitch)
        #     target_heading = wrap_PI(init_state.plane_state.yaw + delta_heading)
        #     target_vt = init_state.plane_state.vt + delta_vt

        # last_obs = self._get_controller_obs(state.plane_state, target_pitch, target_heading, target_vt)
        # last_obs = jnp.transpose(last_obs)
        # last_done = jnp.zeros((self.num_agents), dtype=bool)
        # ac_in = (
        #     last_obs[np.newaxis, :],
        #     last_done[np.newaxis, :],
        # )
        # hstate, pi, _ = controller.apply(controller_params, state.hstate, ac_in)
        # pi_throttle, pi_elevator, pi_aileron, pi_rudder = pi

        # # 使用controller_key进行采样
        # controller_key, key_throttle = jax.random.split(controller_key)
        # action_throttle = pi_throttle.sample(seed=key_throttle)
        # controller_key, key_elevator = jax.random.split(controller_key)
        # action_elevator = pi_elevator.sample(seed=key_elevator)
        # controller_key, key_aileron = jax.random.split(controller_key)
        # action_aileron = pi_aileron.sample(seed=key_aileron)
        # controller_key, key_rudder = jax.random.split(controller_key)
        # action_rudder = pi_rudder.sample(seed=key_rudder)

        # action = jnp.concatenate([action_throttle[:, :, np.newaxis], 
        #                          action_elevator[:, :, np.newaxis], 
        #                          action_aileron[:, :, np.newaxis], 
        #                          action_rudder[:, :, np.newaxis]], axis=-1)
        # state = state.replace(hstate=hstate)
        # action = action.squeeze(0)
        # action = jax.vmap(self._decode_discrete_actions)(action)
        # return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(action)
        else:
            ego_delta_pitch = self.norm_delta_pitch[actions[:self.num_allies, 0]]
            ego_delta_heading = self.norm_delta_heading[actions[:self.num_allies, 1]]
            ego_delta_vt = self.norm_delta_velocity[actions[:self.num_allies, 2]]
            
            ego_x = init_state.plane_state.north[self.num_allies:]
            ego_y = init_state.plane_state.east[self.num_allies:]
            ego_z = init_state.plane_state.altitude[self.num_allies:]

            ego_vx = init_state.plane_state.vel_x[self.num_allies:]
            ego_vy = init_state.plane_state.vel_y[self.num_allies:]
            
            enm_x = init_state.plane_state.north[:self.num_allies]
            enm_y = init_state.plane_state.east[:self.num_allies]
            enm_z = init_state.plane_state.altitude[:self.num_allies]
            
            # delta pitch - enemies point toward allies based on relative position with randomness
            delta_z = enm_z - ego_z  # Altitude difference
            delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
            horizontal_dist = jnp.sqrt(delta_x**2 + delta_y**2)  # Horizontal distance
            target_pitch = jnp.arctan2(delta_z, horizontal_dist)  # Angle to pitch to point at target
            ego_pitch = init_state.plane_state.pitch[self.num_allies:]
            
            # Add randomness to pitch
            pitch_key, heading_key, vt_key = jax.random.split(key, 3)
            pitch_random = jax.random.uniform(pitch_key, shape=target_pitch.shape, minval=-jnp.pi/72, maxval=jnp.pi/72)
            enm_delta_pitch = target_pitch - ego_pitch + pitch_random  # Add random component
            
            # delta heading with randomness
            ego_v = jnp.linalg.norm(jnp.vstack((ego_vx, ego_vy)), axis=0)
            R = jnp.linalg.norm(jnp.vstack((delta_x, delta_y)), axis=0)
            proj_dist = delta_x * ego_vx + delta_y * ego_vy
            ego_AO = jnp.arccos(jnp.clip(proj_dist / (R * ego_v + 1e-6), -1, 1))
            side_flag = jnp.sign(ego_vx * delta_y - ego_vy * delta_x)
            
            # Add randomness to heading
            heading_random = jax.random.uniform(heading_key, shape=ego_AO.shape, minval=-jnp.pi/45, maxval=jnp.pi/45)
            enm_delta_heading = ego_AO * side_flag + heading_random  # Add random component
            
            # delta velocity with randomness
            base_delta_vt = init_state.plane_state.vt[:self.num_allies] - init_state.plane_state.vt[self.num_allies:]
            
            # Add randomness to velocity
            vt_random = jax.random.uniform(vt_key, shape=base_delta_vt.shape, minval=-5.0, maxval=5.0)
            enm_delta_vt = base_delta_vt + vt_random  # Add random component
            
            # NOTE:过大的值似乎容易导致飞机crash
            enm_delta_pitch = jnp.clip(enm_delta_pitch, -jnp.pi/10 * move_rate, jnp.pi/10 * move_rate)
            enm_delta_heading = jnp.clip(enm_delta_heading, -jnp.pi/10 * move_rate, jnp.pi/10 * move_rate)
            enm_delta_vt = jnp.clip(enm_delta_vt, -20 * move_rate, 20 * move_rate)

            delta_pitch = jnp.hstack((ego_delta_pitch, enm_delta_pitch))
            delta_heading = jnp.hstack((ego_delta_heading, enm_delta_heading))
            delta_vt = jnp.hstack((ego_delta_vt, enm_delta_vt))
            
            target_pitch = wrap_PI(init_state.plane_state.pitch + delta_pitch)
            target_heading = wrap_PI(init_state.plane_state.yaw + delta_heading)
            target_vt = init_state.plane_state.vt + delta_vt
        
        last_obs = self._get_controller_obs(state.plane_state, target_pitch, target_heading, target_vt)
        last_obs = jnp.transpose(last_obs)
        last_done = jnp.zeros((self.num_agents), dtype=bool)
        ac_in = (
            last_obs[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        hstate, pi, _ = controller.apply(controller_params, state.hstate, ac_in)
        pi_throttle, pi_elevator, pi_aileron, pi_rudder = pi

        key, key_throttle = jax.random.split(key)
        action_throttle = pi_throttle.sample(seed=key_throttle)
        key, key_elevator = jax.random.split(key)
        action_elevator = pi_elevator.sample(seed=key_elevator)
        key, key_aileron = jax.random.split(key)
        action_aileron = pi_aileron.sample(seed=key_aileron)
        key, key_rudder = jax.random.split(key)
        action_rudder = pi_rudder.sample(seed=key_rudder)

        action = jnp.concatenate([action_throttle[:, :, np.newaxis], 
                                  action_elevator[:, :, np.newaxis], 
                                  action_aileron[:, :, np.newaxis], 
                                  action_rudder[:, :, np.newaxis]], axis=-1)
        state = state.replace(hstate=hstate)
        action = action.squeeze(0)
        action = jax.vmap(self._decode_discrete_actions)(action)
        return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(action)


    @property
    def default_params(self) -> HierarchicalCombatTaskParams:
        return HierarchicalCombatTaskParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_termination(
        self,
        state: HierarchicalCombatTaskState,
        params: HierarchicalCombatTaskParams,
    ) -> Tuple[HierarchicalCombatTaskState, Dict[AgentName, bool]]:
        dones = jnp.zeros(self.num_agents, dtype=jnp.bool_)
        successes = jnp.zeros(self.num_agents, dtype=jnp.bool_)
        for termination_condition in self.termination_conditions:
            new_done, new_success = jax.vmap(
                termination_condition, in_axes=(None, None, 0)
            )(state, params, jnp.arange(self.num_agents))
            dones = jnp.logical_or(dones, new_done)
            successes = jnp.logical_or(successes, new_success)

        # NOTE: 在combat任务中，我方胜利才视作胜利
        state = state.replace(
            done=jnp.all(dones[:self.num_allies]) | jnp.all(dones[self.num_allies:]),
            success=jnp.all(jnp.where(jnp.arange(self.num_agents) < self.num_allies, successes, True))
        )
            
        dones = {
            agent: dones[i] for i, agent in enumerate(self.agents)
        }
        return state, dones

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: HierarchicalCombatTaskParams
    ) -> HierarchicalCombatTaskState:
        state = super()._init_state(key, params)
        init_hstate = ScannedLSTM.initialize_carry(self.num_agents, config["GRU_HIDDEN_DIM"])
        # Initialize enemy FSM states (e.g., to PATROL_STATE)
        # Assuming PATROL_STATE = 0, will define constants later
        init_enemy_fsm_state = jnp.full((self.num_enemies,), PATROL_STATE, dtype=jnp.int32)
        # Initialize prev_enemy_alive - all enemies start alive
        init_prev_enemy_alive = jnp.ones(self.num_enemies, dtype=jnp.bool_)
        state = HierarchicalCombatTaskState.create(state, init_hstate, init_enemy_fsm_state, init_prev_enemy_alive)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: HierarchicalCombatTaskState,
        params: HierarchicalCombatTaskParams,
    ) -> HierarchicalCombatTaskState:
        """Task-specific reset."""

        state = self._generate_formation(key, state, params)
        yaw = jnp.where(jnp.arange(self.num_agents) < self.num_allies, 0.0, jnp.pi)
        q0 = jnp.where(jnp.arange(self.num_agents) < self.num_allies, 1.0, 0.0)
        q3 = jnp.where(jnp.arange(self.num_agents) < self.num_allies, 0.0, 1.0)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        
        # 重置敌方FSM状态为PATROL
        init_enemy_fsm_state = jnp.full((self.num_enemies,), PATROL_STATE, dtype=jnp.int32)

        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=yaw,
                vt=vt,
                q0=q0,
                q3=q3,
            ),
            enemy_fsm_state=init_enemy_fsm_state,  # 更新敌方FSM状态
            prev_enemy_alive=jnp.ones(self.num_enemies, dtype=jnp.bool_)  # 重置敌机历史状态为全部存活
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: HierarchicalCombatTaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: HierarchicalCombatTaskParams,
    ) -> Tuple[HierarchicalCombatTaskState, Dict[str, Any]]:
        """Task-specific step transition."""
        # 硬编码常量
        num_allies = 2
        num_enemies = 2
        total_agents = 4
        
        # 创建掩码
        agent_indices = jnp.arange(total_agents)
        ally_mask = agent_indices < num_allies
        enemy_mask = ~ally_mask
        
        # 获取状态
        is_alive = state.plane_state.is_alive
        is_locked = state.plane_state.is_locked
        blood = state.plane_state.blood
        status = state.plane_state.status
        
        # 正确的存活判断：状态为ALIVE(0)或LOCKED(1)
        truly_alive = jnp.logical_or(status == 0, status == 1)
        
        # 计算各方存活情况
        ally_alive_count = jnp.sum(jnp.where(ally_mask, truly_alive, False))
        enemy_alive_count = jnp.sum(jnp.where(enemy_mask, truly_alive, False))
        
        # 计算各方血量总和（只计算存活的）
        ally_total_blood = jnp.sum(jnp.where(
            jnp.logical_and(ally_mask, truly_alive), blood, 0
        ))
        enemy_total_blood = jnp.sum(jnp.where(
            jnp.logical_and(enemy_mask, truly_alive), blood, 0
        ))
        
        # 原有的监控指标
        info['alive_count'] = ally_alive_count
        info['blood'] = ally_total_blood - enemy_total_blood
        info['enemies_locked_count'] = jnp.sum(jnp.where(enemy_mask, is_locked, 0))
        
        # 检测击落事件（为了监控）
        enemy_indices = jnp.arange(num_allies, num_allies + num_enemies)
        current_enemy_status = state.plane_state.status[enemy_indices]
        current_enemy_alive = jnp.logical_or(
            current_enemy_status == 0,  # ALIVE
            current_enemy_status == 1   # LOCKED
        )
        prev_enemy_alive = state.prev_enemy_alive
        enemy_killed_this_step = jnp.logical_and(prev_enemy_alive, ~current_enemy_alive)
        info['enemies_killed_this_step'] = jnp.sum(enemy_killed_this_step)
        
        # 判断各种胜利条件
        # 1. 完胜：我方全部存活(2个)，敌方全部被消灭(0个)
        complete_victory = jnp.logical_and(
            ally_alive_count == num_allies,
            enemy_alive_count == 0
        )
        
        # 2. 大胜：敌方全部被消灭，我方有存活但有损失
        great_victory = jnp.logical_and(
            jnp.logical_and(enemy_alive_count == 0, ally_alive_count > 0),
            ~complete_victory
        )
        
        # 3. 一般胜利：我方存活数量 > 敌方存活数量
        normal_victory = jnp.logical_and(
            ally_alive_count > enemy_alive_count,
            enemy_alive_count > 0  # 敌方还有存活
        )
        
        # 4. 险胜：存活数量相同但我方血量更多
        narrow_victory = jnp.logical_and(
            jnp.logical_and(
                ally_alive_count == enemy_alive_count,
                ally_alive_count > 0  # 双方都有存活
            ),
            ally_total_blood > enemy_total_blood
        )
        
        # 5. 失败：其他所有情况
        failure = ~jnp.logical_or(
            jnp.logical_or(complete_victory, great_victory),
            jnp.logical_or(normal_victory, narrow_victory)
        )
        
        # 添加胜利条件监控指标
        info['complete_victory'] = complete_victory
        info['great_victory'] = great_victory
        info['normal_victory'] = normal_victory
        info['narrow_victory'] = narrow_victory
        info['failure'] = failure
        
        # 兼容原有的简单胜利指标
        info['success_simple'] = jnp.logical_or(complete_victory, great_victory)
        info['success_weak'] = jnp.logical_or(normal_victory, narrow_victory)
        
        # 更新prev_enemy_alive状态以供下一步使用
        # 敌机当前存活状态：状态为ALIVE或LOCKED
        enemy_indices = jnp.arange(num_allies, num_allies + num_enemies)
        current_enemy_status = state.plane_state.status[enemy_indices]
        current_enemy_alive = jnp.logical_or(
            current_enemy_status == 0,  # ALIVE
            current_enemy_status == 1   # LOCKED
        )
        
        # 更新状态中的prev_enemy_alive
        state = state.replace(prev_enemy_alive=current_enemy_alive)
        
        return state, info

    def train_callback(self, metric: chex.Array, writer:tensorboardX.SummaryWriter, train_mode:bool):
        # NOTE: 训练时间长容易int溢出
        # env_steps = metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]
        env_steps = metric["update_steps"]
        if train_mode:
            for k, v in metric["loss"].items():
                writer.add_scalar('loss/{}'.format(k), v, env_steps)
        indexs = metric["returned_episode"]
        valid_index_count = indexs.sum()

        episodic_return = jnp.where(valid_index_count>0, metric["returned_episode_returns"][indexs].mean(), 0.)
        episodic_length = jnp.where(valid_index_count>0, metric["returned_episode_lengths"][indexs].mean(), 0.)
        success_rate = jnp.where(valid_index_count>0, metric["success"][indexs].mean(), 0.)
        
        # 原有的兼容指标
        success_simple_rate = jnp.where(valid_index_count>0, metric["success_simple"][indexs].mean(), 0.)
        success_weak_rate = jnp.where(valid_index_count>0, metric["success_weak"][indexs].mean(), 0.)
        win_rate = success_simple_rate + success_weak_rate
        
        # 新的详细胜利条件指标
        complete_victory_rate = jnp.where(valid_index_count>0, metric["complete_victory"][indexs].mean(), 0.)
        great_victory_rate = jnp.where(valid_index_count>0, metric["great_victory"][indexs].mean(), 0.)
        normal_victory_rate = jnp.where(valid_index_count>0, metric["normal_victory"][indexs].mean(), 0.)
        narrow_victory_rate = jnp.where(valid_index_count>0, metric["narrow_victory"][indexs].mean(), 0.)
        failure_rate = jnp.where(valid_index_count>0, metric["failure"][indexs].mean(), 0.)
        
        # 计算总胜利率
        total_victory_rate = complete_victory_rate + great_victory_rate + normal_victory_rate + narrow_victory_rate
        
        # 其他监控指标
        blood_advantage = jnp.where(valid_index_count>0, metric["blood"][indexs].mean(), 0.)
        alive_count = jnp.where(valid_index_count>0, metric["alive_count"][indexs].mean(), 0.)
        enemies_locked_count = jnp.where(valid_index_count>0, metric["enemies_locked_count"][indexs].mean(), 0.)
        enemies_killed_per_episode = jnp.where(valid_index_count>0, metric["enemies_killed_this_step"][indexs].sum(), 0.)

        # 记录到tensorboard
        writer.add_scalar('eval/episodic_return', episodic_return, env_steps)
        writer.add_scalar('eval/episodic_length', episodic_length, env_steps)
        writer.add_scalar('eval/success_rate', success_rate, env_steps)
        
        # 原有指标
        writer.add_scalar('eval/success_simple_rate', success_simple_rate, env_steps)
        writer.add_scalar('eval/success_weak_rate', success_weak_rate, env_steps)
        writer.add_scalar('eval/win_rate', win_rate, env_steps)
        
        # 新的详细胜利条件指标
        writer.add_scalar('eval/complete_victory_rate', complete_victory_rate, env_steps)
        writer.add_scalar('eval/great_victory_rate', great_victory_rate, env_steps)
        writer.add_scalar('eval/normal_victory_rate', normal_victory_rate, env_steps)
        writer.add_scalar('eval/narrow_victory_rate', narrow_victory_rate, env_steps)
        writer.add_scalar('eval/failure_rate', failure_rate, env_steps)
        writer.add_scalar('eval/total_victory_rate', total_victory_rate, env_steps)
        
        # 其他指标
        writer.add_scalar('eval/blood_advantage', blood_advantage, env_steps)
        writer.add_scalar('eval/alive_count', alive_count, env_steps)
        writer.add_scalar('eval/enemies_locked_count', enemies_locked_count, env_steps)
        writer.add_scalar('eval/enemies_killed_per_episode', enemies_killed_per_episode, env_steps)

        # 打印详细信息
        print(f"EnvStep={env_steps:<5} EpisodeLength={episodic_length:<7.2f} Return={episodic_return:<7.2f}")
        print(f"  Victory Breakdown: Complete={complete_victory_rate:.3f} Great={great_victory_rate:.3f} " + \
              f"Normal={normal_victory_rate:.3f} Narrow={narrow_victory_rate:.3f} Failure={failure_rate:.3f}")
        print(f"  Total Victory Rate={total_victory_rate:.3f} AliveCount={alive_count:>6.3f} " + \
              f"BloodAdvantage={blood_advantage:>8.2f} EnemiesLocked={enemies_locked_count:>5.2f}")
        print(f"  EnemiesKilled/Episode={enemies_killed_per_episode:>5.2f}")
        print(f"  Legacy: SimpleSuccess={success_simple_rate:.3f} WeakSuccess={success_weak_rate:.3f} " + \
              f"WinRate={win_rate:.3f}")
        print("-" * 80)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: HierarchicalCombatTaskState,
        params: HierarchicalCombatTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        pos = jnp.vstack((state.plane_state.north, state.plane_state.east, state.plane_state.altitude)).T
        
        visible_mask = compute_visibility_mask(pos, self.num_allies, comm_radius=50000, find_radius=50000)

        def _get_features(state: EnvState, i, j):
            empty_features = jnp.zeros(shape=(self.unit_features,))
            features = self._observe_features(state, i, j)
            visible1 = visible_mask[i, j]
            visible2 = (i != j)
            return jax.lax.cond(
                # visible1 & visible2 & (state.plane_state.is_alive[i] | state.plane_state.is_locked[i]) & (state.plane_state.is_alive[j_idx] | state.plane_state.is_locked[j_idx]),
                visible1 & visible2 & (state.plane_state.is_alive[i]) & (state.plane_state.is_alive[j]),
                lambda: features,
                lambda: empty_features,
            )
        obs = self._get_own_and_top_k_other_plane_obs(state, _get_features, enable_one_hot=self.enbale_actor_onehot_agent_id)

        return {agent: obs[self.agent_ids[agent]] for agent in self.agents}

    @functools.partial(jax.jit, static_argnums=(0,))
    def _observe_features(self, state: HierarchicalCombatTaskState, i: int, j_idx: int):
        ego_feature = jnp.hstack((state.plane_state.north[i],
                                  state.plane_state.east[i],
                                  state.plane_state.altitude[i],
                                  state.plane_state.vel_x[i],
                                  state.plane_state.vel_y[i],
                                  state.plane_state.vel_z[i]))
        enm_feature = jnp.hstack((state.plane_state.north[j_idx],
                                  state.plane_state.east[j_idx],
                                  state.plane_state.altitude[j_idx],
                                  state.plane_state.vel_x[j_idx],
                                  state.plane_state.vel_y[j_idx],
                                  state.plane_state.vel_z[j_idx]))
        AO, TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature)
        norm_delta_vt = (state.plane_state.vt[j_idx] - state.plane_state.vt[i]) / 340
        norm_delta_altitude = (state.plane_state.altitude[j_idx] - state.plane_state.altitude[i]) / 1000
        norm_distance = R / 10000
        
        # TODO: team_flag 是否必要（只观测敌机）
        team_flag = jnp.where(((i < self.num_allies) & (j_idx < self.num_allies)) | ((i >= self.num_allies) & (j_idx >= self.num_allies)),
                              1.0,
                              0.0)
        
        features = jnp.hstack((norm_delta_vt, norm_delta_altitude, AO, TA, norm_distance, side_flag))
        return features

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_own_features(self, state: HierarchicalCombatTaskState, i: int):
        altitude = state.plane_state.altitude[i]
        vel_x, vel_y, vel_z, vt = state.plane_state.vel_x[i], state.plane_state.vel_y[i], state.plane_state.vel_z[i], state.plane_state.vt[i]
        norm_altitude = altitude / 5000
        norm_vel_x, norm_vel_y, norm_vel_z, norm_vt = vel_x / 340, vel_y / 340, vel_z / 340, vt / 340
        empty_features = jnp.zeros(shape=(self.own_features,))
        features = jnp.hstack((norm_altitude, norm_vel_x, norm_vel_y, norm_vel_z, norm_vt))
        return jax.lax.cond(
            state.plane_state.is_alive[i], lambda: features, lambda: empty_features
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_obs_unit_list(self, state: HierarchicalCombatTaskState) -> Dict[str, chex.Array]:
        """Applies observation function to state."""

        pos = jnp.vstack((state.plane_state.north, state.plane_state.east, state.plane_state.altitude)).T
        
        visible_mask = compute_visibility_mask(pos, self.num_allies, comm_radius=50000, find_radius=50000)

        def get_features(i, j):
            """Get features of unit j as seen from unit i"""
            j = jax.lax.cond(
                i < self.num_allies,
                lambda: j,
                lambda: self.num_agents - j - 1,
            )
            offset = jax.lax.cond(i < self.num_allies, lambda: 1, lambda: -1)
            j_idx = jax.lax.cond(
                ((j < i) & (i < self.num_allies)) | ((j > i) & (i >= self.num_allies)),
                lambda: j,
                lambda: j + offset,
            )
            empty_features = jnp.zeros(shape=(self.unit_features,))
            features = self._observe_features(state, i, j_idx)
            visible = visible_mask[i, j_idx]
            return jax.lax.cond(
                visible & (state.plane_state.is_alive[i]) & (state.plane_state.is_alive[j_idx]),
                lambda: features,
                lambda: empty_features,
            )

        get_all_features_for_unit = jax.vmap(get_features, in_axes=(None, 0))
        get_all_features = jax.vmap(get_all_features_for_unit, in_axes=(0, None))
        other_unit_obs = get_all_features(
            jnp.arange(self.num_agents), jnp.arange(self.num_agents - 1)
        )
        other_unit_obs = other_unit_obs.reshape((self.num_agents, -1))
        get_all_self_features = jax.vmap(self._get_own_features, in_axes=(None, 0))
        own_unit_obs = get_all_self_features(state, jnp.arange(self.num_agents))

        agent_ids = jnp.arange(self.num_agents)
        one_hot_ids = jax.nn.one_hot(agent_ids, num_classes=self.num_agents)

        obs = jnp.concatenate([own_unit_obs, other_unit_obs], axis=-1)

        if self.enbale_actor_onehot_agent_id:
            obs = jnp.concatenate([obs, one_hot_ids], axis=-1)

        return {agent: obs[self.agent_ids[agent]] for agent in self.agents}
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_controller_obs(
        self,
        state: fighterplane.FighterPlaneState,
        target_pitch,
        target_heading,
        target_vt
    ) -> Dict[AgentName, chex.Array]:
        """
        Task-specific observation function to state.

        observation(dim 16):
            0. ego_delta_heading       (unit rad)
            1. ego_delta_pitch         (unit rad)  # Changed from delta_altitude
            2. ego_delta_vt            (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego_vt                  (unit: mh)
            9. ego_alpha_sin
            10. ego_alpha_cos
            11. ego_beta_sin
            12. ego_beta_cos
            13. ego_P                  (unit: rad/s)
            14. ego_Q                  (unit: rad/s)
            15. ego_R                  (unit: rad/s)
        """
        altitude = state.altitude
        roll, pitch, yaw = state.roll, state.pitch, state.yaw
        vt = state.vt
        alpha = state.alpha
        beta = state.beta
        P, Q, R = state.P, state.Q, state.R

        norm_delta_heading = wrap_PI((yaw - target_heading))
        norm_delta_pitch = wrap_PI((pitch - target_pitch))
        norm_delta_vt = (vt - target_vt) / 340
        norm_altitude = altitude / 5000
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        norm_vt = vt / 340
        alpha_sin = jnp.sin(alpha)
        alpha_cos = jnp.cos(alpha)
        beta_sin = jnp.sin(beta)
        beta_cos = jnp.cos(beta)
        obs = jnp.vstack((norm_delta_heading, norm_delta_pitch, norm_delta_vt,
                          norm_altitude, norm_vt,
                          roll_sin, roll_cos, pitch_sin, pitch_cos,
                          alpha_sin, alpha_cos, beta_sin, beta_cos,
                          P, Q, R))
        return obs

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _generate_formation(
            self,
            key: chex.PRNGKey,
            state: HierarchicalCombatTaskState,
            params: HierarchicalCombatTaskParams,
        ) -> HierarchicalCombatTaskState:  # 返回数组而不是字典

        # 生成随机旋转角度和位置偏移
        key, key_rotation, key_offset, key_distance, key_altitude = jax.random.split(key, 5)
        
        # 为双方生成随机旋转角度 (-π 到 π)
        ally_rotation = jax.random.uniform(key_rotation, minval=-jnp.pi, maxval=jnp.pi)
        enemy_rotation = jax.random.uniform(key_rotation, minval=-jnp.pi, maxval=jnp.pi)
        
        # 生成随机位置偏移 (±2000m)
        ally_offset = jax.random.uniform(key_offset, shape=(2,), minval=-2000.0, maxval=2000.0)
        enemy_offset = jax.random.uniform(key_offset, shape=(2,), minval=-2000.0, maxval=2000.0)

        # 根据队形类型选择生成函数
        if self.formation_type == 0:
            ally_positions = wedge_formation(self.num_allies, params.team_spacing)
            enemy_positions = wedge_formation(self.num_enemies, params.team_spacing)
        elif self.formation_type == 1:
            ally_positions = line_formation(self.num_allies, params.team_spacing)
            enemy_positions = line_formation(self.num_enemies, params.team_spacing)
        elif self.formation_type == 2:
            ally_positions = diamond_formation(self.num_allies, params.team_spacing)
            enemy_positions = diamond_formation(self.num_enemies, params.team_spacing)
        else:
            raise ValueError("Provided formation type is not valid")
        
        # 转换为全局坐标并确保安全距离        
        ally_center = jnp.zeros(3)
        enemy_center = jnp.zeros(3)
        
        # 随机距离和高度
        distance = jax.random.uniform(key_distance, minval=params.min_distance, maxval=params.max_distance)
        altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
        
        # 设置中心点位置，加入随机偏移
        ally_center = ally_center.at[0].set(-distance / 2 + ally_offset[0])
        ally_center = ally_center.at[1].set(ally_offset[1])
        ally_center = ally_center.at[2].set(altitude)
        
        enemy_center = enemy_center.at[0].set(distance / 2 + enemy_offset[0])
        enemy_center = enemy_center.at[1].set(enemy_offset[1])
        enemy_center = enemy_center.at[2].set(altitude)
        
        # 应用安全距离约束
        formation_positions = jnp.vstack((
            enforce_safe_distance(ally_positions, ally_center, params.safe_distance),
            enforce_safe_distance(enemy_positions, enemy_center, params.safe_distance)
        ))
        
        state = state.replace(plane_state=state.plane_state.replace(
            north=formation_positions[:, 0],
            east=formation_positions[:, 1],
            altitude=formation_positions[:, 2]
        ))
        return state



def compute_visibility_mask(pos: jnp.ndarray, k: int, comm_radius: float, find_radius: float) -> jnp.ndarray:
    """
    计算每个 agent 最终可以"看见"哪些其他 agent(通过通信半径、发现半径和同阵营多跳共享)

    参数:
    - pos: shape (n, 3)，所有 agent 的位置
    - k: 前 k 个 agent 是 A 方，其余是 B 方
    - comm_radius: 通信半径
    - find_radius: 发现半径

    返回:
    - visible_mask: shape (n, n), bool, [i, j] == True 表示 i 能看见 j
    """

    diff = pos[:, None, :] - pos[None, :, :]  # (n, n, 3)
    dist = jnp.linalg.norm(diff, axis=-1)     # (n, n)

    comm_mask = dist < comm_radius            # shape (n, n)

    n = pos.shape[0]
    team_flag = jnp.arange(n) < k             # shape (n,)
    same_team = team_flag[:, None] == team_flag[None, :]  # shape (n, n)

    # 只允许同阵营间、且在通信半径内传递信息
    share_graph = comm_mask & same_team       # shape (n, n)

    visible = dist < find_radius  # shape (n, n)，bool

    # Step 6: 多跳传播（布尔图邻接传播）
    def body_fn(state):
        # visible[i, j] == True 表示 i 能看到 j
        # share_graph[i, j] == True 表示 i 可以和 j 同阵营通信
        visible, _ = state
        new_visible = (share_graph @ visible) > 0
        updated = visible | new_visible
        return (updated, visible)

    def cond_fn(state):
        updated, prev = state
        return jnp.any(updated != prev)

    visible, _ = jax.lax.while_loop(cond_fn, body_fn, (visible, jnp.zeros_like(visible, dtype=jnp.bool_)))

    return visible  # shape (n, n), bool

# New reward function for locked penalty
def locked_penalty_fn(
    state: HierarchicalCombatTaskState,
    params: HierarchicalCombatTaskParams,
    agent_id: AgentID,
    penalty_value: float = -10.0,
    num_allies: int = 2, # Default from HierarchicalCombatTaskParams
) -> float:
    """
    Applies a penalty if an ally agent is locked.
    """
    is_locked = state.plane_state.is_locked[agent_id]
    is_ally = agent_id < num_allies # Assuming agent_id < num_allies are allies

    # Apply penalty only if the agent is an ally and is locked
    penalty = jnp.where(is_ally & is_locked, penalty_value, 0.0)
    return penalty

# New reward function for crash penalty
def crash_penalty_fn_new(
    state: HierarchicalCombatTaskState,
    params: HierarchicalCombatTaskParams,
    agent_id: AgentID,
    penalty_value: float = -100.0, # Standard crash penalty
) -> float:
    """
    Applies a penalty if an agent has crashed.
    Status 2 typically indicates a crash.
    """
    has_crashed = state.plane_state.status[agent_id] == 2
    penalty = jnp.where(has_crashed, penalty_value, 0.0)
    return penalty

def heading_alignment_reward_fn(
        state: HierarchicalCombatTaskState,
        params: HierarchicalCombatTaskParams,
        agent_id: AgentID,
        reward_scale: float = 0.5,
        num_allies: int = 1,
        num_enemies: int = 1,
    ) -> float:
    """
    奖励机头指向敌方飞机的精确程度。
    只关注机头朝向与连线的夹角(AO)，角度越小奖励越高。
    """
    # 提取己方飞机特征
    ego_feature = jnp.hstack((state.plane_state.north[agent_id],
                              state.plane_state.east[agent_id],
                              state.plane_state.altitude[agent_id],
                              state.plane_state.vel_x[agent_id],
                              state.plane_state.vel_y[agent_id],
                              state.plane_state.vel_z[agent_id]))
    
    # 获取己方飞机的俯仰角和偏航角
    ego_pitch = state.plane_state.pitch[agent_id]
    ego_yaw = state.plane_state.yaw[agent_id]
    
    # 确定目标敌机列表
    enm_list = jax.lax.select(agent_id < num_allies, 
                              jnp.arange(num_allies, num_allies + num_enemies),
                              jnp.arange(num_allies))
    
    def process_enemy(enm):
        # 提取敌方飞机特征
        enm_feature = jnp.hstack((state.plane_state.north[enm],
                                  state.plane_state.east[enm],
                                  state.plane_state.altitude[enm],
                                  state.plane_state.vel_x[enm],
                                  state.plane_state.vel_y[enm],
                                  state.plane_state.vel_z[enm]))
        
        # 获取敌方飞机的俯仰角和偏航角
        enm_pitch = state.plane_state.pitch[enm]
        enm_yaw = state.plane_state.yaw[enm]
        
        # 计算AO (机头与连线夹角)，使用安全版本
        AO, _, R, _ = safe_get_AO_TA_R_pitch_yaw(
            ego_feature, 
            enm_feature, 
            ego_pitch, 
            ego_yaw, 
            enm_pitch, 
            enm_yaw
        )
        
        # 使用高斯函数: exp(-(AO/sigma)²)，随着AO增大，奖励迅速衰减
        # 使用clipped版本避免数值爆炸
        alignment_reward = jnp.exp(-jnp.clip((AO/(jnp.pi/6))**2, 0.0, 10.0))
        
        # 添加距离权重因子：距离越近，对准越重要
        # 使用clipped版本避免数值爆炸
        distance_weight = jnp.exp(-jnp.clip(R / 10000.0, 0.0, 10.0))
        
        # 敌机存活掩码
        mask = state.plane_state.is_alive[enm] | state.plane_state.is_locked[enm]
        
        # 计算并检查奖励
        reward = alignment_reward * distance_weight
        reward = jnp.where(jnp.isnan(reward) | jnp.isinf(reward), 0.0, reward)
        
        return reward * mask
    
    # 使用vmap处理所有敌机
    enemy_rewards = jax.vmap(process_enemy)(enm_list)
    total_alignment_reward = jnp.max(enemy_rewards)
    
    # 应用存活掩码和奖励缩放
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    final_reward = total_alignment_reward * reward_scale * mask
    
    # 防止NaN和Inf
    final_reward = jnp.where(jnp.isnan(final_reward) | jnp.isinf(final_reward), 0.0, final_reward)
    
    return final_reward

def enemy_kill_reward_fn(
    state: HierarchicalCombatTaskState,
    params: HierarchicalCombatTaskParams,
    agent_id: AgentID,
    kill_reward: float = 100.0,
    num_allies: int = 2,
    num_enemies: int = 2,
) -> float:
    """
    击落敌机奖励函数：当有敌机被击落时，所有存活的我方飞机都获得奖励
    
    参数:
        state: 当前环境状态
        params: 环境参数
        agent_id: 当前Agent的ID
        kill_reward: 击落一架敌机的奖励值
        num_allies: 我方飞机数量
        num_enemies: 敌方飞机数量
    """
    # 检查是否为我方飞机
    is_ally = agent_id < num_allies
    
    # 检查当前agent是否存活（只有存活的我方飞机才能获得奖励）
    agent_alive = state.plane_state.is_alive[agent_id]
    
    # 获取当前敌机状态 (敌机索引：num_allies 到 num_allies + num_enemies - 1)
    enemy_indices = jnp.arange(num_allies, num_allies + num_enemies)
    
    # 当前敌机的状态：0=ALIVE, 1=LOCKED, 2=CRASHED, 3=SHOTDOWN, 4=SUCCESS
    current_enemy_status = state.plane_state.status[enemy_indices]
    
    # 当前敌机是否存活（状态为ALIVE或LOCKED）
    current_enemy_alive = jnp.logical_or(
        current_enemy_status == 0,  # ALIVE
        current_enemy_status == 1   # LOCKED
    )
    
    # 上一步敌机存活状态
    prev_enemy_alive = state.prev_enemy_alive
    
    # 检测击落事件：上一步存活，当前步被击落（status=3）
    enemy_shotdown = current_enemy_status == 3  # SHOTDOWN
    enemy_killed = jnp.logical_and(prev_enemy_alive, enemy_shotdown)
    
    # 计算被击落的敌机数量
    num_enemies_killed = jnp.sum(enemy_killed)
    
    # 根据击落数量给予奖励
    base_reward = num_enemies_killed * kill_reward
    
    # 使用jnp.where替代if语句：只给存活的我方飞机奖励
    reward = jnp.where(
        jnp.logical_and(is_ally, agent_alive),
        base_reward,
        0.0
    )
    
    return reward