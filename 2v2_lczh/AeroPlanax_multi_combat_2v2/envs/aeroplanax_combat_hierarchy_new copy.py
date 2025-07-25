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
    # posture_reward_fn,
    event_driven_reward_fn,
)
from .termination_conditions import (
    crashed_fn,
    safe_return_fn,
    timeout_fn,
)
from .utils.utils import  wedge_formation, line_formation, diamond_formation, enforce_safe_distance, get_AO_TA_R, wrap_PI
import orbax.checkpoint as ocp

import jax.numpy as jnp
from envs.aeroplanax import AgentID
from envs.utils.utils import get_AO_TA_R
import jax


# if not os.getcwd().endswith("AeroPlanax-dev-tmp0429_lxy_reform"):
#     raise ValueError("当前运行目录不是AeroPlanax,无法自动获取heading baseline文件夹位置，请手动填写LOADDIR并禁用本行代码！")

print(f'combat_hierarchy policy: load heading_pitch_V model from {os.path.join(os.getcwd(),"/home/lczh/Git Project/AeroPlanax-dev-tmp0429_lxy_reform/envs/models/heading_pitch_V")}')
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
    "LOADDIR": os.path.join(os.getcwd(),"/home/lczh/Git Project/AeroPlanax-dev-tmp0429_lxy_reform/envs/models/baseline/lstm_Yaw_Pitch_V/baseline_stable_2")
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
    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            pre_rewards=env_state.pre_rewards,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            hstate=extra_state,
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
    max_distance: float = 10000.0
    min_distance: float = 10000.0
    team_spacing: float = 600
    safe_distance: float = 100
    posture_reward_scale: float = 100.0
    use_baseline: bool = True

    noise_features: int = 10
    top_k_ego_obs: int = -1
    top_k_enm_obs: int = -1

'''
TODO: 临时的reward,能够在3B step上训练到30%-40%成功率（失败的情况大概也只差一点血）
'''
# def posture_reward_fn(
#     state: HierarchicalCombatTaskState,
#     params: HierarchicalCombatTaskParams,
#     agent_id: AgentID,
#     reward_scale: float = 1.0,
#     num_allies: int = 1,
#     num_enemies: int = 1,
# ) -> float:
#     """
#     Reward is a complex function of AO, TA and R in the last timestep.
#     Added numerical stability checks.
#     """
#     def safe_exp(x):
#         return jnp.exp(jnp.clip(x, -10, 10))
    
#     def safe_divide(a, b):
#         return a / (b + 1e-8)
    
#     ego_feature = jnp.hstack((
#         state.plane_state.north[agent_id],
#         state.plane_state.east[agent_id],
#         state.plane_state.altitude[agent_id],
#         state.plane_state.vel_x[agent_id],
#         state.plane_state.vel_y[agent_id],
#         state.plane_state.vel_z[agent_id]
#     ))
    
#     enm_list = jax.lax.select(
#         agent_id < num_allies,
#         jnp.arange(num_allies, num_allies + num_enemies),
#         jnp.arange(num_allies)
#     )
    
#     def compute_enemy_reward(enm):
#         enm_feature = jnp.hstack((
#             state.plane_state.north[enm],
#             state.plane_state.east[enm],
#             state.plane_state.altitude[enm],
#             state.plane_state.vel_x[enm],
#             state.plane_state.vel_y[enm],
#             state.plane_state.vel_z[enm]
#         ))
        
#         AO, TA, R, _ = get_AO_TA_R(ego_feature, enm_feature)
        
#         # 使用安全的指数计算
#         AO_reward = safe_exp(-AO / (jnp.pi / 7))
#         TA_reward = safe_exp(-TA / (jnp.pi / 7))
        
#         # 使用安全的距离奖励计算
#         R = jnp.clip(R / 1000.0, 0.0, 20.0)  # 限制R的范围
#         range_reward = 1 * (R < 5) + (R >= 5) * jnp.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0.0, 1.0)
        
#         # 使用安全的乘法
#         orientation_reward = jnp.clip(AO_reward * TA_reward, 0.0, 1.0)
        
#         mask = state.plane_state.is_alive[enm] | state.plane_state.is_locked[enm]
#         return orientation_reward * range_reward * mask
    
#     # 使用vmap计算所有敌机的奖励
#     per_target_rewards = jax.vmap(compute_enemy_reward)(enm_list)
#     # 取求和
#     # total_reward = jnp.max(per_target_rewards)
#     total_reward = jnp.sum(per_target_rewards)
    
#     mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
#     return jnp.clip(total_reward * reward_scale * mask, -100.0, 100.0)

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
    """
    new_reward = 0.0
    # feature: (north, east, down, vn, ve, vd)
    ego_feature = jnp.hstack((state.plane_state.north[agent_id],
                              state.plane_state.east[agent_id],
                              state.plane_state.altitude[agent_id],
                              state.plane_state.vel_x[agent_id],
                              state.plane_state.vel_y[agent_id],
                              state.plane_state.vel_z[agent_id]))
    enm_list = jax.lax.select(agent_id < num_allies, 
                              jnp.arange(num_allies, num_allies + num_enemies),
                              jnp.arange(num_allies))
    for enm in enm_list:
        enm_feature = jnp.hstack((state.plane_state.north[enm],
                                  state.plane_state.east[enm],
                                  state.plane_state.altitude[enm],
                                  state.plane_state.vel_x[enm],
                                  state.plane_state.vel_y[enm],
                                  state.plane_state.vel_z[enm]))
        AO, TA, R, _ = get_AO_TA_R(ego_feature, enm_feature)
        orientation_reward = orientation_reward_fn_new(AO, TA)
        range_reward = range_reward_fn(R / 1000.0)
        mask = state.plane_state.is_alive[enm] | state.plane_state.is_locked[enm]
        new_reward += orientation_reward * range_reward * mask
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return new_reward * reward_scale * mask

def range_reward_fn(R):
    reward = 1 * (R < 5) + (R >= 5) * jnp.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0.0, 1.0) \
        + jnp.clip(jnp.exp(-0.16 * R), 0, 0.2)
    return reward

def orientation_reward_fn_new(AO, TA):
    # jax.debug.print("orientation_reward_fn_new START --- AO: {ao}, TA: {ta}", ao=AO, ta=TA)
    # Clip AO and TA to be within a reasonable range for angular values, e.g., [0, pi]
    AO = jnp.clip(AO, 0.0, jnp.pi)
    TA = jnp.clip(TA, 0.0, jnp.pi)

    # AO_reward: Encourage pointing nose at the target.
    # Smaller AO is better. exp(-AO/scale) gives reward from 1 (AO=0) down to 0.
    ao_scale = jnp.pi / 7.0 # Characteristic angle for AO decay
    safe_AO_term_input = jnp.clip(-AO / ao_scale, -20.0, 20.0)
    AO_reward_val = jnp.exp(safe_AO_term_input)
    # jax.debug.print("orientation_reward_fn_new: AO: {ao}, safe_AO_term_input: {inp}, AO_reward_val: {val}", 
    #                 ao=AO, inp=safe_AO_term_input, val=AO_reward_val)

    # TA_reward: Encourage being in the target's rear hemisphere.
    # TA close to jnp.pi (180 deg) is best.
    # (cos(pi - TA) + 1) / 2 maps TA=[0,pi] to reward=[0,1] (0 for head-on, 1 for directly behind)
    base_ta_reward = (jnp.cos(jnp.pi - TA) + 1.0) / 2.0
    TA_reward_val = base_ta_reward ** 2 
    # jax.debug.print("orientation_reward_fn_new: TA: {ta}, base_ta_reward: {btr}, TA_reward_val (powered): {val}", 
    #                 ta=TA, btr=base_ta_reward, val=TA_reward_val)
    
    # Product and sqrt, ensure product is non-negative before sqrt
    product = AO_reward_val * TA_reward_val
    product = jnp.maximum(product, 0.0) # Ensure non-negative before sqrt
    # jax.debug.print("orientation_reward_fn_new: product (AO_reward * TA_reward): {val}", val=product)
    
    reward = jnp.sqrt(product)
    # Clip final reward to [0, 1] as it's a component of a larger reward structure
    reward = jnp.clip(reward, 0.0, 1.0)
    # jax.debug.print("orientation_reward_fn_new END --- AO: {ao}, TA: {ta}, final_reward: {val}", ao=AO, ta=TA, val=reward)
    return reward

'''
更加强调agent学习攻击具体的[一个]敌机
不过效果不佳
'''
def posture_reward_softmax_fn(
        state: HierarchicalCombatTaskState,
        params: HierarchicalCombatTaskParams,
        agent_id: AgentID,
        reward_scale: float = 1.0,
        num_allies: int = 1,
        num_enemies: int = 1,
        temperature: float = 0.3
    ) -> float:
    """
    Reward is a complex function of AO, TA and R in the last timestep.
    """
    # feature: (north, east, down, vn, ve, vd)
    ego_feature = jnp.hstack((state.plane_state.north[agent_id],
                              state.plane_state.east[agent_id],
                              state.plane_state.altitude[agent_id],
                              state.plane_state.vel_x[agent_id],
                              state.plane_state.vel_y[agent_id],
                              state.plane_state.vel_z[agent_id]))
    enm_list = jax.lax.select(agent_id < num_allies, 
                              jnp.arange(num_allies, num_allies + num_enemies),
                              jnp.arange(num_allies))
    
    def compute_enemy_reward(enm_id):
        enm_feature = jnp.hstack((
            state.plane_state.north[enm_id],
            state.plane_state.east[enm_id],
            state.plane_state.altitude[enm_id],
            state.plane_state.vel_x[enm_id],
            state.plane_state.vel_y[enm_id],
            state.plane_state.vel_z[enm_id],
        ))

        AO, TA, R, _ = get_AO_TA_R(ego_feature, enm_feature)
        orientation_reward = orientation_reward_fn_new(AO, TA)
        range_reward = range_reward_fn(R / 1000.0)
        alive_or_locked = state.plane_state.is_alive[enm_id] | state.plane_state.is_locked[enm_id]
        return orientation_reward * range_reward * alive_or_locked

    per_target_rewards = jax.vmap(compute_enemy_reward)(enm_list)

    # 使用 softmax with temperature 计算权重
    weights = jax.nn.softmax(per_target_rewards / temperature)
    weighted_reward = jnp.sum(per_target_rewards * weights)

    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return weighted_reward * reward_scale * mask

def tactical_position_reward_fn(
    state: HierarchicalCombatTaskState,
    params: HierarchicalCombatTaskParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
    num_allies: int = 1,
    num_enemies: int = 1,
) -> float:
    """
    计算战术位置奖励，主要考虑：
    1. 保持在敌机后半球
    2. 保持合适的攻击距离
    3. 保持高度优势
    """
    def safe_exp(x):
        return jnp.exp(jnp.clip(x, -100, 100))
    
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
        
        # 后半球奖励 (TA接近0表示在敌机后半球)
        rear_hemisphere_reward = safe_exp(-TA / (jnp.pi / 4))
        
        # 距离奖励 (保持在最佳攻击距离)
        optimal_distance = params.max_distance / 3  # 假设最佳攻击距离为最大距离的1/3
        # Add epsilon to denominator for distance_reward
        distance_denominator = (2 * (optimal_distance/2)**2) + 1e-8
        distance_reward = safe_exp(-(R - optimal_distance)**2 / distance_denominator)
        
        # 高度优势奖励
        # Add epsilon to denominator for altitude_reward
        altitude_denominator = params.max_altitude + 1e-8
        altitude_diff = (state.plane_state.altitude[agent_id] - state.plane_state.altitude[enm_id]) / altitude_denominator
        altitude_reward = jnp.clip(altitude_diff, -1.0, 1.0)
        
        # 综合奖励
        tactical_reward = (rear_hemisphere_reward * 0.4 + 
                         distance_reward * 0.4 + 
                         altitude_reward * 0.2)
        
        alive_or_locked = state.plane_state.is_alive[enm_id] | state.plane_state.is_locked[enm_id]
        return tactical_reward * alive_or_locked
    
    per_target_rewards = jax.vmap(compute_tactical_reward)(enm_list)
    # 取求和
    total_reward = jnp.sum(per_target_rewards)

    
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return total_reward * reward_scale * mask

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
            # functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
            # TODO: 暂时使用简单形式reward测试
            functools.partial(posture_reward_fn, 
                              reward_scale=1, 
                              num_allies=env_params.num_allies, 
                              num_enemies=env_params.num_enemies),
            # functools.partial(tactical_position_reward_fn,
            #                  reward_scale=0.5,  # 可以调整权重
            #                  num_allies=env_params.num_allies,
            #                  num_enemies=env_params.num_enemies),
            functools.partial(event_driven_reward_fn, fail_reward=-200, narrow_victory_reward=20, normal_victory_reward=50, complete_victory_reward=200),
        ]
        self.is_potential = [False, True]

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
            target_vt = init_state.plane_state.vt + delta_vt * 340
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
            
            # delta pitch - enemies attempt to match allies' pitch plus a random factor
            enm_pitch = init_state.plane_state.pitch[:self.num_allies]
            ego_pitch = init_state.plane_state.pitch[self.num_allies:]
            enm_delta_pitch = enm_pitch - ego_pitch
            
            # delta heading
            ego_v = jnp.linalg.norm(jnp.vstack((ego_vx, ego_vy)), axis=0)
            delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
            R = jnp.linalg.norm(jnp.vstack((delta_x, delta_y)), axis=0)
            proj_dist = delta_x * ego_vx + delta_y * ego_vy
            ego_AO = jnp.arccos(jnp.clip(proj_dist / (R * ego_v + 1e-6), -1, 1))
            # side_flag = jnp.sign(jnp.cross(jnp.vstack((ego_vx, ego_vy)), jnp.vstack((delta_x, delta_y))))
            side_flag = jnp.sign(ego_vx * delta_y - ego_vy * delta_x)
            enm_delta_heading = ego_AO * side_flag
            # delta velocity
            enm_delta_vt = init_state.plane_state.vt[:self.num_allies] - init_state.plane_state.vt[self.num_allies:]
            
            # NOTE:过大的值似乎容易导致飞机crash
            enm_delta_pitch = jnp.clip(enm_delta_pitch, -jnp.pi/10, jnp.pi/10)
            enm_delta_heading = jnp.clip(enm_delta_heading, -jnp.pi / 10, jnp.pi / 10)
            enm_delta_vt = jnp.clip(enm_delta_vt, -20, 20)

            delta_pitch = jnp.hstack((ego_delta_pitch, enm_delta_pitch))
            delta_heading = jnp.hstack((ego_delta_heading, enm_delta_heading))
            delta_vt = jnp.hstack((ego_delta_vt, enm_delta_vt))
            
            target_pitch = init_state.plane_state.pitch + delta_pitch
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
        state = HierarchicalCombatTaskState.create(state, init_hstate)
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

        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=yaw,
                vt=vt,
                q0=q0,
                q3=q3,
            )
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
        alive_mask = state.plane_state.is_alive | state.plane_state.is_locked
        info['alive_count'] = alive_mask[:self.num_allies].sum()
        # NOTE:
        # success(from super()._step_task())表示"完胜"，全部我方飞机都存活、全部敌方飞机都坠毁
        # success_simple表示"普通胜利"，任意我方飞机存活，全部敌方飞机坠毁
        # success_weak表示"险胜"，双方飞机都有存活，但我方飞机剩余总生命值更高
        # blood表示我方飞机剩余总生命值与对方的差值
        info['success_simple'] = jnp.any(alive_mask[:self.num_allies]) & jnp.all(~alive_mask[self.num_allies:])
        
        info['blood'] = jnp.sum(jnp.where(alive_mask, (state.plane_state.blood * jnp.where(jnp.arange(self.num_agents) < self.num_allies, 1., -1.)), 0))

        # info['success_weak'] = info['success_simple'] | (jnp.any(alive_mask[:self.num_allies]) & jnp.any(alive_mask[self.num_allies:]) & (info['blood'] > 1e-4))
        info['success_weak'] = ~info['success_simple'] & (jnp.any(alive_mask[:self.num_allies]) & jnp.any(alive_mask[self.num_allies:]) & (info['blood'] > 1e-4))
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
        success_simple_rate = jnp.where(valid_index_count>0, metric["success_simple"][indexs].mean(), 0.)
        success_weak_rate = jnp.where(valid_index_count>0, metric["success_weak"][indexs].mean(), 0.)
        win_rate = success_simple_rate + success_weak_rate
        blood_advantage = jnp.where(valid_index_count>0, metric["blood"][indexs].mean(), 0.)
        alive_count = jnp.where(valid_index_count>0, metric["alive_count"][indexs].mean(), 0.)

        writer.add_scalar('eval/episodic_return', episodic_return, env_steps)
        writer.add_scalar('eval/episodic_length', episodic_length, env_steps)
        writer.add_scalar('eval/success_rate', success_rate, env_steps)
        writer.add_scalar('eval/success_simple_rate', success_simple_rate, env_steps)
        writer.add_scalar('eval/success_weak_rate', success_weak_rate, env_steps)
        writer.add_scalar('eval/win_rate', win_rate, env_steps)
        writer.add_scalar('eval/blood_advantage', blood_advantage, env_steps)
        writer.add_scalar('eval/alive_count', alive_count, env_steps)

        print(f"EnvStep={env_steps:<5} EpisodeLength={episodic_length:<7.2f} Return={episodic_return:<7.2f} SuccessRate={success_rate:.3f} " + \
              f"SimpleSuccessRate={success_simple_rate:.3f} WeakSuccessRate={success_weak_rate:.3f} WinRate={win_rate:.3f} " + \
              f"AliveCount={alive_count:>6.3f} BloodAdvantage={blood_advantage:>8.2f}")
        
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

        # 根据队形类型选择生成函数
        if self.formation_type == 0:
            ally_positions = wedge_formation(self.num_allies, params.team_spacing)
            enemy_positions = wedge_formation(self.num_enemies, params.team_spacing)
        elif self.formation_type == 1:
            ally_positions = line_formation(self.num_allies, params.team_spacing)
            enemy_positions = line_formation(self.num_enemies, params.team_spacing)
        elif self.formation_type == 1:
            ally_positions = diamond_formation(self.num_allies, params.team_spacing)
            enemy_positions = diamond_formation(self.num_enemies, params.team_spacing)
        else:
            raise ValueError("Provided formation type is not valid")
        
        # 转换为全局坐标并确保安全距离        
        ally_center = jnp.zeros(3)
        enemy_center = jnp.zeros(3)
        key, key_distance, key_altitude = jax.random.split(key, 3)
        distance = jax.random.uniform(key_distance, minval=params.min_distance, maxval=params.max_distance)
        altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
        ally_center =  ally_center.at[0].set(-distance / 2)
        ally_center =  ally_center.at[2].set(altitude)
        enemy_center =  enemy_center.at[0].set(distance / 2)
        enemy_center =  enemy_center.at[2].set(altitude)
        formation_positions = jnp.vstack((enforce_safe_distance(ally_positions, ally_center, params.safe_distance),
                                          enforce_safe_distance(enemy_positions, enemy_center, params.safe_distance)))
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