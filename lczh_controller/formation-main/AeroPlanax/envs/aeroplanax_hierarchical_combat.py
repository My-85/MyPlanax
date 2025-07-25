from typing import Dict, Optional, Tuple, Any
from jax import Array
from jax.typing import ArrayLike
import chex
from .aeroplanax import AgentName, AgentID

import functools
import jax
import jax.numpy as jnp
from flax import struct
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
from gymnax.environments import spaces
from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv
from .aeroplanax_combat import CombatTaskState, CombatTaskParams
from .aeroplanax_heading import HeadingTaskParams
import orbax.checkpoint as ocp

# 加载底层heading控制器
class HeadingController(nn.Module):
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
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return hidden, pi, jnp.squeeze(critic, axis=-1)

@struct.dataclass
class HierarchicalCombatTaskState(CombatTaskState):
    target_heading: ArrayLike
    target_altitude: ArrayLike
    target_speed: ArrayLike
    hstate: ArrayLike

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            target_heading=jnp.zeros(env_state.plane_state.yaw.shape),
            target_altitude=jnp.zeros(env_state.plane_state.altitude.shape),
            target_speed=jnp.zeros(env_state.plane_state.vt.shape),
            hstate=extra_state,
        )

@struct.dataclass
class CombatTaskParams(EnvParams):
    num_allies: int = 4  # 友方无人机数量
    num_enemies: int = 2  # 敌方无人机数量
    formation_type: str = "wedge"  # 编队类型：wedge, line, diamond
    formation_spacing: float = 100.0  # 编队间距（米）
    max_altitude: float = 7000.0  # 最大高度
    min_altitude: float = 3000.0  # 最小高度
    max_speed: float = 390.0  # 最大速度
    min_speed: float = 270.0  # 最小速度
    max_distance: float = 5000.0  # 最大距离
    min_distance: float = 100.0  # 最小距离
    unit_features: int = 4  # 每个单位的特征数
    own_features: int = 13  # 自身特征数

    def __post_init__(self):
        super().__post_init__()
        self.agents = [f"ally_{i}" for i in range(self.num_allies)] + [f"enemy_{i}" for i in range(self.num_enemies)]

class AeroPlanaxHierarchicalCombatEnv(AeroPlanaxEnv[HierarchicalCombatTaskState, CombatTaskParams]):
    def __init__(self, env_params: Optional[CombatTaskParams] = None):
        super().__init__(env_params)
        
        # 初始化底层控制器
        self.heading_controller = HeadingController(4, config={
            "FC_DIM_SIZE": 128,
            "GRU_HIDDEN_DIM": 128,
            "ACTIVATION": "relu"
        })
        
        # 加载预训练的heading控制器参数
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        state = {"params": None, "opt_state": None, "epoch": jnp.array(0)}
        checkpoint = ckptr.restore("path_to_heading_checkpoint", args=ocp.args.StandardRestore(item=state))
        self.heading_params = checkpoint["params"]

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=jnp.float32)  # [target_heading, target_altitude, target_speed]
        }

        self.reward_functions = [
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200),
            self._formation_reward_fn
        ]

        self.termination_conditions = [
            safe_return_fn,
            self._formation_termination_fn
        ]

    def _get_obs_size(self) -> int:
        # 添加编队相关的观测
        formation_features = 3  # 编队中心位置、速度、航向
        return (self.unit_features * (self.num_allies - 1) + 
                self.unit_features * self.num_enemies + 
                self.own_features +
                formation_features)

    def _formation_reward_fn(self, state: HierarchicalCombatTaskState) -> float:
        """计算编队保持的奖励"""
        formation_reward = 0.0
        for i in range(self.env_params.num_allies):
            # 计算与编队中心的距离
            center_pos = jnp.mean(state.plane_state.position[:self.env_params.num_allies], axis=0)
            dist_to_center = jnp.linalg.norm(state.plane_state.position[i] - center_pos)
            
            # 计算与编队中心的速度差
            center_vel = jnp.mean(state.plane_state.velocity[:self.env_params.num_allies], axis=0)
            vel_diff = jnp.linalg.norm(state.plane_state.velocity[i] - center_vel)
            
            # 计算与编队中心的航向差
            center_heading = jnp.mean(state.plane_state.yaw[:self.env_params.num_allies])
            heading_diff = jnp.abs(wrap_PI(state.plane_state.yaw[i] - center_heading))
            
            # 计算编队保持奖励
            formation_reward += (
                -0.1 * jnp.maximum(0, dist_to_center - self.env_params.formation_spacing) +
                -0.05 * vel_diff +
                -0.05 * heading_diff
            )
        
        return formation_reward / self.env_params.num_allies

    def _formation_termination_fn(self, state: HierarchicalCombatTaskState) -> bool:
        """检查编队是否解散"""
        for i in range(self.env_params.num_allies):
            center_pos = jnp.mean(state.plane_state.position[:self.env_params.num_allies], axis=0)
            dist_to_center = jnp.linalg.norm(state.plane_state.position[i] - center_pos)
            if dist_to_center > self.env_params.max_distance:
                return True
        return False

    def _generate_formation(self, rng: chex.PRNGKey) -> jnp.ndarray:
        """生成初始编队位置"""
        positions = []
        if self.env_params.formation_type == "wedge":
            # 楔形编队
            for i in range(self.env_params.num_allies):
                angle = jnp.pi / 4 * (i - (self.env_params.num_allies - 1) / 2)
                x = self.env_params.formation_spacing * jnp.cos(angle)
                y = self.env_params.formation_spacing * jnp.sin(angle)
                positions.append(jnp.array([x, y, 0.0]))
        elif self.env_params.formation_type == "line":
            # 直线编队
            for i in range(self.env_params.num_allies):
                x = self.env_params.formation_spacing * (i - (self.env_params.num_allies - 1) / 2)
                positions.append(jnp.array([x, 0.0, 0.0]))
        else:  # diamond
            # 菱形编队
            n = int(jnp.ceil(jnp.sqrt(self.env_params.num_allies)))
            for i in range(self.env_params.num_allies):
                row = i // n
                col = i % n
                x = self.env_params.formation_spacing * (col - (n - 1) / 2)
                y = self.env_params.formation_spacing * (row - (n - 1) / 2)
                positions.append(jnp.array([x, y, 0.0]))
        
        return jnp.array(positions)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_actions(
        self,
        key: chex.PRNGKey,
        init_state: HierarchicalCombatTaskState,
        state: HierarchicalCombatTaskState,
        actions: Dict[AgentName, chex.Array]
    ):
        # 解包高层动作
        actions = jnp.array([actions[i] for i in self.agents])
        target_heading = actions[:, 0] * jnp.pi  # 归一化到[-pi, pi]
        target_altitude = actions[:, 1] * 2000 + 5000  # 归一化到[3000, 7000]
        target_speed = actions[:, 2] * 60 + 330  # 归一化到[270, 390]

        # 更新状态中的目标值
        state = state.replace(
            target_heading=target_heading,
            target_altitude=target_altitude,
            target_speed=target_speed
        )

        # 使用底层控制器生成具体控制动作
        last_obs = self._get_controller_obs(state.plane_state, target_altitude, target_heading, target_speed)
        last_obs = jnp.transpose(last_obs)
        last_done = jnp.zeros((self.num_agents), dtype=bool)
        ac_in = (
            last_obs[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        hstate, pi, _ = self.heading_controller.apply(self.heading_params, state.hstate, ac_in)
        state = state.replace(hstate=hstate)
        action = pi.sample(seed=key)[0]
        action = jnp.clip(action, min=-1, max=1)

        return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(action)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_controller_obs(
        self,
        state: fighterplane.FighterPlaneState,
        target_altitude,
        target_heading,
        target_speed
    ) -> Dict[AgentName, chex.Array]:
        """获取底层控制器的观测"""
        altitude = state.altitude
        roll, pitch, yaw = state.roll, state.pitch, state.yaw
        vt = state.vt
        alpha = state.alpha
        beta = state.beta
        P, Q, R = state.P, state.Q, state.R

        norm_delta_altitude = (altitude - target_altitude) / 1000
        norm_delta_heading = wrap_PI((yaw - target_heading))
        norm_delta_vt = (vt - target_speed) / 340
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
        obs = jnp.vstack((norm_delta_altitude, norm_delta_heading, norm_delta_vt,
                          norm_altitude, norm_vt,
                          roll_sin, roll_cos, pitch_sin, pitch_cos,
                          alpha_sin, alpha_cos, beta_sin, beta_cos,
                          P, Q, R))
        return obs 