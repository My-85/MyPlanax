from typing import Dict, Optional, Tuple, Any
from jax import Array
from jax.typing import ArrayLike
import chex
from .aeroplanax import AgentName, AgentID

import functools
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv
from .reward_functions import (
    heading_reward_fn,
    altitude_reward_fn,
    event_driven_reward_fn,
    heading_pitch_V3D_reward_fn,
    heading_pitch_v3D_event_reward_fn,
)

from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    unreach_heading_pitch_V3D_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class Heading_Pitch_V3D_TaskState(EnvState):
    target_heading: ArrayLike 
    target_pitch: ArrayLike
    target_vel_x: ArrayLike  # 目标速度x分量
    target_vel_y: ArrayLike  # 目标速度y分量
    target_vel_z: ArrayLike  # 目标速度z分量
    last_check_time: ArrayLike
    heading_turn_counts: ArrayLike

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            target_heading=extra_state[0],
            target_pitch=extra_state[1],
            target_vel_x=extra_state[2],
            target_vel_y=extra_state[3],
            target_vel_z=extra_state[4],
            last_check_time=env_state.time,
            heading_turn_counts=0,
        )


@struct.dataclass(frozen=True)
class Heading_Pitch_V3D_TaskParams(EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 1
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    max_altitude: float = 20000.0
    min_altitude: float = 2000.0
    max_vt: float = 360.0
    min_vt: float = 120.0
    max_heading_increment: float = jnp.pi  # 最大航向变化量(π≈180°)
    max_pitch_increment: float = jnp.pi/6  # 最大俯仰角变化量(30°)
    max_altitude_increment: float = 2100.0
    max_velocities_increment: float = 50.0  # 速度分量最大变化量
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    noise_scale: float = 0.0
    team_spacing: float = 15000       
    safe_distance: float = 3000 # 编队最小安全间距


class AeroPlanaxHeading_Pitch_V3D_Env(AeroPlanaxEnv[Heading_Pitch_V3D_TaskState, Heading_Pitch_V3D_TaskParams]):
    def __init__(self, env_params: Optional[Heading_Pitch_V3D_TaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(heading_pitch_V3D_reward_fn, reward_scale=1.0),
            functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
            functools.partial(event_driven_reward_fn, fail_reward=-50, success_reward=50),
            functools.partial(heading_pitch_v3D_event_reward_fn),
        ]

        # 与 reward_functions 一一对应，表示这些奖励是否做势能差分
        # 这里全部设为 False 即可
        self.is_potential = [False] * len(self.reward_functions)

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            unreach_heading_pitch_V3D_fn,
        ]

        # 课程学习：
        self.increment_size = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10)

    def _get_obs_size(self) -> int:
        return 20  # 观测维度为20，包括速度分量和角速度

    @property
    def default_params(self) -> Heading_Pitch_V3D_TaskParams:
        return Heading_Pitch_V3D_TaskParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: Heading_Pitch_V3D_TaskParams,
    ) -> Heading_Pitch_V3D_TaskState:
        state = super()._init_state(key, params)
        state = Heading_Pitch_V3D_TaskState.create(state, extra_state=jnp.zeros((5, self.num_agents)))  # 5个目标
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: Heading_Pitch_V3D_TaskState,
        params: Heading_Pitch_V3D_TaskParams,
    ) -> Heading_Pitch_V3D_TaskState:
        """Task-specific reset."""
        state = self._generate_formation(key, state, params)
        key, key_vx, key_vy, key_vz = jax.random.split(key, 4)
        
        # 生成初始速度分量
        vx = jax.random.uniform(key, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)  # 主要速度分量
        vy = jax.random.uniform(key_vy, shape=(self.num_agents,), minval=-10, maxval=10)  # 小的随机扰动
        vz = jax.random.uniform(key_vz, shape=(self.num_agents,), minval=-10, maxval=10)  # 小的随机扰动
        
        # 计算总速度
        vt = jnp.sqrt(vx**2 + vy**2 + vz**2)
        
        # 确保速度在合理范围内
        scale = jnp.clip(vt, params.min_vt, params.max_vt) / (vt + 1e-6)
        vx = vx * scale
        vy = vy * scale
        vz = vz * scale
        vt = jnp.sqrt(vx**2 + vy**2 + vz**2)  # 重新计算缩放后的总速度

        state = state.replace(
            plane_state=state.plane_state.replace(
                vel_x=vx,
                vel_y=vy,
                vel_z=vz,
                vt=vt,
            ),
            target_heading=state.plane_state.yaw,  # 初始目标航向=当前航向
            target_pitch=state.plane_state.pitch,  # 初始目标俯仰角=当前俯仰角
            target_vel_x=vx,  # 目标速度x分量
            target_vel_y=vy,  # 目标速度y分量
            target_vel_z=vz,  # 目标速度z分量
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: Heading_Pitch_V3D_TaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: Heading_Pitch_V3D_TaskParams,
    ) -> Tuple[Heading_Pitch_V3D_TaskState, Dict[str, Any]]:
        """Task-specific step transition."""
        key_heading, key_pitch, key_vx, key_vy, key_vz = jax.random.split(key, 5)
        delta = jax.random.uniform(key_heading, shape=(self.num_agents,), minval=0.2, maxval=1.0)
        # delta = 0.6 # 开始时先用小的变化量，使其首先能稳定飞行
        # 随机航向变化量(-π, π)
        delta_heading = jax.random.uniform(key_heading, shape=(self.num_agents,), minval=-params.max_heading_increment, maxval=params.max_heading_increment)
        
        # 根据当前高度限制俯仰角变化范围
        current_altitude = state.plane_state.altitude
        max_pitch = jnp.where(
            current_altitude > params.max_altitude - 1000,
            -params.max_pitch_increment * 0.5,
            params.max_pitch_increment
        )
        min_pitch = jnp.where(
            current_altitude < params.min_altitude + 1000,
            params.max_pitch_increment * 0.5,
            -params.max_pitch_increment
        )
        delta_pitch = jax.random.uniform(key_pitch, shape=(self.num_agents,), minval=min_pitch, maxval=max_pitch)
        
        # 速度分量变化量
        delta_vx = jax.random.uniform(key_vx, shape=(self.num_agents,), minval=-params.max_velocities_increment, maxval=params.max_velocities_increment)
        delta_vy = jax.random.uniform(key_vy, shape=(self.num_agents,), minval=-params.max_velocities_increment, maxval=params.max_velocities_increment)
        delta_vz = jax.random.uniform(key_vz, shape=(self.num_agents,), minval=-params.max_velocities_increment, maxval=params.max_velocities_increment)

        target_heading = wrap_PI(state.plane_state.yaw + delta_heading * delta)
        target_pitch = wrap_PI(state.plane_state.pitch + delta_pitch * delta)
        
        # 计算新的目标速度分量
        target_vx = state.plane_state.vel_x + delta_vx * delta
        target_vy = state.plane_state.vel_y + delta_vy * delta
        target_vz = state.plane_state.vel_z + delta_vz * delta
        
        # 确保目标速度大小在合理范围内
        target_vt = jnp.sqrt(target_vx**2 + target_vy**2 + target_vz**2)
        scale = jnp.clip(target_vt, params.min_vt, params.max_vt) / (target_vt + 1e-6)
        target_vx = target_vx * scale
        target_vy = target_vy * scale
        target_vz = target_vz * scale

        new_state = state.replace(
            plane_state=state.plane_state.replace(
                status=jnp.where(state.plane_state.is_success, 0, state.plane_state.status)
            ),
            success=False,
            target_heading=target_heading,
            target_pitch=target_pitch,
            target_vel_x=target_vx,
            target_vel_y=target_vy,
            target_vel_z=target_vz,
            last_check_time=state.time,
            heading_turn_counts=(state.heading_turn_counts + 1),
        )
        state = jax.lax.cond(state.success, lambda: new_state, lambda: state)
        info["heading_turn_counts"] = state.heading_turn_counts
        return state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: Heading_Pitch_V3D_TaskState,
        params: Heading_Pitch_V3D_TaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        Task-specific observation function to state.

        observation(dim 18):
            0. ego_delta_heading       (unit rad)
            1. ego_delta_pitch         (unit rad)
            2. ego_delta_vel_x         (unit: m/s)
            3. ego_delta_vel_y         (unit: m/s)
            4. ego_delta_vel_z         (unit: m/s)
            5. ego_altitude            (unit: 5km)
            6. ego_roll_sin
            7. ego_roll_cos
            8. ego_pitch_sin
            9. ego_pitch_cos
            10. ego_vel_x              (unit: m/s)
            11. ego_vel_y              (unit: m/s)
            12. ego_vel_z              (unit: m/s)
            13. ego_alpha_sin
            14. ego_alpha_cos
            15. ego_beta_sin
            16. ego_beta_cos
            17. ego_P                  (unit: rad/s)
            18. ego_Q                  (unit: rad/s)
            19. ego_R                  (unit: rad/s)
        """
        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        vel_x = state.plane_state.vel_x
        vel_y = state.plane_state.vel_y
        vel_z = state.plane_state.vel_z
        alpha = state.plane_state.alpha
        beta = state.plane_state.beta
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        norm_delta_heading = wrap_PI((yaw - state.target_heading))
        norm_delta_pitch = wrap_PI((pitch - state.target_pitch))
        norm_delta_vel_x = (vel_x - state.target_vel_x) / 340
        norm_delta_vel_y = (vel_y - state.target_vel_y) / 340
        norm_delta_vel_z = (vel_z - state.target_vel_z) / 340
        norm_altitude = altitude / 5000
        
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        
        norm_vel_x = vel_x / 340
        norm_vel_y = vel_y / 340
        norm_vel_z = vel_z / 340
        
        alpha_sin = jnp.sin(alpha)
        alpha_cos = jnp.cos(alpha)
        beta_sin = jnp.sin(beta)
        beta_cos = jnp.cos(beta)
        
        obs = jnp.vstack((
            norm_delta_heading, norm_delta_pitch,
            norm_delta_vel_x, norm_delta_vel_y, norm_delta_vel_z,
            norm_altitude,
            roll_sin, roll_cos, pitch_sin, pitch_cos,
            norm_vel_x, norm_vel_y, norm_vel_z,
            alpha_sin, alpha_cos, beta_sin, beta_cos,
            P, Q, R
        ))
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}

    @functools.partial(jax.jit, static_argnums=(0,))
    def _generate_formation(
        self,
        key: chex.PRNGKey,
        state: Heading_Pitch_V3D_TaskState,
        params: Heading_Pitch_V3D_TaskParams,
    ) -> Heading_Pitch_V3D_TaskState:
        # 根据队形类型选择生成函数
        if self.formation_type == 0:
            team_positions = wedge_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 1:
            team_positions = line_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 2:
            team_positions = diamond_formation(self.num_allies, params.team_spacing)
        else:
            raise ValueError("Provided formation type is not valid")
        
        # 转换为全局坐标并确保安全距离        
        team_center = jnp.zeros(3)
        key, key_altitude = jax.random.split(key)
        altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
        team_center = team_center.at[2].set(altitude)
        formation_positions = enforce_safe_distance(team_positions, team_center, params.safe_distance)
        state = state.replace(plane_state=state.plane_state.replace(
            north=formation_positions[:, 0],
            east=formation_positions[:, 1],
            altitude=formation_positions[:, 2]
        ))
        return state 