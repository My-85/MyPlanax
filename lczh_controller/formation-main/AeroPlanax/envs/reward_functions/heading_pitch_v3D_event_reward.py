import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI
from ..core.simulators.fighterplane.dynamics import FighterPlaneState


def heading_pitch_v3D_event_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        max_check_interval: int = 5,
        min_check_interval: int = 0.2
    ) -> float:
    """
    计算航向角、俯仰角和速度向量的奖励
    - 满足角度条件奖励20分
    - 满足速度条件奖励20分
    - 同时满足两个条件奖励60分
    """
    plane_state: FighterPlaneState = state.plane_state
    check_time = state.time - state.last_check_time
    max_check_interval = max_check_interval * params.sim_freq / params.agent_interaction_steps
    mask1 = check_time >= max_check_interval

    # 检查是否达到目标航向角（误差在5度以内）
    delta_heading = wrap_PI(state.plane_state.yaw[agent_id] - state.target_heading[agent_id])
    mask_heading = jnp.abs(delta_heading) <= jnp.pi / 72  # 5度

    # 检查是否达到目标俯仰角（误差在5度以内）
    delta_pitch = wrap_PI(state.plane_state.pitch[agent_id] - state.target_pitch[agent_id])
    mask_pitch = jnp.abs(delta_pitch) <= jnp.pi / 72  # 5度

    # 检查是否达到目标速度向量（每个分量误差在10m/s以内）
    delta_vel_x = jnp.abs(state.plane_state.vel_x[agent_id] - state.target_vel_x[agent_id])
    delta_vel_y = jnp.abs(state.plane_state.vel_y[agent_id] - state.target_vel_y[agent_id])
    delta_vel_z = jnp.abs(state.plane_state.vel_z[agent_id] - state.target_vel_z[agent_id])
    mask_velocity = (delta_vel_x <= 10.0) & (delta_vel_y <= 10.0) & (delta_vel_z <= 10.0)

    # 计算奖励
    angle_success = mask1 & mask_heading & mask_pitch  # 角度变化满足
    velocity_success = mask1 & mask_velocity  # 速度变化满足
    
    # 基础奖励为0
    reward = 0.0
    
    # # 如果满足角度条件，加20分
    # reward = jnp.where(angle_success, reward + 20.0, reward)
    
    # # 如果满足速度条件，加20分
    # reward = jnp.where(velocity_success, reward + 20.0, reward)
    
    # 如果同时满足两个条件，额外加20分（总共60分）
    both_success = angle_success & velocity_success
    reward = jnp.where(both_success, reward + 20.0, reward)

    return reward 