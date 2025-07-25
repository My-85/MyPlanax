import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI
from ..core.simulators.fighterplane.dynamics import FighterPlaneState


def unreach_heading_pitch_V3D_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        max_check_interval: int = 5,
        min_check_interval: int = 0.2
    ) -> tuple[bool, bool]:
    """
    检查飞机是否在限定时间内达到目标航向角、俯仰角和速度向量
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

    # 所有条件都满足才算成功
    # angle_success = mask1 & mask_heading & mask_pitch  # 角度变化满足
    # velocity_success = mask1 & mask_velocity  # 速度变化满足
    # success = angle_success | velocity_success  # 满足一个就算成功
    # success = (mask_heading & mask_pitch)|mask_velocity # 满足一个就算成功,不保持规定时间也算成功
    success = mask_heading & mask_pitch & mask_velocity # 同时满足三个条件才算成功
    done = False

    return done, success 