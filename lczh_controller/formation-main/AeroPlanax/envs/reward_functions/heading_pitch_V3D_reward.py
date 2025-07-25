import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI
import jax


def heading_pitch_V3D_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0,
        time_penalty: float = 0.2   # 每步惩罚 0.2
    ) -> float:
    """
    Measure the difference between current and target values for heading, pitch and velocity vector
    """
    roll = state.plane_state.roll[agent_id]
    pitch = state.plane_state.pitch[agent_id]
    yaw = state.plane_state.yaw[agent_id]
    vel_x = state.plane_state.vel_x[agent_id]
    vel_y = state.plane_state.vel_y[agent_id]
    vel_z = state.plane_state.vel_z[agent_id]
    
    # Calculate differences from target values
    delta_heading = wrap_PI(yaw - state.target_heading[agent_id])
    delta_pitch = wrap_PI(pitch - state.target_pitch[agent_id])
    delta_vel_x = vel_x - state.target_vel_x[agent_id]
    delta_vel_y = vel_y - state.target_vel_y[agent_id]
    delta_vel_z = vel_z - state.target_vel_z[agent_id]
    
    # Define error scales for different components
    heading_error_scale = jnp.pi / 72  # radians (5 degrees)
    heading_r = jnp.exp(-((delta_heading / heading_error_scale) ** 2))
    
    pitch_error_scale = jnp.pi / 72  # radians (5 degrees)
    pitch_r = jnp.exp(-((delta_pitch / pitch_error_scale) ** 2))
    
    roll_error_scale = 0.35  # radians ~= 20 degrees
    roll_r = jnp.exp(-((roll / roll_error_scale) ** 2))
    
    velocity_error_scale = 24  # m/s (~10%)
    vel_x_r = jnp.exp(-((delta_vel_x / velocity_error_scale) ** 2))
    vel_y_r = jnp.exp(-((delta_vel_y / velocity_error_scale) ** 2))
    vel_z_r = jnp.exp(-((delta_vel_z / velocity_error_scale) ** 2))
    
    # Combine rewards with weighted geometric mean
    w_heading = 0.3
    w_pitch = 0.2
    w_roll = 0.1
    w_vel = 0.4  # 速度分量权重总和为0.4，每个分量0.133

    reward_target = (
        heading_r**w_heading *
        pitch_r**w_pitch *
        roll_r**w_roll *
        vel_x_r**(w_vel/3) *
        vel_y_r**(w_vel/3) *
        vel_z_r**(w_vel/3)
    )
    
    # Apply mask for alive/locked state
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    
    return (reward_target - time_penalty) * reward_scale * mask 