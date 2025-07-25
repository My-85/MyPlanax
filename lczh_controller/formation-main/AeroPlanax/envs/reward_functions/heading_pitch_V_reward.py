import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI
import jax


def heading_pitch_V_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0
    ) -> float:
    """
    Measure the difference between current and target values for heading, pitch and velocity
    """
    roll = state.plane_state.roll[agent_id]
    pitch = state.plane_state.pitch[agent_id]
    yaw = state.plane_state.yaw[agent_id]
    vt = state.plane_state.vt[agent_id]
    
    # Calculate differences from target values
    delta_heading = wrap_PI(yaw - state.target_heading[agent_id])
    delta_pitch = wrap_PI(pitch - state.target_pitch[agent_id])
    delta_vt = (vt - state.target_vt[agent_id])
    
    # Define error scales for different components
    heading_error_scale = jnp.pi / 72  # radians (5 degrees)
    heading_r = jnp.exp(-((delta_heading / heading_error_scale) ** 2))
    
    pitch_error_scale = jnp.pi / 72  # radians (5 degrees)
    pitch_r = jnp.exp(-((delta_pitch / pitch_error_scale) ** 2))
    
    roll_error_scale = 0.35  # radians ~= 20 degrees
    roll_r = jnp.exp(-((roll / roll_error_scale) ** 2))
    
    speed_error_scale = 24  # mps (~10%)
    speed_r = jnp.exp(-((delta_vt / speed_error_scale) ** 2))
    
    # Combine rewards with geometric mean
    # reward_target = (heading_r * pitch_r * roll_r * speed_r) ** (1 / 4)
    w_heading = 0.4
    w_pitch   = 0.3
    w_roll    = 0.1
    w_speed   = 0.2

    # 改用加权几何平均
    reward_target = (
        heading_r**w_heading *
        pitch_r**w_pitch *
        roll_r**w_roll *
        speed_r**w_speed
    )

    
    # Apply mask for alive/locked state
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    
    return reward_target * reward_scale * mask 