import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI
import jax


def pitch_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0
    ) -> float:
    """
    Measure the difference between the current pitch angle and the target pitch angle
    """
    altitude = state.plane_state.altitude[agent_id]
    pitch = state.plane_state.pitch[agent_id]
    vt = state.plane_state.vt[agent_id]
    
    # Calculate differences from target values
    delta_altitude = (altitude - state.target_altitude[agent_id])
    delta_pitch = wrap_PI(pitch - state.target_pitch[agent_id])
    delta_vt = (vt - state.target_vt[agent_id])
    
    # Define error scales for different components
    pitch_error_scale = jnp.pi / 36  # radians (5 degrees)
    pitch_r = jnp.exp(-((delta_pitch / pitch_error_scale) ** 2))
    
    alt_error_scale = 15.24  # m
    alt_r = jnp.exp(-((delta_altitude / alt_error_scale) ** 2))
    
    speed_error_scale = 24  # mps (~10%)
    speed_r = jnp.exp(-((delta_vt / speed_error_scale) ** 2))
    
    # Combine rewards with geometric mean
    reward_target = (pitch_r * alt_r * speed_r) ** (1 / 3)
    
    # Apply mask for alive/locked state
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    
    return reward_target * reward_scale * mask 