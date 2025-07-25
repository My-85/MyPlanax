import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
import jax


def waypoint_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0,
        approach_reward: float = 0.01,  # Reward for getting closer to waypoint
        distance_penalty: float = 0.01,  # Penalty for moving away from waypoint
        time_penalty: float = 0.1,  # Small time penalty per step
        waypoint_radius: float = 200.0  # Same as in params, but as a default
    ) -> float:
    """
    Reward function for waypoint navigation task.
    
    The reward consists of:
    1. Approach reward: positive reward when getting closer to the waypoint
    2. Distance penalty: negative reward when moving away from the waypoint
    3. Time penalty: small penalty for each step to encourage efficiency
    4. Waypoint bonus: large reward when reaching the waypoint
    5. Stability reward: reward for stable flight (not excessive roll, etc.)
    
    Args:
        state: Current environment state
        params: Environment parameters
        agent_id: ID of the agent
        reward_scale: Overall scale factor for the reward
        approach_reward: Reward multiplier for getting closer to waypoint
        distance_penalty: Penalty multiplier for moving away from waypoint
        time_penalty: Penalty per step
        waypoint_radius: Radius within which a waypoint is considered reached
    
    Returns:
        float: The reward value
    """
    # Extract relevant state variables
    north = state.plane_state.north[agent_id]
    east = state.plane_state.east[agent_id]
    altitude = state.plane_state.altitude[agent_id]
    roll = state.plane_state.roll[agent_id]
    
    # Calculate current distance to waypoint
    delta_north = state.waypoint_north[agent_id] - north
    delta_east = state.waypoint_east[agent_id] - east
    delta_altitude = state.waypoint_altitude[agent_id] - altitude
    
    # Current 3D distance to waypoint
    current_distance = jnp.sqrt(delta_north**2 + delta_east**2 + delta_altitude**2)
    
    # Get previous distance (stored in the state)
    # Handle both scalar and array cases
    if state.distance_to_waypoint.ndim == 0:
        previous_distance = state.distance_to_waypoint
    else:
        previous_distance = state.distance_to_waypoint[agent_id]
    
    # Calculate change in distance
    distance_change = previous_distance - current_distance
    
    # Get waypoint radius (distance at which waypoint is considered reached)
    waypoint_radius = getattr(params, "waypoint_radius", waypoint_radius)
    
    # Calculate approach/distance reward based on the change in distance
    # Positive reward if the aircraft is getting closer to the waypoint
    # Negative reward if the aircraft is moving away from the waypoint
    distance_reward = jnp.where(
        distance_change > 0,
        # Getting closer to waypoint - positive reward that increases as we get closer
        approach_reward * distance_change * (1.0 + params.waypoint_radius / (current_distance + 1e-6)),
        # Moving away from waypoint - negative reward that increases as we get further
        distance_penalty * distance_change * (1.0 + current_distance / params.max_waypoint_distance)
    )
    
    # Bonus reward for being very close to the waypoint
    close_distance_threshold = waypoint_radius * 2  # Threshold for "close" to waypoint
    close_scale = 0.2  # Extra reward multiplier for being close
    close_bonus = close_scale * jnp.exp(-current_distance / close_distance_threshold)
    
    # Large bonus when the aircraft reaches the waypoint
    reached_bonus = jnp.where(
        current_distance <= waypoint_radius,
        5.0,  # Large bonus when reaching waypoint
        0.0
    )
    
    # Stability reward - penalize excessive roll
    # Good flying practice is to maintain level flight especially when navigating
    roll_error_scale = 0.35  # radians ~= 20 degrees
    roll_penalty = 0.2 * (1 - jnp.exp(-(roll / roll_error_scale) ** 2))
    
    # Combine rewards
    reward = distance_reward - roll_penalty - time_penalty
    
    # Apply mask for alive/locked state
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    
    return reward * reward_scale * mask 