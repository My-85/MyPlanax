import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState


def reached_waypoint_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        max_waypoints: int = 50  # Maximum number of waypoints before termination
    ) -> tuple[bool, bool]:
    """
    Check if the aircraft has reached the current waypoint.
    
    Args:
        state: Current environment state
        params: Environment parameters
        agent_id: ID of the agent
        max_waypoints: Maximum number of waypoints to reach before terminating
    
    Returns:
        tuple[bool, bool]: (done, success) flags
            - done: True if episode should terminate
            - success: True if waypoint was reached successfully
    """
    # Get waypoint radius from params or use default
    waypoint_radius = getattr(params, "waypoint_radius", 200.0)
    
    # Calculate distance to waypoint - handle both scalar and array cases
    # For scalar case (single agent environment)
    if state.distance_to_waypoint.ndim == 0:
        distance = state.distance_to_waypoint
    else:
        # For array case (multi-agent environment)
        distance = state.distance_to_waypoint[agent_id]
    
    # Check if aircraft has reached waypoint (within radius)
    reached = distance <= waypoint_radius
    
    # Task completes successfully when reaching the waypoint
    success = reached
    
    # Check if we've reached the maximum number of waypoints
    # Handle both scalar and array cases for waypoint_count
    if state.waypoint_count.ndim == 0:
        max_reached = state.waypoint_count >= max_waypoints
    else:
        max_reached = state.waypoint_count[agent_id] >= max_waypoints
    
    # Episode terminates if max waypoints reached, but not when just reaching one waypoint
    # This allows continuous navigation through multiple waypoints
    done = max_reached
    
    return done, success 