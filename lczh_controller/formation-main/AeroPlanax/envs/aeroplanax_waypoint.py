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
    waypoint_reward_fn,
    altitude_reward_fn,
    event_driven_reward_fn,
)

from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    reached_waypoint_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class WaypointTaskState(EnvState):
    waypoint_north: ArrayLike  # North coordinate of waypoint
    waypoint_east: ArrayLike   # East coordinate of waypoint
    waypoint_altitude: ArrayLike  # Altitude of waypoint
    distance_to_waypoint: ArrayLike  # Current distance to waypoint
    waypoint_count: ArrayLike  # Number of waypoints reached
    last_waypoint_time: ArrayLike  # Time when last waypoint was reached

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            waypoint_north=extra_state[0],
            waypoint_east=extra_state[1],
            waypoint_altitude=extra_state[2],
            distance_to_waypoint=extra_state[3],
            waypoint_count=env_state.time * 0,  # Initialize waypoint count to zero
            last_waypoint_time=env_state.time,
        )


@struct.dataclass(frozen=True)
class WaypointTaskParams(EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 1
    formation_type: int = 0  # 0: wedge, 1: line, 2: diamond
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    max_altitude: float = 20000.0
    min_altitude: float = 2000.0
    max_vt: float = 360.0
    min_vt: float = 120.0
    min_waypoint_distance: float = 500.0  # Minimum distance to generate waypoint (m)
    max_waypoint_distance: float = 3000.0  # Maximum distance to generate waypoint (m)
    max_altitude_difference: float = 300.0  # Maximum altitude difference for waypoints (m)
    waypoint_radius: float = 200.0  # Radius within which a waypoint is considered reached (m)
    waypoint_timeout: int = 200  # Maximum time steps to reach a waypoint before generating a new one
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    noise_scale: float = 0.0
    team_spacing: float = 15000
    safe_distance: float = 3000  # Minimum safe distance between aircraft in formation


class AeroPlanaxWaypointEnv(AeroPlanaxEnv[WaypointTaskState, WaypointTaskParams]):
    def __init__(self, env_params: Optional[WaypointTaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(waypoint_reward_fn, reward_scale=1.0),
            # functools.partial(altitude_reward_fn, reward_scale=0.5, Kv=0.2),
            functools.partial(event_driven_reward_fn, fail_reward=-50, success_reward=50),
        ]

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            reached_waypoint_fn,
        ]

    def _get_obs_size(self) -> int:
        return 20  # Similar observation size as the heading_pitch environment

    @property
    def default_params(self) -> WaypointTaskParams:
        return WaypointTaskParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: WaypointTaskParams,
    ) -> WaypointTaskState:
        state = super()._init_state(key, params)
        state = WaypointTaskState.create(state, extra_state=jnp.zeros((4, self.num_agents)))
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: WaypointTaskState,
        params: WaypointTaskParams,
    ) -> WaypointTaskState:
        """Task-specific reset."""
        state = self._generate_formation(key, state, params)
        key, key_vx, key_vy, key_vz = jax.random.split(key, 4)
        
        # Generate initial velocities
        vx = jax.random.uniform(key, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        vy = jax.random.uniform(key_vy, shape=(self.num_agents,), minval=-10, maxval=10)
        vz = jax.random.uniform(key_vz, shape=(self.num_agents,), minval=-10, maxval=10)
        
        # Calculate total velocity
        vt = jnp.sqrt(vx**2 + vy**2 + vz**2)
        
        # Ensure velocity is within acceptable range
        scale = jnp.clip(vt, params.min_vt, params.max_vt) / (vt + 1e-6)
        vx = vx * scale
        vy = vy * scale
        vz = vz * scale
        vt = jnp.sqrt(vx**2 + vy**2 + vz**2)

        # Generate initial waypoint
        state = self._generate_waypoint(key, state, params)
        
        state = state.replace(
            plane_state=state.plane_state.replace(
                vel_x=vx,
                vel_y=vy,
                vel_z=vz,
                vt=vt,
            ),
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: WaypointTaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: WaypointTaskParams,
    ) -> Tuple[WaypointTaskState, Dict[str, Any]]:
        """Task-specific step transition."""
        # If waypoint was reached, generate a new one
        new_state = jax.lax.cond(
            state.success,
            lambda: self._on_waypoint_reached(key, state, params),
            lambda: state
        )

        # Check if waypoint timeout has been reached (time since last waypoint generation exceeds timeout)
        time_since_last_waypoint = state.time - state.last_waypoint_time
        waypoint_timeout_reached = time_since_last_waypoint >= params.waypoint_timeout
        
        # Generate new waypoint if timeout reached
        key, subkey = jax.random.split(key)
        new_state = jax.lax.cond(
            waypoint_timeout_reached & ~state.success,  # Only if not already succeeded
            lambda: self._on_waypoint_timeout(subkey, new_state, params),
            lambda: new_state
        )
        
        # Calculate current distance to waypoint for each agent
        north_dist = new_state.waypoint_north - new_state.plane_state.north
        east_dist = new_state.waypoint_east - new_state.plane_state.east
        alt_dist = new_state.waypoint_altitude - new_state.plane_state.altitude
        
        distance = jnp.sqrt(north_dist**2 + east_dist**2 + alt_dist**2)
        
        new_state = new_state.replace(
            distance_to_waypoint=distance,
        )
        
        info["waypoint_count"] = new_state.waypoint_count
        info["distance_to_waypoint"] = new_state.distance_to_waypoint
        info["time_since_last_waypoint"] = time_since_last_waypoint
        return new_state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _on_waypoint_timeout(
        self,
        key: chex.PRNGKey,
        state: WaypointTaskState,
        params: WaypointTaskParams,
    ) -> WaypointTaskState:
        """Handle waypoint timeout event by generating a new waypoint."""
        # Generate new waypoint
        new_state = self._generate_waypoint(key, state, params)
        
        # Reset last_waypoint_time but don't increment waypoint_count
        # since we're replacing a waypoint that wasn't reached
        new_state = new_state.replace(
            last_waypoint_time=state.time,
        )
        
        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _on_waypoint_reached(
        self,
        key: chex.PRNGKey,
        state: WaypointTaskState,
        params: WaypointTaskParams,
    ) -> WaypointTaskState:
        """Handle waypoint reached event."""
        # Generate new waypoint
        new_state = self._generate_waypoint(key, state, params)
        
        # Increment waypoint count and reset success flag
        new_state = new_state.replace(
            success=False,
            waypoint_count=state.waypoint_count + 1,
            last_waypoint_time=state.time,
        )
        
        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _generate_waypoint(
        self,
        key: chex.PRNGKey,
        state: WaypointTaskState,
        params: WaypointTaskParams,
    ) -> WaypointTaskState:
        """Generate a new waypoint relative to each aircraft."""
        keys = jax.random.split(key, 4)
        
        # Generate random distances within the specified range
        distances = jax.random.uniform(
            keys[0], 
            shape=(self.num_agents,), 
            minval=params.min_waypoint_distance, 
            maxval=params.max_waypoint_distance
        )
        
        # Generate random angles in the horizontal plane
        angles = jax.random.uniform(
            keys[1],
            shape=(self.num_agents,),
            minval=-0.5*jnp.pi,
            maxval=0.5*jnp.pi
        )
        
        # Generate random altitude differences
        alt_diffs = jax.random.uniform(
            keys[2],
            shape=(self.num_agents,),
            minval=-params.max_altitude_difference/2,
            maxval=params.max_altitude_difference/2
        )
        
        # Calculate waypoint coordinates relative to aircraft
        north_offsets = distances * jnp.cos(angles)
        east_offsets = distances * jnp.sin(angles)
        
        # Set waypoint coordinates
        waypoint_north = state.plane_state.north + north_offsets
        waypoint_east = state.plane_state.east + east_offsets
        waypoint_altitude = jnp.clip(
            state.plane_state.altitude + alt_diffs,
            params.min_altitude,
            params.max_altitude
        )
        
        # Calculate initial distance to waypoint
        distance = jnp.sqrt(north_offsets**2 + east_offsets**2 + alt_diffs**2)
        
        return state.replace(
            waypoint_north=waypoint_north,
            waypoint_east=waypoint_east,
            waypoint_altitude=waypoint_altitude,
            distance_to_waypoint=distance,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: WaypointTaskState,
        params: WaypointTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        Task-specific observation function.

        observation (dim 20):
            0. delta_north             (normalized distance to waypoint north)
            1. delta_east              (normalized distance to waypoint east)
            2. delta_altitude          (normalized distance to waypoint altitude)
            3. distance_to_waypoint    (normalized total distance to waypoint)
            4. waypoint_count          (number of waypoints reached)
            5. ego_altitude            (unit: 5km)
            6. ego_roll_sin
            7. ego_roll_cos
            8. ego_pitch_sin
            9. ego_pitch_cos
            10. ego_yaw_sin
            11. ego_yaw_cos
            12. ego_vel_x              (unit: m/s)
            13. ego_vel_y              (unit: m/s)
            14. ego_vel_z              (unit: m/s)
            15. ego_alpha_sin
            16. ego_alpha_cos
            17. ego_P                  (unit: rad/s)
            18. ego_Q                  (unit: rad/s)
            19. ego_R                  (unit: rad/s)
        """
        # Calculate relative positions to waypoint
        delta_north = state.waypoint_north - state.plane_state.north
        delta_east = state.waypoint_east - state.plane_state.east
        delta_altitude = state.waypoint_altitude - state.plane_state.altitude
        
        # Normalize distances (divide by max waypoint distance for relative coords)
        norm_delta_north = delta_north / params.max_waypoint_distance
        norm_delta_east = delta_east / params.max_waypoint_distance
        norm_delta_altitude = delta_altitude / params.max_waypoint_distance
        norm_distance = state.distance_to_waypoint / params.max_waypoint_distance
        
        # Get other state variables
        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        vel_x = state.plane_state.vel_x
        vel_y = state.plane_state.vel_y
        vel_z = state.plane_state.vel_z
        alpha = state.plane_state.alpha
        beta = state.plane_state.beta
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R
        
        # Normalized altitude
        norm_altitude = altitude / 5000
        
        # Trigonometric features
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        yaw_sin = jnp.sin(yaw)
        yaw_cos = jnp.cos(yaw)
        
        # Normalized velocities
        norm_vel_x = vel_x / 340
        norm_vel_y = vel_y / 340
        norm_vel_z = vel_z / 340
        
        # Angle of attack features
        alpha_sin = jnp.sin(alpha)
        alpha_cos = jnp.cos(alpha)
        
        # Stack all features
        obs = jnp.vstack((
            norm_delta_north, norm_delta_east, norm_delta_altitude, norm_distance,
            state.waypoint_count,
            norm_altitude,
            roll_sin, roll_cos, pitch_sin, pitch_cos, yaw_sin, yaw_cos,
            norm_vel_x, norm_vel_y, norm_vel_z,
            alpha_sin, alpha_cos,
            P, Q, R
        ))
        
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}

    @functools.partial(jax.jit, static_argnums=(0,))
    def _generate_formation(
        self,
        key: chex.PRNGKey,
        state: WaypointTaskState,
        params: WaypointTaskParams,
    ) -> WaypointTaskState:
        # Choose formation based on formation type
        if self.formation_type == 0:
            team_positions = wedge_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 1:
            team_positions = line_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 2:
            team_positions = diamond_formation(self.num_allies, params.team_spacing)
        else:
            raise ValueError("Provided formation type is not valid")
        
        # Convert to global coordinates and ensure safe distance
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