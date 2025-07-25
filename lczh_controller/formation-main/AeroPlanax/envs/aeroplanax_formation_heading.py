from typing import Dict, Optional, Tuple, Any
from jax import Array
from jax.typing import ArrayLike
import chex

import functools
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv, AgentName, AgentID
from .reward_functions import (
    heading_reward_fn,
    altitude_reward_fn,
    event_driven_reward_fn,
    formation_reward_fn,
    crash_penalty_fn,
)
from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    unreach_heading_fn,
    unreach_formation_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class FormationHeadingTaskState(EnvState):
    # Target heading, altitude and speed for the formation
    target_heading: ArrayLike
    target_altitude: ArrayLike
    target_vt: ArrayLike
    # Relative positions each agent should maintain in formation
    formation_positions: ArrayLike
    last_check_time: ArrayLike
    heading_turn_counts: ArrayLike
    
    @classmethod
    def create(cls, env_state: EnvState, formation_positions: Array, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            formation_positions=formation_positions,
            target_heading=extra_state[0],
            target_altitude=extra_state[1],
            target_vt=extra_state[2],
            last_check_time=env_state.time,
            heading_turn_counts=0,
        )


@struct.dataclass(frozen=True)
class FormationHeadingTaskParams(EnvParams):
    num_allies: int = 3  # Default to 3 agents in formation
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 1  # Discrete actions
    formation_type: int = 0  # 0: wedge, 1: line, 2: diamond
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    max_altitude: float = 9000.0
    min_altitude: float = 4200.0
    max_vt: float = 360.0
    min_vt: float = 120.0
    max_heading_increment: float = jnp.pi  # Maximum heading change (π≈180°)
    max_altitude_increment: float = 2100.0
    max_velocities_u_increment: float = 100.0
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    noise_scale: float = 0.0
    team_spacing: float = 15000  # Distance between agents in formation
    safe_distance: float = 3000  # Minimum safe distance between agents
    formation_reward_weight: float = 1.0  # Weight for formation keeping reward
    heading_reward_weight: float = 0.5  # Weight for heading following reward
    formation_position_error_scale: float = 50.0  # Scale for formation position error


class AeroPlanaxFormationHeadingEnv(AeroPlanaxEnv[FormationHeadingTaskState, FormationHeadingTaskParams]):
    def __init__(self, env_params: Optional[FormationHeadingTaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        # Combined reward functions for both formation keeping and heading following
        self.reward_functions = [
            functools.partial(formation_reward_fn, reward_scale=env_params.formation_reward_weight, 
                             position_error_scale=env_params.formation_position_error_scale),
            functools.partial(heading_reward_fn, reward_scale=env_params.heading_reward_weight),
            functools.partial(altitude_reward_fn, reward_scale=0.5, Kv=0.2),
            functools.partial(crash_penalty_fn, reward_scale=1.0, penalty_scale=-10000.0),
        ]

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            unreach_heading_fn,
            unreach_formation_fn,
        ]

        # Curriculum learning: gradually increase difficulty
        self.increment_size = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10)

    def _get_obs_size(self) -> int:
        # Include formation position error in observation
        return 19  

    @property
    def default_params(self) -> FormationHeadingTaskParams:
        return FormationHeadingTaskParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: FormationHeadingTaskParams,
    ) -> FormationHeadingTaskState:
        state = super()._init_state(key, params)
        # Initialize with zero formation positions, will be set in reset_task
        state = FormationHeadingTaskState.create(state, formation_positions=jnp.zeros((self.num_agents, 3)), 
                                              extra_state=jnp.zeros((3, self.num_agents)))
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: FormationHeadingTaskState,
        params: FormationHeadingTaskParams,
    ) -> FormationHeadingTaskState:
        """Task-specific reset."""
        # Generate the formation positions
        key, key_formation = jax.random.split(key)
        team_positions = self._generate_formation_positions(key_formation, params)
        
        # Generate random velocities
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        vel_x = vt
        
        # Set formation center at a random position
        team_center = jnp.zeros(3)
        key, key_altitude = jax.random.split(key)
        altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
        team_center = team_center.at[2].set(altitude)
        
        # Enforce safe distance between agents
        formation_positions = enforce_safe_distance(team_positions, team_center, params.safe_distance)
        
        # Update aircraft positions
        state = state.replace(
            plane_state=state.plane_state.replace(
                north=formation_positions[:, 0],
                east=formation_positions[:, 1],
                altitude=formation_positions[:, 2],
                vel_x=vel_x,
                vt=vt,
            ),
            formation_positions=formation_positions,
            target_heading=state.plane_state.yaw,  # Initial target heading = current heading
            target_altitude=state.plane_state.altitude,  # Initial target altitude = current altitude
            target_vt=vt,  # Initial target speed = random initial speed
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: FormationHeadingTaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: FormationHeadingTaskParams,
    ) -> Tuple[FormationHeadingTaskState, Dict[str, Any]]:
        """Task-specific step transition."""
        # Change task heading, altitude, and speed periodically
        key_heading, key_altitude_increment, key_vt_increment = jax.random.split(key, 3)
        delta = self.increment_size[state.heading_turn_counts]  # Progressive increment factor
        
        # Random heading change
        delta_heading = jax.random.uniform(key_heading, shape=(1,), 
                                          minval=-params.max_heading_increment, 
                                          maxval=params.max_heading_increment)
        # Altitude change
        delta_altitude = jax.random.uniform(key_altitude_increment, shape=(1,), 
                                           minval=-params.max_altitude_increment, 
                                           maxval=params.max_altitude_increment)
        # Speed change
        delta_vt = jax.random.uniform(key_vt_increment, shape=(1,), 
                                     minval=-params.max_velocities_u_increment, 
                                     maxval=params.max_velocities_u_increment)

        # Same target for all agents in formation
        target_altitude = jnp.ones((self.num_agents,)) * (state.target_altitude[0] + delta_altitude * delta)
        target_heading = jnp.ones((self.num_agents,)) * wrap_PI(state.target_heading[0] + delta_heading * delta)
        target_vt = jnp.ones((self.num_agents,)) * (state.target_vt[0] + delta_vt * delta)

        # Update formation targets when success criteria met
        new_state = state.replace(
            plane_state=state.plane_state.replace(
                status=jnp.where(state.plane_state.is_success, 0, state.plane_state.status)
            ),
            success=False,
            target_heading=target_heading,
            target_altitude=target_altitude,
            target_vt=target_vt,
            last_check_time=state.time,
            heading_turn_counts=(state.heading_turn_counts + 1),
        )
        
        # Transition to new task only if previous was successful
        state = jax.lax.cond(state.success, lambda: new_state, lambda: state)
        info["heading_turn_counts"] = state.heading_turn_counts
        return state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: FormationHeadingTaskState,
        params: FormationHeadingTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        Task-specific observation function for formation flight.
        
        observation (dim 19):
            0. ego_delta_north       (unit: km) - formation position error
            1. ego_delta_east        (unit: km) - formation position error
            2. ego_delta_altitude    (unit: km) - formation position error
            3. ego_delta_altitude    (unit: km) - from target altitude
            4. ego_delta_heading     (unit: rad) - from target heading
            5. ego_delta_vt          (unit: mh) - from target speed
            6. ego_altitude          (unit: 5km)
            7. ego_roll_sin
            8. ego_roll_cos
            9. ego_pitch_sin
            10. ego_pitch_cos
            11. ego_vt               (unit: mh)
            12. ego_alpha_sin
            13. ego_alpha_cos
            14. ego_beta_sin
            15. ego_beta_cos
            16. ego_P                (unit: rad/s)
            17. ego_Q                (unit: rad/s)
            18. ego_R                (unit: rad/s)
        """
        # Extract aircraft state variables
        north = state.plane_state.north
        east = state.plane_state.east
        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        vt = state.plane_state.vt
        alpha = state.plane_state.alpha
        beta = state.plane_state.beta
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        # Calculate normalized observations
        # Formation-specific observations (position errors)
        norm_delta_north = (north - state.formation_positions[:, 0]) / 1000
        norm_delta_east = (east - state.formation_positions[:, 1]) / 1000
        norm_delta_altitude_formation = (altitude - state.formation_positions[:, 2]) / 1000
        
        # Heading task observations
        norm_delta_altitude = (altitude - state.target_altitude) / 1000
        norm_delta_heading = wrap_PI(yaw - state.target_heading)
        norm_delta_vt = (vt - state.target_vt) / 340
        
        # Common aircraft state observations
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
        
        # Stack all observations
        obs = jnp.vstack((
            norm_delta_north, norm_delta_east, norm_delta_altitude_formation,
            norm_delta_altitude, norm_delta_heading, norm_delta_vt,
            norm_altitude, roll_sin, roll_cos, pitch_sin, pitch_cos, norm_vt,
            alpha_sin, alpha_cos, beta_sin, beta_cos,
            P, Q, R
        ))
        
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _generate_formation_positions(
        self,
        key: chex.PRNGKey,
        params: FormationHeadingTaskParams,
    ) -> jnp.ndarray:
        """Generate relative formation positions based on formation type"""
        # Create formation positions based on the specified formation type
        if self.formation_type == 0:
            team_positions = wedge_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 1:
            team_positions = line_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 2:
            team_positions = diamond_formation(self.num_allies, params.team_spacing)
        else:
            # Default to wedge if invalid formation type is provided
            team_positions = wedge_formation(self.num_allies, params.team_spacing)
            
        return team_positions 