import functools
import jax
import jax.numpy as jnp
from typing import Dict, Optional, Tuple, Any
from flax import struct

from envs.aeroplanax_combat_hierarchy_FSM import (
    AeroPlanaxHierarchicalCombatEnv, 
    HierarchicalCombatTaskState, 
    HierarchicalCombatTaskParams
)
from envs.aeroplanax import AgentName
from envs.core.simulators import fighterplane


class AeroPlanaxContinuousHierarchicalCombatEnv(AeroPlanaxHierarchicalCombatEnv):
    """
    Extension of the hierarchical combat environment to support continuous actions.
    This environment modifies the _decode_actions method to handle continuous actions
    from the policy, while still maintaining compatibility with the underlying
    discrete controller and FSM logic for enemy agents.
    """
    
    def __init__(self, env_params: Optional[HierarchicalCombatTaskParams] = None):
        super().__init__(env_params)
        self.action_ranges = {
            'pitch': {'min': -jnp.pi/6, 'max': jnp.pi/6},    # -30° to +30° change in pitch
            'heading': {'min': -jnp.pi/2, 'max': jnp.pi/2},  # -90° to +90° change in heading
            'velocity': {'min': -100, 'max': 100},           # -100 to +100 m/s change in velocity
        }
        
        # Update action spaces to be continuous
        # This step might require changes to gymnax.environments.spaces
        # For now, we'll just use the existing discrete spaces as a placeholder
        
    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_actions(
        self,
        key: jax.Array,
        init_state: HierarchicalCombatTaskState,
        state: HierarchicalCombatTaskState,
        actions: Dict[AgentName, jax.Array]
    ):
        """
        Decode continuous actions from the policy to control the aircraft.
        
        Args:
            key: JAX PRNGKey
            init_state: Initial state
            state: Current state
            actions: Dictionary of actions for each agent
            
        Returns:
            Updated state and control state
        """
        # Convert actions dictionary to array
        actions_array = jnp.array([actions[i] for i in self.agents])
        
        # Clip actions to ensure they're in [-1, 1] range
        continuous_actions = jnp.clip(actions_array, -1.0, 1.0)
        
        # Map from [-1, 1] range to actual action ranges
        def map_to_range(x, range_min, range_max):
            return range_min + (x + 1.0) * 0.5 * (range_max - range_min)
        
        # Extract action ranges
        pitch_range = self.action_ranges['pitch']
        heading_range = self.action_ranges['heading']
        velocity_range = self.action_ranges['velocity']
        
        # Apply mapping to each action dimension
        delta_pitch = map_to_range(continuous_actions[:, 0], pitch_range['min'], pitch_range['max'])
        delta_heading = map_to_range(continuous_actions[:, 1], heading_range['min'], heading_range['max'])
        delta_vt = map_to_range(continuous_actions[:, 2], velocity_range['min'], velocity_range['max'])
        
        # For non-ally agents, apply FSM logic if using baseline
        if self.use_baseline:
            ego_delta_pitch_cmd = delta_pitch[:self.num_allies]
            ego_delta_heading_cmd = delta_heading[:self.num_allies]
            ego_delta_vt_cmd = delta_vt[:self.num_allies]
            
            # Handle enemy FSM logic (from the original implementation)
            fsm_key, controller_key = jax.random.split(key)
            enemy_keys = jax.random.split(fsm_key, self.num_enemies)
            
            # Process each enemy agent with the existing FSM logic
            def single_enemy_fsm_logic(enm_local_idx, current_fsm_state, enm_key):
                enm_global_idx = self.num_allies + enm_local_idx
                enm_ps = init_state.plane_state
                
                # Extract enemy agent state (same as in original implementation)
                enm_north = enm_ps.north[enm_global_idx]
                enm_east = enm_ps.east[enm_global_idx]
                enm_alt = enm_ps.altitude[enm_global_idx]
                enm_vx = enm_ps.vel_x[enm_global_idx]
                enm_vy = enm_ps.vel_y[enm_global_idx]
                enm_vz = enm_ps.vel_z[enm_global_idx]
                enm_pitch = enm_ps.pitch[enm_global_idx]
                enm_yaw = enm_ps.yaw[enm_global_idx]
                enm_vt = enm_ps.vt[enm_global_idx]
                enm_is_alive = enm_ps.is_alive[enm_global_idx]
                enm_blood = enm_ps.blood[enm_global_idx]
                enm_max_blood = 100.0
                
                enm_feature_6d = jnp.array([enm_north, enm_east, enm_alt, enm_vx, enm_vy, enm_vz])
                
                # Process ally agent states (same as original)
                ally_indices = jnp.arange(self.num_allies)
                ally_norths = enm_ps.north[ally_indices]
                ally_easts = enm_ps.east[ally_indices]
                ally_alts = enm_ps.altitude[ally_indices]
                ally_vxs = enm_ps.vel_x[ally_indices]
                ally_vys = enm_ps.vel_y[ally_indices]
                ally_vzs = enm_ps.vel_z[ally_indices]
                ally_is_alives = enm_ps.is_alive[ally_indices]
                ally_bloods = enm_ps.blood[ally_indices]
                
                # Calculate metrics to allies (same as original)
                def calculate_metrics_to_ally(ally_idx):
                    from envs.utils.utils import get_AO_TA_R
                    ally_feature_6d = jnp.array([
                        ally_norths[ally_idx], ally_easts[ally_idx], ally_alts[ally_idx],
                        ally_vxs[ally_idx], ally_vys[ally_idx], ally_vzs[ally_idx]
                    ])
                    ao, ta, r, side_flag = get_AO_TA_R(enm_feature_6d, ally_feature_6d)
                    return r, ao, ta, side_flag, ally_is_alives[ally_idx], ally_bloods[ally_idx]
                
                # Apply metrics calculation to all allies (same as original)
                ally_metrics = jax.vmap(calculate_metrics_to_ally)(jnp.arange(self.num_allies))
                distances, aos, tas, side_flags, is_alives, bloods = ally_metrics
                
                # Filter for alive allies only (same as original)
                masked_distances = jnp.where(is_alives, distances, jnp.inf)
                
                # Find closest ally (same as original)
                any_ally_alive = jnp.any(is_alives)
                closest_ally_idx = jnp.where(any_ally_alive, jnp.argmin(masked_distances), 0)
                closest_ally_distance = masked_distances[closest_ally_idx]
                closest_ally_ao = aos[closest_ally_idx]
                closest_ally_ta = tas[closest_ally_idx]
                closest_ally_is_alive = is_alives[closest_ally_idx]
                
                # Default to invalid values if no ally is alive (same as original)
                closest_ally_distance = jnp.where(closest_ally_is_alive, closest_ally_distance, jnp.inf)
                closest_ally_ao = jnp.where(closest_ally_is_alive, closest_ally_ao, jnp.pi)
                
                # FSM state transition logic (same as original)
                # Import constants
                PATROL_STATE = 0
                ENGAGE_STATE = 1
                ATTACK_STATE = 2
                EVADE_STATE = 3
                ENGAGE_RANGE = 10000.0
                ATTACK_RANGE = 4000.0
                ATTACK_AO_THRESHOLD = jnp.pi / 4
                EVADE_HEALTH_THRESHOLD_RATIO = 0.4
                MAX_TURN_RATE_PER_STEP = jnp.pi / 36
                MAX_PITCH_RATE_PER_STEP = jnp.pi / 36
                MAX_VT_DELTA_PER_STEP = 50.0
                MAX_SPEED = 300.0
                PATROL_SPEED_FACTOR = 0.7
                ATTACK_SPEED_FACTOR = 1.0
                EVADE_SPEED_FACTOR = 1.0
                PATROL_TURN_RANDOMNESS = 0.3
                PATROL_ALTITUDE_RANGE = [5000.0, 10000.0]
                
                # State transition logic (same as original)
                next_fsm_state = current_fsm_state
                is_low_health = enm_is_alive & (enm_blood < (enm_max_blood * EVADE_HEALTH_THRESHOLD_RATIO))
                next_fsm_state = jnp.where(is_low_health, EVADE_STATE, next_fsm_state)
                no_targets_alive = ~jnp.any(is_alives)
                next_fsm_state = jnp.where(no_targets_alive, PATROL_STATE, next_fsm_state)
                can_engage = enm_is_alive & (closest_ally_distance < ENGAGE_RANGE) & closest_ally_is_alive
                is_patrol = current_fsm_state == PATROL_STATE
                patrol_to_engage = is_patrol & can_engage & ~is_low_health
                next_fsm_state = jnp.where(patrol_to_engage, ENGAGE_STATE, next_fsm_state)
                can_attack = enm_is_alive & (closest_ally_distance < ATTACK_RANGE) & (closest_ally_ao < ATTACK_AO_THRESHOLD) & closest_ally_is_alive
                is_engage = current_fsm_state == ENGAGE_STATE
                engage_to_attack = is_engage & can_attack & ~is_low_health
                engage_to_patrol = is_engage & ~can_engage & ~is_low_health
                next_fsm_state = jnp.where(engage_to_attack, ATTACK_STATE, next_fsm_state)
                next_fsm_state = jnp.where(engage_to_patrol, PATROL_STATE, next_fsm_state)
                is_attack = current_fsm_state == ATTACK_STATE
                attack_to_engage = is_attack & ~can_attack & can_engage & ~is_low_health
                attack_to_patrol = is_attack & ~can_attack & ~can_engage & ~is_low_health
                next_fsm_state = jnp.where(attack_to_engage, ENGAGE_STATE, next_fsm_state)
                next_fsm_state = jnp.where(attack_to_patrol, PATROL_STATE, next_fsm_state)
                is_evade = current_fsm_state == EVADE_STATE
                evade_to_patrol = is_evade & ~is_low_health
                next_fsm_state = jnp.where(evade_to_patrol, PATROL_STATE, next_fsm_state)
                next_fsm_state = jnp.where(~enm_is_alive, PATROL_STATE, next_fsm_state)
                
                # Define behavior functions (same as original)
                def patrol_action():
                    key_turn, key_alt = jax.random.split(enm_key)
                    random_turn = jax.random.uniform(key_turn, minval=-MAX_TURN_RATE_PER_STEP*PATROL_TURN_RANDOMNESS, 
                                                    maxval=MAX_TURN_RATE_PER_STEP*PATROL_TURN_RANDOMNESS)
                    target_alt = jnp.mean(jnp.array(PATROL_ALTITUDE_RANGE))
                    alt_diff = target_alt - enm_alt
                    d_pitch = jnp.clip(alt_diff / 1000.0, -MAX_PITCH_RATE_PER_STEP/3, MAX_PITCH_RATE_PER_STEP/3)
                    target_vt = MAX_SPEED * PATROL_SPEED_FACTOR
                    d_vt = jnp.clip(target_vt - enm_vt, -MAX_VT_DELTA_PER_STEP/2, MAX_VT_DELTA_PER_STEP/2)
                    return d_pitch, random_turn, d_vt
                
                def engage_action():
                    target_north = ally_norths[closest_ally_idx]
                    target_east = ally_easts[closest_ally_idx]
                    target_alt = ally_alts[closest_ally_idx]
                    delta_n = target_north - enm_north
                    delta_e = target_east - enm_east
                    desired_heading = jnp.arctan2(delta_e, delta_n)
                    from envs.utils.utils import wrap_PI
                    d_heading = wrap_PI(desired_heading - enm_yaw)
                    height_advantage = 500.0
                    target_alt_with_advantage = target_alt + height_advantage
                    alt_diff = target_alt_with_advantage - enm_alt
                    d_pitch = jnp.clip(alt_diff / 1000.0, -MAX_PITCH_RATE_PER_STEP/2, MAX_PITCH_RATE_PER_STEP/2)
                    target_vt = MAX_SPEED * 0.85
                    d_vt = jnp.clip(target_vt - enm_vt, -MAX_VT_DELTA_PER_STEP/2, MAX_VT_DELTA_PER_STEP/2)
                    return d_pitch, d_heading, d_vt
                
                def attack_action():
                    target_north = ally_norths[closest_ally_idx]
                    target_east = ally_easts[closest_ally_idx]
                    target_alt = ally_alts[closest_ally_idx]
                    target_vx = ally_vxs[closest_ally_idx]
                    target_vy = ally_vys[closest_ally_idx]
                    prediction_time = 1.0
                    pred_north = target_north + target_vx * prediction_time
                    pred_east = target_east + target_vy * prediction_time
                    delta_n = pred_north - enm_north
                    delta_e = pred_east - enm_east
                    desired_heading = jnp.arctan2(delta_e, delta_n)
                    from envs.utils.utils import wrap_PI
                    d_heading = wrap_PI(desired_heading - enm_yaw)
                    alt_diff = target_alt - enm_alt
                    d_pitch = jnp.clip(alt_diff / 800.0, -MAX_PITCH_RATE_PER_STEP, MAX_PITCH_RATE_PER_STEP)
                    target_vt = MAX_SPEED * ATTACK_SPEED_FACTOR
                    d_vt = jnp.clip(target_vt - enm_vt, -MAX_VT_DELTA_PER_STEP, MAX_VT_DELTA_PER_STEP)
                    return d_pitch, d_heading, d_vt
                
                def evade_action():
                    threat_north = ally_norths[closest_ally_idx]
                    threat_east = ally_easts[closest_ally_idx]
                    delta_n = threat_north - enm_north
                    delta_e = threat_east - enm_east
                    threat_heading = jnp.arctan2(delta_e, delta_n)
                    from envs.utils.utils import wrap_PI
                    evade_heading = wrap_PI(threat_heading + jnp.pi)
                    d_heading = wrap_PI(evade_heading - enm_yaw)
                    key_pitch, key_speed = jax.random.split(enm_key)
                    random_pitch_factor = jax.random.uniform(key_pitch, minval=-1.0, maxval=1.0)
                    d_pitch = random_pitch_factor * MAX_PITCH_RATE_PER_STEP
                    target_vt = MAX_SPEED * EVADE_SPEED_FACTOR
                    d_vt = jnp.clip(target_vt - enm_vt, 0, MAX_VT_DELTA_PER_STEP)
                    return d_pitch, d_heading, d_vt
                
                # Select appropriate behavior based on FSM state (same as original)
                p_pitch, p_heading, p_vt = patrol_action()
                e_pitch, e_heading, e_vt = engage_action()
                a_pitch, a_heading, a_vt = attack_action()
                ev_pitch, ev_heading, ev_vt = evade_action()
                
                d_pitch = jnp.select([next_fsm_state == PATROL_STATE,
                                      next_fsm_state == ENGAGE_STATE,
                                      next_fsm_state == ATTACK_STATE,
                                      next_fsm_state == EVADE_STATE],
                                     [p_pitch, e_pitch, a_pitch, ev_pitch],
                                     default=0.0)
                
                d_heading = jnp.select([next_fsm_state == PATROL_STATE,
                                       next_fsm_state == ENGAGE_STATE,
                                       next_fsm_state == ATTACK_STATE,
                                       next_fsm_state == EVADE_STATE],
                                      [p_heading, e_heading, a_heading, ev_heading],
                                      default=0.0)
                
                d_vt = jnp.select([next_fsm_state == PATROL_STATE,
                                   next_fsm_state == ENGAGE_STATE,
                                   next_fsm_state == ATTACK_STATE,
                                   next_fsm_state == EVADE_STATE],
                                  [p_vt, e_vt, a_vt, ev_vt],
                                  default=0.0)
                
                # Apply limits and handle agent death (same as original)
                d_pitch = jnp.clip(d_pitch, -MAX_PITCH_RATE_PER_STEP, MAX_PITCH_RATE_PER_STEP)
                d_heading = jnp.clip(d_heading, -MAX_TURN_RATE_PER_STEP, MAX_TURN_RATE_PER_STEP)
                d_vt = jnp.clip(d_vt, -MAX_VT_DELTA_PER_STEP, MAX_VT_DELTA_PER_STEP)
                
                d_pitch = jnp.where(enm_is_alive, d_pitch, 0.0)
                d_heading = jnp.where(enm_is_alive, d_heading, 0.0)
                d_vt = jnp.where(enm_is_alive, d_vt, 0.0)
                
                return next_fsm_state, d_pitch, d_heading, d_vt
            
            # Apply FSM logic to all enemy agents (same as original)
            vmapped_fsm_results = jax.vmap(
                single_enemy_fsm_logic, in_axes=(0, 0, 0)
            )(jnp.arange(self.num_enemies), state.enemy_fsm_state, enemy_keys)
            
            # Unpack FSM results (same as original)
            new_enemy_fsm_states = vmapped_fsm_results[0]
            enm_delta_pitch = vmapped_fsm_results[1]
            enm_delta_heading = vmapped_fsm_results[2]
            enm_delta_vt = vmapped_fsm_results[3]
            
            # Update FSM states (same as original)
            state = state.replace(enemy_fsm_state=new_enemy_fsm_states)
            
            # Combine ally and enemy actions (same as original)
            delta_pitch = jnp.hstack((ego_delta_pitch_cmd, enm_delta_pitch))
            delta_heading = jnp.hstack((ego_delta_heading_cmd, enm_delta_heading))
            delta_vt = jnp.hstack((ego_delta_vt_cmd, enm_delta_vt))
        
        # Calculate target states for control (same as original)
        target_pitch = init_state.plane_state.pitch + delta_pitch
        from envs.utils.utils import wrap_PI
        target_heading = wrap_PI(init_state.plane_state.yaw + delta_heading)
        target_vt = init_state.plane_state.vt + delta_vt
        
        # Get controller observations (same as original)
        last_obs = self._get_controller_obs(state.plane_state, target_pitch, target_heading, target_vt)
        last_obs = jnp.transpose(last_obs)
        last_done = jnp.zeros((self.num_agents), dtype=bool)
        
        # Apply controller (same as original)
        ac_in = (
            last_obs[jnp.newaxis, :],
            last_done[jnp.newaxis, :],
        )
        
        # Use the controller to get low-level actions (same as original)
        hstate, pi, _ = self.controller.apply(self.controller_params, state.hstate, ac_in)
        pi_throttle, pi_elevator, pi_aileron, pi_rudder = pi
        
        # Sample actions from distributions (same as original)
        controller_key, key_throttle = jax.random.split(key)
        action_throttle = pi_throttle.sample(seed=key_throttle)
        controller_key, key_elevator = jax.random.split(controller_key)
        action_elevator = pi_elevator.sample(seed=key_elevator)
        controller_key, key_aileron = jax.random.split(controller_key)
        action_aileron = pi_aileron.sample(seed=key_aileron)
        controller_key, key_rudder = jax.random.split(controller_key)
        action_rudder = pi_rudder.sample(seed=key_rudder)
        
        # Combine actions (same as original)
        action = jnp.concatenate([action_throttle[:, :, jnp.newaxis], 
                                 action_elevator[:, :, jnp.newaxis], 
                                 action_aileron[:, :, jnp.newaxis], 
                                 action_rudder[:, :, jnp.newaxis]], axis=-1)
        
        # Update state and process actions (same as original)
        state = state.replace(hstate=hstate)
        action = action.squeeze(0)
        action = jax.vmap(self._decode_discrete_actions)(action)
        
        # Return updated state and control state (same as original)
        return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(action) 