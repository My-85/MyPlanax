import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import get_AO_TA_R
import jax


def posture_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0,
        num_allies: int = 1,
        num_enemies: int = 1,
    ) -> float:
    """
    Reward is a complex function of AO, TA and R in the last timestep.
    """
    new_reward = 0.0
    # feature: (north, east, down, vn, ve, vd)
    ego_feature = jnp.hstack((state.plane_state.north[agent_id],
                              state.plane_state.east[agent_id],
                              state.plane_state.altitude[agent_id],
                              state.plane_state.vel_x[agent_id],
                              state.plane_state.vel_y[agent_id],
                              state.plane_state.vel_z[agent_id]))
    enm_list = jax.lax.select(agent_id < num_allies, 
                              jnp.arange(num_allies, num_allies + num_enemies),
                              jnp.arange(num_allies))
    for enm in enm_list:
        enm_feature = jnp.hstack((state.plane_state.north[enm],
                                  state.plane_state.east[enm],
                                  state.plane_state.altitude[enm],
                                  state.plane_state.vel_x[enm],
                                  state.plane_state.vel_y[enm],
                                  state.plane_state.vel_z[enm]))
        AO, TA, R, _ = get_AO_TA_R(ego_feature, enm_feature)
        orientation_reward = orientation_reward_fn(AO, TA)
        range_reward = range_reward_fn(R / 1000.0)
        mask = state.plane_state.is_alive[enm] | state.plane_state.is_locked[enm]
        # mask = state.plane_state.is_alive[enm] & state.plane_state.is_locked[enm]
        new_reward += orientation_reward * range_reward * mask
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return new_reward * reward_scale * mask

def safe_arctanh(x, eps=1e-4):
    x = jnp.clip(x, -1 + eps, 1 - eps)
    return jnp.arctanh(x)

# def orientation_reward_fn(AO, TA):
#     AO = jnp.clip(AO, -jnp.pi, jnp.pi)
#     denominator = 50 * AO / jnp.pi + 2
#     safe_denominator = jnp.maximum(jnp.abs(denominator), 1e-2) * jnp.sign(denominator)

#     ta_ratio = jnp.clip(2 * TA / jnp.pi, 1e-4, 0.999)
#     arctanh_term = safe_arctanh(1. - ta_ratio) / (2 * jnp.pi)
    
#     reward = 1 / safe_denominator + 0.5 + jnp.minimum(arctanh_term, 1e-4) + 0.5
#     return reward
def orientation_reward_fn(AO, TA):
    """
    cos(TA) ↑ → 你正瞄准敌人  
    cos(AO) ↓ → 敌人越偏离你越好  
    """
    score = jnp.cos(TA) - 0.3 * jnp.cos(AO)     # 权重 0.3 可调
    # 归一化到 [-1, 1]
    return jnp.clip(score, -1.0, 1.0)


def range_reward_fn(R):
    reward = 1 * (R < 5) + (R >= 5) * jnp.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0.0, 1.0) \
        + jnp.clip(jnp.exp(-0.16 * R), 0, 0.2)
    return reward
# def range_reward_fn(R):
#     """R 单位 km，输出 [-0.2, 1.0] 区间"""
#     near = (R < 5) * 1.0                       # 最佳区
#     mid  = (R >= 5) * (-0.032*R**2 + 0.284*R + 0.18)  # 下移 0.2
#     far  = jnp.exp(-0.16 * R) - 0.2
#     return jnp.clip(near + mid + far, -0.2, 1.0)
