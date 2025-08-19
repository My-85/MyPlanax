# ... 新建文件 ...
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI


def heading_alignment_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0,
    ) -> float:
    """
    保持目标航向的密集奖励  
        r = exp(-(Δheading / σ)^2)，σ = 15°。
    """
    delta_heading = wrap_PI(state.plane_state.yaw[agent_id] - state.target_heading)
    heading_error_scale = jnp.pi / 12          # 15°
    reward = jnp.exp(- (delta_heading / heading_error_scale) ** 2)

    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * reward_scale * mask