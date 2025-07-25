from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState
import jax.numpy as jnp
from ..utils.utils import wrap_PI


def unreach_heading_pitch_V_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    max_check_interval: int = 5,
    min_check_interval: int = 0.2
) -> Tuple[bool, bool]:
    """
    检查飞机是否在限定时间内达到目标航向角、俯仰角和速度
    """
    plane_state: FighterPlaneState = state.plane_state
    yaw = plane_state.yaw[agent_id]
    altitude = plane_state.altitude[agent_id]
    vt = plane_state.vt[agent_id]
    check_time = state.time - state.last_check_time
    # 判断时间
    max_check_interval = max_check_interval * params.sim_freq / params.agent_interaction_steps # 50*50/10=250
    # min_check_interval = min_check_interval * params.sim_freq / params.agent_interaction_steps # 0.2*50/10=1
    # mask1 = check_time >= max_check_interval
    mask1 = check_time >= max_check_interval
    

    # 检查是否达到目标航向角（误差在5度以内）
    delta_heading = wrap_PI(state.plane_state.yaw[agent_id] - state.target_heading[agent_id])
    mask_heading = jnp.abs(delta_heading) <= jnp.pi / 72  # 5度

    # 检查是否达到目标俯仰角（误差在5度以内）
    delta_pitch = wrap_PI(state.plane_state.pitch[agent_id] - state.target_pitch[agent_id])
    mask_pitch = jnp.abs(delta_pitch) <= jnp.pi / 72  # 5度

    # 检查是否达到目标速度（误差在10m/s以内）
    delta_velocity = jnp.abs(state.plane_state.vt[agent_id] - state.target_vt[agent_id])
    mask_velocity = delta_velocity <= 10.0  # 10m/s

    # 在满足时间间隔的基础上，只要完成任意一个目标就算成功
    # success = mask1 & (mask_heading | mask_pitch | mask_velocity)
    success = mask1 & mask_heading & mask_pitch & mask_velocity
    # 如果达到检查时间但未达到任何目标，则失败
    fail = mask1 & (~mask_heading & ~mask_pitch & ~mask_velocity)

    # 如果飞机已经死亡，则任务结束
    # done = ~state.plane_state.is_alive[agent_id]
    done = False

    return done, success 