from ..aeroplanax import TEnvState, TEnvParams, AgentID


def heading_pitch_v_event_driven_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        fail_reward: float = -10,
        success_reward: float = 10
    ) -> float:
    reward = state.success * success_reward + state.done * fail_reward
    return reward
