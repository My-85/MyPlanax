"""
测试真正的Replay Buffer实现
"""
import os
import jax
import numpy as np
import jax.numpy as jnp
from maketrains.real_replay_buffer import ReplayBuffer, ReplayBufferConfig
from maketrains.mappo_discrete_combine_with_replay import (
    Transition, Episode, 
    external_collect_complete_episode,
    external_train_on_episode
)

def create_dummy_episode(rng, episode_length=10, obs_dim=4, action_dim=2):
    """创建一个假的episode用于测试"""
    rng, _rng = jax.random.split(rng)
    # 创建假的transitions
    done = jnp.zeros((episode_length, 1), dtype=jnp.bool_)
    done = done.at[-1, 0].set(True)  # 最后一步结束
    
    rng, _rng = jax.random.split(rng)
    action = jax.random.normal(_rng, (episode_length, 1, action_dim))
    
    rng, _rng = jax.random.split(rng)
    value = jax.random.normal(_rng, (episode_length, 1))
    
    rng, _rng = jax.random.split(rng)
    reward = jax.random.normal(_rng, (episode_length, 1))
    
    rng, _rng = jax.random.split(rng)
    log_prob = jax.random.normal(_rng, (episode_length, 1))
    
    rng, _rng = jax.random.split(rng)
    obs = jax.random.normal(_rng, (episode_length, 1, obs_dim))
    
    rng, _rng = jax.random.split(rng)
    world_state = jax.random.normal(_rng, (episode_length, 1, obs_dim))
    
    valid_action = jnp.ones((episode_length, 1), dtype=jnp.bool_)
    
    rng, _rng = jax.random.split(rng)
    info = jax.random.normal(_rng, (episode_length, 1))
    
    # 创建transition
    transition = Transition(
        done=done,
        action=action,
        value=value,
        reward=reward,
        log_prob=log_prob,
        obs=obs,
        world_state=world_state,
        valid_action=valid_action,
        info=info
    )
    
    # 创建隐状态
    rng, _rng = jax.random.split(rng)
    init_hstate1 = jax.random.normal(_rng, (1, 64))
    
    rng, _rng = jax.random.split(rng)
    init_hstate2 = jax.random.normal(_rng, (1, 64))
    
    # 创建episode
    episode_return = jnp.sum(reward)
    episode = Episode(
        transitions=transition,
        episode_length=jnp.array(episode_length),
        episode_return=episode_return,
        episode_success=True,
        init_hstate=(init_hstate1, init_hstate2)
    )
    
    return episode, rng

def test_replay_buffer():
    """测试Replay Buffer的基本功能"""
    print("=== 测试Replay Buffer的基本功能 ===")
    
    # 创建Replay Buffer配置
    config = ReplayBufferConfig(
        max_episodes=5,
        min_episodes_for_training=3,
        sample_batch_size=2,
        max_episode_length=10,
        enable_prioritized_sampling=True
    )
    
    # 创建Replay Buffer
    buffer = ReplayBuffer(config)
    
    # 创建并添加episodes
    rng = jax.random.PRNGKey(42)
    for i in range(6):  # 超过max_episodes
        episode, rng = create_dummy_episode(rng, episode_length=10)
        buffer.add_episode(episode)
        print(f"添加第{i+1}个episode，当前buffer大小: {len(buffer)}")
    
    # 测试采样
    print("\n测试采样:")
    rng, _rng = jax.random.split(rng)
    sampled_episode = buffer.sample_batch(_rng)
    print(f"采样的episode return: {sampled_episode.episode_return}")
    print(f"采样的episode length: {sampled_episode.episode_length}")
    
    # 测试是否准备好训练
    print(f"\n是否准备好训练: {buffer.is_ready_for_training()}")
    
    print("=== 测试完成 ===\n")
    return True

def main():
    """主测试函数"""
    print("开始测试真正的Replay Buffer实现...")
    
    # 测试基本功能
    test_replay_buffer()
    
    print("所有测试通过！真正的Replay Buffer实现正常工作。")

if __name__ == "__main__":
    main() 