'''
真正的Replay Buffer实现，用于存储和采样完整episodes
'''
import jax
import numpy as np
import jax.numpy as jnp
from collections import deque
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class ReplayBufferConfig:
    """Replay Buffer配置"""
    max_episodes: int = 100  # 最大存储的完整episode数量
    min_episodes_for_training: int = 10  # 开始训练所需的最小episode数量
    sample_batch_size: int = 32  # 每次采样的batch大小
    max_episode_length: int = 10000  # 单个episode的最大长度
    enable_prioritized_sampling: bool = True  # 是否启用优先级采样


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.episodes = deque(maxlen=config.max_episodes)
        self.episode_priorities = deque(maxlen=config.max_episodes)
        print(f"初始化Replay Buffer: 最大容量={config.max_episodes}个episodes")
        
    def add_episode(self, episode):
        """添加完整的episode"""
        self.episodes.append(episode)
        
        # 计算优先级（基于episode return）
        # 使用abs确保负回报的episode也有机会被采样
        priority = float(abs(episode.episode_return)) + 1e-6
        self.episode_priorities.append(priority)
        
        print(f"添加episode到buffer: return={float(episode.episode_return):.2f}, priority={priority:.2f}, buffer大小={len(self.episodes)}/{self.config.max_episodes}")
        
    def sample_batch(self, rng: jax.random.PRNGKey):
        """从buffer中采样batch"""
        if len(self.episodes) < self.config.min_episodes_for_training:
            print(f"Buffer中episodes数量({len(self.episodes)})不足最小训练要求({self.config.min_episodes_for_training})")
            return []
        
        batch_size = min(self.config.sample_batch_size, len(self.episodes))
            
        # 选择episode
        if self.config.enable_prioritized_sampling and len(self.episode_priorities) > 0:
            # 优先级采样
            priorities = np.array(list(self.episode_priorities))
            probs = priorities / priorities.sum()
            indices = jax.random.choice(
                rng, len(self.episodes), 
                shape=(batch_size,),
                p=jnp.array(probs), 
                replace=True
            )
            print(f"优先级采样: 选择了{batch_size}个episodes")
        else:
            # 均匀采样
            indices = jax.random.choice(
                rng, len(self.episodes),
                shape=(batch_size,),
                replace=True
            )
            print(f"均匀采样: 选择了{batch_size}个episodes")
        
        # 转换为Python索引并获取episodes
        sampled_episodes = []
        for idx in indices:
            idx_py = int(idx)
            sampled_episodes.append(self.episodes[idx_py])
            
        return sampled_episodes
    
    def __len__(self):
        return len(self.episodes)
    
    def is_ready_for_training(self):
        ready = len(self.episodes) >= self.config.min_episodes_for_training
        if ready:
            print(f"Replay Buffer已准备好训练: {len(self.episodes)}/{self.config.min_episodes_for_training}个episodes")
        return ready 