""" Wrappers for use with jaxmarl baselines. """
import os
import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
import gymnax
# from gymnax.environments import environment, spaces
from gymnax.environments.spaces import Box as BoxGymnax, Discrete as DiscreteGymnax
from typing import Dict, Optional, List, Tuple, Union, Any
from .aeroplanax import AeroPlanaxEnv, EnvState


class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers."""

    def __init__(self, env: AeroPlanaxEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name) # 提供环境属性访问

    # def _batchify(self, x: dict):
    #     x = jnp.stack([x[a] for a in self._env.agents])
    #     return x.reshape((self._env.num_agents, -1))

    def _batchify_floats(self, x: dict): # 定义_batchify_floats将字典转为数组
        return jnp.stack([x[a] for a in self._env.agents])


@struct.dataclass
class LogEnvState: # 数据类
    env_state: EnvState
    episode_returns: float # 记录累积奖励
    episode_lengths: int # 回合长度
    returned_episode_returns: float # 返回累积奖励
    returned_episode_lengths: int # 返回回合长度


class LogWrapper(JaxMARLWrapper):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env: AeroPlanaxEnv, replace_info: bool = False, rng: chex.PRNGKey = None):
        super().__init__(env)
        self.replace_info = replace_info
        # @UNCHECKED
        # NOTE:据说global_obs cat一个高斯分布噪声有助于探索，暂且放在这里

        if hasattr(self._env,'noise_features') and self._env.noise_features > 0:
            self.noise_features = self._env.noise_features
            noise_amplifier = 10.0
            self.noise_vectors = jax.random.uniform(rng, shape=(self._env.num_agents, self.noise_features)) * noise_amplifier
        else:
            self.noise_features = 0

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]: # 初始化状态和统计数据
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.num_allies,)),
            0,
            jnp.zeros((self._env.num_allies,)),
            0,
        )
        return obs, state
    
    @property
    def global_obs_size(self) -> int:
        return self._env._get_global_obs_size()
    
    @property
    def ego_obs_size(self) -> int:
        return self._env._get_obs_size()
    
    def get_env_information_for_config(self):
        env_informations = {
            "EGO_OBS_DIM": self._env.own_features,
            "OTHER_OBS_DIM": self._env.unit_features,
            "OBS_DIM": self.ego_obs_size,
            "GLOBAL_OBS_DIM": self.global_obs_size,

            "NUM_ACTORS": self._env.num_agents,
            "NUM_VALID_AGENTS": self._env.num_allies,
        }
        return env_informations
    
    @partial(jax.jit, static_argnums=(0,))
    def get_global_obs(
        self,
        state: LogEnvState,
    ) -> Dict[str, chex.Array]:
        """获取全局观察，支持嵌套的包装器状态"""
        # 获取内部环境状态
        env_state = state.env_state
        
        # 处理内部可能是 RewardNormState 的情况
        if hasattr(env_state, 'env_state'):
            # 如果内部状态也有嵌套结构，进一步提取
            env_state = env_state.env_state
            
        # 获取原始全局观察
        global_obs = self._env.get_raw_global_obs(env_state)
        
        # 添加噪声（如果配置）
        if self.noise_features > 0:
            global_obs = jnp.concatenate([global_obs, self.noise_vectors], axis=-1)
            
        # 返回格式化的观察字典
        return {agent: global_obs[self._env.agent_ids[agent]] for agent in self._env.agents}
        
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, dict, chex.Array, dict]:
        # 执行环境步骤
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )

        # 检查回合是否结束
        ep_done = done["__all__"]

        # 更新累积奖励和回合长度
        new_episode_return = state.episode_returns + self._batchify_floats(reward).reshape(-1)[:self._env.num_allies]
        new_episode_length = state.episode_lengths + 1

        # 更新状态
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        # 更新info字典
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = ep_done
        info["success"] = info["success"]
        return obs, state, reward, done, info

@struct.dataclass
class RewardNormState:
    env_state: Any
    reward_mean: float = 0.0
    reward_var: float = 1.0
    reward_count: float = 0.0


class RewardNormWrapper(JaxMARLWrapper):
    """
    Normalizes rewards using running statistics.
    
    This wrapper maintains a running mean and standard deviation of rewards
    and uses them to normalize rewards to have mean 0 and standard deviation 1.
    
    JAX compatible version that stores statistics in a separate state field.
    """
    
    def __init__(self, env, gamma=0.99, epsilon=1e-8, clip_reward=10.0):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            gamma: Discount factor for the running mean and std calculation
            epsilon: Small constant to avoid division by zero
            clip_reward: Maximum absolute value for normalized rewards
        """
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_reward = clip_reward
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        """Reset the environment and return the initial state."""
        # Get initial state and observations from wrapped env
        obs, env_state = self._env.reset(key, params)
        
        # Create normalization state
        norm_state = RewardNormState(
            env_state=env_state,
            reward_mean=0.0,
            reward_var=1.0,
            reward_count=0.0
        )
        
        # Return observations in original format and wrapped state
        return obs, norm_state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions, params=None):
        """
        Take a step in the environment and normalize the rewards.
        
        Args:
            key: A random key
            state: The current state (RewardNormState)
            actions: Actions to take
            params: Optional parameters
        
        Returns:
            obs: The observation
            state: The new state
            rewards: Normalized rewards
            dones: Done flags
            infos: Additional information
        """
        # Extract the base environment state
        env_state = state.env_state
        
        # Call the underlying environment's step method with the base state
        obs, env_state, rewards, dones, infos = self._env.step(key, env_state, actions, params)
        
        # Extract current statistics
        reward_mean = state.reward_mean
        reward_var = state.reward_var
        reward_count = state.reward_count
        
        # Convert rewards dict to list for easier processing
        agents = list(self._env.agents)
        reward_values = jnp.array([rewards[agent] for agent in agents])
        
        # Update statistics using JAX-compatible operations
        # First, create masks for valid rewards
        valid_rewards = ~(jnp.isnan(reward_values) | jnp.isinf(reward_values))
        
        # Update count based on valid rewards
        new_count = reward_count + jnp.sum(valid_rewards)
        
        # Create safe divisor to avoid divide-by-zero
        safe_divisor = jnp.maximum(new_count, 1.0)
        
        # Calculate deltas for valid rewards (masked)
        deltas = jnp.where(valid_rewards, 
                           reward_values - reward_mean, 
                           0.0)
        
        # Update mean (sum deltas and divide by count)
        delta_sum = jnp.sum(deltas)
        new_mean = reward_mean + delta_sum / safe_divisor
        
        # Calculate second deltas
        deltas2 = jnp.where(valid_rewards, 
                            reward_values - new_mean, 
                            0.0)
        
        # Update variance using Welford's algorithm
        delta_prod_sum = jnp.sum(deltas * deltas2)
        factor = jnp.where(new_count > 0, 1.0 - 1.0 / safe_divisor, 0.0)
        new_var = factor * reward_var + delta_prod_sum / safe_divisor
        
        # Normalize rewards with JAX-compatible operations
        normalized_rewards = {}
        std = jnp.sqrt(new_var + self.epsilon)
        
        for i, agent in enumerate(agents):
            # Get original reward
            reward = rewards[agent]
            
            # Normalize reward (handle NaN/Inf with where)
            is_valid = ~(jnp.isnan(reward) | jnp.isinf(reward))
            norm_reward = jnp.where(is_valid, 
                                    (reward - new_mean) / std, 
                                    reward)
            
            # Clip if needed
            if self.clip_reward > 0:
                norm_reward = jnp.where(is_valid,
                                       jnp.clip(norm_reward, -self.clip_reward, self.clip_reward),
                                       norm_reward)
            
            normalized_rewards[agent] = norm_reward
        
        # Create new state with updated statistics
        new_state = RewardNormState(
            env_state=env_state,
            reward_mean=new_mean,
            reward_var=new_var,
            reward_count=new_count
        )
        
        # Return with expected order: obs, state, rewards, dones, infos
        return obs, new_state, normalized_rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def get_global_obs(self, state):
        """
        Pass the internal environment state to the wrapped environment's get_raw_global_obs method.
        
        Args:
            state: RewardNormState containing the environment state
            
        Returns:
            Global observations dictionary
        """
        # Extract the base environment state from our wrapper state
        env_state = state.env_state
        
        # Get global observations from the base environment
        global_obs = self._env.get_raw_global_obs(env_state)
        
        # Return formatted as a dictionary for each agent
        return {agent: global_obs[self._env.agent_ids[agent]] for agent in self._env.agents}