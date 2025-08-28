import jax
import chex
import distrax
import numpy as np
import jax.numpy as jnp
from typing import Sequence, Tuple

def unzip_discrete_action(rng: chex.PRNGKey, pi: Sequence[distrax.Categorical]) -> Tuple[chex.PRNGKey, jax.Array, jax.Array]:
    rngs = jax.random.split(rng, len(pi) + 1)
    rng, subkeys = rngs[0], rngs[1:]
    
    # 使用温度参数来软化过于确定性的分布
    temperature = 1.0  # 1.0为不变，大于1增加随机性
    
    # 为每个分布添加最小概率保护
    def sample_with_min_prob(dist, key, min_prob=1e-6):
        # 获取原始logits
        logits = dist.logits
        
        # 应用温度缩放
        scaled_logits = logits / temperature
        
        # 将logits转换为概率
        probs = jax.nn.softmax(scaled_logits)
        
        # 确保所有概率至少为min_prob
        safe_probs = jnp.maximum(probs, min_prob)
        
        # 重新归一化
        safe_probs = safe_probs / jnp.sum(safe_probs, axis=-1, keepdims=True)
        
        # 创建新的分布并采样
        safe_dist = distrax.Categorical(probs=safe_probs)
        action = safe_dist.sample(seed=key)
        
        # 计算对数概率，但用原始分布评估（保持PPO算法的一致性）
        # 添加裁剪以避免对数概率太小
        log_prob = jnp.maximum(dist.log_prob(action), jnp.log(min_prob))
        
        return action, log_prob
    
    # 对每个分布执行安全采样
    actions_and_log_probs = [sample_with_min_prob(dist, k) for dist, k in zip(pi, subkeys)]
    actions = [a_lp[0] for a_lp in actions_and_log_probs]
    log_probs = [a_lp[1] for a_lp in actions_and_log_probs]

    action = jnp.stack(actions, axis=-1)  # shape: [B, T, num_action_dim]
    log_prob = jnp.sum(jnp.stack(log_probs, axis=-1), axis=-1)  # shape: [B, T]
    
    return rng, action, log_prob
