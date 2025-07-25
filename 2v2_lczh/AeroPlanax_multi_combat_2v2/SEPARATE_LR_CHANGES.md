# 分别设置Actor和Critic学习率的修改说明

## 修改概述

本次修改允许为Actor和Critic网络分别设置不同的学习率，同时保持向后兼容性。

## 主要修改

### 1. 主训练文件 (main_train_hierarchy_combat_new.py)

**修改前:**
```python
config = {
    "LR": 3e-4,  # Slightly lower learning rate due to larger action space
    # ... other config
}
```

**修改后:**
```python
config = {
    "ACTOR_LR": 3e-4,  # Actor learning rate
    "CRITIC_LR": 1e-3,  # Critic learning rate (higher than actor)
    # ... other config
}
```

### 2. MAPPO RNN网络初始化 (networks/mappoRNN_discrete.py)

**主要修改:**
- 添加了对`ACTOR_LR`和`CRITIC_LR`的支持
- 创建分别的优化器用于actor和critic网络
- 保持向后兼容性，仍支持旧的`LR`参数
- 为学习率退火创建了分别的调度函数

**关键代码:**
```python
# Support both new separate learning rates and backward compatibility
if "ACTOR_LR" in config and "CRITIC_LR" in config:
    actor_lr = config["ACTOR_LR"]
    critic_lr = config["CRITIC_LR"]
    print(f"Using separate learning rates - Actor: {actor_lr}, Critic: {critic_lr}")
elif "LR" in config:
    actor_lr = critic_lr = config["LR"]
    print(f"Using shared learning rate: {actor_lr}")
else:
    raise ValueError("Either specify LR for shared learning rate or ACTOR_LR and CRITIC_LR for separate learning rates")

# Create separate optimizers for actor and critic
ac_tx = optax.chain(
    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
    optax.adam(actor_lr, eps=1e-5),
)
cr_tx = optax.chain(
    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
    optax.adam(critic_lr, eps=1e-5),
)
```

### 3. Pooling Encoder网络 (networks/pooling_encoder.py)

- 与MAPPO RNN网络相同的修改
- 支持分别的actor和critic学习率
- 保持向后兼容性

### 4. PPO RNN网络 (networks/ppoRNN_discrete.py)

**特殊处理:**
- 由于这是一个合并的ActorCritic网络，只有一个优化器
- 优先使用`ACTOR_LR`，如果不存在则使用`LR`
- 输出相应的日志信息

```python
# For combined ActorCritic network, use actor learning rate if available
if "ACTOR_LR" in config:
    network_lr = config["ACTOR_LR"]
    print(f"Using actor learning rate for combined ActorCritic network: {network_lr}")
elif "LR" in config:
    network_lr = config["LR"]
    print(f"Using shared learning rate for combined ActorCritic network: {network_lr}")
```

## 配置参数说明

### 新参数
- `ACTOR_LR`: Actor网络的学习率 (例如: 3e-4)
- `CRITIC_LR`: Critic网络的学习率 (例如: 1e-3)

### 向后兼容
- `LR`: 如果没有指定`ACTOR_LR`和`CRITIC_LR`，则两个网络共享此学习率

## 使用方法

### 方法1: 分别设置学习率 (推荐)
```python
config = {
    "ACTOR_LR": 3e-4,    # Actor学习率
    "CRITIC_LR": 1e-3,   # Critic学习率
    # ... 其他配置
}
```

### 方法2: 共享学习率 (向后兼容)
```python
config = {
    "LR": 3e-4,          # 共享学习率
    # ... 其他配置
}
```

## 优点

1. **灵活性**: 可以根据需要为actor和critic设置不同的学习率
2. **性能优化**: Critic网络通常可以使用更高的学习率来更快地学习价值函数
3. **向后兼容**: 现有代码无需修改即可继续工作
4. **清晰性**: 日志输出明确显示使用的学习率设置

## 典型的学习率设置建议

- **Actor LR**: 通常较低 (如 3e-4)，因为策略更新需要更加谨慎
- **Critic LR**: 可以较高 (如 1e-3)，因为价值函数的学习通常更加稳定

## 测试

运行 `python test_separate_lr.py` 来验证修改是否正确工作。测试包括：
- 分别学习率的解析
- 向后兼容性测试
- 错误配置的处理
- 主训练文件配置的验证

## 注意事项

1. 如果同时提供了`ACTOR_LR`/`CRITIC_LR`和`LR`，系统将优先使用`ACTOR_LR`/`CRITIC_LR`
2. 对于合并的ActorCritic网络，系统会优先使用`ACTOR_LR`
3. 所有修改都支持学习率退火(`ANNEAL_LR`配置)
4. 梯度裁剪设置对两个网络都生效 