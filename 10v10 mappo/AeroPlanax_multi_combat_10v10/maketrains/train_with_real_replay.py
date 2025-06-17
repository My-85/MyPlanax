'''
使用真正Replay Buffer的训练函数
'''
import jax
import numpy as np
import jax.numpy as jnp
import functools
import tensorboardX
from typing import Dict, Tuple

from envs.wrappers_mul import LogWrapper
from flax.training.train_state import TrainState
from networks import ScannedRNN, unzip_discrete_action

from maketrains.utils import batchify, unbatchify
from maketrains.real_replay_buffer import ReplayBuffer, ReplayBufferConfig
from maketrains.mappo_discrete_combine_with_replay import (
    Transition, Episode, 
    save_model as save_train,
    external_collect_complete_episode,
    external_train_on_episode
)

# 使用外部函数别名
collect_complete_episode = external_collect_complete_episode
train_on_episode = external_train_on_episode

def make_train_with_real_replay(
    config: Dict,
    env: LogWrapper,
    networks: Tuple,
    replay_config: ReplayBufferConfig = None,
    train_mode: bool = True,
    save_epochs: int = -1,
):
    """使用真正Replay Buffer的训练函数
    
    Args:
        config: 训练配置
        env: 环境
        networks: 网络模型，(actor, critic)
        replay_config: Replay Buffer配置
        train_mode: 是否为训练模式
        save_epochs: 每隔多少个epoch保存一次模型，-1表示不保存
        
    Returns:
        train: 训练函数
    """
    # 解析网络和配置
    (main_network, critic_network) = networks
    
    # 处理agent数量的配置，兼容不同的配置格式
    if "NUM_VALID_AGENTS" not in config:
        config["NUM_VALID_AGENTS"] = config["NUM_ACTORS"]
    
    # 计算有效和无效的agent数量
    valid_agent_num = config["NUM_ENVS"] * config["NUM_VALID_AGENTS"]
    invalid_agent_num = config["NUM_ENVS"] * (config["NUM_ACTORS"] - config["NUM_VALID_AGENTS"])
    
    # 创建Replay Buffer
    if replay_config is None:
        replay_config = ReplayBufferConfig()
    replay_buffer = ReplayBuffer(replay_config)
    print(f"初始化Replay Buffer: 最大容量={replay_config.max_episodes}个episodes")
    
    # 配置TensorBoard
    if train_mode:
        summary_writer = tensorboardX.SummaryWriter(config["SAVE_PATH"])
    else:
        summary_writer = None
    
    # 保存配置到TensorBoard
    if train_mode:
        for k, v in config.items():
            if isinstance(v, (int, float)):
                summary_writer.add_scalar(f"config/{k}", v, 0)
            else:
                summary_writer.add_text(f"config/{k}", str(v), 0)
    
    # 主训练函数
    def train(rng, train_states: Tuple[TrainState, TrainState], start_epoch: int = 0):
        """训练函数
        
        Args:
            rng: JAX随机数种子
            train_states: 训练状态，(actor_state, critic_state)
            start_epoch: 起始epoch
            
        Returns:
            train_states: 更新后的训练状态
            metrics: 训练指标
        """
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        rng_reset = jax.random.split(_rng, config["NUM_ENVS"])
        env_state = jax.vmap(env.reset)(rng_reset)
        
        # 使用正确的方法获取观察
        # AeroPlanaxHierarchicalCombatEnv使用_get_obs而不是get_obs
        init_obs_dict = jax.vmap(env._get_obs, in_axes=(0, None))(env_state, env.default_params)
        init_global_obs_dict = jax.vmap(env.get_global_obs, in_axes=(0))(env_state)
        
        # 对输入进行batch处理
        init_obs = batchify(init_obs_dict, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        init_global_obs = batchify(init_global_obs_dict, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        
        # 只保留有效的agent
        init_obs = init_obs[:valid_agent_num]
        init_global_obs = init_global_obs[:valid_agent_num]
        init_done = jnp.zeros((valid_agent_num,), dtype=jnp.bool_)
        
        # 初始化隐藏状态
        init_ac_hstate = main_network.initialize_carry(valid_agent_num)
        if critic_network is not None:
            init_cr_hstate = critic_network.initialize_carry(valid_agent_num)
        else:
            init_cr_hstate = jnp.zeros((valid_agent_num, 1))
        init_hstates = (init_ac_hstate, init_cr_hstate)
        
        # 训练时的更新步骤
        def _update_step(update_runner_state, unused):
            """单个更新步骤"""
            (train_states, env_state, obs, done, hstates, rng, epoch, metrics) = update_runner_state
            
            # 1. 收集一个完整episode
            (curr_obs, curr_global_obs) = obs
            rng, _rng = jax.random.split(rng)
            episode, env_state, next_obs, next_done, next_hstates, _rng = collect_complete_episode(
                _rng, train_states, env_state, curr_obs, curr_global_obs, done, hstates,
                env, config, replay_config, networks, valid_agent_num, invalid_agent_num
            )
            
            # 2. 将episode添加到replay buffer
            replay_buffer.add_episode(episode)
            
            # 3. 如果buffer中有足够的episode，进行训练
            if replay_buffer.is_ready_for_training():
                # 多次从buffer中采样并训练
                training_iterations = config.get("TRAINING_ITERATIONS", 1)
                total_loss_info = {}
                
                for _ in range(training_iterations):
                    # 从buffer中采样episode
                    rng, _rng = jax.random.split(rng)
                    sampled_episode = replay_buffer.sample_batch(_rng)
                    
                    # 对采样的episode进行训练
                    rng, _rng = jax.random.split(rng)
                    train_states, loss_info, _rng = train_on_episode(
                        _rng, train_states, sampled_episode, networks, config
                    )
                    
                    # 累加loss
                    for k, v in loss_info.items():
                        if k in total_loss_info:
                            total_loss_info[k] += v
                        else:
                            total_loss_info[k] = v
                
                # 计算平均loss
                for k in total_loss_info:
                    total_loss_info[k] /= training_iterations
            else:
                total_loss_info = {}
            
            # 4. 更新metrics
            metrics["episode_length"] = episode.episode_length
            metrics["episode_return"] = episode.episode_return
            metrics["buffer_size"] = len(replay_buffer)
            
            for k, v in total_loss_info.items():
                metrics[k] = v
            
            # 5. 定期保存模型
            if save_epochs > 0 and (epoch) % save_epochs == 0:
                save_train(train_states, epoch, config["SAVE_PATH"])
            
            # 6. 准备下一轮训练的状态
            next_epoch = epoch + 1
            next_metrics = metrics
            
            next_update_runner_state = (
                train_states, env_state, next_obs, next_done, next_hstates,
                _rng, next_epoch, next_metrics
            )
            
            # 将metrics回传给TensorBoard
            if train_mode:
                callback_fn = lambda: callback(metrics)
                jax.debug.callback(callback_fn)
            
            return next_update_runner_state, metrics
        
        # TensorBoard回调函数
        def callback(metric):
            """将指标写入TensorBoard"""
            if not train_mode or summary_writer is None:
                return
            
            def safe_scalar(x):
                """安全地转换为scalar"""
                if hasattr(x, "item"):
                    return x.item()
                if hasattr(x, "tolist"):
                    return x.tolist()
                return x
            
            # 写入当前epoch的指标
            epoch = int(safe_scalar(metric.get("epoch", start_epoch)))
            for k, v in metric.items():
                if k != "epoch":
                    summary_writer.add_scalar(f"train/{k}", safe_scalar(v), epoch)
            
            # 刷新writer
            summary_writer.flush()
        
        # 初始化runner state
        runner_state = (
            train_states, env_state, (init_obs, init_global_obs), init_done, init_hstates,
            rng, start_epoch, {"epoch": start_epoch}
        )
        
        # 运行训练循环
        epochs = config.get("EPOCHS", 1000)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=epochs
        )
        
        # 获取最终状态
        train_states = runner_state[0]
        
        # 关闭TensorBoard
        if train_mode and summary_writer is not None:
            summary_writer.close()
        
        return train_states, metrics
    
    # 返回训练函数
    return train 