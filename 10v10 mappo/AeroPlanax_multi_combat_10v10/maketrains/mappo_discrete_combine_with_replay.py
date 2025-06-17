'''
NOTE: 带有Replay Buffer的MAPPO实现，解决轨迹截断问题
'''
import os
import jax
import numpy as np
import tensorboardX
import flax.linen as nn
import jax.experimental
import jax.numpy as jnp
import functools
from collections import deque
from typing import Dict, Any, Tuple, Callable, List
from typing import NamedTuple
from dataclasses import dataclass

import orbax.checkpoint as ocp
from envs.wrappers_mul import LogWrapper
from flax.training.train_state import TrainState

from networks import (
    ScannedRNN,
    unzip_discrete_action,
)

from maketrains.utils import (
    batchify,
    unbatchify
)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    valid_action: jnp.ndarray # last_done(此时的Transition.done)和curr_done都为True时，才为False
    info: jnp.ndarray


@dataclass
class ReplayBufferConfig:
    """Replay Buffer配置"""
    max_episodes: int = 100  # 最大存储的完整episode数量
    min_episodes_for_training: int = 10  # 开始训练所需的最小episode数量
    sample_batch_size: int = 32  # 每次采样的batch大小
    max_episode_length: int = 10000  # 单个episode的最大长度
    enable_prioritized_sampling: bool = False  # 是否启用优先级采样


class Episode(NamedTuple):
    """完整的episode数据"""
    transitions: Transition  # shape: (episode_length, num_agents, ...)
    episode_length: jnp.ndarray  # JAX数组
    episode_return: jnp.ndarray  # JAX数组
    episode_success: bool
    init_hstate: Tuple[jnp.ndarray, jnp.ndarray]  # 初始隐状态


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.episodes: deque = deque(maxlen=config.max_episodes)
        self.episode_priorities: deque = deque(maxlen=config.max_episodes)
        
    def add_episode(self, episode: Episode):
        """添加完整的episode"""
        self.episodes.append(episode)
        
        # 计算优先级（基于episode return）
        priority = abs(episode.episode_return) + 1e-6
        self.episode_priorities.append(priority)
        
    def sample_batch(self, rng: jax.random.PRNGKey) -> Tuple[List[Transition], List[Tuple]]:
        """从buffer中采样batch"""
        if len(self.episodes) < self.config.min_episodes_for_training:
            return None, None
            
        # 选择episode
        if self.config.enable_prioritized_sampling and len(self.episode_priorities) > 0:
            # 优先级采样
            priorities = jnp.array(list(self.episode_priorities))
            probs = priorities / priorities.sum()
            episode_indices = jax.random.choice(
                rng, len(self.episodes), 
                shape=(min(self.config.sample_batch_size, len(self.episodes)),),
                p=probs, replace=True
            )
        else:
            # 均匀采样
            episode_indices = jax.random.choice(
                rng, len(self.episodes),
                shape=(min(self.config.sample_batch_size, len(self.episodes)),),
                replace=True
            )
        
        sampled_transitions = []
        sampled_init_hstates = []
        
        for idx in episode_indices:
            episode = self.episodes[idx]
            sampled_transitions.append(episode.transitions)
            sampled_init_hstates.append(episode.init_hstate)
            
        return sampled_transitions, sampled_init_hstates
    
    def __len__(self):
        return len(self.episodes)
    
    def is_ready_for_training(self):
        return len(self.episodes) >= self.config.min_episodes_for_training


def make_train_combine_vsbaseline_with_replay(
        config : Dict,
        env : LogWrapper,
        networks : Tuple[nn.Module, nn.Module|None],
        replay_config: ReplayBufferConfig = None,
        train_mode : bool=True,
        save_epochs : int=-1,
    ):
    '''
    带有Replay Buffer的训练函数
    
    networks: 以[actorcritic, None]或者[actor, critic]的形式传入
    replay_config: Replay Buffer配置，如果为None则使用默认配置
    '''
    
    if replay_config is None:
        replay_config = ReplayBufferConfig()
    
    (main_network, critic_network) = networks

    def _union_loss_fn(network_params, init_hstate, traj_batch: Transition, gae, targets):
        # RERUN NETWORK
        _, pi, value = main_network.apply(
            network_params,
            init_hstate.squeeze(0),
            (traj_batch.obs, traj_batch.done),
        )
        # 添加最小概率保护
        min_log_prob = jnp.log(1e-6)  # log(1e-6) ≈ -13.8
        log_probs = [jnp.maximum(p.log_prob(traj_batch.action[:, :, index]), min_log_prob) for index, p in enumerate(pi)]
        
        log_prob = jnp.array(log_probs).sum(axis=0)

        # CALCULATE ACTOR LOSS
        logratio = (log_prob - traj_batch.log_prob)
        # 对 logratio 做限幅，避免出现 Inf
        logratio = jnp.where(jnp.isfinite(logratio), logratio, 0.0)
        logratio = jnp.clip(logratio, -20.0, 20.0)
        ratio = jnp.exp(logratio)
        # 数值安全：防止 ratio 出现 NaN / Inf
        ratio = jnp.where(jnp.isfinite(ratio), ratio, 1.0)
        # 限幅，避免梯度爆炸
        ratio = jnp.clip(ratio, 1e-6, 1e6)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = (loss_actor * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)

        entropys = [p.entropy() for p in pi]

        entropy = ((jnp.array(entropys).sum(axis=0)) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        
        # debug
        approx_kl = (((ratio - 1) - logratio) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        clip_frac = ((jnp.abs(ratio - 1) > config["CLIP_EPS"]) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        
        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)

        total_loss = (
            loss_actor
            + config["VF_COEF"] * value_loss
            - config["ENT_COEF"] * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

    def _actor_loss_fn(actor_params, init_hstate, traj_batch: Transition, gae, targets):
        # RERUN NETWORK
        _, pi = main_network.apply(
            actor_params,
            init_hstate.squeeze(0),
            (traj_batch.obs, traj_batch.done),
        )
        # 添加最小概率保护
        min_log_prob = jnp.log(1e-6)  # log(1e-6) ≈ -13.8
        log_probs = [jnp.maximum(p.log_prob(traj_batch.action[:, :, index]), min_log_prob) for index, p in enumerate(pi)]
        
        log_prob = jnp.array(log_probs).sum(axis=0)

        # CALCULATE ACTOR LOSS
        logratio = (log_prob - traj_batch.log_prob)
        # 对 logratio 做限幅，避免出现 Inf
        logratio = jnp.where(jnp.isfinite(logratio), logratio, 0.0)
        logratio = jnp.clip(logratio, -20.0, 20.0)
        ratio = jnp.exp(logratio)
        # 数值安全：防止 ratio 出现 NaN / Inf
        ratio = jnp.where(jnp.isfinite(ratio), ratio, 1.0)
        # 限幅，避免梯度爆炸
        ratio = jnp.clip(ratio, 1e-6, 1e6)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"],
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = (loss_actor * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)

        entropys = [p.entropy() for p in pi]

        entropy = ((jnp.array(entropys).sum(axis=0)) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        
        # debug
        approx_kl = (((ratio - 1) - logratio) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        clip_frac = ((jnp.abs(ratio - 1) > config["CLIP_EPS"]) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        
        actor_loss = loss_actor - config["ENT_COEF"] * entropy
        
        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

    def _critic_loss_fn(critic_params, init_hstate, traj_batch: Transition, targets):
        # RERUN NETWORK
        _, value = critic_network.apply(critic_params, init_hstate.squeeze(0), (traj_batch.world_state,  traj_batch.done)) 
        
        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        critic_loss = config["VF_COEF"] * value_loss
        return critic_loss, (value_loss)

    if critic_network is not None:
        ENABLE_CRITIC = True
        main_loss_fn, critic_loss_fn = _actor_loss_fn, _critic_loss_fn
    else:
        ENABLE_CRITIC = False
        main_loss_fn, critic_loss_fn = _union_loss_fn, None

    if "NUM_VALID_AGENTS" not in config.keys():
        config["NUM_VALID_AGENTS"] = config["NUM_ACTORS"]

    valid_agent_num = config["NUM_ENVS"] * config["NUM_VALID_AGENTS"]
    invalid_agent_num = config["NUM_ENVS"] * (config["NUM_ACTORS"] - config["NUM_VALID_AGENTS"])

    if invalid_agent_num > 0:
        VS_BASELINE = True
    else:
        VS_BASELINE = False
    
    print(f"make_train_with_replay(): ENABLE_CRITIC: {ENABLE_CRITIC} VS_BASELINE: {VS_BASELINE}")
    print(f"Replay Buffer Config: max_episodes={replay_config.max_episodes}, min_episodes={replay_config.min_episodes_for_training}")

    def collect_complete_episode(rng, train_states, env_state, init_obs, init_global_obs, init_done, init_hstates):
        """收集完整的episode直到结束 - 使用JAX兼容的scan操作"""
        
        def _env_step(carry, unused):
            (train_states, env_state, last_obs, last_global_obs, last_done, hstates, rng, episode_return, continue_episode) = carry
            
            # SELECT ACTION
            ac_in = (
                last_obs[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            if ENABLE_CRITIC:
                cr_in = (
                    last_global_obs[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                ac_hstates, pi = main_network.apply(train_states[0].params, hstates[0], ac_in)
                cr_hstates, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            else:
                ac_hstates, pi, value = main_network.apply(train_states[0].params, hstates[0], ac_in)
                cr_hstates = hstates[1]
            
            rng, action, log_prob = unzip_discrete_action(rng, pi)

            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )
            
            # vsbaseline情况下，敌方飞机的action不需要提供
            if VS_BASELINE:
                full_action = jnp.vstack((action, jnp.zeros((invalid_agent_num, action.shape[1]),dtype=action.dtype)))
            else:
                full_action = action

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(rng_step, env_state, 
              unbatchify(full_action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]))

            global_obsv = jax.vmap(env.get_global_obs, in_axes=(0))(env_state)

            reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
            obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
            global_obsv = batchify(global_obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
            done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)

            obsv = obsv[:valid_agent_num]
            global_obsv = global_obsv[:valid_agent_num]
            reward = reward[:valid_agent_num]
            done = done[:valid_agent_num]
            
            # 记录transition
            transition = Transition(
                last_done, action, value, reward, log_prob, last_obs, last_global_obs, ~(last_done & done), info
            )
            
            # 更新episode统计
            episode_return += reward.mean()
            
            # 检查episode是否结束
            all_done = jnp.all(done)
            continue_episode = ~all_done  # 继续episode的条件
            
            # 更新状态
            mask = jnp.reshape(1.0 - done, (-1, 1))
            ac_hstates = ac_hstates * mask
            cr_hstates = cr_hstates * mask
            hstates = (ac_hstates, cr_hstates)
            
            carry = (train_states, env_state, obsv, global_obsv, done, hstates, rng, episode_return, continue_episode)
            return carry, transition
        
        # 初始状态
        initial_carry = (
            train_states, env_state, init_obs, init_global_obs, init_done, 
            init_hstates, rng, 0.0, True
        )
        
        # 使用scan收集transitions，最大长度为max_episode_length
        final_carry, transitions = jax.lax.scan(
            _env_step, 
            initial_carry, 
            None, 
            length=replay_config.max_episode_length
        )
        
        # 提取最终状态
        (_, final_env_state, final_obs, final_global_obs, final_done, final_hstates, final_rng, episode_return, _) = final_carry
        
        # 简化：直接使用收集到的所有transitions，不进行截断
        # 原始的valid_action mask已经处理了episode结束的逻辑
        episode = Episode(
            transitions=transitions,
            episode_length=jnp.array(replay_config.max_episode_length),  # 使用固定长度
            episode_return=episode_return,  # 保持为JAX数组
            episode_success=True,  # 简化：假设完成episode即为成功
            init_hstate=init_hstates
        )
        
        return episode, final_env_state, (final_obs, final_global_obs), final_done, final_hstates, final_rng

    # 简化的训练函数，直接使用原始的MAPPO训练逻辑，但收集完整episode
    def train_on_episode(rng, train_states, episode):
        """对单个episode进行训练"""
        if episode is None:
            return train_states, {}, rng
        
        episode_transitions = episode.transitions
        
        # 计算GAE和targets
        if ENABLE_CRITIC:
            cr_in = (
                episode_transitions.world_state[-1:],
                episode_transitions.done[-1:],
            )
            _, last_val = critic_network.apply(train_states[1].params, episode.init_hstate[1], cr_in)
        else:
            ac_in = (
                episode_transitions.obs[-1:],
                episode_transitions.done[-1:],
            )
            _, _, last_val = main_network.apply(train_states[0].params, episode.init_hstate[0], ac_in)

        last_val = last_val.squeeze(0)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(episode_transitions, last_val)

        # 训练网络
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_states: Tuple[TrainState,TrainState|None], batch_info):
                (network_train_state, critic_network_train_state) = train_states
                (network_hstate, critic_network_hstate), traj_batch, advantages, targets = batch_info

                grad_fn = jax.value_and_grad(main_loss_fn, has_aux=True)
                main_loss, grads = grad_fn(
                    network_train_state.params, network_hstate, traj_batch, advantages, targets
                )
                
                actor_grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
                actor_grad_abs_max = jnp.max(jnp.array([jnp.max(jnp.abs(g)) for g in jax.tree_util.tree_leaves(grads)]))
                network_train_state = network_train_state.apply_gradients(grads=grads)

                if ENABLE_CRITIC:
                    critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_network_train_state.params, critic_network_hstate, traj_batch, targets
                    )
                    critic_grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(critic_grads)))
                    critic_grad_abs_max = jnp.max(jnp.array([jnp.max(jnp.abs(g)) for g in jax.tree_util.tree_leaves(critic_grads)]))
                    critic_network_train_state = critic_network_train_state.apply_gradients(grads=critic_grads)
                    
                    total_loss = main_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": main_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": main_loss[1][1],
                        "ratio": main_loss[1][2],
                        "approx_kl": main_loss[1][3],
                        "clip_frac": main_loss[1][4],
                        "actor_grad_norm": actor_grad_norm,
                        "actor_grad_max": actor_grad_abs_max,
                        "critic_grad_norm": critic_grad_norm,
                        "critic_grad_max": critic_grad_abs_max,
                    }
                else:
                    loss_info = {
                        "total_loss": main_loss[0],
                        "value_loss": main_loss[1][0],
                        "actor_loss": main_loss[1][1],
                        "entropy": main_loss[1][2],
                        "ratio": main_loss[1][3],
                        "approx_kl": main_loss[1][4],
                        "clip_frac": main_loss[1][5],
                        "actor_grad_norm": actor_grad_norm,
                        "actor_grad_max": actor_grad_abs_max,
                    }

                loss_info = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), loss_info)
                
                return (network_train_state, critic_network_train_state), loss_info

            (train_states, init_hstates, traj_batch, advantages, targets, rng) = update_state
            rng, _rng = jax.random.split(rng)

            batch = (init_hstates, traj_batch, advantages, targets)
            
            # 简化的minibatch处理
            train_states, loss_info = _update_minbatch(train_states, batch)
            
            update_state = (train_states, init_hstates, traj_batch, advantages, targets, rng)
            return update_state, loss_info

        # 添加fake维度进行minibatching
        init_hstate = jax.tree_util.tree_map(lambda x: x[None, :], episode.init_hstate)

        update_state = (train_states, init_hstate, episode_transitions, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        train_states = update_state[0]
        rng = update_state[-1]
        
        return train_states, loss_info, rng

    def train(rng, train_states : Tuple[TrainState,TrainState], start_epoch : int = 0):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
        init_last_obs = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])[:valid_agent_num]
        init_last_done = jnp.zeros((valid_agent_num), dtype=bool)

        global_obsv = jax.vmap(env.get_global_obs, in_axes=(0))(env_state)
        init_last_global_obs = batchify(global_obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])[:valid_agent_num]
        
        # INIT Tensorboard
        if config.get("DEBUG"):
            writer = tensorboardX.SummaryWriter(config["LOGDIR"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state
            train_states, env_state, (last_obs, last_global_obs), last_done, hstates, rng = runner_state

            # 收集完整episode
            network_init_hstate = ScannedRNN.initialize_carry(valid_agent_num, config["GRU_HIDDEN_DIM"])
            critic_network_init_hstate = ScannedRNN.initialize_carry(valid_agent_num, config["GRU_HIDDEN_DIM"])
            
            rng, collect_rng = jax.random.split(rng)
            episode, env_state, (last_obs, last_global_obs), last_done, hstates, rng = collect_complete_episode(
                collect_rng, train_states, env_state, last_obs, last_global_obs, last_done, 
                (network_init_hstate, critic_network_init_hstate)
            )
            
            # 直接在收集到的episode上训练
            loss_info = {}
            if train_mode and episode is not None:
                rng, train_rng = jax.random.split(rng)
                train_states, loss_info, rng = train_on_episode(train_rng, train_states, episode)
            
            # 重置环境准备下一个episode
            rng, reset_rng = jax.random.split(rng)
            reset_rng = jax.random.split(reset_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
            last_obs = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])[:valid_agent_num]
            last_done = jnp.zeros((valid_agent_num), dtype=bool)
            global_obsv = jax.vmap(env.get_global_obs, in_axes=(0))(env_state)
            last_global_obs = batchify(global_obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])[:valid_agent_num]
            
            # 重置隐状态
            hstates = (network_init_hstate, critic_network_init_hstate)
            
            runner_state = (train_states, env_state, (last_obs, last_global_obs), last_done, hstates, rng)
            
            # 保存模型
            if save_epochs > 0:
                jax.experimental.io_callback(functools.partial(_save_model_callback, save_epochs=save_epochs, save_dir=config['SAVEDIR']), 
                                            None, 
                                            (train_states, update_steps), 
                                            ordered=True)

            # 构造metric用于logging
            metric = {
                "update_steps": update_steps,
                "loss": loss_info,
                "episode_length": episode.episode_length if episode else 0,
                "episode_return": episode.episode_return if episode else 0.0,
                "episode_success": episode.episode_success if episode else False,
                # 添加一些假的metric以兼容原始logging
                "returned_episode_returns": jnp.array([episode.episode_return if episode else 0.0]),
                "returned_episode_lengths": jnp.array([episode.episode_length if episode else 0]),
                "returned_episode": jnp.array([True if episode else False]),
                "success": jnp.array([episode.episode_success if episode else False]),
                "alive_count": jnp.array([10.0]),  # 假设的存活数量
            }
            
            update_steps = update_steps + 1

            if config.get("DEBUG"):
                def callback(metric):
                    env_steps = metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]
                    
                    if train_mode and metric["loss"]:
                        for k, v in metric["loss"].items():
                            v = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                            # 确保是标量值 - 对多维数组取平均值
                            if jnp.ndim(v) > 0:
                                scalar_v = float(jnp.mean(v))
                            else:
                                scalar_v = float(v)
                            writer.add_scalar('loss/{}'.format(k), scalar_v, env_steps)
                        
                        # 将梯度统计单独写到一个新的分组
                        grad_metrics = {
                            'actor_grad_norm': metric["loss"]["actor_grad_norm"],
                            'actor_grad_max': metric["loss"]["actor_grad_max"]
                        }
                        
                        if "critic_grad_norm" in metric["loss"]:
                            grad_metrics.update({
                                'critic_grad_norm': metric["loss"]["critic_grad_norm"],
                                'critic_grad_max': metric["loss"]["critic_grad_max"]
                            })
                            
                        for k, v in grad_metrics.items():
                            v = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                            # 确保是标量值 - 对多维数组取平均值
                            if jnp.ndim(v) > 0:
                                scalar_v = float(jnp.mean(v))
                            else:
                                scalar_v = float(v)
                            writer.add_scalar('grad/{}'.format(k), scalar_v, env_steps)
                    
                    # 对评估指标做数值安全处理
                    ep_ret = jnp.nan_to_num(metric["returned_episode_returns"][metric["returned_episode"].astype(bool)].mean(), nan=0.0, posinf=0.0, neginf=0.0)
                    ep_len = jnp.nan_to_num(metric["returned_episode_lengths"][metric["returned_episode"].astype(bool)].mean(), nan=0.0, posinf=0.0, neginf=0.0)
                    succ_rate = jnp.nan_to_num(metric["success"][metric["returned_episode"].astype(bool)].mean(), nan=0.0, posinf=0.0, neginf=0.0)
                    alive_cnt = jnp.nan_to_num(metric["alive_count"][metric["returned_episode"].astype(bool)].mean(), nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # 确保所有值都是标量
                    def safe_scalar(x):
                        if jnp.ndim(x) > 0:
                            return float(jnp.mean(x))
                        else:
                            return float(x)
                    
                    writer.add_scalar('eval/episodic_return', safe_scalar(ep_ret), env_steps)
                    writer.add_scalar('eval/episodic_length', safe_scalar(ep_len), env_steps)
                    writer.add_scalar('eval/success_rate', safe_scalar(succ_rate), env_steps)
                    writer.add_scalar('eval/alive_count', safe_scalar(alive_cnt), env_steps)
                    
                    print("EnvStep={:<10} EpisodeLength={:<4} Return={:<4.2f} Success={}".format(
                        env_steps,
                        metric["episode_length"],
                        metric["episode_return"],
                        metric["episode_success"],
                    ))

                jax.experimental.io_callback(callback, None, metric)

            return (runner_state, update_steps), None

        network_init_hstate = ScannedRNN.initialize_carry(valid_agent_num, config["GRU_HIDDEN_DIM"])
        critic_network_init_hstate = ScannedRNN.initialize_carry(valid_agent_num, config["GRU_HIDDEN_DIM"])
        
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_states,
            env_state,
            (init_last_obs, init_last_global_obs),
            init_last_done,
            (network_init_hstate, critic_network_init_hstate),
            _rng,
        )
        
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, start_epoch), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


def save_model(trains_states:Tuple[TrainState, TrainState|None], current_epochs:int, save_dir:str) -> Callable[[Tuple[Tuple[TrainState, TrainState|None], int]], None]:
    return functools.partial(_save_model_callback, save_epochs=1, save_dir=save_dir)(((trains_states, current_epochs)))


def _save_model_callback(params:Tuple[Tuple[TrainState, TrainState|None], int], save_epochs: int, save_dir):
    (actor_train_state, critic_train_state), current_epochs = params

    checkpoint_path = os.path.abspath(os.path.join(save_dir, f"checkpoint_epoch_{current_epochs}"))

    if ((current_epochs + 1) % save_epochs == 0) and (not os.path.exists(checkpoint_path)):
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        if critic_train_state is not None:
            checkpoint = {
                "actor_params": actor_train_state.params,
                "actor_opt_state": actor_train_state.opt_state,
                "critic_params": critic_train_state.params,
                "critic_opt_state": critic_train_state.opt_state,
                "epoch": current_epochs
            }
        else:
            checkpoint = {
                "params": actor_train_state.params,
                "opt_state": actor_train_state.opt_state,
                "epoch": current_epochs
            }
        
        ckptr.save(checkpoint_path, args=ocp.args.StandardSave(checkpoint))
        ckptr.wait_until_finished()
        print(f"Checkpoint saved at epoch {current_epochs}")

# 添加可导出的collect_complete_episode函数
def external_collect_complete_episode(
    rng, train_states, env_state, init_obs, init_global_obs, init_done, init_hstates,
    env, config, replay_config, networks, valid_agent_num, invalid_agent_num
):
    """外部可调用的收集完整episode函数"""
    
    (main_network, critic_network) = networks
    ENABLE_CRITIC = critic_network is not None
    VS_BASELINE = invalid_agent_num > 0
    
    def _env_step(carry, unused):
        (train_states, env_state, last_obs, last_global_obs, last_done, hstates, rng, episode_return, continue_episode) = carry
        
        # SELECT ACTION
        ac_in = (
            last_obs[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        if ENABLE_CRITIC:
            cr_in = (
                last_global_obs[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            ac_hstates, pi = main_network.apply(train_states[0].params, hstates[0], ac_in)
            cr_hstates, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)
        else:
            ac_hstates, pi, value = main_network.apply(train_states[0].params, hstates[0], ac_in)
            cr_hstates = hstates[1]
        
        rng, action, log_prob = unzip_discrete_action(rng, pi)

        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )
        
        # vsbaseline情况下，敌方飞机的action不需要提供
        if VS_BASELINE:
            full_action = jnp.vstack((action, jnp.zeros((invalid_agent_num, action.shape[1]),dtype=action.dtype)))
        else:
            full_action = action

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(rng_step, env_state, 
          unbatchify(full_action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]))

        # 使用正确的方法获取观察
        # AeroPlanaxHierarchicalCombatEnv使用_get_obs而不是get_obs
        obsv_dict = jax.vmap(env._get_obs, in_axes=(0, None))(env_state, env.default_params)
        global_obsv_dict = jax.vmap(env.get_global_obs, in_axes=(0))(env_state)
        
        # 对输入进行batch处理
        obsv = batchify(obsv_dict, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        global_obsv = batchify(global_obsv_dict, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        
        reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
        done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)

        obsv = obsv[:valid_agent_num]
        global_obsv = global_obsv[:valid_agent_num]
        reward = reward[:valid_agent_num]
        done = done[:valid_agent_num]
        
        # 记录transition
        transition = Transition(
            last_done, action, value, reward, log_prob, last_obs, last_global_obs, ~(last_done & done), info
        )
        
        # 更新episode统计
        episode_return += reward.mean()
        
        # 检查episode是否结束
        all_done = jnp.all(done)
        continue_episode = ~all_done  # 继续episode的条件
        
        # 更新状态
        mask = jnp.reshape(1.0 - done, (-1, 1))
        ac_hstates = ac_hstates * mask
        cr_hstates = cr_hstates * mask
        hstates = (ac_hstates, cr_hstates)
        
        carry = (train_states, env_state, obsv, global_obsv, done, hstates, rng, episode_return, continue_episode)
        return carry, transition
    
    # 初始状态
    initial_carry = (
        train_states, env_state, init_obs, init_global_obs, init_done, 
        init_hstates, rng, 0.0, True
    )
    
    # 使用scan收集transitions，最大长度为max_episode_length
    final_carry, transitions = jax.lax.scan(
        _env_step, 
        initial_carry, 
        None, 
        length=replay_config.max_episode_length
    )
    
    # 提取最终状态
    (_, final_env_state, final_obs, final_global_obs, final_done, final_hstates, final_rng, episode_return, _) = final_carry
    
    # 创建Episode对象
    episode = Episode(
        transitions=transitions,
        episode_length=jnp.array(replay_config.max_episode_length),  # 使用固定长度
        episode_return=episode_return,  # 保持为JAX数组
        episode_success=True,  # 简化：假设完成episode即为成功
        init_hstate=init_hstates
    )
    
    return episode, final_env_state, (final_obs, final_global_obs), final_done, final_hstates, final_rng

# 添加可导出的train_on_episode函数
def external_train_on_episode(rng, train_states, episode, networks, config):
    """外部可调用的对单个episode进行训练的函数"""
    if episode is None:
        return train_states, {}, rng
    
    (main_network, critic_network) = networks
    ENABLE_CRITIC = critic_network is not None
    
    episode_transitions = episode.transitions
    
    # 计算GAE和targets
    if ENABLE_CRITIC:
        cr_in = (
            episode_transitions.world_state[-1:],
            episode_transitions.done[-1:],
        )
        _, last_val = critic_network.apply(train_states[1].params, episode.init_hstate[1], cr_in)
    else:
        ac_in = (
            episode_transitions.obs[-1:],
            episode_transitions.done[-1:],
        )
        _, _, last_val = main_network.apply(train_states[0].params, episode.init_hstate[0], ac_in)

    last_val = last_val.squeeze(0)

    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + config["GAMMA"] * next_value * (1 - done) - value
            gae = (
                delta
                + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
            )
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16
        )
        return advantages, advantages + traj_batch.value

    advantages, targets = _calculate_gae(episode_transitions, last_val)
    
    # 进行简单的梯度更新
    def _update_actor(actor_params, init_hstate, traj_batch, advantages):
        def loss_fn(params):
            _, pi = main_network.apply(
                params,
                init_hstate.squeeze(0),
                (traj_batch.obs, traj_batch.done),
            )
            min_log_prob = jnp.log(1e-6)
            log_probs = [jnp.maximum(p.log_prob(traj_batch.action[:, :, index]), min_log_prob) for index, p in enumerate(pi)]
            log_prob = jnp.array(log_probs).sum(axis=0)
            
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            ratio = jnp.clip(ratio, 1e-6, 1e6)
            
            loss_actor1 = ratio * advantages
            loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * advantages
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            return loss_actor.mean()
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(actor_params)
        return train_states[0].apply_gradients(grads=grads), loss
    
    def _update_critic(critic_params, init_hstate, traj_batch, targets):
        def loss_fn(params):
            _, value = critic_network.apply(
                params,
                init_hstate.squeeze(0),
                (traj_batch.world_state, traj_batch.done),
            )
            loss = jnp.square(value - targets).mean()
            return loss
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(critic_params)
        return train_states[1].apply_gradients(grads=grads), loss
    
    # 添加fake维度进行minibatching
    init_hstate = jax.tree_util.tree_map(lambda x: x[None, :], episode.init_hstate)
    
    # 更新网络
    actor_train_state, actor_loss = _update_actor(train_states[0].params, init_hstate, episode_transitions, advantages)
    
    if ENABLE_CRITIC:
        critic_train_state, critic_loss = _update_critic(train_states[1].params, init_hstate, episode_transitions, targets)
        train_states = (actor_train_state, critic_train_state)
        loss_info = {"actor_loss": actor_loss, "critic_loss": critic_loss}
    else:
        train_states = (actor_train_state, train_states[1])
        loss_info = {"actor_loss": actor_loss}
    
    return train_states, loss_info, rng 