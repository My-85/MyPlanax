'''
NOTE: 统一mappo和ppo的实现方式
'''
import os
import jax
import numpy as np
import tensorboardX
import flax.linen as nn
import jax.experimental
import jax.numpy as jnp
import functools

import orbax.checkpoint as ocp
from typing import Dict, Any, Tuple, Callable
from typing import NamedTuple
from envs.wrappers_mul import LogWrapper
from flax.training.train_state import TrainState
import optax # （用于手动全局梯度裁剪）
from networks import (
    ScannedRNN,
    unzip_discrete_action,
)

from maketrains.utils import (
    batchify,
    unbatchify
)

# ===== helper: safe clip for scalars =====
def _clip_scalar(x, lo, hi):
    return jnp.minimum(jnp.maximum(x, lo), hi)


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


def make_train_combine_vsbaseline(
        config : Dict,
        env : LogWrapper,
        networks : Tuple[nn.Module, nn.Module|None],
        train_mode : bool=True,
        save_epochs : int=-1,
    ):
    '''
    networks: 以[actorcritic, None]或者[actor, critic]的形式传入,对于network.apply:  
        - actorcritic返回(h_states, pi, value),  
        - actor返回(h_states, pi),  
        - critic返回(h_states, value)  
    
    save_epoch: 当目前epoch%save_epoch时保存,默认为-1不保存  

    NOTE: ["NUM_VALID_AGENTS"] in config.keys()
    '''
    (main_network, critic_network) = networks

    ######################################################################################################
    def _huber(x, delta):
        ax = jnp.abs(x)
        quad = jnp.minimum(ax, delta)
        lin = ax - quad
        return 0.5 * quad * quad + delta * lin
    ######################################################################################################

    def _union_loss_fn(network_params, init_hstate, traj_batch: Transition, gae, targets, ent_coef):
        # RERUN NETWORK
        _, pi, value = main_network.apply(
            network_params,
            init_hstate.squeeze(0),
            (traj_batch.obs, traj_batch.done),
        )

        ################
        # === 数值与掩码准备 ===
        mask = traj_batch.valid_action.astype(jnp.float32)
        denom = mask.sum() + 1e-8
        ################

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
        # gae = (gae - gae.mean()) / (gae.std() + 1e-8)
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
        # loss_actor = (loss_actor * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        loss_actor  = (loss_actor * mask).sum() / denom

        entropys = [p.entropy() for p in pi]

        # entropy = ((jnp.array(entropys).sum(axis=0)) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        entropy  = ((jnp.array(entropys).sum(axis=0)) * mask).sum() / denom
        
        # debug
        # approx_kl = (((ratio - 1.0) - logratio) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        # clip_frac = ((jnp.abs(ratio - 1) > config["CLIP_EPS"]) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        approx_kl = (((ratio - 1.0) - logratio) * mask).sum() / denom
        clip_frac = ((jnp.abs(ratio - 1.0) > config["CLIP_EPS"]) * mask).sum() / denom
        
        # CALCULATE VALUE LOSS
        ############################################################################################
        # value_pred_clipped = traj_batch.value + (
        #     value - traj_batch.value
        # ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        # value_losses = jnp.square(value - targets)
        # value_losses_clipped = jnp.square(value_pred_clipped - targets)
        # value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        
        # --- 值函数：clip + Huber ---
        value = jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        vf_clip = config["VF_CLIP_EPS"]
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-vf_clip, vf_clip)

        err        = value - targets
        err_clip   = value_pred_clipped - targets
        delta      = config["HUBER_DELTA"]

        vloss      = _huber(err,      delta)
        vloss_clip = _huber(err_clip, delta)

        vloss_comb = jnp.maximum(vloss, vloss_clip)                     # PPO 风格：取更大的那个
        value_loss = (0.5 * vloss_comb * mask).sum() / denom

        ############################################################################################
        total_loss = (loss_actor
                    + config["VF_COEF"] * value_loss
                    - ent_coef * entropy)   # <- 这里
        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

    def _actor_loss_fn(actor_params, init_hstate, traj_batch: Transition, gae, targets, ent_coef):
        # RERUN NETWORK
        _, pi = main_network.apply(
            actor_params,
            init_hstate.squeeze(0),
            (traj_batch.obs, traj_batch.done),
        )

        # 掩码与分母
        mask  = traj_batch.valid_action.astype(jnp.float32)
        denom = mask.sum() + 1e-8

        # 最小 log prob 保护
        min_log_prob = jnp.log(1e-6)
        log_probs = [jnp.maximum(p.log_prob(traj_batch.action[:, :, index]), min_log_prob)
                     for index, p in enumerate(pi)]
        log_prob = jnp.array(log_probs).sum(axis=0)

        # PPO 比率
        logratio = (log_prob - traj_batch.log_prob)
        logratio = jnp.where(jnp.isfinite(logratio), logratio, 0.0)
        logratio = jnp.clip(logratio, -20.0, 20.0)
        ratio    = jnp.exp(logratio)
        ratio    = jnp.where(jnp.isfinite(ratio), ratio, 1.0)
        ratio    = jnp.clip(ratio, 1e-6, 1e6)

        # 注意：gae 已在 _calculate_gae 中做过（带掩码）标准化，这里不要再归一化
        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
        loss_actor  = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor  = (loss_actor * mask).sum() / denom

        # 熵与统计
        entropys = [p.entropy() for p in pi]
        entropy  = ((jnp.array(entropys).sum(axis=0)) * mask).sum() / denom

        approx_kl = (((ratio - 1.0) - logratio) * mask).sum() / denom
        clip_frac = ((jnp.abs(ratio - 1.0) > config["CLIP_EPS"]) * mask).sum() / denom

        actor_loss = loss_actor - ent_coef * entropy
        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)


    ###################################################################################################################
    # def _critic_loss_fn(critic_params, init_hstate, traj_batch: Transition, targets):
    #     # RERUN NETWORK
    #     _, value = critic_network.apply(critic_params, init_hstate.squeeze(0), (traj_batch.world_state,  traj_batch.done)) 
        
    #     ###############################################
    #     # 数值安全：把 NaN/Inf 先变成 0，避免后续全链路 NaN
    #     value = jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    #     ###############################################

    #     # CALCULATE VALUE LOSS
    #     value_pred_clipped = traj_batch.value + (
    #         value - traj_batch.value
    #     ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
    #     value_losses = jnp.square(value - targets)
    #     value_losses_clipped = jnp.square(value_pred_clipped - targets)
    #     value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
    #     critic_loss = config["VF_COEF"] * value_loss
    #     return critic_loss, (value_loss)

    def _critic_loss_fn(critic_params, init_hstate, traj_batch: Transition, targets):
        # 重新前向（critic 用全局观测）
        _, value = critic_network.apply(
            critic_params,
            init_hstate.squeeze(0),
            (traj_batch.world_state, traj_batch.done)
        )

        # === 数值与掩码准备 ===
        value = jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        mask  = traj_batch.valid_action.astype(jnp.float32)
        denom = mask.sum() + 1e-8

        # --- 值函数：clip + Huber ---
        vf_clip = config["VF_CLIP_EPS"]
        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-vf_clip, vf_clip)

        err        = value - targets
        err_clip   = value_pred_clipped - targets
        delta      = config["HUBER_DELTA"]

        vloss      = _huber(err,      delta)
        vloss_clip = _huber(err_clip, delta)

        vloss_comb = jnp.maximum(vloss, vloss_clip)
        value_loss = (0.5 * vloss_comb * mask).sum() / denom

        critic_loss = config["VF_COEF"] * value_loss
        return critic_loss, (value_loss)
    ##################################################################################################################


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
    
    print(f"make_train(): ENABLE_CRITIC: {ENABLE_CRITIC} VS_BASELINE: {VS_BASELINE}")

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
            # COLLECT TRAJECTORIES
            # runner_state, update_steps = update_runner_state
            (runner_state, sched_state), update_steps = update_runner_state
            ent_coef, lr_mult, stop_flag = sched_state
            # 每个 update 内部重新开始早停判据；跨 update 不继承
            stop_flag = jnp.array(False, dtype=jnp.bool_)

            def _env_step(runner_state, unused):
                train_states, env_state, (last_obs, last_global_obs), last_done, hstates, rng = runner_state

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
                
                #############################################################################################
                # transition = Transition(
                #     last_done, action, value, reward, log_prob, last_obs, last_global_obs,
                #     # ~(last_done & done), info
                #     jnp.ones_like(done, dtype=bool),  # 不再屏蔽，避免整批loss被乘为0从而造成梯度消失
                #     info
                # )

                # mask = jnp.reshape(1.0 - done, (-1, 1))
                # ac_hstates = ac_hstates * mask
                # cr_hstates = cr_hstates * mask

                #############################################################################################
                # === 计算当前步的 valid_action 掩码 ===
                # 语义：如果“上一时刻已 done 且此刻也 done”，这条样本就无效（跨段/填充不训练）
                valid_action = jnp.logical_not(jnp.logical_and(last_done, done))  # ~(last_done & done)

                transition = Transition(
                    last_done,                     # 这里保留 last_done（RNN 的 resets 用它更合理）
                    action, value, reward, log_prob, last_obs, last_global_obs,
                    valid_action,                  # 用上面的掩码，不再是全 1
                    info
                )

                # === 在 done 边界上“断梯度 + 清零”隐藏态（而不是简单乘 mask）===
                def _reset_h(h):
                    zeros = jnp.zeros_like(h)
                    # done 的地方把 state 硬置零，且 stop_gradient，防止梯度穿越终止边界
                    return jnp.where(done[:, None], jax.lax.stop_gradient(zeros), h)

                ac_hstates = _reset_h(ac_hstates)
                cr_hstates = _reset_h(cr_hstates)
                #############################################################################################



                runner_state = (train_states, env_state, (obsv, global_obsv), done, (ac_hstates, cr_hstates), rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            metric = traj_batch.info   

            if train_mode:
                # CALCULATE ADVANTAGE
                train_states, env_state, (last_obs, last_global_obs), last_done, hstates, rng = runner_state

                if ENABLE_CRITIC:
                    cr_in = (
                        last_global_obs[np.newaxis, :],
                        last_done[np.newaxis, :],
                    )
                    _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
                else:
                    cr_in = (
                        last_obs[np.newaxis, :],
                        last_done[np.newaxis, :],
                    )
                    _, _, last_val = main_network.apply(train_states[0].params, hstates[0], cr_in)

                last_val = last_val.squeeze(0)

                def _calculate_gae(traj_batch, last_val):
                    def _get_advantages(gae_and_next_value, transition):
                        gae, next_value = gae_and_next_value
                        done, value, reward = (
                            transition.done,
                            transition.value,
                            transition.reward,
                        )

                        ###############################################
                        # 数值安全：先把 reward/value/next_value 清洗为有限值
                        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
                        value = jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                        next_value = jnp.nan_to_num(next_value, nan=0.0, posinf=0.0, neginf=0.0)
                        ###############################################

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
                    # return advantages, advantages + traj_batch.value
                    ##################################################
                    # === 关键：targets 用“未归一化”的优势 ===
                    advantages_raw = advantages
                    targets = advantages_raw + traj_batch.value

                    # === 仅对优势做一次（带掩码）的标准化 ===
                    mask = traj_batch.valid_action.astype(jnp.float32)
                    count = mask.sum() + 1e-8
                    adv_mean = (advantages_raw * mask).sum() / count
                    adv_var  = ((advantages_raw - adv_mean)**2 * mask).sum() / count
                    adv_std  = jnp.sqrt(adv_var + 1e-8)
                    advantages = (advantages_raw - adv_mean) / (adv_std + 1e-8)

                    return advantages, targets
                    ##################################################

                advantages, targets = _calculate_gae(traj_batch, last_val)

                # UPDATE NETWORK
                def _update_epoch(update_state, unused):

                    # 解包时多解三个量
                    (train_states, init_hstates, traj_batch, advantages, targets, rng,
                    ent_coef, lr_mult, stop_flag) = update_state

                    rng, _rng = jax.random.split(rng)

                    def _update_minbatch(carry, minibatch):
                        # carry 带上 ent_coef / lr_mult / do_update
                        (network_train_state, critic_network_train_state, ent_coef, lr_mult, do_update) = carry
                        
                        (network_hstate, critic_network_hstate), traj_batch, advantages, targets = minibatch

                        #################################################################################################################################
                        # ====== ACTOR ======
                        grad_fn = jax.value_and_grad(main_loss_fn, has_aux=True)
                        main_loss, grads = grad_fn(
                            network_train_state.params, network_hstate, traj_batch, advantages, targets, ent_coef  # <- 传入 ent_coef
                        )

                        #################################################################################################################################
                        # === 数值安全：把梯度里的 NaN/Inf 清理掉 ===
                        grads = jax.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads)

                        # === 全局梯度裁剪（actor）===
                        gn_actor = optax.global_norm(grads)
                        scale_actor = jnp.minimum(1.0, config["MAX_GRAD_NORM"] / (gn_actor + 1e-9))
                        # === 学习率乘子，等价缩放 lr ===
                        grads = jax.tree_map(lambda g: g * scale_actor, grads)

                        # ====== 学习率乘子（真正生效）======
                        grads = jax.tree_map(lambda g: g * lr_mult, grads)

                        # ====== 早停：用一个 0/1 mask 跳过参数更新（保持指标可记录）======
                        update_mask = jnp.asarray(do_update, dtype=jnp.float32)
                        grads = jax.tree_map(lambda g: g * update_mask, grads)

                        network_train_state = network_train_state.apply_gradients(grads=grads)

                        #################################################################################################################################

                        # 计算actor梯度统计
                        actor_grad_norm = gn_actor
                        # 获取最大梯度
                        actor_grad_abs_max = jnp.max(jnp.array([jnp.max(jnp.abs(g)) for g in jax.tree_util.tree_leaves(grads)]))

                        # 监控坏梯度（actor）
                        actor_bad_grad = sum([(~jnp.isfinite(g)).sum() for g in jax.tree_util.tree_leaves(grads)]).astype(jnp.float32)
                        actor_nan_grad = sum([jnp.isnan(g).sum() for g in jax.tree_util.tree_leaves(grads)]).astype(jnp.float32)
                        actor_inf_grad = sum([jnp.isinf(g).sum() for g in jax.tree_util.tree_leaves(grads)]).astype(jnp.float32)

                        #################################################################################################################################
                        # === critic（同样缩放）===
                        if ENABLE_CRITIC:

                            #################################################################################################################################
                            critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                            critic_loss, critic_grads = critic_grad_fn(
                                critic_network_train_state.params, critic_network_hstate, traj_batch, targets
                            )
                            #################################################################################################################################
                            # 清洗 + 全局裁剪（critic）
                            critic_grads = jax.tree_map(lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), critic_grads)
                            gn_critic = optax.global_norm(critic_grads)
                            scale_critic = jnp.minimum(1.0, config["MAX_GRAD_NORM"] / (gn_critic + 1e-9))
                            critic_grads = jax.tree_map(lambda g: g * scale_critic, critic_grads)

                            # 学习率乘子（critic 同样生效）
                            critic_grads = jax.tree_map(lambda g: g * lr_mult, critic_grads)

                            # 早停 mask
                            critic_grads = jax.tree_map(lambda g: g * update_mask, critic_grads)
                            critic_network_train_state = critic_network_train_state.apply_gradients(grads=critic_grads)
                            #################################################################################################################################
                            # 获取最大梯度
                            critic_grad_abs_max = jnp.max(jnp.array([jnp.max(jnp.abs(g)) for g in jax.tree_util.tree_leaves(critic_grads)]))

                            # 监控坏梯度（critic）
                            critic_bad_grad = sum([(~jnp.isfinite(g)).sum() for g in jax.tree_util.tree_leaves(critic_grads)]).astype(jnp.float32)
                            critic_nan_grad = sum([jnp.isnan(g).sum() for g in jax.tree_util.tree_leaves(critic_grads)]).astype(jnp.float32)
                            critic_inf_grad = sum([jnp.isinf(g).sum() for g in jax.tree_util.tree_leaves(critic_grads)]).astype(jnp.float32)

                            
                            #################################################################################################################################
                            
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
                                "critic_grad_norm": gn_critic,
                                "critic_grad_max": critic_grad_abs_max,
                                # 坏梯度监控汇总与分解
                                "bad_grad": actor_bad_grad + critic_bad_grad,
                                "bad_grad_actor": actor_bad_grad,
                                "bad_grad_critic": critic_bad_grad,
                                "nan_grad_actor": actor_nan_grad,
                                "inf_grad_actor": actor_inf_grad,
                                "nan_grad_critic": critic_nan_grad,
                                "inf_grad_critic": critic_inf_grad,
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
                                # 仅 actor 的坏梯度
                                "bad_grad": actor_bad_grad,
                                "bad_grad_actor": actor_bad_grad,
                                "nan_grad_actor": actor_nan_grad,
                                "inf_grad_actor": actor_inf_grad,
                            }

                        # 全局数值清理，防止 NaN / Inf 流入 logger
                        loss_info = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), loss_info)
                        return (network_train_state, critic_network_train_state, ent_coef, lr_mult, do_update), loss_info

                    # ===== 这里是 _update_epoch 的外层解包 =====
                    (
                        train_states,
                        init_hstates,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                        ent_coef,
                        lr_mult,
                        stop_flag,   # << 新增：沿着 epoch 传下去
                    ) = update_state
                    rng, _rng = jax.random.split(rng)

                    batch = (
                        init_hstates,
                        traj_batch,
                        advantages,
                        targets,
                    )
                    permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )

                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )

                    #########################################################
                    # do_update: 如果上一个 epoch 触发了早停，则本 epoch 直接跳过参数更新
                    do_update = jnp.logical_not(stop_flag)
                    #########################################################

                    # train_states, total_loss = jax.lax.scan(
                    #     _update_minbatch, train_states, minibatches
                    # )

                    # ======= 扫描所有 minibatch =======
                    # 明确展开/组回 carry
                    (actor_ts, critic_ts, ent_coef, lr_mult, _), loss_stack = jax.lax.scan(
                        _update_minbatch,
                        (train_states[0], train_states[1], ent_coef, lr_mult, do_update),
                        minibatches
                    )
                    train_states = (actor_ts, critic_ts)

                    ######################################################################################
                    # ============ 这一段是新的：依据 epoch 的 KL 调整 ENT / LR，并决定是否“早停” ============
                    # total_loss["approx_kl"] 形状是 [NUM_MINIBATCHES]，取平均
                    kl_mean = jnp.mean(loss_stack["approx_kl"])
                    ent_mean = jnp.mean(loss_stack["entropy"])    

                    # KL 早停判据（作用于“后续 epoch”）
                    target_kl = jnp.asarray(config["TARGET_KL"], dtype=jnp.float32)
                    stop_mult = jnp.asarray(config["KL_STOP_MULT"], dtype=jnp.float32)
                    new_stop = kl_mean > (target_kl * stop_mult)
                    stop_flag = jnp.logical_or(stop_flag, new_stop)          

                    # 自适应熵系数：KL 小 -> 增大熵，KL 大 -> 减小熵
                    ent_lo = jnp.asarray(config["ENT_COEF_MIN"], dtype=jnp.float32)
                    ent_hi = jnp.asarray(config["ENT_COEF_MAX"], dtype=jnp.float32)
                    ent_adj = jnp.asarray(config["ENT_ADJ_RATE"], dtype=jnp.float32)

                    # KL 太小（变化慢/可能趋于确定）=> 增大熵权重，鼓励探索
                    ent_coef = jnp.where(
                        kl_mean < (0.5 * target_kl),
                        _clip_scalar(ent_coef * ent_adj, ent_lo, ent_hi),
                        ent_coef
                    )
                    # KL 太大（策略抖动过猛）=> 略减熵权重（主要还是靠 LR 与早停控住）
                    ent_coef = jnp.where(
                        kl_mean > (1.5 * target_kl),
                        _clip_scalar(ent_coef / ent_adj, ent_lo, ent_hi),
                        ent_coef
                    )

                    # ======= 学习率退火（multiplicative），不低于 MIN_LR_MULT =======
                    # 学习率乘子：KL 太大大幅降（软早停），KL 太小微升
                    lr_decay = jnp.asarray(config["LR_DECAY"], dtype=jnp.float32)
                    lr_min   = jnp.asarray(config["MIN_LR_MULT"], dtype=jnp.float32)
                    lr_mult  = jnp.maximum(lr_min, lr_mult * lr_decay)

                    # 回填 update_state（把新的调度变量带到下一个 epoch）
                    update_state = (
                        train_states,
                        init_hstates,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                        ent_coef,
                        lr_mult,
                        stop_flag,
                    )

                    return update_state, loss_stack
                    ######################################################################################

                ######################################################################################
                """
                改 3：在每个 rollout 片段起点对 hstate 做 stop_gradient（BPTT 截断到 NUM_STEPS）(这一步确保不会把第 k 段的梯度沿着 hstate 传回到第 k-1 段，否则时间链太长，RNN 训练极不稳定。)

                在 _update_step 里，jax.lax.scan(_env_step, ...) 扫完 trajectory 后，你会进入“计算 GAE + 更新网络”的分支。
                在把 initial_hstate 加一维 [None, :] 前，补一个 stop_gradient：
                """
                # adding an additional "fake" dimensionality to perform minibatching correctly

                # 先对片段起点的 hstate 断梯度（BPTT 截断）
                initial_hstate = jax.tree_util.tree_map(jax.lax.stop_gradient, initial_hstate)

                initial_hstate = jax.tree_util.tree_map(
                    lambda x: x[None, :],
                    initial_hstate,
                )
                ######################################################################################

                update_state = (
                    train_states,
                    initial_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    ent_coef, lr_mult, stop_flag
                )

                ######################################################################################
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                train_states = update_state[0]
                
                #############
                # 取回新的调度变量
                ent_coef, lr_mult, stop_flag = update_state[6], update_state[7], update_state[8]

                # 也记录到 metric 里，方便 TensorBoard 观察
                metric["ent_coef"] = ent_coef
                metric["lr_mult"]  = lr_mult
                metric["kl_mean_epoch"] = jnp.mean(loss_info["approx_kl"])
                metric["kl_stop"]  = stop_flag.astype(jnp.float32)   # <- 新增
                #############

                loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
                loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

                metric["loss"] = loss_info
                rng = update_state[5]
                
                runner_state = (train_states, env_state, (last_obs, last_global_obs), last_done, hstates, rng)

                if save_epochs > 0:
                    jax.experimental.io_callback(functools.partial(_save_model_callback, save_epochs=save_epochs, save_dir=config['SAVEDIR']), 
                                                None, 
                                                (train_states, update_steps), 
                                                ordered=True)

            
            metric["update_steps"] = update_steps
            update_steps = update_steps + 1 

            ##########################################################################
            # 在 JIT 内计算，避免回调里触碰 Tracer
            if train_mode:
                mask = traj_batch.valid_action.astype(jnp.float32)
                count = mask.sum() + 1e-8
                adv_mean = (advantages * mask).sum() / count
                adv_var  = (((advantages - adv_mean)**2) * mask).sum() / count
                metric["debug_adv_std"] = jnp.sqrt(adv_var + 1e-8)
            else:
                metric["debug_adv_std"] = jnp.array(0.0, dtype=jnp.float32)
            metric["debug_valid_action_sum"] = traj_batch.valid_action.sum()
            metric["debug_valid_action_ratio"] = traj_batch.valid_action.sum() / traj_batch.valid_action.size

            # 同一段 JIT 内，计算 valid_action 的 size
            metric["debug_valid_action_size"] = jnp.array(traj_batch.valid_action.size, dtype=jnp.float32)
            ##########################################################################

            if config.get("DEBUG"):
                def callback(metric):

                    ########################################################################################
                    """
                    之前用 env_steps = metric["update_steps"] * NUM_ENVS * NUM_STEPS：只是把横轴显示成“累计环境交互步数”（env steps）。
                    每做一次 PPO 更新（update_steps += 1），就大约消费 NUM_ENVS * NUM_STEPS 个交互样本，所以乘起来更像“真实采样量”的时间轴。

                    改成 env_steps = metric["update_steps"]：为了“两个回调用同一套步长”，避免同一条曲线一半用 env steps、一半用 update steps 导致图上错位、被覆盖，看起来像“突然归零/异常跳变”。
                    现在训练器回调和环境回调都用 update_steps，WandB/TensorBoard 的步长一致
                    """
                    # env_steps = metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]
                    env_steps = metric["update_steps"] # 
                    ########################################################################################

                    if train_mode:
                        for k, v in metric["loss"].items():
                            v = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                            writer.add_scalar('loss/{}'.format(k), v, env_steps)

                        ########################################################################################
                        # 在 default_callback 内、写 loss 的同一处再加一份，避免被环境回调覆盖
                        writer.add_scalar('loss_shadow/actor_loss', metric["loss"]["actor_loss"], env_steps)
                        writer.add_scalar('loss_shadow/value_loss', metric["loss"]["value_loss"], env_steps)
                        writer.add_scalar('loss_shadow/entropy',    metric["loss"]["entropy"],    env_steps)
                        ########################################################################################
                        
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


                        # 坏梯度监控（存在即写）
                        if "bad_grad" in metric["loss"]:
                            grad_metrics['bad_grad'] = metric["loss"]["bad_grad"]
                        if "bad_grad_actor" in metric["loss"]:
                            grad_metrics['bad_grad_actor'] = metric["loss"]["bad_grad_actor"]
                        if "bad_grad_critic" in metric["loss"]:
                            grad_metrics['bad_grad_critic'] = metric["loss"]["bad_grad_critic"]
                        if "nan_grad_actor" in metric["loss"]:
                            grad_metrics['nan_grad_actor'] = metric["loss"]["nan_grad_actor"]
                        if "inf_grad_actor" in metric["loss"]:
                            grad_metrics['inf_grad_actor'] = metric["loss"]["inf_grad_actor"]
                        if "nan_grad_critic" in metric["loss"]:
                            grad_metrics['nan_grad_critic'] = metric["loss"]["nan_grad_critic"]
                        if "inf_grad_critic" in metric["loss"]:
                            grad_metrics['inf_grad_critic'] = metric["loss"]["inf_grad_critic"]
           
                            
                        for k, v in grad_metrics.items():
                            v = jnp.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                            writer.add_scalar('grad/{}'.format(k), v, env_steps)

                        # ====== Schedulers ======
                        writer.add_scalar('sched/ent_coef',  float(metric["ent_coef"]),  env_steps)
                        writer.add_scalar('sched/lr_mult',   float(metric["lr_mult"]),   env_steps)
                        writer.add_scalar('sched/kl_epoch',  float(metric["kl_mean_epoch"]), env_steps)
                        writer.add_scalar('sched/target_kl', float(config["TARGET_KL"]), env_steps)
                        writer.add_scalar('sched/kl_stop',   float(metric["kl_stop"]),   env_steps)

                    # 对评估指标做数值安全处理
                    ep_ret = jnp.nan_to_num(metric["returned_episode_returns"][metric["returned_episode"].astype(bool)].mean(), nan=0.0, posinf=0.0, neginf=0.0)
                    ep_len = jnp.nan_to_num(metric["returned_episode_lengths"][metric["returned_episode"].astype(bool)].mean(), nan=0.0, posinf=0.0, neginf=0.0)
                    succ_rate = jnp.nan_to_num(metric["success"][metric["returned_episode"].astype(bool)].mean(), nan=0.0, posinf=0.0, neginf=0.0)
                    alive_cnt = jnp.nan_to_num(metric["alive_count"][metric["returned_episode"].astype(bool)].mean(), nan=0.0, posinf=0.0, neginf=0.0)
                    writer.add_scalar('eval/episodic_return', ep_ret, env_steps)
                    writer.add_scalar('eval/episodic_length', ep_len, env_steps)
                    writer.add_scalar('eval/success_rate', succ_rate, env_steps)
                    writer.add_scalar('eval/alive_count', alive_cnt, env_steps)

                    ##########################################################################
                    # 监控valid_action掩码，检查是否出现全部被掩码掉的情况
                    # 原：advantages.std() / traj_batch.valid_action.sum()（会泄漏 Tracer）
                    # 改：使用 metric 中的宿主标量
                    writer.add_scalar('debug/adv_std', metric["debug_adv_std"], env_steps)
                    writer.add_scalar('debug/valid_action_sum', metric["debug_valid_action_sum"], env_steps)
                    writer.add_scalar('debug/valid_action_ratio', metric["debug_valid_action_ratio"], env_steps)

                    writer.add_scalar('debug/valid_action_size', metric["debug_valid_action_size"], env_steps)
                    ##########################################################################
                    
                    # 同时更新打印的内容
                    print("EnvStep={:<10} EpisodeLength={:<4.2f} Return={:<4.2f} SuccessRate={:.3f} AliveCount:{:.3f}".format(
                        # metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"],
                        env_steps,
                        ep_len,
                        ep_ret,
                        succ_rate,
                        alive_cnt,
                    ))
                # if hasattr(env, 'train_callback') and callable(getattr(env, 'train_callback')):
                #     print(f'检测到{type(env._env).__name__}拥有自定义的train callback！')
                #     callback = functools.partial(env.train_callback, writer=writer, train_mode=train_mode)
                # jax.experimental.io_callback(callback, None, metric)

                ##########################################################################
                # 把两种回调“都执行”
                default_callback = callback
                if hasattr(env, 'train_callback') and callable(getattr(env, 'train_callback')):
                    print(f'检测到{type(env._env).__name__}拥有自定义的train callback！')
                    def combined_cb(metric):
                        default_callback(metric)  # 先写 loss/debug/grad
                        return env.train_callback(metric, writer=writer, train_mode=False)  # 再写环境指标,直接把传给环境回调的 train_mode 置为 False，让它只写评估与奖励指标，不写 loss/grad。
                    callback = combined_cb
                jax.experimental.io_callback(callback, None, metric)
                ##########################################################################

            # return (runner_state, update_steps), None
            # 把更新后的调度器状态一并带回去（stop_flag 不跨 update）
            return ((runner_state, (ent_coef, lr_mult, jnp.array(False, dtype=jnp.bool_))), update_steps), None
            # return (runner_state, update_steps), metric

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
        # runner_state, metric = jax.lax.scan(
        ######################################################################################
        """
        ENT_COEF_INIT 没配时，用你现在的 ENT_COEF 当起始值（再不行就 1e-3）。
        lr_mult 起始为 1.0，后续每个 update 乘 LR_DECAY，不低于 MIN_LR_MULT。
        stop_flag 用来做 KL 早停（跨 epoch）。
        """
        # ===== 初始化调度器状态 =====
        ent_coef0 = jnp.array(config.get("ENT_COEF_INIT", config.get("ENT_COEF", 1e-3)), dtype=jnp.float32)
        lr_mult0  = jnp.array(1.0, dtype=jnp.float32)
        stop_flag0 = jnp.array(False)

        # ===== 把 (runner_state, sched_state, start_epoch) 作为 scan 的携带状态 =====
        runner_state, _ = jax.lax.scan(
            _update_step,
            ((runner_state, (ent_coef0, lr_mult0, stop_flag0)), start_epoch),
            None,
            config["NUM_UPDATES"]
        )

        
        # runner_state, _ = jax.lax.scan(
        #     _update_step, (runner_state, start_epoch), None, config["NUM_UPDATES"]
        # )
        return {"runner_state": runner_state}
        ######################################################################################
        # return {"runner_state": runner_state, "metric": metric}

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