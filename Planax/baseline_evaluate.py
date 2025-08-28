# baseline_evaluate.py
# 直接运行：python baseline_evaluate.py
# 改参数：见“用户可改参数”一节

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'
from typing import Sequence, Dict, Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import optax
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp
import functools

from envs.wrappers import LogWrapper
from envs.aeroplanax_heading_pitch_V import (
    AeroPlanaxHeading_Pitch_V_Env,
    Heading_Pitch_V_TaskParams,
)

# ======================
# 用户可改参数（无需命令行）
# ======================
CKPT_PATH     = "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/Planax/envs/models/RNN_baseline(no_fc2_no_layer_norm)/checkpoints/checkpoint_epoch_1000"
NUM_EPISODES  = 3
STEPS_LIMIT   = 2000
SEED          = 42
GREEDY_ACTION = False   # True=贪心 mode()；False=按分布采样
NUM_ENVS      = 1      # 建议评估用1，后续想并行可以改大（代码已兼容）
# ======================


# ==============
# 网络定义（与训练一致：GRU + scan）
# ==============
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        # carry: (B, H)
        # x: (ins, resets) 其中 ins: (B, D)，resets: (B,)
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],                     # (B,1)
            self.initialize_carry(*rnn_state.shape),    # (B,H)
            rnn_state
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        # x: (obs, dones)  obs: (T,B,ObsDim)  dones: (T,B)
        act = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh
        obs, dones = x

        # 前端 MLP（与训练相同）
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = act(embedding)

        # GRU（时间维T在最前，使用scan）
        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))  # hidden: (B,H); embedding: (T,B,H)

        # 策略头（四个离散动作头）
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = act(actor_mean)

        def head(n):
            return nn.Dense(n, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        pi_throttle = distrax.Categorical(logits=head(self.action_dim[0]))
        pi_elevator = distrax.Categorical(logits=head(self.action_dim[1]))
        pi_aileron  = distrax.Categorical(logits=head(self.action_dim[2]))
        pi_rudder   = distrax.Categorical(logits=head(self.action_dim[3]))

        # 价值头
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = act(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)  # (T,B,1)

        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)  # (T,B)


# =========
# 与训练一致的打包函数
# =========
def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])         # (num_actors, num_envs, dim)
    return x.reshape((num_actors * num_envs, -1))     # (B, dim)

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))         # (num_actors, num_envs, dim)
    return {a: x[i] for i, a in enumerate(agent_list)}


# =========
# 评估函数
# =========
def evaluate_checkpoint() -> Dict[str, Any]:
    # —— 与训练关键超参一致（影响参数树/初始化的部分一定要匹配）
    config = {
        "SEED": SEED,
        "NUM_ENVS": NUM_ENVS,
        "NUM_ACTORS": 1,        # 本任务单智能体
        "FC_DIM_SIZE": 128,
        "GRU_HIDDEN_DIM": 128,
        "MAX_GRAD_NORM": 2,
        "ACTIVATION": "relu",
        "LR": 3e-4,             # 仅用于构建 TrainState 作为 restore 的 target
    }
    action_dims = [31, 41, 41, 41]  # 与训练时一致

    # 环境
    env_params = Heading_Pitch_V_TaskParams()
    env = LogWrapper(AeroPlanaxHeading_Pitch_V_Env(env_params))

    # 构图并初始化参数（shape 要与训练完全一致）
    net = ActorCriticRNN(action_dims, config=config)
    rng = jax.random.PRNGKey(config["SEED"])
    obs_dim = env.observation_space(env.agents[0], env_params).shape

    init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], *obs_dim)),  # (T=1,B,ObsDim)
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),            # (T=1,B)
    )
    init_h = ScannedRNN.initialize_carry(
        config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
    )
    params = net.init(rng, init_h, init_x)

    # 用 target 恢复，避免不安全警告
    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                     optax.adam(config["LR"], eps=1e-5))
    ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
    state_item = {"params": ts.params, "opt_state": ts.opt_state, "epoch": jnp.array(0)}

    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    restored = ckptr.restore(CKPT_PATH, args=ocp.args.StandardRestore(item=state_item))
    params = restored["params"]
    restored_epoch = int(restored.get("epoch", jnp.array(-1)))
    print(f"[Evaluate] Restored epoch: {restored_epoch}")

    rng, _rng = jax.random.split(rng)

    # 跨 episode 统计
    ep_returns, ep_lengths = [], []
    ep_pmax_means, ep_margin_means, ep_entropy_means, ep_change_rates, ep_pmax_ge09 = [], [], [], [], []

    for ep in range(NUM_EPISODES):
        # reset
        reset_keys = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
        obs = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])  # (B,ObsDim)
        done = jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool)
        h = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

        # per-episode 累积
        ret_sum = 0.0
        steps_count = 0

        pmax_list, margin_list, entropy_list, pmax_ge09_list = [], [], [], []
        change_count = jnp.zeros((4,), dtype=jnp.float32)
        prev_modes = None

        for t in range(STEPS_LIMIT):
            ac_in = (obs[None, :], done[None, :])           # (T=1,B,Obs) & (1,B)
            h, pis, value = net.apply(params, h, ac_in)
            pi_th, pi_el, pi_ai, pi_ru = pis

            # 指标
            def head_metrics(pi):
                probs = jax.nn.softmax(pi.logits, axis=-1)  # (1,B,A)
                probs = jnp.clip(probs, 1e-9, 1.0)
                pmax = probs.max(axis=-1)                   # (1,B)
                top2 = jnp.sort(probs, axis=-1)[..., -2:]   # (1,B,2)
                margin = top2[..., 1] - top2[..., 0]        # (1,B)
                ent = pi.entropy()                          # (1,B)
                ge09 = (pmax >= 0.9).astype(jnp.float32)    # (1,B)
                return pmax.mean(), margin.mean(), ent.mean(), ge09.mean()

            m = [head_metrics(p) for p in [pi_th, pi_el, pi_ai, pi_ru]]
            p_m, m_m, e_m, ge_m = zip(*m)
            pmax_list.append(jnp.stack(p_m))       # (4,)
            margin_list.append(jnp.stack(m_m))     # (4,)
            entropy_list.append(jnp.stack(e_m))    # (4,)
            pmax_ge09_list.append(jnp.stack(ge_m)) # (4,)

            # 动作（贪心/采样）
            if GREEDY_ACTION:
                a_th = pi_th.mode(); a_el = pi_el.mode(); a_ai = pi_ai.mode(); a_ru = pi_ru.mode()
            else:
                rng, sk = jax.random.split(rng); a_th = pi_th.sample(seed=sk)
                rng, sk = jax.random.split(rng); a_el = pi_el.sample(seed=sk)
                rng, sk = jax.random.split(rng); a_ai = pi_ai.sample(seed=sk)
                rng, sk = jax.random.split(rng); a_ru = pi_ru.sample(seed=sk)

            # 去掉时间维 -> (B,)
            a_th = a_th.squeeze(0); a_el = a_el.squeeze(0); a_ai = a_ai.squeeze(0); a_ru = a_ru.squeeze(0)
            actions = jnp.stack([a_th, a_el, a_ai, a_ru], axis=-1)  # (B,4)

            # 模式变更率
            if prev_modes is not None:
                change_count = change_count + jnp.mean((actions != prev_modes).astype(jnp.float32), axis=0)
            prev_modes = actions

            # env.step（注意：第三个参数要 unbatchify 成 {agent: (NUM_ENVS, 4)}）
            rng, _rng = jax.random.split(rng)
            step_keys = jax.random.split(_rng, config["NUM_ENVS"])
            action_dict = unbatchify(actions, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
            ob, env_state, rew, dn, info = jax.vmap(env.step, in_axes=(0, 0, 0))(step_keys, env_state, action_dict)

            r = jnp.stack([rew[a] for a in env.agents]).reshape(-1)     # (B,)
            d = jnp.stack([dn[a]  for a in env.agents]).reshape(-1)     # (B,)
            obs = batchify(ob, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])

            ret_sum += float(r.mean())
            done = d
            steps_count += 1

            if bool(d.any()):  # 单环境下 episode 结束
                break

        # 本 episode 汇总
        pmax_arr    = jnp.stack(pmax_list)       # (T,4)
        margin_arr  = jnp.stack(margin_list)     # (T,4)
        entropy_arr = jnp.stack(entropy_list)    # (T,4)
        ge09_arr    = jnp.stack(pmax_ge09_list)  # (T,4)
        change_rate = change_count / max(steps_count - 1, 1)  # (4,)

        ep_returns.append(ret_sum)
        ep_lengths.append(steps_count)
        ep_pmax_means.append(np.array(pmax_arr.mean(axis=0)))
        ep_margin_means.append(np.array(margin_arr.mean(axis=0)))
        ep_entropy_means.append(np.array(entropy_arr.mean(axis=0)))
        ep_change_rates.append(np.array(change_rate))
        ep_pmax_ge09.append(np.array(ge09_arr.mean(axis=0)))

        print(f"[Episode {ep+1}/{NUM_EPISODES}] steps={steps_count}  return(sum)={ret_sum:.4f}")

    # 跨 episode 平均
    ep_returns = np.array(ep_returns)
    ep_lengths = np.array(ep_lengths)
    result = {
        "episodes": NUM_EPISODES,
        "return_sum_mean": float(ep_returns.mean()),
        "return_sum_std": float(ep_returns.std()),
        "length_mean": float(ep_lengths.mean()),
        "length_std": float(ep_lengths.std()),
        "pmax_mean_per_head": np.stack(ep_pmax_means).mean(axis=0),
        "pmax_ge_0.9_per_head": np.stack(ep_pmax_ge09).mean(axis=0),
        "margin_mean_per_head": np.stack(ep_margin_means).mean(axis=0),
        "entropy_mean_per_head": np.stack(ep_entropy_means).mean(axis=0),
        "mode_change_rate_per_head": np.stack(ep_change_rates).mean(axis=0),
    }
    return result


def main():
    report = evaluate_checkpoint()
    print("\n=== Policy Evaluation Report ===")
    print(f"Episodes: {report['episodes']}")
    print(f"Return(sum) mean ± std: {report['return_sum_mean']:.4f} ± {report['return_sum_std']:.4f}")
    print(f"Length mean ± std:      {report['length_mean']:.2f} ± {report['length_std']:.2f}")
    print("pmax_mean_per_head [throttle, elevator, aileron, rudder]:", report["pmax_mean_per_head"])
    print("pmax>=0.9 fraction per head:", report["pmax_ge_0.9_per_head"])
    print("margin_mean_per_head:", report["margin_mean_per_head"])
    print("entropy_mean_per_head:", report["entropy_mean_per_head"])
    print("mode_change_rate_per_head:", report["mode_change_rate_per_head"])


if __name__ == "__main__":
    main()
