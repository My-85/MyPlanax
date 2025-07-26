# jax.config.update('jax_platform_name', 'cpu')  # 注释掉强制使用CPU的配置
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple, Dict
from envs.wrappers_mul import LogWrapper
from envs.aeroplanax_formation import (
    # AeroPlanaxFormationEnv as Env,
    # FormationTaskParams as TaskParams,
    FormationTaskState
)
# from envs.aeroplanax_combat_hierarchy import(
#     AeroPlanaxHierarchicalCombatEnv as Env,
#     HierarchicalCombatTaskParams as TaskParams,
#     HierarchicalCombatTaskState
# )
from envs.aeroplanax_combat_hierarchy_FSM import(
    AeroPlanaxHierarchicalCombatEnv as Env,
    HierarchicalCombatTaskParams as TaskParams,
    HierarchicalCombatTaskState
)
from networks import (
    init_network_mappoRNN_discrete,
    ScannedRNN,
    unzip_discrete_action
)
from maketrains import (
    RENDER_CONFIG
)
from maketrains.utils import batchify, unbatchify


# PPO_DISCRETE_HIERARCHY_DEFAULT_DIMS = [3, 5, 3]
PPO_DISCRETE_HIERARCHY_DEFAULT_DIMS = [30, 30, 30]
DEFUALT_DIMS = PPO_DISCRETE_HIERARCHY_DEFAULT_DIMS

env_params = TaskParams()
env = Env(env_params)


# config = {
#     "SEED": 42,
#     "NOISE_SEED": 42,
#     "OUTPUT_TRACK": True,
#     "LOADDIR": "/home/lczh/Git Project/results/combat_hierarchy_pitch_2025-05-18-18-52/checkpoints/checkpoint_epoch_60" 
# }
from datetime import datetime

str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
config = {
    "LR": 2e-4,  # Slightly lower learning rate due to larger action space
    "FC_DIM_SIZE": 256,  # Increased network capacity
    "GRU_HIDDEN_DIM": 256,  # Increased network capacity
    "UPDATE_EPOCHS": 16,
    "NUM_MINIBATCHES": 5,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 1e-3,
    "VF_COEF": 1,
    "MAX_GRAD_NORM": 2,
    "ACTIVATION": "relu",
    "ANNEAL_LR": False,
    "DEBUG": True,
    "NUM_ENVS": 1,
    "NUM_STEPS": 2000,
    "TOTAL_TIMESTEPS": 1e4,
    "SEED": 42,
    "NOISE_SEED": 42,
    "OUTPUT_TRACK": True,
    "GROUP": "combat_hierarchy_pitch",
    "FOR_LOOP_EPOCHS": 1,
    "WANDB": True,
    "TRAIN": False,
    "WANDB_API_KEY" : "f4316927feb010654a1d429360c2f2c824e84387",
    "OUTPUTDIR": "results/" + "combat_hierarchy_pitch_" + str_date_time,
    "LOGDIR": "results/" + "combat_hierarchy_pitch_" + str_date_time + "/logs",
    "SAVEDIR": "results/" + "combat_hierarchy_pitch_" + str_date_time + "/checkpoints",
    # "LOADDIR": "/home/lczh/Git Project/results/combat_hierarchy_pitch_2025-05-23-14-35/checkpoints/checkpoint_epoch_1360" 
    "LOADDIR": "/home/dqy/NeuralPlanex/Planax_lczh/Planax_lczh/results/combat_hierarchy_pitch_2V22025-07-25-10-52/checkpoints/checkpoint_epoch_480" 
}
# config = config | RENDER_CONFIG
config["NUM_UPDATES"] = (
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)

rng = jax.random.PRNGKey(config["SEED"])
if "NOISE_SEED" in config.keys():
    _noise_rng = jax.random.PRNGKey(config["NOISE_SEED"])
else:
    rng, _noise_rng = jax.random.split(rng)

env = Env(env_params)
env = LogWrapper(env, rng=_noise_rng)

# NOTE:从wrappers_mul中取得obs_dim、num_agents等数据
config = config | env.get_env_information_for_config()

(network, _), (ac_train_state, _), _ = init_network_mappoRNN_discrete(config, DEFUALT_DIMS)

network_params = ac_train_state.params

print()
print(f'current env: {Env.__name__}')
print(f'{config["NUM_ACTORS"]} agents per env, where {config["NUM_VALID_AGENTS"]} agents can be played.')
print(f'output tracks in folder \'./tracks/\': {config["OUTPUT_TRACK"]}')

valid_agent_num = config["NUM_VALID_AGENTS"] * config["NUM_ENVS"]
invalid_agent_num = config["NUM_ENVS"] * (config["NUM_ACTORS"] - config["NUM_VALID_AGENTS"])

if invalid_agent_num > 0:
    VS_BASELINE = True
else:
    VS_BASELINE = False

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def test(config: Dict, rng):
    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
    init_obs = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])[:valid_agent_num]
    if config["OUTPUT_TRACK"]:
        env.render(env_state.env_state, env_params, {'__all__': False}, './tracks/')
    init_hstate = ScannedRNN.initialize_carry(valid_agent_num, config["GRU_HIDDEN_DIM"])

    # TEST LOOP
    def _env_step(test_state):
        env_state, last_obs, last_done, hstate, rng = test_state
        rng, _rng = jax.random.split(rng)

        # last_obs : ArrayLike
        # last_obs = last_obs.at[:,10].set(0.)
        # last_obs = last_obs.at[:,11].set(0.)
        ac_in = (
            last_obs[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        output = network.apply(network_params, hstate, ac_in)
        
        hstate, pi = output[0], output[1]

        rng, action, log_prob = unzip_discrete_action(_rng, pi)
        
        action, log_prob = (
            action.squeeze(0),
            log_prob.squeeze(0),
        )
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
        if config["OUTPUT_TRACK"]:
            env.render(env_state.env_state, env_params, done, './tracks/')

        reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
        obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
        
        obsv = obsv[:valid_agent_num]
        reward = reward[:valid_agent_num]
        done = done[:valid_agent_num]

        transition = Transition(
            last_done, action, reward, log_prob, last_obs, info
        )

        mask = jnp.reshape(1.0 - done, (-1, 1))
        hstate = hstate * mask

        test_state = (env_state, obsv, done, hstate, rng)
        return test_state, transition

    rng, _rng = jax.random.split(rng)
    test_state = (
        env_state,
        init_obs,
        jnp.zeros((valid_agent_num), dtype=bool),
        init_hstate,
        _rng,
    )
    for _ in range(5000):
        test_state, traj_batch = _env_step(test_state)
        env_state = test_state[0].env_state

        if isinstance(env_state, HierarchicalCombatTaskState):
            print(f'Time: {env_state.time}, Done: {test_state[2]}, Reward: {traj_batch.reward}, Episodic_return: {test_state[0].returned_episode_returns}, Status: {env_state.plane_state.status}  blood: {env_state.plane_state.blood}')
        elif isinstance(env_state, FormationTaskState):
            delta_N = env_state.plane_state.north - env_state.formation_positions[:,:,0]
            delta_E = env_state.plane_state.east - env_state.formation_positions[:,:,1]
            delta_alt = env_state.plane_state.altitude - env_state.formation_positions[:,:,2]
            delta_heading = env_state.plane_state.yaw - env_state.target_heading
            delta_vt = env_state.plane_state.vt - env_state.target_vt

            distance = (delta_N**2+delta_E**2+delta_alt**2)**(1/2)

            # print(f'{env_state.time} alpha:{env_state.plane_state.alpha[0]}')
            # print(f't: {env_state.time}, alive: {env_state.plane_state.is_crashed}, Done: {env_state.done}, Reward: {traj_batch.reward}')
            # print(f't: {env_state.time}, dist: {distance}, alive: {env_state.plane_state.is_crashed}, Done: {env_state.done}, Reward: {traj_batch.reward}')
            crashed_status = ["T" if x else "F" for x in env_state.plane_state.is_crashed[0]]
            print(f'{env_state.time}, dist: {distance[0]}, crashed: {crashed_status}')
            # print(f'{env_state.time}, dist: {distance[0]}, crashed: {['T' if x else 'F' for x in env_state.plane_state.is_crashed[0]]} pos: {env_state.plane_state.north[0]},{env_state.plane_state.east[0]}')
        else:
            print(f'Time: {env_state.time}, Done: {test_state[2]}, Reward: {traj_batch.reward}')

        
    return {"test_state": test_state, "trajectory": traj_batch}

out = test(config, rng)
