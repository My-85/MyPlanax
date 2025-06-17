import jax
import optax
import distrax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import orbax.checkpoint as ocp
from typing import Sequence, Dict, Tuple, List, Any
from flax.linen.initializers import constant, orthogonal

from flax.training.train_state import TrainState
from networks.scannedRNN import ScannedRNN

MAPPO_DISCRETE_DEFAULT_DIMS = [41, 41, 41, 41]

'''
@desperated
暂时保留以读取旧模型
'''
class ActorRNNOldPattern(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.LayerNorm()(embedding)
        embedding = activation(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)


        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_throttle_mean = nn.Dense(
            self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_elevator_mean = nn.Dense(
            self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_aileron_mean = nn.Dense(
            self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_rudder_mean = nn.Dense(
            self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi_throttle = distrax.Categorical(logits=actor_throttle_mean)
        pi_elevator = distrax.Categorical(logits=actor_elevator_mean)
        pi_aileron = distrax.Categorical(logits=actor_aileron_mean)
        pi_rudder = distrax.Categorical(logits=actor_rudder_mean)

        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder)
    
class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        obs = nn.LayerNorm()(obs)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.LayerNorm()(embedding)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)


        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.LayerNorm()(actor_mean)
        actor_mean = activation(actor_mean)

        pi_list = []
        for i, dim in enumerate(self.action_dim):
            logits = nn.Dense(
                dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name=f"actor_head_{i}"
            )(actor_mean)
            pi_list.append(distrax.Categorical(logits=logits))

        return hidden, pi_list
class CriticRNN(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        world_state = nn.LayerNorm()(world_state)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.LayerNorm()(embedding)
        embedding = activation(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.LayerNorm()(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)
    

def init_network(config : Dict[str, Any], discrete_action_dims : List[int] = MAPPO_DISCRETE_DEFAULT_DIMS, old_model=False) -> Tuple[Tuple[nn.Module, nn.Module], Tuple[TrainState, TrainState], int]:
    rng = jax.random.PRNGKey(42)

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

    def linear_schedule_actor(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return actor_lr * frac
    
    def linear_schedule_critic(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return critic_lr * frac
    
    if old_model:
        actor_network = ActorRNNOldPattern(discrete_action_dims, config=config)
    else:
        actor_network = ActorRNN(discrete_action_dims, config=config)

    critic_network = CriticRNN(config=config)
    rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

    ac_init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], config["OBS_DIM"])),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
    if "GLOBAL_OBS_DIM" not in config.keys():
        config["GLOBAL_OBS_DIM"] = config["OBS_DIM"]
    cr_init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], config["GLOBAL_OBS_DIM"])),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)
    
    # Create separate optimizers for actor and critic
    if config["ANNEAL_LR"]:
        ac_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule_actor, eps=1e-5),
        )
        cr_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule_critic, eps=1e-5),
        )
    else:
        ac_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(actor_lr, eps=1e-5),
        )
        cr_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(critic_lr, eps=1e-5),
        )
    
    ac_train_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_network_params,
        tx=ac_tx,
    )
    cr_train_state = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_network_params,
        tx=cr_tx,
    )

    if "LOADDIR" in config:
        state = {
            "actor_params": ac_train_state.params,
            "actor_opt_state": ac_train_state.opt_state,
            "critic_params": cr_train_state.params,
            "critic_opt_state": cr_train_state.opt_state,
            "epoch": jnp.array(0)
        }
        checkpoint = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()).restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))

        actor_params, actor_opt_state = checkpoint["actor_params"], checkpoint["actor_opt_state"]
        ac_train_state = ac_train_state.replace(params=actor_params, opt_state=actor_opt_state)

        critic_params, critic_opt_state = checkpoint["critic_params"], checkpoint["critic_opt_state"]
        cr_train_state = cr_train_state.replace(params=critic_params, opt_state=critic_opt_state)
        
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    return (actor_network, critic_network), (ac_train_state, cr_train_state), start_epoch