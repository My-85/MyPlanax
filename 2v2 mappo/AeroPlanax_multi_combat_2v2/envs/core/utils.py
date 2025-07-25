import jax
import functools
import jax.numpy as jnp
from .base_dataclass import BasePlaneState, BaseMissileState


def check_collision(state: BasePlaneState, agent_id, R=20):
    alive = state.is_alive | state.is_locked
    cur_pos = jnp.hstack((state.north[agent_id], state.east[agent_id], state.altitude[agent_id]))
    cur_pos = cur_pos.reshape(-1, 1)
    position = jnp.vstack((state.north, state.east, state.altitude))
    distance = jnp.linalg.norm(cur_pos - position, axis=0)
    distance = distance.at[agent_id].set(jnp.finfo(jnp.float32).max)
    distance = jnp.where(alive, distance, jnp.finfo(jnp.float32).max)
    done = jnp.any(distance < R)
    # jax.debug.callback(lambda x: print("Collision is", x), done)
    return done

# def check_extreme_state(state: BasePlaneState, agent_id, min_alpha=-20, max_alpha=45, min_beta=-5.0, max_beta=5.0):
#     alpha = state.alpha[agent_id] * 180 / jnp.pi
#     beta = state.beta[agent_id] * 180 / jnp.pi
#     mask1 = (alpha < min_alpha) | (alpha > max_alpha)
#     mask2 = (beta < min_beta) | (beta > max_beta)
#     done = mask1 | mask2
#     return done

def check_extreme_state(state: BasePlaneState, agent_id, rotation_limit=1000.0):
    P, Q, R = state.P[agent_id], state.Q[agent_id], state.R[agent_id]
    done = jnp.sqrt(P**2 + Q**2 + R**2) > rotation_limit
    return done

def check_high_speed(state: BasePlaneState, agent_id, max_velocity=3):
    velocity = state.vt[agent_id] / 340
    done = velocity > max_velocity
    return done

def check_low_speed(state: BasePlaneState, agent_id, min_velocity=0.01):
    velocity = state.vt[agent_id] / 340
    done = velocity < min_velocity
    return done

def check_high_altitude(state: BasePlaneState, agent_id, altitude_limit=1e9):
    altitude = state.altitude[agent_id]
    done = altitude > altitude_limit
    return done

def check_low_altitude(state: BasePlaneState, agent_id, altitude_limit=2500.0):
    altitude = state.altitude[agent_id]
    done = altitude < altitude_limit
    return done

def check_overload(state: BasePlaneState, agent_id, max_overload=10.0):
    # done = state.az[agent_id] < -max_overload
    mask1 = jnp.abs(state.ax[agent_id]) >= max_overload
    mask2 = jnp.abs(state.ay[agent_id]) >= max_overload
    mask3 = jnp.abs(state.az[agent_id]) >= max_overload
    done = mask1 | mask2 | mask3
    return done

def check_crashed(state: BasePlaneState, agent_id):
    mask1 = check_collision(state, agent_id)
    mask2 = check_extreme_state(state, agent_id)
    mask3 = check_high_speed(state, agent_id)
    mask4 = check_low_speed(state, agent_id)
    mask5 = check_low_altitude(state, agent_id)
    mask6 = check_overload(state, agent_id)
    mask7 = check_high_altitude(state, agent_id)
    crashed = mask1 | mask2 | mask3 | mask4 | mask5 | mask6 | mask7
    return crashed

def check_locked(teams, state: BasePlaneState, agent_id, R=1000, angle=jnp.pi/16): # 本来是R=10000, angle=jnp.pi/8
    cur_pos = jnp.hstack((state.north[agent_id], state.east[agent_id], state.altitude[agent_id]))
    cur_pos = cur_pos.reshape(-1, 1)
    enemy_pos = jnp.vstack((state.north, state.east, state.altitude))
    relative_vector = cur_pos - enemy_pos
    
    # 计算敌机的朝向向量
    st = jnp.sin(state.pitch)
    ct = jnp.cos(state.pitch)
    spsi = jnp.sin(state.yaw)
    cpsi = jnp.cos(state.yaw)
    heading_vector = jnp.vstack((ct * cpsi, ct * spsi, st))
    
    # 计算相对向量和敌机朝向向量的点积
    dot_product = jnp.sum(relative_vector * heading_vector, axis=0)
    
    # 计算自机和敌机之间的距离
    distance = jnp.linalg.norm(relative_vector, axis=0)
    
    # 计算夹角的cos值，如果夹角小于阈值且距离小于锁定距离，则认为被锁定
    angle_cos = dot_product / (distance + 1e-6)  # 防止除以零
    angle_condition = (angle_cos) > jnp.cos(angle)
    distance_condition = distance < R
    mask = angle_condition & distance_condition # mask为1，则说明被锁定
    
    # 只考虑敌方飞机的锁定（不同阵营的飞机）
    current_team = teams[agent_id]
    enemy_mask = teams != current_team
    mask = mask & enemy_mask
    
    alive = state.is_alive | state.is_locked
    mask = jnp.where(alive, mask, False)
    locked = jnp.any(mask)
    # jax.debug.print("id:{id} mask:{mask} angle_cos:{angle_cos} lock:{locked}", id=agent_id,mask=mask,angle_cos=angle_cos,locked=locked)
    return locked

def check_shotdown(state: BasePlaneState, agent_id):
    shotdown = state.blood[agent_id] < 0
    return shotdown

def check_shotdown_by_missile(plane_state: BasePlaneState, missile_state: BaseMissileState, agent_id, Rc=300):
    alive = missile_state.is_alive
    cur_pos = jnp.hstack((plane_state.north[agent_id], plane_state.east[agent_id], plane_state.altitude[agent_id]))
    cur_pos = cur_pos.reshape(-1, 1)
    position = jnp.vstack((missile_state.north, missile_state.east, missile_state.altitude))
    distance = jnp.linalg.norm(cur_pos - position, axis=0)
    distance = jnp.where(alive, distance, jnp.finfo(jnp.float32).max)
    shotdown = jnp.any(distance < Rc)
    return shotdown

def check_miss(state: BaseMissileState, agent_id, t_max=60.0, v_min=150.0):
    timeout = state.time[agent_id] > t_max
    lowspeed = state.vt[agent_id] < v_min
    miss = timeout | lowspeed
    return miss

def check_hit(plane_state: BasePlaneState, missile_state: BaseMissileState, agent_id, Rc=300):
    alive = plane_state.is_alive
    cur_pos = jnp.hstack((missile_state.north[agent_id], missile_state.east[agent_id], missile_state.altitude[agent_id]))
    cur_pos = cur_pos.reshape(-1, 1)
    position = jnp.vstack((plane_state.north, plane_state.east, plane_state.altitude))
    distance = jnp.linalg.norm(cur_pos - position, axis=0)
    distance = jnp.where(alive, distance, jnp.finfo(jnp.float32).max)
    hit = jnp.any(distance < Rc)
    return hit

def count_locked_by(teams, state: BasePlaneState, agent_id, R=10000, angle=jnp.pi/8):
    """
    计算有多少架敌机锁定了目标飞机
    
    Args:
        teams: 阵营数组，0表示友军，1表示敌军
        state: 飞机状态
        agent_id: 目标飞机ID
        R: 锁定距离阈值
        angle: 锁定角度阈值
    
    Returns:
        锁定当前飞机的敌机数量
    """
    cur_pos = jnp.hstack((state.north[agent_id], state.east[agent_id], state.altitude[agent_id]))
    cur_pos = cur_pos.reshape(-1, 1)
    enemy_pos = jnp.vstack((state.north, state.east, state.altitude))
    relative_vector = cur_pos - enemy_pos
    
    # 计算敌机的朝向向量
    st = jnp.sin(state.pitch)
    ct = jnp.cos(state.pitch)
    spsi = jnp.sin(state.yaw)
    cpsi = jnp.cos(state.yaw)
    heading_vector = jnp.vstack((ct * cpsi, ct * spsi, st))
    
    # 计算相对向量和敌机朝向向量的点积
    dot_product = jnp.sum(relative_vector * heading_vector, axis=0)
    
    # 计算自机和敌机之间的距离
    distance = jnp.linalg.norm(relative_vector, axis=0)
    
    # 计算夹角的cos值，如果夹角小于阈值且距离小于锁定距离，则认为被锁定
    angle_cos = dot_product / (distance + 1e-6)  # 防止除以零
    angle_condition = (angle_cos) > jnp.cos(angle)
    distance_condition = distance < R
    mask = angle_condition & distance_condition
    
    # 只考虑敌方飞机的锁定（不同阵营的飞机）
    current_team = teams[agent_id]
    enemy_mask = teams != current_team
    mask = mask & enemy_mask
    
    # 只考虑存活的敌机（活着或被锁定的飞机才能锁定别人）
    alive = state.is_alive_or_locked
    mask = jnp.where(alive, mask, False)
    
    # 计算锁定的敌机数量
    locked_count = jnp.sum(mask.astype(jnp.float32))
    return locked_count

def update_blood(state: BasePlaneState, agent_id, dt, teams, damage_per_lock=1.0):
    """
    更新飞机血量，支持多机锁定时血量叠加
    
    Args:
        state: 飞机状态
        agent_id: 飞机ID
        dt: 时间步长
        teams: 阵营数组，0表示友军，1表示敌军
        damage_per_lock: 每个锁定造成的伤害率
    
    Returns:
        更新后的血量
    """
    # 检查飞机是否还活着或被锁定，如果已经死亡则不扣血
    is_alive_or_locked = state.is_alive_or_locked[agent_id]
    
    # 只有活着或被锁定的飞机才会被继续锁定和扣血
    new_blood = jax.lax.cond(
        is_alive_or_locked,
        lambda: _update_blood_when_alive(state, agent_id, dt, teams, damage_per_lock),
        lambda: state.blood[agent_id]  # 如果已经死亡，血量不变
    )
    
    return new_blood

def _update_blood_when_alive(state: BasePlaneState, agent_id, dt, teams, damage_per_lock=1.0):
    """
    当飞机还活着时更新血量的辅助函数
    """
    # 计算有多少架敌机锁定了当前飞机
    locked_count = count_locked_by(teams, state, agent_id)
    
    # 根据锁定数量计算总伤害
    total_damage = damage_per_lock * dt * locked_count
    
    # 更新血量
    new_blood = state.blood[agent_id] - total_damage
    
    return new_blood