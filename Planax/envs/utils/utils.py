import os
import yaml
import jax.numpy as jnp
import jax


a = 6378137
b = 6356752.3142
f = (a - b) / a
e_sq = f * (2-f)
pi = jnp.pi


def parse_config(filename):
    """Parse F16Sim config file.

    Args:
        config (str): config file name

    Returns:
        (EnvConfig): a custom class which parsing dict into object.
    """
    filepath = os.path.join(get_root_dir(), 'configs', f'{filename}.yaml')
    assert os.path.exists(filepath), \
        f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
    with open(filepath, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    return type('EnvConfig', (object,), config_data)

def get_root_dir():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')

def geodetic_to_ecef(lat, lon, h):
    # (lat, lon) in degrees
    # h in meters
    lamb = jnp.radians(lat)
    phi = jnp.radians(lon)
    s = jnp.sin(lamb)
    N = a / jnp.sqrt(1 - e_sq * s * s)

    sin_lambda = jnp.sin(lamb)
    cos_lambda = jnp.cos(lamb)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e_sq) * N) * sin_lambda

    return x, y, z

def ecef_to_enu(x, y, z, lat0, lon0, h0):
    lamb = jnp.radians(lat0)
    phi = jnp.radians(lon0)
    s = jnp.sin(lamb)
    N = a / jnp.sqrt(1 - e_sq * s * s)
    sin_lambda = jnp.sin(lamb)
    cos_lambda = jnp.cos(lamb)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda
    xd = x - x0
    yd = y - y0
    zd = z - z0
    t = -cos_phi * xd -  sin_phi * yd
    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = t * sin_lambda  + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd
    return xEast, yNorth, zUp

def enu_to_ecef(xEast, yNorth, zUp, lat0, lon0, h0):
    lamb = jnp.radians(lat0)
    phi = jnp.radians(lon0)
    s = jnp.sin(lamb)
    N = a / jnp.sqrt(1 - e_sq * s * s)
    sin_lambda = jnp.sin(lamb)
    cos_lambda = jnp.cos(lamb)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda
    t = cos_lambda * zUp - sin_lambda * yNorth
    zd = sin_lambda * zUp + cos_lambda * yNorth
    xd = cos_phi * t - sin_phi * xEast 
    yd = sin_phi * t + cos_phi * xEast
    x = xd + x0 
    y = yd + y0 
    z = zd + z0 
    return x, y, z

def ecef_to_geodetic(x, y, z):
   # Convert from ECEF cartesian coordinates to 
   # latitude, longitude and height.  WGS-84
    x2 = x ** 2 
    y2 = y ** 2 
    z2 = z ** 2 
    a = 6378137.0000    # earth radius in meters
    b = 6356752.3142    # earth semiminor in meters 
    e = jnp.sqrt (1 - (b / a) ** 2) 
    b2 = b * b 
    e2 = e ** 2 
    ep = e * (a / b) 
    r = jnp.sqrt(x2 + y2) 
    r2 = r * r 
    E2 = a ** 2 - b ** 2 
    F = 54 * b2 * z2 
    G = r2 + (1 - e2) * z2 - e2 * E2 
    c = (e2 * e2 * F * r2) / (G * G * G) 
    s = (1 + c + jnp.sqrt(c * c + 2 * c)) ** (1 / 3) 
    P = F / (3 * (s + 1 / s + 1) ** 2 * G * G) 
    Q = jnp.sqrt(1 + 2 * e2 * e2 * P) 
    ro = -(P * e2 * r) / (1 + Q) + jnp.sqrt((a * a / 2) * (1 + 1 / Q) - (P * (1 - e2) * z2) / (Q * (1 + Q)) - P * r2 / 2) 
    tmp = (r - e2 * ro) ** 2 
    U = jnp.sqrt(tmp + z2) 
    V = jnp.sqrt(tmp + (1 - e2) * z2) 
    zo = (b2 * z) / (a * V) 
    height = U * (1 - b2 / (a * V)) 
    lat = jnp.atan((z + ep * ep *zo) / r) 
    temp = jnp.atan(y / x)
    if x >= 0 :    
        long = temp 
    elif (x < 0) & (y >= 0):
        long = pi + temp 
    else :
        long = temp - pi 
    lat0 = lat/(pi/180) 
    lon0 = long/(pi/180) 
    h0 = height
    return lat0, lon0, h0

def geodetic_to_enu(lat, lon, h, lat_ref, lon_ref, h_ref):
    x, y, z = geodetic_to_ecef(lat, lon, h)
    return ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)

def enu_to_geodetic(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref):
    x,y,z = enu_to_ecef(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref)
    return ecef_to_geodetic(x,y,z)

def wrap_2PI(angle):
    res = angle % (2 * jnp.pi)
    mask1 = res < 0
    res += 2 * jnp.pi * mask1
    return res

def wrap_PI(angle):
    res = wrap_2PI(angle)
    mask1 = res > jnp.pi
    res -= 2 * jnp.pi * mask1
    return res

def get_AO_TA_R_pitch_yaw(ego_feature, enm_feature, ego_pitch, ego_yaw, enm_pitch, enm_yaw):
    """Get AO & TA angles and relative distance between two agent using heading vectors.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd)
        ego_pitch: 己方飞机的俯仰角(rad)
        ego_yaw: 己方飞机的偏航角(rad)
        enm_pitch: 敌方飞机的俯仰角(rad)
        enm_yaw: 敌方飞机的偏航角(rad)

    Returns:
        (tuple): ego_AO, ego_TA, R, side_flag
    """
    # 提取位置数据
    ego_x, ego_y, ego_z = ego_feature[0], ego_feature[1], ego_feature[2]
    enm_x, enm_y, enm_z = enm_feature[0], enm_feature[1], enm_feature[2]
    
    # 计算相对位置向量
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = jnp.linalg.norm(jnp.array([delta_x, delta_y, delta_z]))
    
    # 安全处理：当R非常小时的特殊情况
    R_is_small = R < 1.0  # 1米以内视为极近

    # 计算己方飞机机头方向向量(单位向量)
    ego_heading_x = jnp.cos(ego_pitch) * jnp.cos(ego_yaw)
    ego_heading_y = jnp.cos(ego_pitch) * jnp.sin(ego_yaw)
    ego_heading_z = jnp.sin(ego_pitch)
    ego_heading = jnp.array([ego_heading_x, ego_heading_y, ego_heading_z])
    
    # 计算敌方飞机机头方向向量(单位向量)
    enm_heading_x = jnp.cos(enm_pitch) * jnp.cos(enm_yaw)
    enm_heading_y = jnp.cos(enm_pitch) * jnp.sin(enm_yaw)
    enm_heading_z = jnp.sin(enm_pitch)
    enm_heading = jnp.array([enm_heading_x, enm_heading_y, enm_heading_z])
    
    # # 计算连线的单位向量
    # los_vector = jnp.array([delta_x, delta_y, delta_z])
    # los_unit = los_vector / (R + 1e-6)  # 避免除零错误
    
    # # 计算AO: 己方机头方向与连线的夹角
    # ego_AO = jnp.arccos(jnp.clip(jnp.dot(ego_heading, los_unit), -1.0, 1.0))
    
    # # 计算TA: 敌方机头方向与连线的夹角
    # # 注意TA的连线是从敌方到己方，所以取负向量
    # enm_TA = jnp.arccos(jnp.clip(jnp.dot(enm_heading, -los_unit), -1.0, 1.0))
    # # 计算侧向标志(水平面上)
    # side_flag = jnp.sign(jnp.cross(jnp.array([ego_heading_x, ego_heading_y]), jnp.array([delta_x, delta_y])))
    # 安全计算连线的单位向量
    safe_R = jnp.maximum(R, 1.0)  # 确保分母至少为1.0
    los_unit = jnp.array([delta_x, delta_y, delta_z]) / safe_R
    
    # 当R很小时使用默认值
    ego_AO = jnp.where(R_is_small, 
                       jnp.pi/2,  # 默认值：π/2
                       jnp.arccos(jnp.clip(jnp.dot(ego_heading, los_unit), -0.9999, 0.9999)))
    
    enm_TA = jnp.where(R_is_small,
                       jnp.pi/2,  # 默认值：π/2
                       jnp.arccos(jnp.clip(jnp.dot(enm_heading, -los_unit), -0.9999, 0.9999)))
    
    # 安全计算侧向标志
    side_flag = jnp.where(R_is_small,
                          0.0,  # 当R很小时默认为0
                          jnp.sign(jnp.cross(jnp.array([ego_heading_x, ego_heading_y]), 
                                             jnp.array([delta_x, delta_y]))))
    return ego_AO, enm_TA, R, side_flag

def get_AO_TA_R(ego_feature, enm_feature):
    """Get AO & TA angles and relative distance between two agent.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd)

    Returns:
        (tuple): ego_AO, ego_TA, R
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_velocity = jnp.hstack((ego_vx, ego_vy, ego_vz))
    ego_v = jnp.linalg.norm(ego_velocity)
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_velocity = jnp.hstack((enm_vx, enm_vy, enm_vz))
    enm_v = jnp.linalg.norm(enm_velocity)
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = jnp.linalg.norm(jnp.hstack((delta_x, delta_y, delta_z)))

    proj_dist = delta_x * ego_vx + delta_y * ego_vy + delta_z * ego_vz
    ego_AO = jnp.arccos(jnp.clip(proj_dist / (R * ego_v + 1e-6), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy + delta_z * enm_vz
    ego_TA = jnp.arccos(jnp.clip(proj_dist / (R * enm_v + 1e-6), -1, 1))

    side_flag = jnp.sign(jnp.cross(jnp.hstack((ego_vx, ego_vy)), jnp.hstack((delta_x, delta_y))))
    return ego_AO, ego_TA, R, side_flag


def get2d_AO_TA_R(ego_feature, enm_feature):
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_velocity = jnp.hstack((ego_vx, ego_vy))
    ego_v = jnp.linalg.norm(ego_velocity)
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_velocity = jnp.hstack((enm_vx, enm_vy))
    enm_v = jnp.linalg.norm(enm_velocity)
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = jnp.linalg.norm(jnp.hstack((delta_x, delta_y)))

    proj_dist = delta_x * ego_vx + delta_y * ego_vy
    ego_AO = jnp.arccos(jnp.clip(proj_dist / (R * ego_v + 1e-6), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy
    ego_TA = jnp.arccos(jnp.clip(proj_dist / (R * enm_v + 1e-6), -1, 1))

    side_flag = jnp.sign(jnp.cross(jnp.hstack((ego_vx, ego_vy)), jnp.hstack((delta_x, delta_y))))
    return ego_AO, ego_TA, R, side_flag

def wedge_formation(num_agents, spacing):
    max_layers = num_agents  # 最大层数
    positions = jnp.zeros((num_agents, 3))  # 预分配空间

    def layer_loop(i, carry):
        layers, positions, count = carry
        layer_capacity = 2 * layers
        current_layer = jnp.minimum(num_agents - count, layer_capacity)
        layer_spacing = spacing * (1.0 / layers)

        def agent_loop(j, carry):
            positions, count = carry
            dx = jax.lax.cond(
                j % 2 == 0,
                lambda: -((j // 2) + 1) * layer_spacing,
                lambda: ((j // 2) + 1) * layer_spacing
            )
            dy = layers * spacing
            new_position = jnp.array([dx, dy, 0.0])
            return positions.at[count].set(new_position), count + 1

        positions, count = jax.lax.fori_loop(0, current_layer, agent_loop, (positions, count))
        return layers + 1, positions, count

    # 使用 fori_loop 替代 while_loop
    _, positions, _ = jax.lax.fori_loop(0, max_layers, layer_loop, (1, positions, 0))
    return positions

def line_formation(num_agents, spacing):
    positions = jnp.zeros((num_agents, 3))  # 预分配空间
    start_x = -(num_agents - 1) * spacing / 2
    positions = positions.at[:, 0].set(start_x + spacing * jnp.arange(num_agents))
    return positions

def diamond_formation(num_agents, spacing):
    max_layers = num_agents  # 最大层数
    positions = jnp.zeros((num_agents, 3))  # 预分配空间
    positions = positions.at[0].set(jnp.array([0.0, 0.0, 0.0]))  # 添加长机位置

    def layer_loop(i, carry):
        layer, positions, count = carry
        directions = jnp.array([[-1, 1], [1, 1], [0, 2]])

        def direction_loop(j, carry):
            positions, count = carry
            dx, dy = directions[j]
            new_position = jnp.array([dx * layer * spacing, dy * layer * spacing, 0.0])
            return positions.at[count].set(new_position), count + 1

        positions, count = jax.lax.fori_loop(0, len(directions), direction_loop, (positions, count))
        return layer + 1, positions, count

    # 使用 fori_loop 替代 while_loop
    _, positions, _ = jax.lax.fori_loop(0, max_layers, layer_loop, (1, positions, 1))
    return positions

def enforce_safe_distance(positions, center, safe_distance):
    def agent_loop(i, carry):
        formation_positions, positions = carry
        pos = positions[i] + center

        def distance_loop(j, pos):
            existing = formation_positions[j]
            dist = jnp.linalg.norm(pos - existing)
            return jnp.where(
                dist < safe_distance,
                existing + (pos - existing) * safe_distance / dist,
                pos
            )

        pos = jax.lax.fori_loop(0, i, distance_loop, pos)
        formation_positions = formation_positions.at[i].set(pos)
        return formation_positions, positions

    formation_positions = jnp.zeros_like(positions)
    formation_positions, _ = jax.lax.fori_loop(0, len(positions), agent_loop, (formation_positions, positions))
    return formation_positions


###############################################################
# Planax里面原版本

# import os
# import yaml
# import jax.numpy as jnp
# import jax


# a = 6378137
# b = 6356752.3142
# f = (a - b) / a
# e_sq = f * (2-f)
# pi = jnp.pi


# def parse_config(filename):
#     """Parse F16Sim config file.

#     Args:
#         config (str): config file name

#     Returns:
#         (EnvConfig): a custom class which parsing dict into object.
#     """
#     filepath = os.path.join(get_root_dir(), 'configs', f'{filename}.yaml')
#     assert os.path.exists(filepath), \
#         f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
#     with open(filepath, 'r', encoding='utf-8') as f:
#         config_data = yaml.load(f, Loader=yaml.FullLoader)

#     return type('EnvConfig', (object,), config_data)

# def get_root_dir():
#     return os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')

# def geodetic_to_ecef(lat, lon, h):
#     # (lat, lon) in degrees
#     # h in meters
#     lamb = jnp.radians(lat)
#     phi = jnp.radians(lon)
#     s = jnp.sin(lamb)
#     N = a / jnp.sqrt(1 - e_sq * s * s)

#     sin_lambda = jnp.sin(lamb)
#     cos_lambda = jnp.cos(lamb)
#     sin_phi = jnp.sin(phi)
#     cos_phi = jnp.cos(phi)

#     x = (h + N) * cos_lambda * cos_phi
#     y = (h + N) * cos_lambda * sin_phi
#     z = (h + (1 - e_sq) * N) * sin_lambda

#     return x, y, z

# def ecef_to_enu(x, y, z, lat0, lon0, h0):
#     lamb = jnp.radians(lat0)
#     phi = jnp.radians(lon0)
#     s = jnp.sin(lamb)
#     N = a / jnp.sqrt(1 - e_sq * s * s)
#     sin_lambda = jnp.sin(lamb)
#     cos_lambda = jnp.cos(lamb)
#     sin_phi = jnp.sin(phi)
#     cos_phi = jnp.cos(phi)
#     x0 = (h0 + N) * cos_lambda * cos_phi
#     y0 = (h0 + N) * cos_lambda * sin_phi
#     z0 = (h0 + (1 - e_sq) * N) * sin_lambda
#     xd = x - x0
#     yd = y - y0
#     zd = z - z0
#     t = -cos_phi * xd -  sin_phi * yd
#     xEast = -sin_phi * xd + cos_phi * yd
#     yNorth = t * sin_lambda  + cos_lambda * zd
#     zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd
#     return xEast, yNorth, zUp

# def enu_to_ecef(xEast, yNorth, zUp, lat0, lon0, h0):
#     lamb = jnp.radians(lat0)
#     phi = jnp.radians(lon0)
#     s = jnp.sin(lamb)
#     N = a / jnp.sqrt(1 - e_sq * s * s)
#     sin_lambda = jnp.sin(lamb)
#     cos_lambda = jnp.cos(lamb)
#     sin_phi = jnp.sin(phi)
#     cos_phi = jnp.cos(phi)
#     x0 = (h0 + N) * cos_lambda * cos_phi
#     y0 = (h0 + N) * cos_lambda * sin_phi
#     z0 = (h0 + (1 - e_sq) * N) * sin_lambda
#     t = cos_lambda * zUp - sin_lambda * yNorth
#     zd = sin_lambda * zUp + cos_lambda * yNorth
#     xd = cos_phi * t - sin_phi * xEast 
#     yd = sin_phi * t + cos_phi * xEast
#     x = xd + x0 
#     y = yd + y0 
#     z = zd + z0 
#     return x, y, z

# def ecef_to_geodetic(x, y, z):
#    # Convert from ECEF cartesian coordinates to 
#    # latitude, longitude and height.  WGS-84
#     x2 = x ** 2 
#     y2 = y ** 2 
#     z2 = z ** 2 
#     a = 6378137.0000    # earth radius in meters
#     b = 6356752.3142    # earth semiminor in meters 
#     e = jnp.sqrt (1 - (b / a) ** 2) 
#     b2 = b * b 
#     e2 = e ** 2 
#     ep = e * (a / b) 
#     r = jnp.sqrt(x2 + y2) 
#     r2 = r * r 
#     E2 = a ** 2 - b ** 2 
#     F = 54 * b2 * z2 
#     G = r2 + (1 - e2) * z2 - e2 * E2 
#     c = (e2 * e2 * F * r2) / (G * G * G) 
#     s = (1 + c + jnp.sqrt(c * c + 2 * c)) ** (1 / 3) 
#     P = F / (3 * (s + 1 / s + 1) ** 2 * G * G) 
#     Q = jnp.sqrt(1 + 2 * e2 * e2 * P) 
#     ro = -(P * e2 * r) / (1 + Q) + jnp.sqrt((a * a / 2) * (1 + 1 / Q) - (P * (1 - e2) * z2) / (Q * (1 + Q)) - P * r2 / 2) 
#     tmp = (r - e2 * ro) ** 2 
#     U = jnp.sqrt(tmp + z2) 
#     V = jnp.sqrt(tmp + (1 - e2) * z2) 
#     zo = (b2 * z) / (a * V) 
#     height = U * (1 - b2 / (a * V)) 
#     lat = jnp.atan((z + ep * ep *zo) / r) 
#     temp = jnp.atan(y / x)
#     if x >= 0 :    
#         long = temp 
#     elif (x < 0) & (y >= 0):
#         long = pi + temp 
#     else :
#         long = temp - pi 
#     lat0 = lat/(pi/180) 
#     lon0 = long/(pi/180) 
#     h0 = height
#     return lat0, lon0, h0

# def geodetic_to_enu(lat, lon, h, lat_ref, lon_ref, h_ref):
#     x, y, z = geodetic_to_ecef(lat, lon, h)
#     return ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)

# def enu_to_geodetic(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref):
#     x,y,z = enu_to_ecef(xEast, yNorth, zUp, lat_ref, lon_ref, h_ref)
#     return ecef_to_geodetic(x,y,z)

# def wrap_2PI(angle):
#     res = angle % (2 * jnp.pi)
#     mask1 = res < 0
#     res += 2 * jnp.pi * mask1
#     return res

# def wrap_PI(angle):
#     res = wrap_2PI(angle)
#     mask1 = res > jnp.pi
#     res -= 2 * jnp.pi * mask1
#     return res

# def get_AO_TA_R(ego_feature, enm_feature):
#     """Get AO & TA angles and relative distance between two agent.

#     Args:
#         ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd)

#     Returns:
#         (tuple): ego_AO, ego_TA, R
#     """
#     ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
#     ego_velocity = jnp.hstack((ego_vx, ego_vy, ego_vz))
#     ego_v = jnp.linalg.norm(ego_velocity)
#     enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
#     enm_velocity = jnp.hstack((enm_vx, enm_vy, enm_vz))
#     enm_v = jnp.linalg.norm(enm_velocity)
#     delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
#     R = jnp.linalg.norm(jnp.hstack((delta_x, delta_y, delta_z)))

#     proj_dist = delta_x * ego_vx + delta_y * ego_vy + delta_z * ego_vz
#     ego_AO = jnp.arccos(jnp.clip(proj_dist / (R * ego_v + 1e-6), -1, 1))
#     proj_dist = delta_x * enm_vx + delta_y * enm_vy + delta_z * enm_vz
#     ego_TA = jnp.arccos(jnp.clip(proj_dist / (R * enm_v + 1e-6), -1, 1))

#     side_flag = jnp.sign(jnp.cross(jnp.hstack((ego_vx, ego_vy)), jnp.hstack((delta_x, delta_y))))
#     return ego_AO, ego_TA, R, side_flag


# def get2d_AO_TA_R(ego_feature, enm_feature):
#     ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
#     ego_velocity = jnp.hstack((ego_vx, ego_vy))
#     ego_v = jnp.linalg.norm(ego_velocity)
#     enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
#     enm_velocity = jnp.hstack((enm_vx, enm_vy))
#     enm_v = jnp.linalg.norm(enm_velocity)
#     delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
#     R = jnp.linalg.norm(jnp.hstack((delta_x, delta_y)))

#     proj_dist = delta_x * ego_vx + delta_y * ego_vy
#     ego_AO = jnp.arccos(jnp.clip(proj_dist / (R * ego_v + 1e-6), -1, 1))
#     proj_dist = delta_x * enm_vx + delta_y * enm_vy
#     ego_TA = jnp.arccos(jnp.clip(proj_dist / (R * enm_v + 1e-6), -1, 1))

#     side_flag = jnp.sign(jnp.cross(jnp.hstack((ego_vx, ego_vy)), jnp.hstack((delta_x, delta_y))))
#     return ego_AO, ego_TA, R, side_flag

# def wedge_formation(num_agents, spacing):
#     max_layers = num_agents  # 最大层数
#     positions = jnp.zeros((num_agents, 3))  # 预分配空间

#     def layer_loop(i, carry):
#         layers, positions, count = carry
#         layer_capacity = 2 * layers
#         current_layer = jnp.minimum(num_agents - count, layer_capacity)
#         layer_spacing = spacing * (1.0 / layers)

#         def agent_loop(j, carry):
#             positions, count = carry
#             dx = jax.lax.cond(
#                 j % 2 == 0,
#                 lambda: -((j // 2) + 1) * layer_spacing,
#                 lambda: ((j // 2) + 1) * layer_spacing
#             )
#             dy = layers * spacing
#             new_position = jnp.array([dx, dy, 0.0])
#             return positions.at[count].set(new_position), count + 1

#         positions, count = jax.lax.fori_loop(0, current_layer, agent_loop, (positions, count))
#         return layers + 1, positions, count

#     # 使用 fori_loop 替代 while_loop
#     _, positions, _ = jax.lax.fori_loop(0, max_layers, layer_loop, (1, positions, 0))
#     return positions

# def line_formation(num_agents, spacing):
#     positions = jnp.zeros((num_agents, 3))  # 预分配空间
#     start_x = -(num_agents - 1) * spacing / 2
#     positions = positions.at[:, 0].set(start_x + spacing * jnp.arange(num_agents))
#     return positions

# def diamond_formation(num_agents, spacing):
#     max_layers = num_agents  # 最大层数
#     positions = jnp.zeros((num_agents, 3))  # 预分配空间
#     positions = positions.at[0].set(jnp.array([0.0, 0.0, 0.0]))  # 添加长机位置

#     def layer_loop(i, carry):
#         layer, positions, count = carry
#         directions = jnp.array([[-1, 1], [1, 1], [0, 2]])

#         def direction_loop(j, carry):
#             positions, count = carry
#             dx, dy = directions[j]
#             new_position = jnp.array([dx * layer * spacing, dy * layer * spacing, 0.0])
#             return positions.at[count].set(new_position), count + 1

#         positions, count = jax.lax.fori_loop(0, len(directions), direction_loop, (positions, count))
#         return layer + 1, positions, count

#     # 使用 fori_loop 替代 while_loop
#     _, positions, _ = jax.lax.fori_loop(0, max_layers, layer_loop, (1, positions, 1))
#     return positions

# def enforce_safe_distance(positions, center, safe_distance):
#     def agent_loop(i, carry):
#         formation_positions, positions = carry
#         pos = positions[i] + center

#         def distance_loop(j, pos):
#             existing = formation_positions[j]
#             dist = jnp.linalg.norm(pos - existing)
#             return jnp.where(
#                 dist < safe_distance,
#                 existing + (pos - existing) * safe_distance / dist,
#                 pos
#             )

#         pos = jax.lax.fori_loop(0, i, distance_loop, pos)
#         formation_positions = formation_positions.at[i].set(pos)
#         return formation_positions, positions

#     formation_positions = jnp.zeros_like(positions)
#     formation_positions, _ = jax.lax.fori_loop(0, len(positions), agent_loop, (formation_positions, positions))
#     return formation_positions