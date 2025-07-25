import jax
import jax.numpy as jnp
import numpy as np

class EnvAdapter:
    """
    适配器类，将AeroPlanax环境状态转换为可被PID控制器使用的格式。
    这个类提供了与原始PyTorch控制器相同的接口，但使用JAX数组。
    """
    def __init__(self, state):
        """
        初始化适配器
        Args:
            state: 环境状态对象，包含plane_state等
        """
        self.state = state
        # 将model设为self，以便直接通过env.model()方法访问
        self.model = self
        
    def get_posture(self):
        """获取飞机姿态（横滚、俯仰、偏航）"""
        roll = jnp.array([[self.state.plane_state.roll[0]]])
        pitch = jnp.array([[self.state.plane_state.pitch[0]]])
        yaw = jnp.array([[self.state.plane_state.yaw[0]]])
        return roll, pitch, yaw
            
    def get_TAS(self):
        """获取真空速"""
        return jnp.array([self.state.plane_state.vt[0]])
            
    def get_EAS2TAS(self):
        """获取等效空速到真空速的比值"""
        # 使用1.0作为默认值
        return jnp.array([1.0])
            
    def get_euler_angular_velocity(self):
        """获取欧拉角速度"""
        P = jnp.array([self.state.plane_state.P[0]])
        Q = jnp.array([self.state.plane_state.Q[0]])
        R = jnp.array([self.state.plane_state.R[0]])
        return P, Q, R

    def get_position(self):
        """获取位置（北、东、高度）"""
        north = jnp.array([self.state.plane_state.north[0]])
        east = jnp.array([self.state.plane_state.east[0]])
        altitude = jnp.array([self.state.plane_state.altitude[0]])
        return north, east, altitude
            
    def get_climb_rate(self):
        """获取爬升率"""
        return jnp.array([self.state.plane_state.vel_z[0]])
            
    def get_ground_speed(self):
        """获取地面速度"""
        return jnp.array([self.state.plane_state.vel_x[0]]), jnp.array([self.state.plane_state.vel_y[0]])
            
    def get_acceleration(self):
        """获取加速度"""
        if hasattr(self.state.plane_state, 'ax'):
            return jnp.array([self.state.plane_state.ax[0]]), jnp.array([self.state.plane_state.ay[0]]), jnp.array([self.state.plane_state.az[0]])
        else:
            # 默认为零加速度
            return jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0])
        
    def get_AOA(self):
        """获取攻角"""
        return jnp.array([self.state.plane_state.alpha[0]])
        
    def get_AOS(self):
        """获取侧滑角"""
        return jnp.array([self.state.plane_state.beta[0]])
        
    def get_G(self):
        """获取G值（过载）"""
        # 如果没有具体过载信息，可以通过加速度计算或返回默认值
        return jnp.array([1.0])  # 默认为1G
        
    def get_thrust(self):
        """获取推力"""
        if hasattr(self.state.plane_state, 'T'):
            return jnp.array([self.state.plane_state.T[0]])
        else:
            return jnp.array([0.0])
        
    def get_control_surface(self):
        """获取控制面（升降舵、副翼、方向舵、前缘襟翼）"""
        if hasattr(self.state.plane_state, 'el'):
            el = jnp.array([self.state.plane_state.el[0]])
            ail = jnp.array([self.state.plane_state.ail[0]])
            rud = jnp.array([self.state.plane_state.rud[0]])
            # 前缘襟翼可能不存在
            lef = jnp.array([0.0])
            return el, ail, rud, lef
        else:
            # 默认值
            return jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0])
        
    def get_vt(self):
        """获取真空速"""
        return jnp.array([self.state.plane_state.vt[0]]) 