import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os
import sys
from . import EnvAdapter
from .controller import Controller
from .utils import wrap_PI


def _t2n(x):
    return x.detach().cpu().numpy()

episode_rewards = 0
device = "cuda:0"

env = ControlEnv(num_envs=1, config="heading", model='F16', random_seed=0, device=device)

controller = Controller(dt=env.model.dt, n=env.n, device=device)

print("Start render")
ego_obs = env.reset()
# 状态量
npos, epos, altitude = env.model.get_position()
npos_buf = np.mean(_t2n(npos))
epos_buf = np.mean(_t2n(epos))
altitude_buf = np.mean(_t2n(altitude))

roll, pitch, yaw = env.model.get_posture()
roll_buf = np.mean(_t2n(roll))
pitch_buf = np.mean(_t2n(pitch))
yaw_buf = np.mean(_t2n(yaw))

yaw_rate = env.model.get_extended_state()[:, 5]
yaw_rate_buf = np.mean(_t2n(yaw_rate))

roll_dem_buf = np.array(0)
pitch_dem_buf = np.array(0)
yaw_rate_dem_buf = np.array(0)

vt = env.model.get_vt()
vt_buf = np.mean(_t2n(vt))

alpha = env.model.get_AOA()
alpha_buf = np.mean(_t2n(alpha))

beta = env.model.get_AOS()
beta_buf = np.mean(_t2n(beta))

G = env.model.get_G()
G_buf = np.mean(_t2n(G))
# 控制量
T = env.model.get_thrust()
T_buf = np.mean(_t2n(T))
throttle_buf = np.mean(_t2n(T * 0.3048 / 82339 / 0.225))

el, ail, rud, lef = env.model.get_control_surface()
el_buf = np.mean(_t2n(el))
ail_buf = np.mean(_t2n(ail))
rud_buf = np.mean(_t2n(rud))
# 目标
target_altitude_buf = np.mean(_t2n(env.task.target_altitude))
target_heading_buf = np.mean(_t2n(env.task.target_heading))
target_vt_buf = np.mean(_t2n(env.task.target_vt))

counts = 0
start = time.time()

class HeadingPIDController:
    def __init__(self, sim_freq=50):
        # 创建PID控制器
        self.controller = Controller(dt=1.0/sim_freq, n=1)
        
        # 目标值
        self.target_heading = 0.0
        self.target_altitude = 6000.0  # 默认目标高度 6000m
        self.target_speed = 200.0      # 默认目标速度 200m/s
        
    def get_control(self, state):
        # 从状态中提取当前值
        current_heading = state.plane_state.yaw[0]
        current_altitude = state.plane_state.altitude[0]
        current_speed = state.plane_state.vt[0]
        
        # 创建环境适配器对象
        env = EnvAdapter(state)
        
        # 更新航向控制
        navigation_heading = jnp.array(self.target_heading)
        self.controller.update_heading_hold(navigation_heading, env)
        
        # 计算高度和速度控制
        hgt_dem = jnp.array([[self.target_altitude]])
        TAS_dem = jnp.array([[self.target_speed]])
        self.controller.cal_pitch_throttle(hgt_dem, TAS_dem, env)
        
        # 稳定控制
        self.controller.stabilize(env)
        
        # 获取控制动作
        action_array = self.controller.get_action()
        
        # 转换为JAX数组并调整范围
        throttle = float(action_array[0, 0])  # 范围已在controller中处理
        elevator = float(action_array[0, 1])  # 范围已在controller中处理 
        aileron = float(action_array[0, 2])   # 范围已在controller中处理
        rudder = float(action_array[0, 3])    # 范围已在controller中处理
        
        # 返回控制指令 [油门, 升降舵, 副翼, 方向舵]
        return jnp.array([throttle, elevator, aileron, rudder])

def test(env, env_params, render_dir='./tracks/', max_steps=3000):
    # 初始化PID控制器
    controller = HeadingPIDController(sim_freq=env_params.sim_freq)
    
    # 初始化环境
    rng = jax.random.PRNGKey(42)
    obsv, env_state = env.reset(rng)
    
    # 创建渲染目录
    Path(render_dir).mkdir(parents=True, exist_ok=True)
    
    # 渲染初始状态
    env.render(env_state.env_state, env_params, {'__all__': False}, render_dir)
    
    # 保存结果数据
    results = {
        'time': [],
        'target_heading': [],
        'current_heading': [],
        'heading_error': [],
        'roll': [],
        'pitch': [],
        'altitude': [],
        'speed': [],
        'reward': []
    }
    
    total_reward = 0
    
    # 测试循环
    for step in range(max_steps):
        # 更新目标航向
        if step > 0 and step % 500 == 0:
            # 每500步更新一次目标航向
            rng, _rng = jax.random.split(rng)
            new_target = float(jax.random.uniform(_rng, minval=-np.pi, maxval=np.pi))
            controller.target_heading = new_target
            print(f"新的目标航向: {np.degrees(controller.target_heading):.2f}°")
            
            # 如果有目标高度和速度，也可以更新
            rng, _rng = jax.random.split(rng)
            controller.target_altitude = float(jax.random.uniform(_rng, minval=5000, maxval=7000))
            
            rng, _rng = jax.random.split(rng)
            controller.target_speed = float(jax.random.uniform(_rng, minval=150, maxval=250))
            
            print(f"新的目标高度: {controller.target_altitude:.2f}m, 目标速度: {controller.target_speed:.2f}m/s")
            
        # 获取控制指令
        action = controller.get_control(env_state.env_state)
        
        # 执行环境步进
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, {'agent_0': action})
        
        # 记录数据
        current_time = float(env_state.env_state.time * env_params.agent_interaction_steps / env_params.sim_freq)
        current_heading = float(env_state.env_state.plane_state.yaw[0])
        heading_error = float(wrap_PI(controller.target_heading - current_heading))
        roll = float(env_state.env_state.plane_state.roll[0])
        pitch = float(env_state.env_state.plane_state.pitch[0])
        altitude = float(env_state.env_state.plane_state.altitude[0])
        speed = float(env_state.env_state.plane_state.vt[0])
        
        results['time'].append(current_time)
        results['target_heading'].append(controller.target_heading)
        results['current_heading'].append(current_heading)
        results['heading_error'].append(heading_error)
        results['roll'].append(roll)
        results['pitch'].append(pitch)
        results['altitude'].append(altitude)
        results['speed'].append(speed)
        results['reward'].append(float(reward))
        
        total_reward += float(reward)
        
        # 渲染当前状态
        env.render(env_state.env_state, env_params, done, render_dir)
        
        # 打印信息
        if step % 100 == 0:
            print(f'Time: {current_time:.2f}, Heading Error: {np.degrees(heading_error):.2f}°, '
                  f'Altitude: {altitude:.2f}m, Speed: {speed:.2f}m/s, Reward: {reward:.4f}')
        
        if done['__all__']:
            print("任务结束!")
            break
    
    # 绘制结果
    plot_results(results)
    print(f"总奖励: {total_reward:.2f}")
    
    return results

def plot_results(results):
    # 创建图表目录
    plots_dir = './plots/'
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置图表风格
    plt.style.use('ggplot')
    
    # 创建一个2x2的子图
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    
    # 航向控制图
    axs[0, 0].plot(results['time'], np.degrees(results['target_heading']), 'r-', label='目标航向')
    axs[0, 0].plot(results['time'], np.degrees(results['current_heading']), 'b-', label='当前航向')
    axs[0, 0].set_title('航向控制')
    axs[0, 0].set_xlabel('时间 (秒)')
    axs[0, 0].set_ylabel('航向角 (度)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # 姿态角图
    axs[0, 1].plot(results['time'], np.degrees(results['roll']), 'g-', label='横滚角')
    axs[0, 1].plot(results['time'], np.degrees(results['pitch']), 'm-', label='俯仰角')
    axs[0, 1].set_title('姿态角')
    axs[0, 1].set_xlabel('时间 (秒)')
    axs[0, 1].set_ylabel('角度 (度)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # 高度控制图
    axs[1, 0].plot(results['time'], results['altitude'], 'b-')
    axs[1, 0].set_title('高度控制')
    axs[1, 0].set_xlabel('时间 (秒)')
    axs[1, 0].set_ylabel('高度 (米)')
    axs[1, 0].grid(True)
    
    # 速度控制图
    axs[1, 1].plot(results['time'], results['speed'], 'r-')
    axs[1, 1].set_title('速度控制')
    axs[1, 1].set_xlabel('时间 (秒)')
    axs[1, 1].set_ylabel('速度 (米/秒)')
    axs[1, 1].grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'./plots/pid_control_results_{timestamp}.png')
    plt.close(fig)
