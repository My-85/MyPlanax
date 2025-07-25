import jax
import jax.numpy as jnp
import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from . import pid
from .utils import parse_config


class PitchController:
    def __init__(self, config='pitchcontroller', dt=0.01, n=1):
        self.config = parse_config(config)
        Kp = getattr(self.config, 'Kp')
        Ki = getattr(self.config, 'Ki')
        Kd = getattr(self.config, 'Kd')
        Kff = getattr(self.config, 'Kff')
        Kimax = getattr(self.config, 'Kimax')
        self.rate_pid = pid.PID(Kp=Kp, Ki=Ki, Kd=Kd, Kff=Kff, Kimax=Kimax, dt=dt, n=n)
        self.dt = dt
        self.n = n
        self.tau = getattr(self.config, 'tau')
        self.rmax_pos = getattr(self.config, 'rmax_pos')
        self.rmax_neg = getattr(self.config, 'rmax_neg')
        self.roll_ff = getattr(self.config, 'roll_ff')
        self.gravity = getattr(self.config, 'gravity')
        self.last_out = jnp.zeros((self.n, 1))
    
    def get_rate_out(self, desired_rate, scaler, env):
        eas2tas = env.model().get_EAS2TAS().reshape(-1, 1)
        roll_rate, pitch_rate, yaw_rate = env.model().get_euler_angular_velocity()
        pitch_rate = pitch_rate.reshape(-1, 1)
        limit_I = jnp.abs(self.last_out) > 45
        self.rate_pid.update_all(desired_rate * scaler * scaler, pitch_rate * scaler * scaler, limit_I)
        ff_out = self.rate_pid.get_ff() / (scaler * eas2tas + 1e-8)
        # ff_out = self.rate_pid.get_ff() / scaler
        p_out = self.rate_pid.get_p()
        i_out = self.rate_pid.get_i()
        d_out = self.rate_pid.get_d()
        out = ff_out + p_out + i_out + d_out
        out = 180 * out / jnp.pi
        self.last_out = out
        out = jnp.clip(out, -45, 45)
        return out
    
    def get_coordination_rate_offset(self, env):
        TAS = env.model().get_TAS()
        roll, pitch, yaw = env.model().get_posture()
        eas2tas = env.model().get_EAS2TAS().reshape(-1, 1)
        vt = TAS.reshape(-1, 1)
        roll = roll.reshape(-1, 1)
        pitch = pitch.reshape(-1, 1)
        mask1 = jnp.abs(roll) < (jnp.pi / 2)
        mask2 = roll >= (jnp.pi / 2)
        mask3 = roll <= (-jnp.pi / 2)
        roll1 = jnp.clip(roll, -4 * jnp.pi / 9, 4 * jnp.pi / 9)
        roll2 = jnp.clip(roll, 5 * jnp.pi / 9, jnp.pi)
        roll3 = jnp.clip(roll, -jnp.pi, -5 * jnp.pi / 9)
        inverted = ~mask1
        roll = mask1 * roll1 + mask2 * roll2 + mask3 * roll3
        mask1 = jnp.abs(pitch) <= (7 * jnp.pi / 18)
        # w = self.gravity * jnp.tan(roll) / vt * eas2tas
        # Q = w * jnp.sin(pitch)
        # R = w * jnp.cos(roll) * jnp.cos(pitch)
        # rate_offset = mask1 * (Q * jnp.cos(roll) - R * jnp.sin(roll)) * self.roll_ff
        rate_offset = mask1 * jnp.cos(pitch) * jnp.abs(self.gravity / vt * jnp.tan(roll) * jnp.sin(roll) * eas2tas) * self.roll_ff
        rate_offset = rate_offset * ~inverted - rate_offset * inverted
        return inverted, rate_offset

    def get_servo_out(self, angle_err, scaler, env):
        if self.tau < 0.05:
            self.tau = 0.05
        desired_rate = angle_err / self.tau
        inverted, rate_offset = self.get_coordination_rate_offset(env)
        desired_rate1 = desired_rate + rate_offset
        if self.rmax_pos:
            desired_rate1 = jnp.clip(desired_rate1, max=self.rmax_pos)
        if self.rmax_neg:
            desired_rate1 = jnp.clip(desired_rate1, min=-self.rmax_neg)
        desired_rate = ~inverted * desired_rate1 + inverted * (rate_offset - desired_rate)

        roll, pitch, yaw = env.model().get_posture()
        roll = roll.reshape(-1, 1)
        roll_wrapped = jnp.abs(roll)
        pitch = pitch.reshape(-1, 1)
        pitch_wrapped = jnp.abs(pitch)
        mask = roll_wrapped > (jnp.pi / 2)
        roll_wrapped = mask * (jnp.pi - roll_wrapped) + (~mask) * roll_wrapped
        mask = (roll_wrapped > (5 * jnp.pi / 18)) & (pitch_wrapped < (7 * jnp.pi / 18))
        roll_prop = (roll_wrapped - 5 * jnp.pi / 18) / (4 * jnp.pi / 18)
        roll_prop = roll_prop * mask
        desired_rate = desired_rate * (1 - roll_prop)
        return self.get_rate_out(desired_rate, scaler, env)
