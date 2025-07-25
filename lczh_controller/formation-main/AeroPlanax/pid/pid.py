import jax
import jax.numpy as jnp


class PID:
    def __init__(self, Kp=0, Ki=0, Kd=0, Kff=0, Kimax=0, dt=0.01, n=1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kff = Kff
        self.Kimax = Kimax
        self.dt = dt
        self.n = n
        self.reset = True
        
        # Initialize state variables as JAX arrays
        self.target = jnp.zeros((n, 1))
        self.error = jnp.zeros((n, 1))
        self.derivative = jnp.zeros((n, 1))
        self.integrator = jnp.zeros((n, 1))
    
    def update_all(self, target, measurement, limit):
        # Check for NaN or infinite values
        if jnp.any(jnp.isnan(target)) or jnp.any(jnp.isinf(target)):
            return jnp.zeros((self.n, 1))
        if jnp.any(jnp.isnan(measurement)) or jnp.any(jnp.isinf(measurement)):
            return jnp.zeros((self.n, 1))
            
        if self.reset:
            # First call after reset
            self.target = target
            self.error = target - measurement
            self.derivative = jnp.zeros((self.n, 1))
            self.integrator = jnp.zeros((self.n, 1))
            self.reset = False
        else:
            # Normal update
            last_error = self.error
            self.target = target
            self.error = target - measurement
            self.derivative = (self.error - last_error) / self.dt
            
        # Update integrator
        self.update_i(limit)
        
        # Return PID output
        return self.error * self.Kp + self.derivative * self.Kd + self.integrator

    def update_i(self, limit):
        if self.Ki != 0 and self.dt > 0:
            # Only integrate when not limited or error is reducing
            integrate_mask = ~limit | (self.error * self.dt < 0)
            self.integrator = self.integrator + self.error * self.Ki * self.dt * integrate_mask
            self.integrator = jnp.clip(self.integrator, -self.Kimax, self.Kimax)
        else:
            self.integrator = jnp.zeros((self.n, 1))
    
    def get_p(self):
        return self.error * self.Kp
    
    def get_i(self):
        return self.integrator
    
    def get_d(self):
        return self.derivative * self.Kd
    
    def get_ff(self):
        return self.target * self.Kff
    
    def reset_I(self):
        self.integrator = jnp.zeros((self.n, 1))
