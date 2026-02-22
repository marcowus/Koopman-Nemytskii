import numpy as np
from scipy.integrate import solve_ivp

class TankSystem:
    def __init__(self, A=1.0, a=0.1, g=9.81):
        self.A = A
        self.a = a
        self.g = g

    def dynamics(self, t, h, u):
        """
        Single tank dynamics.
        dh/dt = (1/A) * (u - a*sqrt(2gh))
        """
        h_val = h[0] if isinstance(h, np.ndarray) else h
        u_val = u[0] if (isinstance(u, np.ndarray) and len(u) > 0) else u

        # Prevent sqrt of negative
        if h_val < 0:
            h_val = 0

        dhdt = (1.0 / self.A) * (u_val - self.a * np.sqrt(2 * self.g * h_val))
        return [dhdt]

    def step(self, h0, u, dt=0.1):
        """
        Simulate one time step.
        """
        sol = solve_ivp(lambda t, y: self.dynamics(t, y, u), [0, dt], [h0], t_eval=[dt])
        return sol.y[0, -1]
