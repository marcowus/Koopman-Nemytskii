import numpy as np
from scipy.integrate import solve_ivp

class LandauStuart:
    def __init__(self, mu=0.1, omega=1.0):
        self.mu = mu
        self.omega = omega

    def dynamics(self, t, z, u=0):
        """
        z: [x, y]
        u: control input (scalar or vector)
        """
        x, y = z
        r2 = x**2 + y**2

        dxdt = (self.mu - r2) * x - self.omega * y
        dydt = (self.mu - r2) * y + self.omega * x

        # Add control if provided
        if np.isscalar(u) or len(u) == 1:
            # Assume control acts on x
            dxdt += u
        elif len(u) == 2:
            dxdt += u[0]
            dydt += u[1]

        return [dxdt, dydt]

    def simulate(self, z0, t_span, dt, u=None):
        """
        Simulate the system.
        z0: initial state
        t_span: (t_start, t_end)
        dt: time step
        u: control input function u(t) or constant
        """
        t_eval = np.arange(t_span[0], t_span[1], dt)

        def func(t, z):
            if callable(u):
                ut = u(t)
            elif u is not None:
                ut = u
            else:
                ut = 0
            return self.dynamics(t, z, ut)

        sol = solve_ivp(func, t_span, z0, t_eval=t_eval, method='RK45')
        return sol.t, sol.y.T
