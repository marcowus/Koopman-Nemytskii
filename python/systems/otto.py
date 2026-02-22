import numpy as np
from scipy.integrate import solve_ivp

class OttoCycle:
    def __init__(self, J=0.5, Vm=0.005, Vd=0.002, RT=287*300):
        self.J = J # Inertia
        self.Vm = Vm # Manifold volume
        self.Vd = Vd # Displacement volume
        self.RT = RT # Gas constant * Temperature
        self.Cd = 0.6
        self.Ath_max = 0.001
        self.P_amb = 101325
        self.T_amb = 300
        self.eta_vol = 0.9
        self.Q_LHV = 44e6
        self.AFR = 14.7
        self.eta_th = 0.3 # Thermal efficiency

    def flow_throttle(self, theta, Pm):
        # Choked flow approximation
        pr = Pm / self.P_amb
        if pr < 0.528:
            psi = np.sqrt(1.4 * (2/(1.4+1))**((1.4+1)/(1.4-1)))
        else:
            psi = np.sqrt(2 * 1.4 / (1.4-1) * (pr**(2/1.4) - pr**((1.4+1)/1.4)))

        # Area function of throttle angle (linear approx)
        Ath = self.Ath_max * (1 - np.cos(theta))

        m_dot = self.Cd * Ath * self.P_amb / np.sqrt(self.RT) * psi
        return m_dot

    def torque_ind(self, Pm, w):
        # Indicated torque from fuel energy
        # m_cyl depends on Pm and w
        m_cyl_dot = self.eta_vol * self.Vd * w / (4 * np.pi * self.RT) * Pm
        m_fuel_dot = m_cyl_dot / self.AFR
        power = self.eta_th * m_fuel_dot * self.Q_LHV
        if w > 1e-1:
            torque = power / w
        else:
            torque = 0
        return torque

    def dynamics(self, t, x, u):
        """
        x: [Pm, w] (Manifold Pressure, Engine Speed)
        u: [theta, T_load] (Throttle angle, Load torque)
        """
        Pm, w = x
        theta, T_load = u

        # Manifold Dynamics
        m_in = self.flow_throttle(theta, Pm)
        m_out = self.eta_vol * self.Vd * w / (4 * np.pi * self.RT) * Pm
        dPm = (self.RT / self.Vm) * (m_in - m_out)

        # Rotational Dynamics
        T_ind = self.torque_ind(Pm, w)
        dw = (T_ind - T_load) / self.J

        return [dPm, dw]

    def step(self, x0, u, dt=0.01):
        """
        Simulate one time step.
        """
        sol = solve_ivp(lambda t, y: self.dynamics(t, y, u), [0, dt], x0, t_eval=[dt])
        return sol.y[:, -1]
