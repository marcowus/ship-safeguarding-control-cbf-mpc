import numpy as np
from .utils import ssa

class PIDController:
    def __init__(self, wn, zeta, T_f, dt):
        """
        Nonlinear MIMO PID regulator for dynamic positioning (DP).

        Args:
            wn: Natural frequency (3x3 diagonal or scalar)
            zeta: Damping ratio (3x3 diagonal or scalar)
            T_f: Filter time constant
            dt: Sampling time
        """
        self.wn = np.array(wn)
        self.zeta = np.array(zeta)
        self.T_f = T_f
        self.dt = dt

        self.eta_d = np.zeros(3)
        self.z_int = np.zeros(3)

    def step(self, eta, nu, eta_ref, M):
        """
        Compute control input tau.

        Args:
            eta: Position [N, E, psi] (3,)
            nu: Velocity [u, v, r] (3,)
            eta_ref: Reference position [N_ref, E_ref, psi_ref] (3,)
            M: Mass matrix (3x3)

        Returns:
            tau: Control force [Tx, Ty, Tn] (3,)
        """

        # Ensure inputs are numpy arrays
        eta = np.array(eta).flatten()
        nu = np.array(nu).flatten()
        eta_ref = np.array(eta_ref).flatten()

        # Rotation matrix in yaw
        psi = eta[2]
        c = np.cos(psi)
        s = np.sin(psi)
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

        # MIMO pole placement
        M_diag = np.diag(np.diag(M))

        # Kp = M_diag * wn^2
        Kp = M_diag @ (self.wn @ self.wn)

        # Kd = M_diag * 2 * zeta * wn
        Kd = M_diag @ (2 * self.zeta @ self.wn)

        # Ki = 0.1 * Kp * wn
        Ki = 0.1 * Kp @ self.wn

        # Error
        e = eta - self.eta_d
        e[2] = ssa(e[2])

        # Control law
        # tau_PID = -R' * ( Kp * e + Ki * z_int ) - Kd * nu
        term1 = Kp @ e + Ki @ self.z_int
        term2 = Kd @ nu
        tau_PID = -R.T @ term1 - term2

        # Update integral state
        # z_int = z_int + dt * (eta - eta_d)
        # Note: In the original MATLAB code, ssa is not applied to (eta - eta_d) here.
        self.z_int = self.z_int + self.dt * (eta - self.eta_d)

        # Update low-pass filter
        # eta_d = eta_d + dt * (eta_ref - eta_d)/ T_f
        self.eta_d = self.eta_d + self.dt * (eta_ref - self.eta_d) / self.T_f

        return tau_PID
