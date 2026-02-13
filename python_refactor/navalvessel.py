import numpy as np
from .params import get_params

def thrConfig(config, lx, ly):
    """
    Compute thrust configuration matrix T_thr.

    Args:
        config: List of thruster types ('T', 'T', alpha1, alpha2)
        lx: x-coordinates of thrusters
        ly: y-coordinates of thrusters

    Returns:
        T_thr: 3x4 thrust configuration matrix
    """
    n_thrusters = len(lx)
    T_thr = np.zeros((3, n_thrusters))

    for i in range(n_thrusters):
        # Determine force direction
        if config[i] == 'T':
            # Tunnel thruster, force in Y (sway)
            fx = 0
            fy = 1
        else:
            # Azimuth thruster
            alpha = config[i]
            fx = np.cos(alpha)
            fy = np.sin(alpha)

        # Moment
        # N = x * Fy - y * Fx
        mz = lx[i] * fy - ly[i] * fx

        T_thr[0, i] = fx
        T_thr[1, i] = fy
        T_thr[2, i] = mz

    return T_thr

def navalvessel(x, ui, Vc, betaVc):
    """
    Nonlinear maneuvering model.

    Args:
        x: State vector [u, v, p, r, phi, psi] (6,)
        ui: Control input [n1, n2, n3, n4, alpha1, alpha2] (6,)
        Vc: Current speed
        betaVc: Current direction

    Returns:
        xdot: State derivative (6,)
    """
    # Get parameters
    h = get_params()

    # State unpacking
    u = x[0]
    v = x[1]
    p = x[2]
    r = x[3]
    b = x[4] # phi
    psi = x[5]

    # Current velocity in body frame
    u_c = Vc * np.cos(betaVc - psi)
    v_c = Vc * np.sin(betaVc - psi)

    u_rel = u - u_c
    v_rel = v - v_c

    # Auxiliary variables
    au = np.abs(u_rel)
    av = np.abs(v_rel)
    ar = np.abs(r)
    ap = np.abs(p)

    # Mass Matrix M
    M = np.zeros((6, 6))
    M[0, 0] = h['m'] - h['Xudot']

    M[1, 1] = h['m'] - h['Yvdot']
    M[1, 2] = -(h['m'] * h['zG'] + h['Ypdot'])
    M[1, 3] = h['m'] * h['xG'] - h['Yrdot']

    M[2, 1] = -(h['m'] * h['zG'] + h['Kvdot'])
    M[2, 2] = h['Ixx'] - h['Kpdot']
    M[2, 3] = -h['Krdot']

    M[3, 1] = h['m'] * h['xG'] - h['Nvdot']
    M[3, 2] = -h['Npdot']
    M[3, 3] = h['Izz'] - h['Nrdot']

    M[4, 4] = 1.0
    M[5, 5] = 1.0

    # Thrust configuration
    K_max = np.array([300e3, 300e3, 655e3, 655e3])
    n_max = np.array([140, 140, 150, 150])
    K_thr = np.diag(K_max / n_max**2)

    l_x = [37, 35, -51.5/2, -51.5/2]
    l_y = [0, 0, 7, -7]

    # Thrust inputs
    n_prop = ui[0:4]
    alpha = ui[4:6]

    # u_thr = abs(n) * n (element-wise)
    u_thr = np.abs(n_prop) * n_prop

    # Thruster configuration
    config = ['T', 'T', alpha[0], alpha[1]]
    T_thr_mat = thrConfig(config, l_x, l_y)

    tau_3dof = T_thr_mat @ K_thr @ u_thr # [Xe, Ye, Ne]

    Xe = tau_3dof[0]
    Ye = tau_3dof[1]
    Ke = 0
    Ne = tau_3dof[2]

    # Hydrodynamic forces
    Xh = h['Xuau'] * u_rel * au + h['Xvr'] * v_rel * r

    Yh = (h['Yauv'] * au * v_rel + h['Yur'] * u_rel * r + h['Yvav'] * v_rel * av +
          h['Yvar'] * v_rel * ar + h['Yrav'] * r * av +
          h['Ybauv'] * b * np.abs(u_rel * v_rel) + h['Ybaur'] * b * np.abs(u_rel * r) +
          h['Ybuu'] * b * u_rel**2)

    Kh = (h['Kauv'] * au * v_rel + h['Kur'] * u_rel * r + h['Kvav'] * v_rel * av +
          h['Kvar'] * v_rel * ar + h['Krav'] * r * av +
          h['Kbauv'] * b * np.abs(u_rel * v_rel) + h['Kbaur'] * b * np.abs(u_rel * r) +
          h['Kbuu'] * b * u_rel**2 + h['Kaup'] * au * p +
          h['Kpap'] * p * ap + h['Kp'] * p + h['Kbbb'] * b**3 -
          (h['rho_water'] * h['g'] * h['gm'] * h['disp']) * b)

    Nh = (h['Nauv'] * au * v_rel + h['Naur'] * au * r + h['Nrar'] * r * ar +
          h['Nrav'] * r * av + h['Nbauv'] * b * np.abs(u_rel * b) +
          h['Nbuar'] * b * u_rel * ar + h['Nbuau'] * b * u_rel * au)

    # Rigid-body centripetal accelerations
    Xc = h['m'] * (r * v + h['xG'] * r**2 - h['zG'] * p * r)
    Yc = -h['m'] * u * r
    Kc = h['m'] * h['zG'] * u * r
    Nc = -h['m'] * h['xG'] * u * r

    F1 = Xh + Xc + Xe
    F2 = Yh + Yc + Ye
    F4 = Kh + Kc + Ke
    F6 = Nh + Nc + Ne

    # RHS vector
    RHS = np.array([F1, F2, F4, F6, p, r])

    # Solve M * xdot = RHS
    xdot = np.linalg.solve(M, RHS)

    return xdot
