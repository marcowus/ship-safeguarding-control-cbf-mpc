import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from python_refactor.params import get_params
from python_refactor.pid_controller import PIDController
from python_refactor.navalvessel import navalvessel
from python_refactor.utils import Rzyx

def run_simulation():
    # Get parameters
    params = get_params()

    # Simulation settings
    sim_t = 60
    dt = 0.05

    # Obstacles
    xo_1 = -10
    yo_1 = -10
    xo_2 = -7
    yo_2 = -2
    d = 3

    # CBF parameters
    cbf_gamma0 = 2
    cbf_gamma0_2 = 2
    cbf_rate = 0.2
    cbf_rate2 = 0.15

    time_interval1 = 15

    Vc = 0
    betaVc = np.deg2rad(-45)

    # --- Build Symbolic CBF-QP Solver ---

    # States
    u = ca.SX.sym('u')
    v = ca.SX.sym('v')
    p = ca.SX.sym('p')
    r = ca.SX.sym('r')
    b = ca.SX.sym('b') # phi
    n = ca.SX.sym('n') # psi
    x_p = ca.SX.sym('x_p')
    y_p = ca.SX.sym('y_p')

    x1 = ca.vertcat(u, v, p, r, b, n, x_p, y_p)

    # Auxiliary variables
    au = ca.fabs(u)
    av = ca.fabs(v)
    ar = ca.fabs(r)
    ap = ca.fabs(p)

    # Mass-Inertia Matrix
    M = ca.SX.zeros(6, 6)
    M[0, 0] = params['m'] - params['Xudot']

    M[1, 1] = params['m'] - params['Yvdot']
    M[1, 2] = -(params['m'] * params['zG'] + params['Ypdot'])
    M[1, 3] = params['m'] * params['xG'] - params['Yrdot']

    M[2, 1] = -(params['m'] * params['zG'] + params['Kvdot'])
    M[2, 2] = params['Ixx'] - params['Kpdot']
    M[2, 3] = -params['Krdot']

    M[3, 1] = params['m'] * params['xG'] - params['Nvdot']
    M[3, 2] = -params['Npdot']
    M[3, 3] = params['Izz'] - params['Nrdot']

    M[4, 4] = 1.0
    M[5, 5] = 1.0

    # Hydrodynamic + Centripetal
    Xh = params['Xuau']*u*au + params['Xvr']*v*r

    Yh = (params['Yauv']*au*v + params['Yur']*u*r + params['Yvav']*v*av + params['Yvar']*v*ar + params['Yrav']*r*av +
          params['Ybauv']*b*ca.fabs(u*v) + params['Ybaur']*b*ca.fabs(u*r) + params['Ybuu']*b*u**2)

    Kh = (params['Kauv']*au*v + params['Kur']*u*r + params['Kvav']*v*av + params['Kvar']*v*ar + params['Krav']*r*av +
          params['Kbauv']*b*ca.fabs(u*v) + params['Kbaur']*b*ca.fabs(u*r) + params['Kbuu']*b*u**2 + params['Kaup']*au*p +
          params['Kpap']*p*ap + params['Kp']*p + params['Kbbb']*b**3 -
          (params['rho_water']*params['g']*params['gm']*params['disp'])*b)

    Nh = (params['Nauv']*au*v + params['Naur']*au*r + params['Nrar']*r*ar + params['Nrav']*r*av +
          params['Nbauv']*b*ca.fabs(u*b) + params['Nbuar']*b*u*ar + params['Nbuau']*b*u*au)

    Xc =  params['m']*(r*v + params['xG']*r**2 - params['zG']*p*r)
    Yc = -params['m']*u*r
    Kc =  params['m']*params['zG']*u*r
    Nc = -params['m']*params['xG']*u*r

    F1 = Xh + Xc
    F2 = Yh + Yc
    F4 = Kh + Kc
    F6 = Nh + Nc

    # f in x_dot = f(x) + g(x)*tau
    # sola = M \ [F1; F2; F4; F6; p; r]
    RHS = ca.vertcat(F1, F2, F4, F6, p, r)
    sola = ca.solve(M, RHS)

    # Position kinematics
    x_dot_y_dot = ca.vertcat(u*ca.cos(n) - v*ca.sin(n),
                             u*ca.sin(n) + v*ca.cos(n))

    f = ca.vertcat(sola, x_dot_y_dot)     # 8x1 drift vector

    # g in x_dot = f(x) + g(x)*tau
    Minv = ca.inv(M)
    g_body = Minv[:, [0, 1, 3]]  # we have forces/moment in Xe, Ye, Ne (surge, sway, yaw) (Columns 0, 1, 3)

    g = ca.vertcat(g_body, ca.SX.zeros(2, 3))   # extend to 8x3

    # CBFs and Lie Derivatives
    e1_1 = x_p - xo_1
    e2_1 = y_p - yo_1
    e_1 = ca.vertcat(e1_1, e2_1)

    B1_1 = -(e1_1)**2 - (e2_1)**2 + d**2    # B1 for obstacle 1
    B1_1_dot = -2 * ca.mtimes(e_1.T, x_dot_y_dot)

    cbf_1 = B1_1_dot + cbf_gamma0 * B1_1   # B2 for obstacle 1

    e1_2 = x_p - xo_2
    e2_2 = y_p - yo_2
    e_2 = ca.vertcat(e1_2, e2_2)

    B1_2 = -(e1_2)**2 - (e2_2)**2 + d**2  # B1 for obstacle 2
    B1_2_dot = -2*ca.mtimes(e_2.T, x_dot_y_dot)

    cbf_2 = B1_2_dot + cbf_gamma0_2 * B1_2  # B2 for obstacle 2

    # Lie Derivatives
    dcbf_1 = ca.jacobian(cbf_1, x1)
    lf_cbf_1 = ca.mtimes(dcbf_1, f)
    lg_cbf_1 = ca.mtimes(dcbf_1, g)

    cbf_1_func = ca.Function('cbf1', [x1], [cbf_1])
    B1_1_func = ca.Function('B1_1', [x1], [B1_1])

    dcbf_2 = ca.jacobian(cbf_2, x1)
    lf_cbf_2 = ca.mtimes(dcbf_2, f)
    lg_cbf_2 = ca.mtimes(dcbf_2, g)

    cbf_2_func = ca.Function('cbf2', [x1], [cbf_2])
    B1_2_func = ca.Function('B1_2', [x1], [B1_2])

    # CBF-QP problem setup
    tau_var = ca.SX.sym('tau', 3)  # Optimization variables
    tau_ref = ca.SX.sym('tau_ref', 3)

    P = ca.vertcat(x1, tau_ref)

    # ---------------Create CBF solver for subproblem 1-------------------------
    A1 = lg_cbf_1
    b1 = lf_cbf_1 + cbf_rate * cbf_1

    H = ca.SX.eye(3)
    obj = ca.mtimes([(tau_var - tau_ref).T, H, (tau_var - tau_ref)])

    # Constraint: A*tau + b <= 0.
    # In qpsol: g = A*x. lb <= g <= ub.
    # We want lg_cbf_1 * tau + b1 <= 0.
    # So lg_cbf_1 * tau <= -b1.
    # Wait, MATLAB: g: A*tau + b. lbg = -inf, ubg = 0.
    # A*tau + b <= 0. Correct.

    prob_struct1 = {'x': tau_var, 'f': obj, 'g': ca.mtimes(A1, tau_var) + b1, 'p': P}
    opts = {'printLevel': 'none'}

    solver1 = ca.qpsol('solver', 'qpoases', prob_struct1, opts)

    # ---------------Create CBF solver for subproblem 2-------------------------
    A2 = lg_cbf_2
    b2 = lf_cbf_2 + cbf_rate2 * cbf_2

    prob_struct2 = {'x': tau_var, 'f': obj, 'g': ca.mtimes(A2, tau_var) + b2, 'p': P}

    solver1_2 = ca.qpsol('solver', 'qpoases', prob_struct2, opts)

    # -----------Build control allocation symbolic below this line ---------%%
    alpha = ca.SX.sym('alpha', 2)
    u_alloc = ca.SX.sym('u', 4)
    s = ca.SX.sym('s', 3)
    x2 = ca.vertcat(alpha, u_alloc, s)    # Optimization variables

    az_max = np.deg2rad(60)
    l_x = [37, 35, -51.5/2, -51.5/2]
    l_y = [0, 0, 7, -7]
    K_max_val = [300e3, 300e3, 655e3, 655e3]
    K_max = ca.diag(K_max_val)
    n_max = np.array([140, 140, 150, 150])

    lbx = np.concatenate([[-az_max, -az_max], [-1]*4, [-np.inf]*3])
    ubx = np.concatenate([[az_max, az_max], [1]*4, [np.inf]*3])

    alpha_old = ca.SX.sym('alpha_old', 2)
    u_old = ca.SX.sym('u_old', 4)
    tau_pid = ca.SX.sym('tau_pid', 3)

    P_alloc = ca.vertcat(alpha_old, u_old, tau_pid)

    w1 = 1
    w2 = 100
    w3 = 1
    w4 = 0.1

    obj_alloc = w1 * ca.norm_2(u_alloc)**2 + w2 * ca.norm_2(s)**2 + w3 * ca.norm_2(alpha - alpha_old)**2 + w4 * ca.norm_2(u_alloc - u_old)**2

    max_rate_alpha = 0.3
    max_rate_u = 0.1

    # Constraints
    c1 = (alpha[0] - alpha_old[0]) / dt - max_rate_alpha
    c2 = -(alpha[0] - alpha_old[0]) / dt - max_rate_alpha
    c3 = (alpha[1] - alpha_old[1]) / dt - max_rate_alpha
    c4 = -(alpha[1] - alpha_old[1]) / dt - max_rate_alpha

    c5 = (u_alloc[0] - u_old[0]) / dt - max_rate_u
    c6 = -(u_alloc[0] - u_old[0]) / dt - max_rate_u
    c7 = (u_alloc[1] - u_old[1]) / dt - max_rate_u
    c8 = -(u_alloc[1] - u_old[1]) / dt - max_rate_u
    c9 = (u_alloc[2] - u_old[2]) / dt - max_rate_u
    c10 = -(u_alloc[2] - u_old[2]) / dt - max_rate_u
    c11 = (u_alloc[3] - u_old[3]) / dt - max_rate_u
    c12 = -(u_alloc[3] - u_old[3]) / dt - max_rate_u

    # Equality constraint
    T_thr = ca.SX.zeros(3, 4)
    T_thr[:, 0] = ca.vertcat(0, 1, l_x[0])
    T_thr[:, 1] = ca.vertcat(0, 1, l_x[1])
    # T_thr[:,2] = [cos(alpha(1)) sin(alpha(1)) l_x(3)*sin(alpha(1))-l_y(3)*cos(alpha(1))]';
    T_thr[:, 2] = ca.vertcat(ca.cos(alpha[0]), ca.sin(alpha[0]), l_x[2]*ca.sin(alpha[0]) - l_y[2]*ca.cos(alpha[0]))
    T_thr[:, 3] = ca.vertcat(ca.cos(alpha[1]), ca.sin(alpha[1]), l_x[3]*ca.sin(alpha[1]) - l_y[3]*ca.cos(alpha[1]))

    ceq = ca.mtimes([T_thr, K_max, u_alloc]) - tau_pid + s

    g_alloc = ca.vertcat(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, ceq)

    # lb for c1-c11: -inf, ub: 0
    # ceq: 0 to 0
    lbg = np.concatenate([[-np.inf]*12, [0, 0, 0]])
    ubg = np.concatenate([[0]*12, [0, 0, 0]])

    prob_struct_alloc = {'x': x2, 'f': obj_alloc, 'g': g_alloc, 'p': P_alloc}

    opts_alloc = {
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.max_iter': 100,
        'ipopt.tol': 1e-6
    }

    solver2 = ca.nlpsol('solver', 'ipopt', prob_struct_alloc, opts_alloc)

    x0_alloc = np.array([np.deg2rad(-28), np.deg2rad(28), 0, 0, 0, 0, 0, 0, 0])

    # --- Initial State & Loop ---

    x_ref = 0
    y_ref = 0
    psi_ref = np.deg2rad(0)
    eta_ref = np.array([x_ref, y_ref, psi_ref])

    wn = 0.1 * np.diag([1, 1, 3])
    zeta = 1.0 * np.diag([1, 1, 1])
    T_f = 30

    pid = PIDController(wn, zeta, T_f, dt)

    total_k = int(np.ceil(sim_t / dt))
    t = 0

    x = np.array([0, 0, 0, 0, 0, np.deg2rad(0)]) # [u v p r phi psi]
    eta = np.array([-15, -15, np.deg2rad(0)]) # [N, E, psi]

    alpha_old_val = np.deg2rad([-28, 28])
    u_old_val = np.array([0, 0, 0, 0])

    # Storage
    ts = np.zeros(total_k)
    xs = np.zeros((total_k, 6))
    etas = np.zeros((total_k, 3))
    us = np.zeros((total_k-1, 3))
    bs1 = np.zeros(total_k-1)
    bs2 = np.zeros(total_k-1)
    hs1 = np.zeros(total_k-1)
    hs2 = np.zeros(total_k-1)
    uis = np.zeros((total_k-1, 6))

    xs[0] = x
    etas[0] = eta
    ts[0] = t

    print("Starting simulation...")

    for k in range(total_k - 1):
        if k % 100 == 0:
            print(f"Progress: {k/total_k*100:.1f}%")

        nu = np.array([x[0], x[1], x[3]]) # [u, v, r]

        # M used for PID is 3x3 diagonal from 6x6 M
        # M([1 2 4],[1 2 4]) in MATLAB (indices 1, 2, 4) -> 0, 1, 3 in Python
        # Note: M is symbolic (SX), but it contains constants.
        # We need numerical M for PID.
        M_num = ca.DM(M).full()
        M_pid = M_num[[0, 1, 3]][:, [0, 1, 3]]

        tau_pid_val = pid.step(eta, nu, eta_ref, M_pid)

        # CBF-QP Solver
        # P = [x; eta(1); eta(2); tau_pid]
        # x is [u v p r phi psi x_p y_p] ?
        # Wait, in MATLAB: x1 = [u; v; p; r; b; n; x_p; y_p];
        # In loop: `sol1 = solver1('p', [x; eta(1); eta(2); tau_pid], ...)`
        # x passed is [u v p r phi psi] (6x1).
        # eta(1), eta(2) are N, E.
        # So input to solver P corresponds to:
        # u, v, p, r, b (phi), n (psi), x_p, y_p, tau_pid
        # x_p -> eta(1) (North/x), y_p -> eta(2) (East/y).

        p_val = np.concatenate([x, [eta[0], eta[1]], tau_pid_val])

        if ts[k] < time_interval1:
            sol1 = solver1(p=p_val, lbg=-np.inf, ubg=0)
        else:
            sol1 = solver1_2(p=p_val, lbg=-np.inf, ubg=0)

        tau_opt = sol1['x'].full().flatten()
        us[k] = tau_opt

        # Log CBF values
        # B1_1_func([x; eta(1); eta(2)])
        x_aug = np.concatenate([x, [eta[0], eta[1]]])
        bs1[k] = B1_1_func(x_aug).full().item()
        bs2[k] = B1_2_func(x_aug).full().item()
        hs1[k] = cbf_1_func(x_aug).full().item()
        hs2[k] = cbf_2_func(x_aug).full().item()

        # Control Allocation
        # P = [alpha_old; u_old; tau_opt]
        p_alloc = np.concatenate([alpha_old_val, u_old_val, tau_opt])

        sol2 = solver2(x0=x0_alloc, p=p_alloc, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        sol_opt = sol2['x'].full().flatten()
        alpha_c = sol_opt[0:2]
        u_c = sol_opt[2:6]

        alpha_old_val = alpha_c
        u_old_val = u_c

        # Controls ui = [n_c, alpha_c]
        # u_c = n_max^2 * u_c (normalized u_c?)
        # In MATLAB: u_c = n_max.^2 .* u_c;
        # n_c = sign(u_c) .* sqrt(abs(u_c));
        u_c_scaled = (n_max**2) * u_c
        n_c = np.sign(u_c_scaled) * np.sqrt(np.abs(u_c_scaled))

        ui = np.concatenate([n_c, alpha_c])
        uis[k] = ui

        # Dynamics
        # ode45 over [t, t+dt]
        # We can implement a simple RK4 or Euler.
        # MATLAB uses ode45.

        def dynamics(t_inner, x_inner):
            return navalvessel(x_inner, ui, Vc, betaVc)

        # RK4
        k1 = dynamics(t, x)
        k2 = dynamics(t + 0.5*dt, x + 0.5*dt*k1)
        k3 = dynamics(t + 0.5*dt, x + 0.5*dt*k2)
        k4 = dynamics(t + dt, x + dt*k3)

        x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        # Update eta
        # eta = eta + dt * Rzyx * [u, v, r]
        # Rzyx(0, 0, x[5])
        # x[5] is psi from previous step.
        R = Rzyx(0, 0, x[5])
        vel_body = np.array([x[0], x[1], x[3]])
        eta_dot = R @ vel_body
        eta_next = eta + dt * eta_dot

        x = x_next
        eta = eta_next
        t = t + dt

        xs[k+1] = x
        etas[k+1] = eta
        ts[k+1] = t

    print("Simulation finished.")

    # Save results to NPZ
    np.savez("python_refactor/simulation_results.npz", ts=ts, xs=xs, etas=etas, us=us, bs1=bs1, bs2=bs2, hs1=hs1, hs2=hs2, uis=uis)

    # Plotting
    plot_results(ts, xs, etas, us, bs1, bs2, hs1, hs2, uis, xo_1, yo_1, xo_2, yo_2, d)

def plot_results(ts, xs, etas, us, bs1, bs2, hs1, hs2, uis, xo_1, yo_1, xo_2, yo_2, d):
    # Figure 1: Trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(etas[0, 1], etas[0, 0], 'bo', label='Start Ship')
    plt.plot(yo_1, xo_1, 'ro', label='Obstacle 1 Origin')
    plt.plot(etas[:, 1], etas[:, 0], 'b-', linewidth=2, label='Ship Trajectory')
    plt.plot(yo_2, xo_2, 'r*', label='Obstacle 2 Origin')

    theta = np.linspace(0, 2*np.pi, 200)
    circle_x = yo_1 + d * np.cos(theta)
    circle_y = xo_1 + d * np.sin(theta)
    plt.plot(circle_x, circle_y, 'r--', label='Safety Radius (3 m)')

    circle_x_2 = yo_2 + d * np.cos(theta)
    circle_y_2 = xo_2 + d * np.sin(theta)
    plt.plot(circle_x_2, circle_y_2, 'r--')

    plt.xlabel('East [m]')
    plt.ylabel('North [m]')
    plt.legend()
    plt.title('Trajectory of Evasive Ship and Obstacle')
    plt.grid(True)
    plt.savefig('python_refactor/trajectory.png')

    # Figure 2: Control Inputs
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(ts[:-1], us[:, 0], 'b')
    plt.title('Surge Control Force (Xe)')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(ts[:-1], us[:, 1], 'g')
    plt.title('Sway Control Force (Ye)')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(ts[:-1], np.zeros_like(us[:, 2]), 'm')
    plt.title('Roll Control Moment (Ke)')
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(ts[:-1], us[:, 2], 'r')
    plt.title('Yaw Control Moment (Ne)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('python_refactor/control_inputs.png')

    # Figure 3: CBF
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(ts[:-1], bs1, 'r')
    plt.title('CBF (B1) Obs 1')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(ts[:-1], bs2, 'r')
    plt.title('CBF (B1) Obs 2')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(ts[:-1], hs1, 'b')
    plt.title('CBF (B2) Obs 1')
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(ts[:-1], hs2, 'b')
    plt.title('CBF (B2) Obs 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('python_refactor/cbf.png')

    # Check violations
    dist1 = np.sqrt((etas[:,0] - xo_1)**2 + (etas[:,1] - yo_1)**2)
    dist2 = np.sqrt((etas[:,0] - xo_2)**2 + (etas[:,1] - yo_2)**2)

    viol1 = dist1 < d
    viol2 = dist2 < d

    if np.any(viol1):
        print("Violation Obs 1 at:", ts[viol1])
    if np.any(viol2):
        print("Violation Obs 2 at:", ts[viol2])
    if not np.any(viol1) and not np.any(viol2):
        print("No safety distance violations detected.")

if __name__ == "__main__":
    run_simulation()
