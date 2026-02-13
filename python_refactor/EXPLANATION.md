# Naval Vessel Control Simulation Explanation

This project simulates the control of a naval vessel using a combination of PID control for trajectory tracking, Control Barrier Functions (CBF) for obstacle avoidance, and Control Allocation for thrust distribution.

## 1. Naval Vessel Dynamics

The vessel is modeled as a 3-DOF (Surge, Sway, Yaw) system, though the full state includes Roll (6-DOF reduced). The dynamics are governed by the equation of motion:

$$ M \dot{\nu} + C(\nu)\nu + D(\nu)\nu = \tau $$

where:
- $\nu = [u, v, r]^T$ is the velocity vector in the body frame (Surge, Sway, Yaw rate).
- $M$ is the mass-inertia matrix including added mass.
- $C(\nu)$ is the Coriolis-centripetal matrix.
- $D(\nu)$ is the damping matrix (linear and nonlinear).
- $\tau = [X, Y, N]^T$ is the control force/moment vector.

In this simulation, the 6-DOF model includes roll ($\phi$) and its coupling effects. The state vector is $x = [u, v, p, r, \phi, \psi]^T$.

## 2. PID Control

A MIMO (Multiple Input Multiple Output) nonlinear PID controller is used for trajectory tracking. The control law is designed to drive the vessel to a reference pose $\eta_{ref} = [x_{ref}, y_{ref}, \psi_{ref}]^T$.

The control law is given by:

$$ \tau_{PID} = -R(\psi)^T ( K_p e + K_i z_{int} ) - K_d \nu $$

where:
- $e = \eta - \eta_d$ is the position error.
- $z_{int} = \int e \, dt$ is the integral error.
- $R(\psi)$ is the rotation matrix from body to NED frame.
- $K_p, K_i, K_d$ are the proportional, integral, and derivative gain matrices.

## 3. Control Barrier Functions (CBF) for Obstacle Avoidance

CBFs are used to ensure safety (collision avoidance) by constraining the control input $\tau$. A barrier function $h(x)$ is defined such that the set $\mathcal{C} = \{x \in \mathbb{R}^n : h(x) \geq 0\}$ is the safe set.

For a circular obstacle with radius $d$ at $(x_o, y_o)$, the barrier function is:

$$ h(x) = (x_p - x_o)^2 + (y_p - y_o)^2 - d^2 $$

To ensure forward invariance of $\mathcal{C}$, the control input $u$ (here $\tau$) must satisfy:

$$ \dot{h}(x) \geq -\gamma h(x) $$

Expanding $\dot{h}(x)$:

$$ \frac{\partial h}{\partial x} (f(x) + g(x)\tau) \geq -\gamma h(x) $$

This leads to a linear constraint on $\tau$:

$$ L_g h(x) \tau \geq -\gamma h(x) - L_f h(x) $$

This constraint is enforced using a Quadratic Program (QP) that minimizes the deviation from the nominal PID control $\tau_{ref}$:

$$
\begin{aligned}
\min_{\tau} \quad & \frac{1}{2} ||\tau - \tau_{ref}||^2 \\
\text{s.t.} \quad & A_{cbf} \tau \leq b_{cbf}
\end{aligned}
$$

The QP is solved at each time step using `casadi` with `qpoases`.

## 4. Control Allocation

The generalized forces $\tau = [X, Y, N]^T$ computed by the CBF-QP must be distributed to the vessel's thrusters (2 tunnel thrusters, 2 azimuth thrusters).

The relationship between thruster inputs and generalized forces is:

$$ \tau = T(\alpha) K u $$

where:
- $u$ contains the normalized squared propeller speeds.
- $\alpha$ contains the azimuth angles.
- $T(\alpha)$ is the thrust configuration matrix.

This is an over-actuated system, solved as a constrained nonlinear optimization problem:

$$
\min_{u, \alpha, s} \quad w_1 ||u||^2 + w_2 ||s||^2 + w_3 ||\Delta \alpha||^2 + w_4 ||\Delta u||^2
$$

subject to:
- Rate limits on $\alpha$ and $u$.
- Thrust saturation limits.
- The equality constraint $T(\alpha) K u - \tau_{cmd} + s = 0$ (with slack variable $s$).

This optimization is solved using `casadi` with `ipopt`.

## 5. Simulation Results

The simulation runs for 60 seconds. The vessel starts at $(-15, -15)$ and aims for $(0, 0)$. Two obstacles are placed at $(-10, -10)$ and $(-7, -2)$. The CBF controller successfully modifies the PID output to steer the vessel around the obstacles while maintaining stability.

Outputs:
- **Trajectory**: Shows the path of the vessel avoiding the obstacles.
- **Control Inputs**: Shows the surge, sway forces and yaw moment applied.
- **CBF Values**: Shows the value of $h(x)$ (B1) and the constraint satisfaction (B2).
