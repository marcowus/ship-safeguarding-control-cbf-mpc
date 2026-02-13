import numpy as np
import sys
import os

# Add the current directory to path so we can import python_refactor
sys.path.append(os.getcwd())

from python_refactor.params import get_params
from python_refactor.utils import ssa, Rzyx
from python_refactor.pid_controller import PIDController
from python_refactor.navalvessel import navalvessel

def test_modules():
    print("Testing params...")
    params = get_params()
    assert 'm' in params
    print("Params OK.")

    print("Testing utils...")
    angle = ssa(np.pi + 0.1)
    # ssa maps to [-pi, pi)
    # pi + 0.1 -> -pi + 0.1
    assert abs(angle - (-np.pi + 0.1)) < 1e-6
    R = Rzyx(0, 0, np.pi/2)
    # R should be [0, -1, 0; 1, 0, 0; 0, 0, 1] roughly
    # cos(pi/2) = 0, sin(pi/2) = 1
    # [0, -1, 0]
    # [1, 0, 0]
    # [0, 0, 1]
    assert abs(R[0,1] + 1) < 1e-6
    print("Utils OK.")

    print("Testing PIDController...")
    wn = np.diag([1, 1, 1])
    zeta = np.diag([1, 1, 1])
    pid = PIDController(wn, zeta, 10, 0.1)
    eta = np.zeros(3)
    nu = np.zeros(3)
    eta_ref = np.array([1, 1, 0])
    M = np.eye(3)
    tau = pid.step(eta, nu, eta_ref, M)
    assert tau.shape == (3,)
    print("PIDController OK.")

    print("Testing navalvessel...")
    x = np.zeros(6)
    ui = np.zeros(6)
    xdot = navalvessel(x, ui, 0, 0)
    assert xdot.shape == (6,)
    print("Navalvessel OK.")

if __name__ == "__main__":
    test_modules()
