import numpy as np

def ssa(angle):
    """
    Smallest signed angle. Maps angle to [-pi, pi).
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def Rzyx(phi, theta, psi):
    """
    Rotation matrix from body to NED.
    R = Rz(psi) * Ry(theta) * Rx(phi)

    Args:
        phi: roll angle (rad)
        theta: pitch angle (rad)
        psi: yaw angle (rad)

    Returns:
        R: 3x3 rotation matrix
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    R = np.array([
        [cpsi*cth, -spsi*cphi + cpsi*sth*sphi, spsi*sphi + cpsi*sth*cphi],
        [spsi*cth, cpsi*cphi + spsi*sth*sphi, -cpsi*sphi + spsi*sth*cphi],
        [-sth,     cth*sphi,                   cth*cphi]
    ])
    return R
