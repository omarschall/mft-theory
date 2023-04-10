import numpy as np
from functools import partial
from theory.theory_utils import gaussian_integral

### --- Key functions and their derivatives for MFT --- ###
#Assumes phi is tanh() and thus Phi is log(cosh())

def Phi_sqrt(Delta_0, z):
    """Second integrand of Eq. 2.14 from Mastrogiuseppe thesis, assuming phi
    is tanh nonlinearity, and thus Phi is log(cosh()). Multivariate function of
    both Delta_0 and z."""
    return np.log(np.cosh(np.sqrt(Delta_0) * z))

def Phi_sqrt_prime(Delta_0, z):
    """Derivative of Phi_Sqrt with respect to Delta_0 (for Newton's method)"""
    return np.tanh(np.sqrt(Delta_0) * z) * (z / (2 * np.sqrt(Delta_0)))

def squared_Phi_sqrt(Delta_0, z):
    """First integrand of Eq. 2.14."""
    return np.square(Phi_sqrt(Delta_0, z))

def squared_Phi_sqrt_prime(Delta_0, z):
    """Derivative of Squared_Phi_Sqrt with respect to Delta_0."""
    phi_sqrt = np.tanh(np.sqrt(Delta_0) * z)
    return Phi_sqrt(Delta_0, z) * phi_sqrt * (z / np.sqrt(Delta_0))

def phi_sqrt(Delta, Delta_0, x, z):
    """Inner integrand of Eq. 2.8 in Mastrogiuseppe thesis"""
    return np.tanh(np.sqrt(Delta_0 - Delta) * x + np.sqrt(Delta) * z)

### --- Equations for solving for Delta_0 --- ###

def equation_2_14(Delta_0, g=1.2):
    """Move all terms to RHS of Eq. 2.14. When this is 0, we have solved for
    the correct Delta_0."""
    Phi_sqrt_ = partial(Phi_sqrt, Delta_0=Delta_0)
    squared_Phi_sqrt_ = partial(squared_Phi_sqrt, Delta_0=Delta_0)

    return Delta_0**2 / 2 - g**2 * (gaussian_integral(squared_Phi_sqrt_)
                                    - gaussian_integral(Phi_sqrt_) ** 2)

def equation_2_14_prime(Delta_0, g=1.2):
    """Derivative of Equation_2_14 wrt Delta_0, for using Newton's method to
    solve."""

    Phi_sqrt_ = partial(Phi_sqrt, Delta_0=Delta_0)
    Phi_sqrt_prime_ = partial(Phi_sqrt_prime, Delta_0=Delta_0)
    squared_Phi_sqrt_prime_ = partial(squared_Phi_sqrt_prime, Delta_0=Delta_0)

    return Delta_0 - g**2 * (gaussian_integral(squared_Phi_sqrt_prime_)
                             - 2 * gaussian_integral(Phi_sqrt_) * gaussian_integral(Phi_sqrt_prime_))

### --- Numerically solve for Delta_0 --- ###

def solve_for_Delta_0(g, Delta_0_init=2, max_iters=1000):
    """For a given value of g and initial guess for Delta_0, use Newton's method
    to solve for Delta_0."""

    Delta_0 = Delta_0_init
    y = equation_2_14(Delta_0, g=g)
    N_iters = 0
    while np.abs(y - 0) > 0.0001 and N_iters < max_iters:
        y = equation_2_14(Delta_0, g=g)
        Delta_0 = np.maximum(Delta_0 - y / equation_2_14_prime(Delta_0, g=g),
                             0.000001) #keep Delta_0 positive
        N_iters += 1

    return Delta_0

def phi_autocorrelation(Delta, Delta_0, dz=0.01):
    """Numerical answer to Eq. 2.8 in Mastrogiuseppe thesis, for a given value
    of Delta and Delta_0. Uses Hermite polynomials."""

    gaussian_norm = 1 / np.sqrt(np.pi)
    gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
    gauss_points = gauss_points * np.sqrt(2)
    inner_integrand = np.tanh(np.add.outer(np.sqrt(Delta) * gauss_points,
                              np.sqrt(Delta_0 - Delta) * gauss_points))

    ret = np.square(gaussian_norm * np.dot(inner_integrand, gauss_weights))
    ret = gaussian_norm * np.dot(ret, gauss_weights)

    return ret


def solve_for_Delta_T(g, Delta_0, T=10, dT=0.01, dz=0.01):
    """For a given value of Delta_0 corresponding to the stable MFT solution,
    find the trajectory of Delta_T values for arbitrary time separations.

    Integrate Eq. 2.6 from Mastrogiuseppe thesis using Euler method, for
    Delta(0) = Delta_0 and Delta'(0) = 0, with Delta''(T) following Eq. 2.6.

    Args:
        g (float): value of g for the MFT.
        Delta_0 (float): value of Delta_0 as solved for the given g
        dT (float): Euler integration step size
        T (float): duration of integration"""

    time_vector = np.arange(0, T, dT)
    Delta_T = np.zeros_like(time_vector)
    Delta_T_dot = np.zeros_like(time_vector)
    Delta_T[0] = Delta_0
    for i_t, t in enumerate(time_vector):

        if i_t == 0:
            continue

        Delta_T[i_t] = Delta_T[i_t-1] + dT * Delta_T_dot[i_t-1]
        Delta_T_dotdot = Delta_T[i_t] - g**2 * phi_autocorrelation(Delta_T[i_t],
                                                                   Delta_0,
                                                                   dz=dz)
        Delta_T_dot[i_t] = Delta_T_dot[i_t-1] + dT * Delta_T_dotdot

    return time_vector, Delta_T, Delta_T_dot

def Delta_potential(g, Delta, Delta_0, dz=0.01):
    """Numerical answer to Eq. 2.11 in Mastrogiuseppe thesis, for a given value
    of Delta and Delta_0.

    NEEDS CHANGING WITH HERMITE POLYNOMIALS."""

    x = np.arange(-4, 4, dz)
    z = np.arange(-4, 4, dz)
    gauss_pdf_x = 1 / np.sqrt(2 * np.pi) * np.exp(-x**2/2)
    gauss_pdf_z = 1 / np.sqrt(2 * np.pi) * np.exp(-z**2/2)
    inner_integrand = np.log(np.cosh((np.add.outer(np.sqrt(Delta) * z,
                                      np.sqrt(Delta_0 - Delta) * x))))

    ret = np.square(np.sum(inner_integrand * gauss_pdf_x * dz, 1))
    ret = np.sum(ret * gauss_pdf_z * dz)
    ret = -Delta**2/2 + g**2 * ret

    return ret
