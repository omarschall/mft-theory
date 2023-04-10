import numpy as np

def gaussian_integral(f):
    """Numerically solve a gaussian integral of a function f, for standard gaussian
    parameters mu = 0 and sigma = 1, via Hermite polynomials.

    The function f must have kwarg 'z'."""

    gaussian_norm = (1 / np.sqrt(np.pi))
    gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
    gauss_points = gauss_points * np.sqrt(2)
    integrand = f(z=gauss_points)

    return gaussian_norm * np.dot(integrand, gauss_weights)