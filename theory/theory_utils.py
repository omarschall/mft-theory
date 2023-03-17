import numpy as np

def gaussian_integral(f, mu=0, sigma=1, dz=0.001, z_max=4):
    """Numerically solve a gaussian integral of a function f, for given gaussian
    parameters mu and sigma, via Riemann integration.

    Args:
        f (function): f must be a function of a single variable, with name z
        mu (float): mean of gaussian measure
        sigma (float): standard deviation of gaussian measure
        dz (float): spacing between rectangles for Riemannian integral
        z_max (float): range of z values about mean for numerical integration"""

    z = np.arange(mu - z_max*sigma, mu + z_max*sigma, dz)
    gauss_pdf = 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(z - mu)**2/(2 * sigma**2))
    return np.sum(gauss_pdf * f(z) * dz)

def gaussian_integral_2d(f, mu=np.array([0, 0]), sigma=np.eye(2),
                         dx=0.001, x_max=4, dz=0.001, z_max=4):
    """Numerically solve a 2D gaussian integral of a function f, for """

    z = np.arange(mu - z_max*sigma, mu + z_max*sigma, dz)
    gauss_pdf = 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(z - mu)**2/(2 * sigma**2))

    pass