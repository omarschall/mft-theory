import numpy as np

def gaussian_integral(f, n_hermite=200):
    """Numerically solve a gaussian integral of a function f, for standard gaussian
    parameters mu = 0 and sigma = 1, via Hermite polynomials.

    The function f must have kwarg 'z'."""

    gaussian_norm = (1 / np.sqrt(np.pi))
    gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(n_hermite)
    gauss_points = gauss_points * np.sqrt(2)
    integrand = f(z=gauss_points)

    return gaussian_norm * np.dot(integrand, gauss_weights)

def fft(x, dt):
    n_dim = len(x.shape)
    rescaling = (dt / np.sqrt(2 * np.pi)) ** n_dim
    if n_dim == 1:
        return rescaling * np.fft.fft(x)
    elif n_dim == 2:
        return rescaling * np.fft.fft2(x)


def ifft(x, dt):
    n_dim = len(x.shape)
    samp_freq = 1 / dt
    rescaling = (samp_freq * np.sqrt(2 * np.pi)) ** n_dim
    if n_dim == 1:
        return rescaling * np.fft.ifft(x)
    elif n_dim == 2:
        return rescaling * np.fft.ifft2(x)