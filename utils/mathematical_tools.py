import numpy as np

def get_spectral_radius(M):
    """Calculates the spectral radius of a matrix."""

    eigs, _ = np.linalg.eig(M)

    return np.amax(np.absolute(eigs))

def norm(z):
    """Computes the L2 norm of a numpy array."""

    return np.sqrt(np.sum(np.square(z)))

import numpy as np

def generate_real_matrix(eigenvalues):
    # Initialize an empty block diagonal matrix
    B = np.zeros((0, 0))

    # Process each eigenvalue
    for eigenvalue in eigenvalues:
        if np.iscomplex(eigenvalue):
            # For complex eigenvalues, ensure they come in conjugate pairs and form a 2x2 block
            a = eigenvalue.real
            b = eigenvalue.imag
            block = np.array([[a, -b], [b, a]])
            # Append the block to the diagonal of B
            B = np.block([[B, np.zeros((B.shape[0], block.shape[1]))],
                          [np.zeros((block.shape[0], B.shape[1])), block]])
        else:
            # For real eigenvalues, form a 1x1 block
            block = np.array([[eigenvalue]])
            # Append the block to the diagonal of B
            B = np.block([[B, np.zeros((B.shape[0], block.shape[1]))],
                          [np.zeros((block.shape[0], B.shape[1])), block]])

    # Generate a random invertible matrix P
    P = np.random.randn(B.shape[0], B.shape[0])
    while np.linalg.det(P) == 0:
        P = np.random.randn(B.shape[0], B.shape[0])  # Ensure P is invertible

    # Compute the inverse of P
    P_inv = np.linalg.inv(P)

    # Apply the similarity transformation M = P^-1 * B * P
    M = P_inv @ B @ P

    return M


def invert_PR_by_newton(PR, initial_guess=1.0, tolerance=1e-5, max_iterations=100):
    """Thanks chat gpt 4o"""

    # Define the function and its derivative
    def f(beta):
        return (1 / beta) * np.tanh(beta) - PR

    def f_prime(beta):
        return (-1 / (beta ** 2)) * np.tanh(beta) + (1 / beta) * (1 - np.tanh(beta) ** 2)

    beta = initial_guess
    for _ in range(max_iterations):
        beta_new = beta - f(beta) / f_prime(beta)
        if abs(beta_new - beta) < tolerance:
            return beta_new
        beta = beta_new

    raise ValueError("Newton's method did not converge")

from scipy.interpolate import interp1d
def G_i(i, N, beta):
    return (i / N) ** beta

def compute_PR_(N, beta):
    G = np.array([G_i(i, N, beta) for i in range(1, N + 1)])
    sum_G = np.sum(G)
    sum_G2 = np.sum(G ** 2)
    PR = (sum_G ** 2) / (N * sum_G2)
    return PR


def invert_PR_approx(PR):
    return 2 * (1 - PR) / PR

def get_exact_beta_for_PR(PR, N=1000, beta_range=(0.01, 10), num_points=1000):
    beta_values = np.linspace(beta_range[0], beta_range[1], num_points)
    PR_values = [compute_PR_(N, beta) for beta in beta_values]

    # Create an interpolation function
    interp_func = interp1d(PR_values, beta_values, kind='cubic', fill_value='extrapolate')

    # Get the initial approximation
    beta_approx = invert_PR_approx(PR)

    # Refine the approximation using interpolation
    beta_exact = interp_func(PR)

    return beta_approx, beta_exact


def compute_PR(beta, N):
    D = np.array([(i / N) ** beta for i in range(1, N + 1)])
    PR = (D.sum() ** 2) / (np.sum(D ** 2) * N)
    return PR

def find_beta_for_PR(desired_PR, N, tolerance=0.001, learning_rate=0.01):
    """Thanks chatgpt 4o for writing this."""
    # Start with the approximate formula
    beta = -2 * (1 - desired_PR) / desired_PR

    # Define a function to compute the error
    def error(beta):
        current_PR = compute_PR(beta, N)
        return current_PR - desired_PR

    # Perform a simple iterative method to refine beta
    max_iterations = 10000
    iteration = 0

    while abs(error(beta)) > tolerance and iteration < max_iterations:
        beta -= learning_rate * error(beta)
        iteration += 1

    if iteration == max_iterations:
        print("Warning: Maximum iterations reached without achieving desired precision.")

    return beta, compute_PR(beta, N)

def power_law_fit(y_data, x_data=None):
    if x_data is None:
        x_data = np.arange(1, len(y_data) + 1)
    log_y = np.log(y_data)
    log_x = np.log(x_data)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    return slope, intercept