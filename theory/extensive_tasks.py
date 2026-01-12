import numpy as np
import torch
from scipy.linalg import block_diag
from .util import *

def update_extensive_tasks_2x2(d, C_rr, C, S, N_samples, dt, alpha, conj_Z=False):
    C_ft, S_ft = (uni_rfft(param, dt) for param in (C, S))

    I = torch.tensor([1, 0, 0, 1], dtype=torch.float32).to(0)
    Z_inv = I[None,:,None] - np.sqrt(2*np.pi)*d[None,None,:]*C_rr[None,:,:]*S_ft[:,None,None]
    det = Z_inv[:,0,:]*Z_inv[:,3,:] - Z_inv[:,1,:]*Z_inv[:,2,:]
    Z = torch.stack([Z_inv[:,3,:], -Z_inv[:,1,:], -Z_inv[:,2,:], Z_inv[:,0,:]], dim=1) / det[:,None,:]
    Q_r_sum_ft = alpha * (d**2 * torch.sum(torch.abs(Z)**2, dim=1)).mean(-1) * C_ft
    if conj_Z:
        R_ft = alpha*torch.sum(d*C_rr[None,(0,2,1,3),:] * torch.conj(Z), dim=1).mean(-1)/np.sqrt(2*np.pi)
    else:
        R_ft = alpha*torch.sum(d*C_rr[None,(0,2,1,3),:] * Z, dim=1).mean(-1)/np.sqrt(2*np.pi)

    Q_r_sum, R = (uni_irfft(param, dt) for param in (Q_r_sum_ft, R_ft))

    eta = sample_gp(Q_r_sum, N_samples, C_in_fourier=False)
    nk = int(50 / dt)
    x = run_dynamics(eta, dt=dt, kernel=R[:nk])
    phi = phi_torch(x)

    C_updated = compute_C(phi, -1)
    S_updated = compute_S(phi, eta, Q_r_sum, dt)

    return C_updated, S_updated

def update_extensive_tasks_general(d, C_rr, C, S, N_samples, dt, alpha, conj_Z=False):
    """
    Updated function for general RxR matrices with an ensemble dimension.

    Parameters:
      d      : torch.Tensor, shape (N_ensemble,)
               Ensemble of scalar factors.
      C_rr   : torch.Tensor, shape (R*R, N_ensemble)
               "Unrolled" RxR matrices (one per ensemble member);
               they will be reshaped to (R, R, N_ensemble).
      C, S   : time-domain signals.
      N_samples, dt, alpha, conj_Z : additional parameters.

    Fourier transforms (uni_rfft and uni_irfft) are assumed to yield:
      - C_ft, S_ft: tensors with shape (N_freq,)
        (i.e. there is one frequency series for S and C, common to all ensemble members).

    Returns:
      C_updated, S_updated : updated signals computed via the dynamics.
    """
    # Compute Fourier transforms of C and S.
    # (These functions should return tensors with shape (N_freq,).)
    C_ft, S_ft = (uni_rfft(param, dt) for param in (C, S))

    # Determine matrix dimension R from C_rr.
    # C_rr is provided with shape (R*R, N_ensemble); reshape it to (R, R, N_ensemble)
    R = int(np.sqrt(C_rr.shape[0]))
    N_ensemble = C_rr.shape[1]
    C_rr_full = C_rr.reshape(R, R, N_ensemble)  # shape: (R, R, N_ensemble)

    # Create the RxR identity matrix.
    I_R = torch.eye(R, dtype=torch.float32, device=C_rr.device)  # shape: (R, R)

    # We now form the frequency-dependent matrices:
    #    Z_inv(f) = I_R - sqrt(2*pi) * d * S_ft(f) * C_rr_full.
    #
    # Here d is shape (N_ensemble,) and S_ft is shape (N_freq,).
    # We wish to obtain Z_inv of shape (N_freq, R, R, N_ensemble).
    # Arrange dimensions as follows:
    #   I_R:        (1, R, R, 1)
    #   d:          (1, 1, 1, N_ensemble)  via d[None, None, None, :]
    #   S_ft:       (N_freq, 1, 1, 1)       via S_ft[:, None, None, None]
    #   C_rr_full:  (1, R, R, N_ensemble)   via C_rr_full[None, :, :, :]
    sqrt_2pi = np.sqrt(2 * np.pi)
    Z_inv = I_R[None, :, :, None] - sqrt_2pi * d[None, None, None, :] * S_ft[:, None, None, None] * C_rr_full[None, :,
                                                                                                    :, :]
    # Z_inv now has shape: (N_freq, R, R, N_ensemble)

    # For batched inversion, permute Z_inv to bring the ensemble dimension next to frequency:
    # New shape: (N_freq, N_ensemble, R, R)
    Z_inv = Z_inv.permute(0, 3, 1, 2)

    # Compute the inverse for each frequency and ensemble.
    Z = torch.inverse(Z_inv)  # shape: (N_freq, N_ensemble, R, R)

    # --- Compute Q_r_sum_ft ---
    #
    # In the original 2x2 code, one computed:
    #   Q_r_sum_ft = alpha * (d^2 * sum(|Z|^2 over matrix dims)).mean(ensemble) * C_ft.
    # For general RxR, sum the squared absolute values over the last two dims.
    Z_norm_sq = torch.sum(torch.abs(Z) ** 2, dim=(-2, -1))  # shape: (N_freq, N_ensemble)
    Q_temp = (d[None, :] ** 2) * Z_norm_sq  # shape: (N_freq, N_ensemble)
    Q_temp_mean = Q_temp.mean(dim=-1)  # shape: (N_freq,)
    Q_r_sum_ft = alpha * Q_temp_mean * C_ft  # shape: (N_freq,)

    # --- Compute R_ft ---
    #
    # In the 2x2 version, one performed a contraction using the "transposed" C_rr.
    # For the general RxR case, we first obtain the transpose of C_rr_full.
    # C_rr_full has shape (R, R, N_ensemble); its transpose (swapping the first two dims)
    # is given by:
    C_rr_full_T = C_rr_full.transpose(0, 1)  # shape: (R, R, N_ensemble)
    # Bring the ensemble dimension to the front so that it matches Z.
    C_rr_full_T = C_rr_full_T.permute(2, 0, 1)  # shape: (N_ensemble, R, R)

    # Now, Z has shape (N_freq, N_ensemble, R, R).
    # We want to compute an elementwise product and sum over the matrix indices.
    # Expand C_rr_full_T to (1, N_ensemble, R, R) and d to (1, N_ensemble, 1, 1):
    C_rr_full_T = C_rr_full_T[None, :, :, :]  # shape: (1, N_ensemble, R, R)
    d_exp = d[None, :, None, None]  # shape: (1, N_ensemble, 1, 1)

    # Multiply elementwise. (For the conjugated case, you could use torch.conj(Z) if needed.)
    if conj_Z:
        prod = d_exp * C_rr_full_T * torch.conj(Z)  # shape: (N_freq, N_ensemble, R, R)
    else:
        prod = d_exp * C_rr_full_T * Z  # shape: (N_freq, N_ensemble, R, R)

    # Sum over the matrix dimensions (last two dims) to get a tensor of shape (N_freq, N_ensemble)
    prod_sum = torch.sum(prod, dim=(-2, -1))  # shape: (N_freq, N_ensemble)
    # Then average over the ensemble dimension:
    prod_mean = prod_sum.mean(dim=-1)  # shape: (N_freq,)

    R_ft = alpha * prod_mean / np.sqrt(2 * np.pi)  # shape: (N_freq,)

    # --- Inverse Fourier transform back to time domain ---
    Q_r_sum = uni_irfft(Q_r_sum_ft, dt)
    R_time = uni_irfft(R_ft, dt)

    # --- Continue with the dynamics (unchanged from your original code) ---
    eta = sample_gp(Q_r_sum, N_samples, C_in_fourier=False)
    nk = int(50 / dt)
    x = run_dynamics(eta, dt=dt, kernel=R_time[:nk])
    phi = phi_torch(x)

    C_updated = compute_C(phi, -1)
    S_updated = compute_S(phi, eta, Q_r_sum, dt)

    return C_updated, S_updated


def sample_W_optimized(sigma_mn_all, D, N, n_var=1, seed=None):
    # Use CPU or GPU depending on availability
    device = 0

    # Get dimensions
    R = sigma_mn_all.shape[0]
    N_tasks = sigma_mn_all.shape[2]

    # Convert inputs to torch tensors with appropriate dtype
    sigma_mn_all = torch.from_numpy(sigma_mn_all.astype(np.float32)).to(device)  # (R, R, N_tasks)
    D_tensor = torch.from_numpy(D.astype(np.float32)).to(device)  # (N_tasks,)

    # Create identity matrices for sigma_mm and sigma_nn
    sigma_mm = torch.eye(R, dtype=torch.float32, device=device).unsqueeze(2).repeat(1, 1, N_tasks)  # (R, R, N_tasks)
    sigma_nn = n_var*torch.eye(R, dtype=torch.float32, device=device).unsqueeze(2).repeat(1, 1, N_tasks)

    # Build full covariance matrices for all tasks
    covariance_top = torch.cat([sigma_mm, sigma_mn_all], dim=1)  # (R, 2R, N_tasks)
    covariance_bot = torch.cat([sigma_mn_all.transpose(0, 1), sigma_nn], dim=1)  # (R, 2R, N_tasks)
    covariance = torch.cat([covariance_top, covariance_bot], dim=0)  # (2R, 2R, N_tasks)
    covariance = (1 / N) * covariance  # Scale by (1/N)

    # Mean vector (zero mean) for all tasks
    mean = torch.zeros(2 * R, N_tasks, dtype=torch.float32, device=device)  # (2R, N_tasks)

    # Create multivariate normal distributions for each task
    if seed is not None:
        torch.manual_seed(seed)
    mvn = torch.distributions.MultivariateNormal(
        mean.T,  # (N_tasks, 2R)
        covariance_matrix=covariance.permute(2, 0, 1)  # (N_tasks, 2R, 2R)
    )

    # Sample loadings: Shape (N_tasks, N, 2R)
    loadings = mvn.rsample((N,))  # (N, N_tasks, 2R)
    loadings = loadings.permute(1, 0, 2)  # (N_tasks, N, 2R)

    # Split loadings into two parts
    loadings_m = loadings[:, :, :R]  # (N_tasks, N, R)
    loadings_n = loadings[:, :, R:]  # (N_tasks, N, R)

    # Multiply loadings_m by D
    D_expanded = D_tensor[:, None, None]  # (N_tasks, 1, 1)
    loadings_m_weighted = D_expanded * loadings_m  # (N_tasks, N, R)

    # Reshape loadings_m_weighted and loadings_n to (N, N_tasks * R)
    loadings_m_weighted_flat = loadings_m_weighted.permute(1, 0, 2).reshape(N, -1)  # (N, N_tasks * R)
    loadings_n_flat = loadings_n.permute(1, 0, 2).reshape(N, -1)  # (N, N_tasks * R)

    # Compute W = loadings_m_weighted_flat @ loadings_n_flat.T
    W = loadings_m_weighted_flat @ loadings_n_flat.T  # (N, N)

    return W, loadings.cpu().numpy()

def sample_W_optimized_with_barcodes(sigma_mn_all, D, N, f=1, n_var=1, seed=None):
    # Use CPU or GPU depending on availability
    device = 0

    # Get dimensions
    R = sigma_mn_all.shape[0]
    N_tasks = sigma_mn_all.shape[2]

    # Convert inputs to torch tensors with appropriate dtype
    sigma_mn_all = torch.from_numpy(sigma_mn_all.astype(np.float32)).to(device)  # (R, R, N_tasks)
    D_tensor = torch.from_numpy(D.astype(np.float32)).to(device)  # (N_tasks,)

    # Create identity matrices for sigma_mm and sigma_nn
    sigma_mm = torch.eye(R, dtype=torch.float32, device=device).unsqueeze(2).repeat(1, 1, N_tasks)  # (R, R, N_tasks)
    sigma_nn = n_var*torch.eye(R, dtype=torch.float32, device=device).unsqueeze(2).repeat(1, 1, N_tasks)

    # Build full covariance matrices for all tasks
    covariance_top = torch.cat([sigma_mm, sigma_mn_all], dim=1)  # (R, 2R, N_tasks)
    covariance_bot = torch.cat([sigma_mn_all.transpose(0, 1), sigma_nn], dim=1)  # (R, 2R, N_tasks)
    covariance = torch.cat([covariance_top, covariance_bot], dim=0)  # (2R, 2R, N_tasks)
    covariance = (1 / N) * covariance  # Scale by (1/N)

    # Mean vector (zero mean) for all tasks
    mean = torch.zeros(2 * R, N_tasks, dtype=torch.float32, device=device)  # (2R, N_tasks)

    # Create multivariate normal distributions for each task
    if seed is not None:
        torch.manual_seed(seed)
    mvn = torch.distributions.MultivariateNormal(
        mean.T,  # (N_tasks, 2R)
        covariance_matrix=covariance.permute(2, 0, 1)  # (N_tasks, 2R, 2R)
    )

    # Sample loadings: Shape (N_tasks, N, 2R)
    loadings = mvn.rsample((N,))  # (N, N_tasks, 2R)
    loadings = loadings.permute(1, 0, 2)  # (N_tasks, N, 2R)

    if f < 1:
        #sample barcodes
        barcodes = torch.bernoulli(f*torch.ones(N_tasks, N, dtype=torch.float32, device=device))
        loadings[:,:,:R] = loadings[:,:,:R] * barcodes[:,:,None]
        loadings[:,:,R:] = loadings[:,:,R:] * barcodes[:,:,None]

    # Split loadings into two parts
    loadings_m = loadings[:, :, :R]  # (N_tasks, N, R)
    loadings_n = loadings[:, :, R:]  # (N_tasks, N, R)

    # Multiply loadings_m by D
    D_expanded = D_tensor[:, None, None]  # (N_tasks, 1, 1)
    loadings_m_weighted = D_expanded * loadings_m  # (N_tasks, N, R)

    # Reshape loadings_m_weighted and loadings_n to (N, N_tasks * R)
    loadings_m_weighted_flat = loadings_m_weighted.permute(1, 0, 2).reshape(N, -1)  # (N, N_tasks * R)
    loadings_n_flat = loadings_n.permute(1, 0, 2).reshape(N, -1)  # (N, N_tasks * R)

    # Compute W = loadings_m_weighted_flat @ loadings_n_flat.T
    W = loadings_m_weighted_flat @ loadings_n_flat.T  # (N, N)

    return W, loadings.cpu().numpy()

def run_z_dynamics(eta, dt, d, C_rr, kernel=None):
    N_z, N_t = eta.shape
    pad = N_t // 2
    z = torch.zeros(N_z, N_t + pad).to(eta.device)
    eta_pad = torch.zeros(N_z, N_t + pad).to(eta.device)
    eta_pad[:, :pad] = eta[:, N_t - pad:]
    eta_pad[:, pad:] = eta
    eta = eta_pad
    z[:, 0] = torch.randn(N_z, device=eta.device)

    nk = len(kernel) if kernel is not None else 0

    for i in range(1, N_t + pad):
        start = max(0, i - nk)
        nk_stop = i - start
        kernel_contribution = dt * torch.trapz(
            torch.flip(z[:, start:i], dims=(1,)) * kernel[None, :nk_stop], dim=1
        ) if nk_stop > 0 else 0.0
        z[:, i] = eta[:, i] + d * torch.matmul(C_rr, kernel_contribution)

    return z[:, pad:]

def upsample_array(arr, ratio):
    # Calculate the number of points in the upsampled array
    new_length = int(len(arr) * ratio)

    # Create the new array by interpolating values
    x_original = np.arange(len(arr))
    x_new = np.linspace(0, len(arr) - 1, new_length)
    upsampled_arr = np.interp(x_new, x_original, arr)

    return upsampled_arr

def generate_gaussian_matrix(R, sigma_on, sigma_off, symmetry_factor, traceless=False):
    """
    Generates an RxR matrix with specified on-diagonal sigma, off-diagonal sigma,
    and a symmetry factor that biases off-diagonal elements.

    Parameters:
    R (int): The size of the matrix (RxR).
    sigma_on (float): Standard deviation for on-diagonal elements.
    sigma_off (float): Standard deviation for off-diagonal elements.
    symmetry_factor (float): Symmetry bias factor for off-diagonal elements.

    Returns:
    numpy.ndarray: The generated RxR matrix.
    """
    # Generate random values for the matrix
    matrix = np.random.normal(0, sigma_off, size=(R, R))

    # Make the matrix symmetric by adjusting off-diagonal terms with the symmetry factor
    matrix = (matrix + symmetry_factor * matrix.T) / (1 + np.abs(symmetry_factor))

    # Fill the diagonal with random values using the on-diagonal sigma
    np.fill_diagonal(matrix, np.random.normal(0, sigma_on, size=R))
    if traceless:
        matrix = matrix - np.trace(matrix) / R * np.eye(R)

    return matrix


def is_positive_definite(matrix):
    """Check if a matrix is positive definite by verifying its eigenvalues."""
    return np.all(np.linalg.eigvals(matrix) > 0)


def generate_positive_definite_covariance(R, sigma_on, sigma_off, symmetry_factor, max_attempts=1000,
                                          traceless=False, report_attempts=False):
    """
    Generates a 2R x 2R positive definite covariance matrix with on-diagonal blocks
    as identity matrices and off-diagonal blocks as Gaussian matrices.

    Parameters:
    R (int): Size of the on-diagonal blocks (RxR).
    sigma_on (float): Standard deviation for the on-diagonal elements of the off-diagonal block.
    sigma_off (float): Standard deviation for the off-diagonal elements of the off-diagonal block.
    symmetry_factor (float): Symmetry bias factor for the off-diagonal blocks.
    max_attempts (int): Maximum number of attempts to sample a positive definite matrix.

    Returns:
    numpy.ndarray: The generated 2R x 2R positive definite covariance matrix.
    """
    identity_block = np.eye(R)

    for attempt in range(max_attempts):
        # Generate the off-diagonal block using the previous function
        off_diag_block = generate_gaussian_matrix(R, sigma_on, sigma_off, symmetry_factor, traceless=traceless)

        # Construct the full 2R x 2R matrix
        covariance_matrix = np.block([
            [identity_block, off_diag_block],
            [off_diag_block.T, identity_block]
        ])

        # Check if the matrix is positive definite
        if is_positive_definite(covariance_matrix):
            if report_attempts:
                return off_diag_block, attempt
            else:
                return off_diag_block

    raise ValueError(f"Failed to generate a positive definite matrix in {max_attempts} attempts.")

def generate_positive_definite_covariance_block_haar(R, gamma, max_attempts=1000, report_attempts=False):
    """
    Generates a 2R x 2R positive definite covariance matrix with on-diagonal blocks
    as identity matrices and off-diagonal blocks as Gaussian matrices.

    Parameters:
    R (int): Size of the on-diagonal blocks (RxR).
    sigma_on (float): Standard deviation for the on-diagonal elements of the off-diagonal block.
    sigma_off (float): Standard deviation for the off-diagonal elements of the off-diagonal block.
    symmetry_factor (float): Symmetry bias factor for the off-diagonal blocks.
    max_attempts (int): Maximum number of attempts to sample a positive definite matrix.

    Returns:
    numpy.ndarray: The generated 2R x 2R positive definite covariance matrix.
    """
    identity_block = np.eye(R)

    for attempt in range(max_attempts):
        # Generate the off-diagonal block using the previous function
        off_diag_block = generate_block_haar_matrix(R, gamma)

        # Construct the full 2R x 2R matrix
        covariance_matrix = np.block([
            [identity_block, off_diag_block],
            [off_diag_block.T, identity_block]
        ])

        # Check if the matrix is positive definite
        if is_positive_definite(covariance_matrix):
            if report_attempts:
                return off_diag_block, attempt
            else:
                return off_diag_block

    raise ValueError(f"Failed to generate a positive definite matrix in {max_attempts} attempts.")

def generate_block_haar_matrix(R, gamma):
    """
    Generates a 2R x 2R block Haar matrix with the specified gamma parameter.

    Parameters:
    R (int): Size of the on-diagonal blocks (RxR).
    gamma (float): Parameter for the block Haar matrix.

    Returns:
    numpy.ndarray: The generated 2R x 2R block Haar matrix.
    """

    if R % 2 != 0:
        raise ValueError("R must be even (i.e. R = 2k).")

    k = R // 2

    # Create a list of k independent 2x2 rotation matrices.
    blocks = []
    for i in range(k):
        theta = np.random.uniform(0, 2 * np.pi)
        rot_block = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
        blocks.append(rot_block)

    # Create the block-diagonal matrix B with these 2x2 blocks.
    B = block_diag(*blocks)

    # Generate a random R x R orthogonal matrix Q.
    # One simple way is to take a random Gaussian matrix and compute its QR decomposition.
    X = np.random.randn(R, R)
    Q, _ = np.linalg.qr(X)

    # Form the similarity transform of B and scale by gamma.
    M = gamma * (Q @ B @ Q.T)

    return M
