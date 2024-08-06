import numpy as np
import torch
from empirics import random_derangement
from .Update_Step import update_step

phi_torch = lambda x: torch.erf((np.sqrt(np.pi)/2)*x)
def estimate_psi(lags, T_sim, dt_save, dt, W, phi_torch=phi_torch,
                 T_save_delay=100, N_batch=1, N_loops=10, runga_kutta=False, noise_sigma=0,
                 mode='tau_tau'):

    assert mode in ['tau_tau', 'tau_0']
    device = W.device
    N = W.shape[0]
    eval_iter = int(dt_save / dt)
    Nt = int(T_sim / dt)
    N_save = int(T_sim / dt_save)
    # define stuff
    r_save = torch.zeros(N_save, N_batch, N, device=device)
    x0 = torch.randn(N_batch, N, device=device) * torch.std(W).item() * np.sqrt(N)
    r_save[0] = phi_torch(x0)
    # run
    idx = random_derangement(N)
    xcovs = torch.zeros(N_loops, len(lags), N_batch, N, device=device)
    # if r_weights is not None:
    #     xcovs_nn = torch.zeros(N_loops, len(lags), N_batch, N, device=device)
    for i_loop in range(N_loops):
        r_save = torch.zeros(N_save, N_batch, N, device=device)
        #x = torch.randn(N_batch, N, device=device) * torch.std(W).item() * np.sqrt(N)
        r_save[0] = phi_torch(x0)
        x = x0
        for i in range(1, Nt):
            x, r = update_step(x, dt=dt, W=W, phi_torch=phi_torch,
                               noise_sigma=noise_sigma, runga_kutta=runga_kutta)

            if i % eval_iter == 0:
                r_save[i // eval_iter] = r
        r_save = r_save[int(T_save_delay/dt_save):]
        T_save_delay = 0
        x0 = x
        xcov = compute_lagged_xcov(r_save[:,:,idx], r_save, lags, dt_save)
        xcovs[i_loop] = xcov
        # if r_weights is not None:
        #     r_save_nn = r_weights * r_save
        #     xcov_nn = compute_lagged_xcov(r_save_nn[:, :, idx], r_save_nn, lags, dt_save)
        #     xcovs_nn[i_loop] = xcov_nn
    xcov_mean = xcovs.mean(dim=0)

    #estimate_psi
    if mode == 'tau_tau':
        psi = N*torch.mean(torch.square(torch.mean(xcov_mean, dim=1)), dim=1)
    elif mode == 'tau_0':
        C_tau = torch.mean(xcov_mean, dim=1)
        C_0 = torch.mean(xcov_mean, dim=1)[0]
        psi = N*torch.mean(C_tau * C_0[None,:], dim=1)

    return psi
    # if r_weights is not None:
    #     xcov_nn_mean = xcovs_nn.mean(dim=0)
    #     psi_nn = N*torch.mean(torch.square(torch.mean(xcov_nn_mean, dim=1)), dim=1)
    #     return psi, psi_nn
    # else:
    #     return psi

def estimate_Psi_with_on_diagonals(lags, T_sim, dt_save, dt, W, phi_torch=phi_torch,
                                   T_save_delay=100, N_batch=1, N_loops=10, runga_kutta=False, noise_sigma=0,
                                   mode='tau_tau', return_raw_cov=False):

    assert mode in ['tau_tau', 'tau_0']
    device = W.device
    N = W.shape[0]
    eval_iter = int(dt_save / dt)
    Nt = int(T_sim / dt)
    N_save = int(T_sim / dt_save)
    # define stuff
    r_save = torch.zeros(N_save, N_batch, N, device=device)
    x0 = torch.randn(N_batch, N, device=device) * torch.std(W).item() * np.sqrt(N)
    r_save[0] = phi_torch(x0)
    # run
    xcov_mean = torch.zeros(len(lags), N_batch, N, N, device=device)
    for i_loop in range(N_loops):
        r_save = torch.zeros(N_save, N_batch, N, device=device)
        #x = torch.randn(N_batch, N, device=device) * torch.std(W).item() * np.sqrt(N)
        r_save[0] = phi_torch(x0)
        x = x0
        for i in range(1, Nt):
            x, r = update_step(x, dt=dt, W=W, phi_torch=phi_torch,
                               noise_sigma=noise_sigma, runga_kutta=runga_kutta)

            if i % eval_iter == 0:
                r_save[i // eval_iter] = r
        r_save = r_save[int(T_save_delay/dt_save):]
        T_save_delay = 0
        x0 = x
        xcov = compute_lagged_xcov(r_save, r_save, lags, dt_save, outer=True)
        xcov_mean += xcov/N_loops
    #estimate_psi
    if mode == 'tau_tau':
        psi = N*torch.mean(torch.square(torch.mean(xcov_mean, dim=1)), dim=(1,2))
    elif mode == 'tau_0':
        C_tau = torch.mean(xcov_mean, dim=1)
        C_0 = torch.mean(xcov_mean, dim=1)[0]
        psi = N*torch.mean(C_tau * C_0[None,:,:], dim=(1,2))
    if not return_raw_cov:
        return psi
    else:
        return torch.mean(xcov_mean, dim=1)

def compute_lagged_xcov(r1, r2, lags, dt_save, outer=False):
    if not outer:
        xcov = torch.zeros(len(lags), r1.shape[1], r1.shape[2], device=r1.device)
    else:
        xcov = torch.zeros(len(lags), r1.shape[1], r1.shape[2], r1.shape[2], device=r1.device)
    for i_lag, lag in enumerate(lags):
        lag_steps = int(lag / dt_save)
        r1_lag = torch.roll(r1, lag_steps, dims=0)[lag_steps:]
        r2_lag = r2[lag_steps:]
        if not outer:
            xcov[i_lag] = torch.einsum('tbi, tbi -> bi', r1_lag, r2_lag)/(r1_lag.shape[0] - 1)
        else:
            xcov[i_lag] = torch.einsum('tbi, tbj -> bij', r1_lag, r2_lag)/(r1_lag.shape[0] - 1)
    return xcov


