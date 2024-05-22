import numpy as np
import torch
from empirics import random_derangement

phi_torch = lambda x: torch.erf((np.sqrt(np.pi)/2)*x)
def estimate_psi(lags, T_sim, dt_save, dt, W, phi_torch=phi_torch,
                 T_save_delay=100, N_batch=1, N_loops=10):

    device = W.device
    N = W.shape[0]
    eval_iter = int(dt_save / dt)
    Nt = int(T_sim / dt)
    N_save = int(T_sim / dt_save)
    # define stuff
    r_save = torch.zeros(N_save, N_batch, N, device=device)
    x = torch.randn(N_batch, N, device=device) * torch.std(W).item() * np.sqrt(N)
    r_save[0] = phi_torch(x)
    # run
    idx = random_derangement(N)
    xcovs = torch.zeros(N_loops, len(lags), N_batch, N, device=device)
    for i_loop in range(N_loops):
        r_save = torch.zeros(N_save, N_batch, N, device=device)
        x = torch.randn(N_batch, N, device=device) * torch.std(W).item() * np.sqrt(N)
        r_save[0] = phi_torch(x)
        for i in range(1, Nt):
            r = phi_torch(x)
            x += dt * (-x + torch.mm(r, W.T))
            if i % eval_iter == 0:
                r_save[i // eval_iter] = r
        r_save = r_save[int(T_save_delay/dt_save):]
        xcov = compute_lagged_xcov(r_save[:,:,idx], r_save, lags, dt_save)
        xcovs[i_loop] = xcov
    xcov_mean = xcovs.mean(dim=0)

    #estimate_psi
    psi = N*torch.mean(torch.square(torch.mean(xcov_mean, dim=1)), dim=1)
    return psi

def compute_lagged_xcov(r1, r2, lags, dt_save):
    xcov = torch.zeros(len(lags), r1.shape[1], r1.shape[2], device=r1.device)
    for i_lag, lag in enumerate(lags):
        lag_steps = int(lag / dt_save)
        r1_lag = torch.roll(r1, lag_steps, dims=0)[lag_steps:]
        r2_lag = r2[lag_steps:]
        xcov[i_lag] = torch.einsum('tbi, tbi -> bi', r1_lag, r2_lag)/(r1_lag.shape[0] -1)
    return xcov


