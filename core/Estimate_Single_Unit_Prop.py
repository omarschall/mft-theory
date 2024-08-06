import numpy as np
import torch
from .Estimate_Psi import compute_lagged_xcov
from .Update_Step import update_step

phi_torch = lambda x: torch.erf((np.sqrt(np.pi)/2)*x)
phi_prime = lambda x: torch.exp(-(np.pi/4)*x**2)
def estimate_single_unit_autocov_and_alpha(lags, T_sim, dt_save, dt, generate_W, N, phi_torch=phi_torch,
                                           phi_prime=phi_prime, T_save_delay=100, N_batch=1, N_disorder=10,
                                           N_loops=10, runga_kutta=False, noise_sigma=0):
    eval_iter = int(dt_save / dt)
    Nt = int(T_sim / dt)
    N_save = int(T_sim / dt_save)
    xcovs = torch.zeros(N_disorder, N_loops, len(lags), N_batch, N)
    alphas = torch.zeros(N_disorder, N_loops)
    for i_disorder in range(N_disorder):
        W = generate_W()
        #W = torch.from_numpy(W).type(torch.FloatTensor).to(0)
        device = W.device
        N = W.shape[0]
        x_save = torch.zeros(N_save, N_batch, N, device=device)
        x0 = torch.randn(N_batch, N, device=device) * torch.std(W).item() * np.sqrt(N)
        for i_loop in range(N_loops):
            x_save = torch.zeros(N_save, N_batch, N, device=device)
            r_save = torch.zeros(N_save, N_batch, N, device=device)
            r_save[0] = phi_torch(x0)
            x = x0
            x_save[0] = x
            for i in range(1, Nt):
                x, r = update_step(x, dt=dt, W=W, phi_torch=phi_torch,
                                   noise_sigma=noise_sigma, runga_kutta=runga_kutta)
                if i % eval_iter == 0:
                    x_save[i // eval_iter] = x
                    r_save[i // eval_iter] = r
            x_save = x_save[int(T_save_delay/dt_save):]
            r_save = r_save[int(T_save_delay/dt_save):]
            T_save_delay = 0
            x0 = x
            gain_save = phi_prime(x_save)
            xcov = compute_lagged_xcov(r_save, r_save, lags, dt_save)
            xcovs[i_disorder, i_loop] = xcov
            alphas[i_disorder, i_loop] = gain_save.mean()

    C_phi_tau = torch.mean(xcovs, dim=(0,1,3,4))
    alpha = torch.mean(alphas)
    return C_phi_tau, alpha