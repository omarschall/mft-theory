import numpy as np
import torch
from .Update_Step import update_step

phi_torch = lambda x: torch.erf((np.sqrt(np.pi)/2)*x)
def estimate_cov_eigs(T_sim, dt_save, dt, W, phi_torch=phi_torch,
                      T_save_delay=100, N_batch=1, N_loops=10, return_vecs=False, x0=None,
                      return_raw_covs=False, runga_kutta=False, noise_sigma=0):

    device = W.device
    N = W.shape[0]
    eval_iter = int(dt_save / dt)
    Nt = int(T_sim / dt)
    N_save = int(T_sim / dt_save)
    # define stuff
    x_save = torch.zeros(N_save, N_batch, N, device=device)
    r_save = torch.zeros(N_save, N_batch, N, device=device)
    x = torch.randn(N_batch, N, device=device) * torch.std(W).item() * np.sqrt(N)
    x_save[0] = x
    r_save[0] = phi_torch(x)
    # run
    #r_xcovs = torch.zeros(N_loops, N_batch, N, N, device=device)
    r_xcovs_mean = torch.zeros(N_batch, N, N, device=device)
    x_xcovs_mean = torch.zeros(N_batch, N, N, device=device)
    #x_xcovs = torch.zeros(N_loops, N_batch, N, N, device=device)
    x_s = torch.zeros(N_batch, N, device=device)
    xx_s = torch.zeros(N_batch, N, N, device=device)
    r_s = torch.zeros(N_batch, N, device=device)
    rr_s = torch.zeros(N_batch, N, N, device=device)
    T_samples = 0
    for i_loop in range(N_loops):
        x_save = torch.zeros(N_save, N_batch, N, device=device)
        r_save = torch.zeros(N_save, N_batch, N, device=device)
        if i_loop == 0:
            if x0 is None:
                x = torch.randn(N_batch, N, device=device) * torch.std(W).item() * np.sqrt(N)
            else:
                x = x0
        x_save[0] = x
        r_save[0] = phi_torch(x)
        for i in range(1, Nt):
            x, r = update_step(x, dt=dt, W=W, phi_torch=phi_torch,
                               noise_sigma=noise_sigma, runga_kutta=runga_kutta)
            if i % eval_iter == 0:
                x_save[i // eval_iter] = x
                r_save[i // eval_iter] = r
        x_save = x_save[int(T_save_delay/dt_save):]
        r_save = r_save[int(T_save_delay/dt_save):]
        T_save_delay = 0

        x_s += x_save.sum(0)
        xx_s += torch.einsum('tbi, tbj -> bij', x_save, x_save)
        r_s += r_save.sum(0)
        rr_s += torch.einsum('tbi, tbj -> bij', r_save, r_save)
        T_samples += x_save.shape[0]

    # Global (pooled) second-moment sums across batches
    XX_all = xx_s.sum(dim=0)  # [N,N]
    RR_all = rr_s.sum(dim=0)  # [N,N]

    # Global sums (for the mean) across batches
    X_all = x_s.sum(dim=0)  # [N]
    R_all = r_s.sum(dim=0)  # [N]

    T_tot = T_samples * N_batch
    mu_x_bar = X_all / T_tot
    mu_r_bar = R_all / T_tot

    # Pooled unbiased covariance across ALL samples in ALL batches
    x_xcov_mean = (XX_all - T_tot * torch.outer(mu_x_bar, mu_x_bar)) / (T_tot - 1)
    r_xcov_mean = (RR_all - T_tot * torch.outer(mu_r_bar, mu_r_bar)) / (T_tot - 1)

    # (optional) small numerical symmetrization
    x_xcov_mean = 0.5 * (x_xcov_mean + x_xcov_mean.T)
    r_xcov_mean = 0.5 * (r_xcov_mean + r_xcov_mean.T)

        #x_xcov = torch.einsum('tbi, tbj -> bij', x_save, x_save) / (x_save.shape[0] - 1)
        #r_xcov = torch.einsum('tbi, tbj -> bij', r_save, r_save)/(r_save.shape[0] -1)
        #x_xcovs_mean += x_xcov/N_loops
        #r_xcovs_mean += r_xcov/N_loops
    #x_xcov_mean = x_xcov_mean.mean(dim=0)
    #r_xcov_mean = r_xcov_mean.mean(dim=0)

    if return_raw_covs:
        return x_xcov_mean.cpu().detach().numpy(), r_xcov_mean.cpu().detach().numpy()

    eigs_x, vecs_x = torch.linalg.eigh(x_xcov_mean)
    eigs_r, vecs_r = torch.linalg.eigh(r_xcov_mean)
    if not return_vecs:
        return eigs_x.cpu().detach().numpy(), eigs_r.cpu().detach().numpy()
    else:
        return (eigs_x.cpu().detach().numpy(),
                eigs_r.cpu().detach().numpy(),
                vecs_x.cpu().detach().numpy(),
                vecs_r.cpu().detach().numpy())


