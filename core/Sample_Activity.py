import numpy as np
import torch
from .Update_Step import update_step

phi_torch = lambda x: torch.erf((np.sqrt(np.pi)/2)*x)
def sample_activity(T_sim, dt_save, dt, W, phi_torch=phi_torch, avg_activity=False, x0=None,
                   N_batch=1, T_save_delay=100, runga_kutta=False, noise_sigma=0):
    """Thanks to David Clark"""

    device = W.device
    N = W.shape[0]
    eval_iter = int(dt_save / dt)
    Nt = int(T_sim / dt)
    N_save = int(T_sim / dt_save)
    #define stuff
    x_save = torch.zeros(N_save, N_batch, N, device=device)
    if x0 is None:
        x = torch.randn(N_batch, N, device=device) * torch.std(W).item() * np.sqrt(N)
    else:
        x = x0
    x_save[0] = x
    r_save = torch.zeros_like(x_save)
    r_save[0] = phi_torch(x)
    #run
    if avg_activity:
        x_to_avg = torch.zeros(eval_iter, N, device=device)
        r_to_avg = torch.zeros(eval_iter, N, device=device)
        x_to_avg[0] = x
        r_to_avg[0] = r_save[0]
    for i in range(1, Nt):
        x, r = update_step(x, dt=dt, W=W, phi_torch=phi_torch,
                           noise_sigma=noise_sigma, runga_kutta=runga_kutta)

        if avg_activity:
            x_to_avg[i % eval_iter] = x
            r_to_avg[i % eval_iter] = r
            if i % eval_iter == 0:
                x_save[i // eval_iter] = torch.mean(x_to_avg, dim=0)
                r_save[i // eval_iter] = torch.mean(r_to_avg, dim=0)
        else:
            if i % eval_iter == 0:
                x_save[i//eval_iter] = x
                r_save[i//eval_iter] = r

    x_ret = x_save[int(T_save_delay/dt_save):].cpu().detach().numpy().squeeze()
    r_ret = r_save[int(T_save_delay/dt_save):].cpu().detach().numpy().squeeze()
    return x_ret, r_ret