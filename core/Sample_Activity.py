import numpy as np
import torch

phi_torch = lambda x: torch.erf((np.sqrt(np.pi)/2)*x)
def sample_activity(T_sim, dt_save, dt, W, phi_torch=phi_torch,
                   N_batch=1, T_save_delay=100):
    """Thanks to David Clark"""

    device = W.device
    N = W.shape[0]
    eval_iter = int(dt_save / dt)
    Nt = int(T_sim / dt)
    N_save = int(T_sim / dt_save)
    #define stuff
    x_save = torch.zeros(N_save, N_batch, N, device=device)
    x = torch.randn(N_batch, N, device=device) * torch.std(W).item() * np.sqrt(N)
    x_save[0] = x
    r_save = torch.zeros_like(x_save)
    r_save[0] = phi_torch(x)
    #run
    for i in range(1, Nt):
        r = phi_torch(x)
        x += dt*(-x + torch.mm(r, W.T))
        if i % eval_iter == 0:
            x_save[i//eval_iter] = x
            r_save[i//eval_iter] = r

    x_ret = x_save[int(T_save_delay/dt_save):].cpu().detach().numpy().squeeze()
    r_ret = r_save[int(T_save_delay/dt_save):].cpu().detach().numpy().squeeze()
    return x_ret, r_ret