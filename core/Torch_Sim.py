import numpy as np
import torch

phi_torch = lambda x: torch.erf((np.sqrt(np.pi)/2)*x)
def run_torch_sim(T_sim, dt_save, dt, W, N_batch=1, T_save_delay=100):
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
    r_lpf = x.clone()
    r_lpf_save = torch.zeros_like(x_save)
    r_lpf_save[0] = r_lpf
    #run
    for i in range(1, Nt):
        r = phi_torch(x)
        r_lpf += dt*(-r_lpf + r)
        x += dt*(-x + torch.mm(r, W.T))
        if i % eval_iter == 0:
            x_save[i//eval_iter] = x
            r_lpf_save[i//eval_iter] = r_lpf

    x_ret = x_save[int(T_save_delay/dt_save):].cpu().detach().numpy().squeeze()
    r_lpf_ret = r_lpf_save[int(T_save_delay/dt_save):].cpu().detach().numpy().squeeze()
    return x_ret, r_lpf_ret

def run_online_cov_sim(T_sim, dt_save, dt, W, N_batch=1, T_save_delay=100,
                       dt_PR=1, compute_PRs=False):
    """Thanks to David Clark"""

    device = W.device
    N = W.shape[0]
    eval_iter = int(dt_save / dt)
    PR_iter = int(dt_PR / dt)
    Nt = int(T_sim / dt)
    N_save = int(T_sim / dt_save)
    #define stuff
    x_save = torch.zeros(N_save, N_batch, N, device=device)
    x = torch.randn(N_batch, N, device=device) * torch.std(W).item() * np.sqrt(N)
    x_save[0] = x
    #run
    cov_sum = torch.zeros(N_batch, N, N, device=device)
    PRs = []
    PR_nums = []
    PR_doms_1 = []
    PR_doms_2 = []
    n_cov = 0
    for i in range(1, Nt):
        r = phi_torch(x)
        x += dt*(-x + torch.mm(r, W.T))
        if i*dt >= T_save_delay and (i % PR_iter == 0):
            n_cov += 1
            cov_sum += torch.einsum('bi, bj -> bij', r, r)
            cov = cov_sum/n_cov
            x_save[i // eval_iter] = x
            if compute_PRs:
                PR_num = torch.trace(cov[0])**2
                PR_dom_1 = torch.trace(cov[0]**2)
                PR_dom_2 = torch.sum(cov[0]**2) - PR_dom_1
                #PR = torch.trace(cov[0])**2 / torch.sum(cov[0]**2) / N
                PR = PR_num / (PR_dom_1 + PR_dom_2) / N
                PRs.append(PR)
                PR_nums.append(PR_num)
                PR_doms_1.append(PR_dom_1)
                PR_doms_2.append(PR_dom_2)

    print(n_cov)
    cov = cov_sum / (n_cov - 1)

    cov_ret = cov.cpu().detach().numpy().squeeze()
    x_ret = x_save[:].cpu().detach().numpy().squeeze()
    if compute_PRs:
        return cov_ret, PRs, PR_nums, PR_doms_1, PR_doms_2
    else:
        return cov_ret, x_ret