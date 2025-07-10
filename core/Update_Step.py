import torch
import numpy as np

def update_step(x, dt, W, phi_torch, noise_sigma=0, runga_kutta=False, input_current=None):
    """General update step that can handle both Euler and Runge-Kutta integration
    for noise driven, non-noise driven, linear and non-linear networks."""

    r = phi_torch(x)
    if not runga_kutta:
        x += dt * (-x + torch.mm(r, W.T))
    else:
        # RK4
        x1 = x
        r1 = phi_torch(x1)
        k1 = -x1 + torch.mm(r1, W.T)

        x2 = x1 + dt * k1 / 2
        r2 = phi_torch(x2)
        k2 = -x2 + torch.mm(r2, W.T)

        x3 = x1 + dt * k2 / 2
        r3 = phi_torch(x3)
        k3 = -x3 + torch.mm(r3, W.T)

        x4 = x1 + dt * k3
        r4 = phi_torch(x4)
        k4 = -x4 + torch.mm(r4, W.T)

        x = x1 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    if noise_sigma > 0:
        x += np.sqrt(dt) * noise_sigma * torch.randn_like(x)

    if input_current is not None:
        x += dt * input_current

    return x, r