import torch
from torch import tensor

# ------------- Utility functions -------------
def phi_torch(x):
    """Saturating nonlinearity: erf((sqrt(pi)/2)*x)."""
    return torch.erf((torch.sqrt(torch.tensor(torch.pi, device=x.device)) / 2) * x)

def phi_prime_torch(x):
    """Derivative of erf((sqrt(pi)/2)*x) = exp(-(pi/4)*x^2)."""
    return torch.exp(-(torch.pi/4) * x**2)

def sample_gp_torch(cov, M, device):
    """Sample M trajectories from a GP with zero mean and covariance 'cov'."""
    T = cov.shape[0]
    L = torch.linalg.cholesky(cov)
    noise = torch.randn(M, T, device=device, dtype=torch.float32)
    return noise @ L.T  # shape (M,T)

def sample_weights_torch(M, M_star, R, cov_mn_list, device):
    """
    Sample the weight sets m^(μ*,r) and n^(μ*,r) for each condensed pattern.
    """
    if isinstance(cov_mn_list, list):
        covs = torch.stack(cov_mn_list, dim=0)  # shape: (M_star, R, R)
    else:
        covs = cov_mn_list
    I_R = torch.eye(R, device=device, dtype=torch.float32).unsqueeze(0).repeat(M_star, 1, 1)
    top = torch.cat([I_R, covs], dim=2)  # (M_star, R, 2R)
    bot = torch.cat([covs.transpose(1,2), I_R], dim=2)  # (M_star, R, 2R)
    cov_full = torch.cat([top, bot], dim=1)  # (M_star, 2R, 2R)
    mean = torch.zeros(M_star, 2*R, device=device, dtype=torch.float32)
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_full)
    samples = mvn.sample((M,))  # shape: (M, M_star, 2R)
    m = samples[..., :R]
    n = samples[..., R:]
    return m, n

# ------------- The main tangent-linear integrators -------------
def integrate_x_with_jacobian(eta_x, m, z_condensed, R_z, dt):
    """
    Integrate the x dynamics *and* compute d x(t)/d I_x(s) in one pass.
    Suppose the x-dynamics is:
      x[t+1] = x[t] + dt * ( -x[t] + sum_{μ*,r} [m^(μ*,r)*z^(μ*)_r(t)] + dt * Σ_{u=0}^{t} R_z[t,u]*phi(x[u]) )
    We store x in shape (M,T), and the Jacobian J_x in shape (M,T,T).

    For each trajectory i, define J_x[i,t,s] = d x[i,t] / d I[i,s].
    We'll assume a *scalar* impulse for each time index. If you want a vector impulse, add more dimensions.
    """
    M, T = eta_x.shape
    device = eta_x.device

    # State and nonlinearity
    x = torch.zeros(M, T, device=device)
    phi_x = torch.zeros(M, T, device=device)
    # Jacobian: J_x[i,t,s] = partial x[i,t] wrt impulse at time s
    J_x = torch.zeros(M, T, T, device=device)

    # For convenience, let's define the "lower-triangular" version of R_z:
    Rz_lower = torch.tril(R_z)

    # Initialize
    phi_x[:, 0] = phi_torch(x[:, 0])

    for t in range(T - 1):
        # 1) main update
        # coupling from condensed patterns:
        coupling = torch.sum(m * z_condensed[:, :, t].unsqueeze(0), dim=(1,2))  # shape (M,)
        # convolution with R_z:
        conv = dt * torch.sum(phi_x[:, :t+1] * Rz_lower[t, :t+1].unsqueeze(0), dim=1)
        # dxdt = (-x + coupling + conv + eta_x[:,t])
        dxdt = -x[:, t] + coupling + conv + eta_x[:, t]

        # Euler step
        x[:, t+1] = x[:, t] + dt * dxdt
        phi_x[:, t+1] = phi_torch(x[:, t+1])

        # 2) derivative update for each s <= t
        # For a purely Euler equation:
        #   x[t+1] = x[t] + dt * F(x[t]),
        # we get
        #   J_x[t+1,s] = J_x[t,s] + dt * dF/dx[t] * J_x[t,s].
        #
        # Let's define partial(F)/partial(x[t]):

        # partial of dxdt wrt x[t] is:
        #   d(-x[t])/dx[t] = -1
        # + partial of conv wrt x[t], which is dt * R_z[t,t] * phi'(x[t]) if t is in range.
        #   Because conv = dt * Σ_{u=0..t} R_z[t,u]*phi(x[u]),
        #   only the term u=t depends on x[t].
        #   So partial wrt x[t] = dt * R_z[t,t] * phi'( x[t] ).
        # coupling is from z_condensed(t) which doesn't depend on x[t] in this single-site eqn
        #   (assuming z_condensed is not updated within the same single-step).
        #   If it does, you'd add partial derivatives there too.

        dFdx = -1.0
        # add the piece from conv:
        dFdx += dt * Rz_lower[t, t] * phi_prime_torch(x[:, t])  # shape (M,)

        # So d x[t+1]/d x[t] = 1 + dt * dFdx.  But we used x[t+1] = x[t] + dt*F(x[t]).
        # Actually in the Euler scheme:
        #   x[t+1] = x[t] + dt * F(...)
        # partial wrt x[t]: = 1 + dt * partial(F)/partial(x[t]).
        # => = 1 + dt*dFdx
        # So let's define:
        partial_x_next_wrt_x = 1.0 + dt * dFdx  # shape (M,)

        for s in range(t+1):
            # J_x[i,t+1,s] = partial_x_next_wrt_x[i] * J_x[i,t,s]
            J_x[:, t+1, s] = partial_x_next_wrt_x * J_x[:, t, s]

        # 3) If you define an impulse at exactly time t (meaning "I[t]" enters x[t+1] directly),
        #    then you add + dt to J_x[t+1, t].
        #    If you want the impulse to appear at x[t], do it differently.
        #
        # In your code, you might say: x[t+1] = x[t] + dt*(F + I[t]).
        # => partial wrt I[t] = dt.  So:
        J_x[:, t+1, t] += dt

    return x, phi_x, J_x

def integrate_z_with_jacobian(eta_z, D, C_mat, S_phi, dt):
    """
    Integrate the z–dynamics AND compute a 5D Jacobian in one pass, vectorized over M.

    z–equation (feed–forward):
      z[t+1] = η_z[t] + D[i] * [ dt * Σ_{u=0..t} S_phi[t,u] * z[i,:,u] ] @ C_mat[i]^T

    We now allow an R–dimensional impulse at each time s. So J_z has shape (M,T,T,R,R):
      J_z[i,t,s,r,r'] = ∂ z[i,t,r] / ∂ I[i,s,r'].

    Parameters:
      eta_z: (M,T)    – the noise for z
      D    : (M,)     – per–trajectory scalar
      C_mat: (M,R,R)  – per–trajectory coupling matrix
      S_phi: (T,T)    – feed–forward temporal kernel
      dt   : float    – time step

    Returns:
      z   : (M,R,T)         – the integrated z over time
      J_z : (M,T,T,R,R)     – the 5D Jacobian
    """
    M, T = eta_z.shape
    R = C_mat.shape[1]
    device = eta_z.device

    # z[i,r,t]
    z = torch.zeros(M, R, T, device=device)

    # J_z[i,t,s,r,r'] = partial z[i,t,r]/partial I[i,s,r']
    J_z = torch.zeros(M, T, T, R, R, device=device)

    # We'll zero out the "above–diagonal" part of S_phi so we only sum up to t>=u
    S_phi_lower = torch.tril(S_phi)

    # Initialize z at t=0 (choose your own initial condition)
    z[:, :, 0] = torch.randn(M, R, device=device)

    # Pre–transpose the coupling matrix for convenience
    # so we can do bmm(..., C_mat_T)
    C_mat_T = C_mat.transpose(1, 2)  # shape (M,R,R)

    for t in range(T - 1):
        # ---------------------------------------------------------------------
        # 1) Compute z[:, :, t+1]
        #    conv[i,:] = D[i] * [ dt * Σ_{u=0..t} S_phi[t,u]*z[i,:,u] ] @ C_mat[i]^T
        # ---------------------------------------------------------------------
        # sum_{u=0..t} S_phi_lower[t,u] * z[i,:,u] => vectorize over M, R
        # shape z[:, :, :t+1] => (M,R,t+1), but let's rearrange to (M,t+1,R) to use einsum
        z_slice = z[:, :, :t+1].transpose(1, 2)  # (M, t+1, R)
        # S_phi_lower[t,:t+1] => shape (t+1,)

        # sum_j[i,r] = dt * Σ_{u=0..t} [ S_phi_lower[t,u] * z[i,r,u] ]
        # vectorized with einsum: 'u, b u r -> b r'
        conv_sum = dt * torch.einsum('u,bur->br', S_phi_lower[t, :t+1], z_slice)
        # shape (M,R)

        # Multiply by C_mat^T and D
        # conv[i,:] = conv_sum[i,:].unsqueeze(0) @ C_mat_T[i] => shape (M,R)
        conv_sum_2d = conv_sum.unsqueeze(1)         # (M,1,R)
        tmp = torch.bmm(conv_sum_2d, C_mat_T)       # (M,1,R)
        tmp = tmp.squeeze(1)                        # (M,R)
        tmp = tmp * D.unsqueeze(1)                  # scale each row by D[i]

        # z[t+1] = tmp + eta_z[t]
        z[:, :, t+1] = tmp + eta_z[:, t].unsqueeze(1)

        # ---------------------------------------------------------------------
        # 2) Compute derivative J_z[:, t+1, s, :, :]
        #    for each s in [0..t], partial z[t+1,r]/partial I[s,r']
        #
        # z[t+1,r] depends on z[u,*], which depends on I[s,r'].
        # => J_z[t+1,s,r,r'] = dt*D[i] * Σ_{u=0..t} [ S_phi[t,u]*J_z[u,s,r,r'] ] @ C_mat[i]^T
        # ---------------------------------------------------------------------
        for s_ in range(t+1):
            # sum_j[i,r,r'] = dt * Σ_{u=0..t} [ S_phi_lower[t,u] * J_z[i,u,s_,r,r'] ]
            # shape J_z[:, :t+1, s_, :, :] => (M, t+1, R, R)
            # We want to sum over u => output (M,R,R).
            j_slice = J_z[:, :t+1, s_, :, :]  # shape (M, t+1, R, R)

            # sum over u with einsum: 'u, buxy->bxy'
            # We'll rename x->r, y->r' for clarity:
            # 'u, b u r r' -> b r r
            sum_j = dt * torch.einsum('u,burp->brp', S_phi_lower[t, :t+1], j_slice)
            # shape (M,R,R)

            # multiply by C_mat^T => (M,R,R)
            tmp2 = torch.bmm(sum_j, C_mat_T)  # shape (M,R,R)
            tmp2 = tmp2 * D.unsqueeze(1).unsqueeze(2)  # scale by D[i]

            # store in J_z[:, t+1, s_, :, :]
            J_z[:, t+1, s_, :, :] = tmp2

        # If you want a direct impulse on z[t+1], e.g. "z[t+1] += dt*I[t]",
        # you'd add J_z[:, t+1, t, :, :] += ??? here.

    return z, J_z



# ------------- Update/Compute Order Parameters -------------
def update_order_parameters_x(phi_x):
    """C^phi(t,t') = 1/M sum_{i} phi_x[i,t]*phi_x[i,t']"""
    M, T = phi_x.shape
    return (phi_x.transpose(0,1) @ phi_x) / M  # (T,T)

def update_condensed_patterns(n, phi_x, D_condensed):
    """
    z^(μ*)_r(t) = average over i of n^(μ*,r)[i] * phi_x[i,t].
    shape(n) = (M, M_star, R), shape(phi_x)=(M,T)
    output shape = (M_star,R,T).
    """
    M, M_star, R = n.shape
    _, T = phi_x.shape
    # z_condensed[m_star, r, t] = sum_i n[i,m_star,r]*phi_x[i,t] / M
    z_condensed = torch.einsum('imr,it->mrt', n, phi_x) / M * D_condensed[:,None,None]
    return z_condensed

def update_order_parameters_z(z, alpha):
    """
    Q^z(t,t') = alpha * sum_r < z_r(t) z_r(t') >_M
    shape(z)=(M,R,T)
    """
    M, R, T = z.shape
    Qz_new = torch.zeros(T, T, device=z.device, dtype=z.dtype)
    for r in range(R):
        Qz_new += z[:, r, :].T @ z[:, r, :]
    Qz_new = alpha * Qz_new / M
    return Qz_new

def update_with_memory(O_old, O_new, gamma):
    return (1 - gamma)*O_old + gamma*O_new

# ------------- Final: Use J_x, J_z to get S_phi, R_z -------------
def compute_S_phi_from_J_x(x, J_x):
    """
    S^phi(t,s) = (1/M) * sum_i [ phi'(x[i,t]) * J_x[i,t,s] ]
    where x.shape=(M,T), J_x.shape=(M,T,T).
    """
    M, T = x.shape
    phi_prime = phi_prime_torch(x)  # shape (M,T)
    # Einsum: 'mt, mts -> ts' sums over the batch dimension 'm' and matches 't',
    # leaving 't,s' as output.
    S_phi = torch.einsum('mt,mts->ts', phi_prime, J_x) / M
    return S_phi


def compute_R_z_from_J_z(D, C_mat, J_z, alpha):
    """
    R^z(t,s) = alpha * average_i [ D[i] * sum_{r,r'} C_mat[i,r,r'] * J_z[i,t,s,r,r'] ].

    Here:
      J_z.shape = (M,T,T,R,R)
      C_mat.shape = (M,R,R)
      D.shape     = (M,)
    Output: (T,T)
    """
    M, T, _, R, _ = J_z.shape
    device = J_z.device

    # We'll broadcast multiply:
    #   C_mat.unsqueeze(1).unsqueeze(1) => shape (M,1,1,R,R)
    #   J_z => shape (M,T,T,R,R)
    # => product => shape (M,T,T,R,R).
    # Then we sum over (r,r') => shape (M,T,T).
    temp = (C_mat.unsqueeze(1).unsqueeze(1) * J_z).sum(dim=(4,3))  # sum over r,r'
    # Now temp.shape = (M,T,T).

    # multiply by D[i], shape => (M,1,1)
    temp = temp * D.view(M,1,1)

    # average over M => (T,T)
    R_z_est = alpha * temp.mean(dim=0)

    return R_z_est


# ------------- Main DMFT with on-the-fly derivative -------------
def simulate_dmft_torch_with_tangent(num_iterations=10, T=50, dt=0.1, M=10, M_star=2, R=3, gamma=0.1, alpha=1.0,
                                     device=None, cov_mn_list=None, D_condensed=None, D_z_all=None, C_mat_all=None):
    """
    A single pass approach: each iteration, we:
      1) sample GP noise for x
      2) sample m,n for condensed patterns
      3) integrate x plus its derivative wrt impulses -> get x, phi_x, J_x
      4) from J_x, compute S^phi
      5) update z_condensed
      6) sample GP noise for z
      7) integrate z plus its derivative wrt impulses -> get z, J_z
      8) from J_z, compute R^z
      9) from z, compute Q^z
      10) from x, compute C^phi
      11) update all order params with memory
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize order parameters
    C_phi = torch.eye(T, device=device)
    S_phi = torch.zeros(T, T, device=device)
    Q_z   = torch.eye(T, device=device)
    R_z   = torch.zeros(T, T, device=device)
    z_condensed = torch.zeros(M_star, R, T, device=device)

    if cov_mn_list is None:
        cov_mn_list = [torch.eye(R, device=device) for _ in range(M_star)]
    if D_condensed is None:
        D_condensed = torch.ones(M_star, device=device)
    if D_z_all is None:
        D_z_all = torch.ones(M, device=device)
    if C_mat_all is None:
        C_mat_all = torch.eye(R, device=device).unsqueeze(0).repeat(M,1,1)

    for it in range(num_iterations):
        print(f"DMFT iteration {it+1}/{num_iterations}")

        # ---- x part ----
        eta_x = sample_gp_torch(Q_z, M, device)  # shape (M,T)
        m, n = sample_weights_torch(M, M_star, R, cov_mn_list, device)
        # Integrate x + derivative
        x, phi_x, J_x = integrate_x_with_jacobian(eta_x, m, z_condensed, R_z, dt)

        if torch.isinf(x).any() or torch.isnan(x).any():
            import matplotlib.pyplot as plt
            import numpy as np
            x_numpy = x.cpu().numpy()
            plt.plot(np.amax(x_numpy, axis=0))
            plt.title('x')
            break
        # from J_x => S_phi
        S_phi_new = compute_S_phi_from_J_x(x, J_x)
        # from x => C^phi
        C_phi_new = update_order_parameters_x(phi_x)
        # from x, n => z_condensed
        z_condensed_new = update_condensed_patterns(n, phi_x, D_condensed)

        # --- memory updates ---
        S_phi = update_with_memory(S_phi, S_phi_new, gamma)
        C_phi = update_with_memory(C_phi, C_phi_new, gamma)
        z_condensed = update_with_memory(z_condensed, z_condensed_new, gamma)

        S_lambda = torch.linalg.eigvals(S_phi)
        C_lambda = torch.linalg.eigvals(C_mat_all)
        lambs = D_z_all[:,None,None] * S_lambda[None,:,None] * C_lambda[:,None,:]
        print(S_lambda)
        print(C_lambda)
        print(torch.amax(torch.abs(lambs)))
        if torch.amax(torch.abs(lambs)) >= 1.0:
            print('unstable S')

        # ---- z part ----
        eta_z = sample_gp_torch(C_phi, M, device)
        z, J_z = integrate_z_with_jacobian(eta_z, D_z_all, C_mat_all, S_phi, dt)

        if torch.isinf(z).any() or torch.isnan(z).any():
            import matplotlib.pyplot as plt
            import numpy as np
            z_numpy = torch.flatten(z,start_dim=0,end_dim=1).cpu().numpy()
            plt.plot(np.amax(np.abs(z_numpy), axis=0))
            plt.title('z')
            break

        # from z => Q^z
        Q_z_new = update_order_parameters_z(z, alpha)
        # from J_z => R^z
        R_z_new = compute_R_z_from_J_z(D_z_all, C_mat_all, J_z, alpha)

        # ---- memory updates ----
        Q_z = update_with_memory(Q_z, Q_z_new, gamma)
        R_z = update_with_memory(R_z, R_z_new, gamma)

        # optionally check norms to see if system is stable
        # e.g., norm_z = z.abs().mean().item()
        # print("z norm:", norm_z)

    return x, z, C_phi, S_phi, Q_z, R_z, z_condensed
