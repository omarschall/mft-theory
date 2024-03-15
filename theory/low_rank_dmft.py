import numpy as np
from scipy.special import erf

#Define nonlinearity and its derivative
phi = lambda x: erf((np.sqrt(np.pi)/2)*x)
phi_prime = lambda x: np.exp(-(np.pi/4)*x**2)

#Define gaussian integral functions
def compute_C(Delta_11, Delta_22, Delta_12): #Delta_mn = variances
    A = (Delta_11 + (2./np.pi)) * (Delta_22 + (2./np.pi))
    in_sqrt = np.clip(A - Delta_12**2, a_min=1e-20, a_max=np.inf)
    return (2./np.pi)*np.arctan(Delta_12 / (np.sqrt(in_sqrt)))

def compute_psi(Delta): #Delta = variance
    in_sqrt = np.clip(1./((np.pi/2.)*Delta + 1), a_min=1e-20, a_max=np.inf)
    return np.sqrt(in_sqrt)

def run_low_rank_dmft(g, s, T=10, dt=0.05, diag=True, sigma_nm=None, sigma_mm=None):
    """Simulate the low-rank DMFT equations."""
    Nt = int(T/dt)
    Delta = np.zeros((Nt, Nt))
    kappa = np.zeros((Nt, 2))
    psi = np.zeros(Nt)
    H = np.zeros((Nt, 2))
    Gamma = np.zeros((Nt, Nt))
    T = s*np.array([[1.6, -0.8],
                    [0.8, 1.6]])
    if sigma_nm is not None:
        T = s*sigma_nm
    U = s*np.eye(2)
    if sigma_mm is not None:
        U = s*sigma_mm
    Delta[0, 0] = 1.
    kappa[0, :] = np.array([0.1, -0.1])
    idx = np.arange(Nt)
    for j in range(1, Nt):
        psi[j] = compute_psi(Delta[j-1,j-1])
        H[j] = psi[j] * T.dot(kappa[j-1])
        kappa[j] = (1 - dt)*kappa[j-1] + dt*H[j]
        Delta[:j,j] = (1 - dt)*Delta[:j,j-1] + dt*Gamma[:j,j-1]
        if diag:
            Delta[j,j] = Gamma[j-1, j-1]
        else:
            Delta[j,j] = (1 - dt)*Delta[j,j-1] + dt*Gamma[j,j-1]
        C_vals = compute_C(Delta[idx,idx][:j], Delta[j,j], Delta[:j,j])
        A_vals = np.einsum('mn,tm,n->t', U, H[:j], H[j])
        for i in range(j+1): #t = i*dt
            Gamma[i,j] = ((1-dt)*Gamma[i-1,j]
                + dt*(g**2)*C_vals[i-1]
                + dt*A_vals[i-1])

    ret = {'Delta': Delta, 'kappa': kappa, 'psi': psi, 'H': H, 'Gamma': Gamma}
    return ret