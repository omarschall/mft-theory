import numpy as np
from scipy.special import erf
from scipy.optimize import root
from scipy.integrate import odeint

phi_numpy = lambda x: erf((np.sqrt(np.pi)/2)*x)
phi_prime_numpy = lambda x: np.exp(-(np.pi/4)*x**2)

def compute_C(Delta_11, Delta_22, Delta_12): #Delta_mn = variances
    A = (Delta_11 + (2./np.pi)) * (Delta_22 + (2./np.pi))
    in_sqrt = np.clip(A - Delta_12**2, a_min=1e-20, a_max=np.inf)
    return (2./np.pi)*np.arctan(Delta_12 / (np.sqrt(in_sqrt)))

def compute_C_simple(Delta_0, Delta):
    return compute_C(Delta_0, Delta_0, Delta)

def compute_C_prime(Delta_0, Delta):
    return 2/np.sqrt((np.pi*Delta_0 + 2)**2 - (np.pi*Delta)**2)

def compute_phi_prime_avg(Delta_0):
    return 1/np.sqrt(1 + (np.pi*Delta_0)/2)

def compute_C_antideriv(Delta_0, Delta):
    A = (Delta_0 + (2/np.pi))**2
    t1 = np.sqrt(A - Delta**2)
    t2 = Delta*np.arctan(Delta / t1)
    stuff = t1 + t2
    return (2/np.pi)*stuff

def compute_potential(Delta, Delta_0, g, A):
    Phi = compute_C_antideriv(Delta_0, Delta)
    V = -.5*Delta**2 + (g**2)*Phi + A*np.abs(Delta)
    return V

C_fn = compute_C_simple
Phi_fn = compute_C_antideriv

def compute_V(Delta_0, Delta, g, leak=1):
    t1 = -0.5*leak*(Delta**2 - Delta_0**2)
    t2 = (g**2)*(Phi_fn(Delta_0, Delta) - Phi_fn(Delta_0, Delta_0))
    return t1+t2

def compute_V_deriv(Delta_0, Delta, g, leak=1):
    t1 = -leak*Delta
    t2 = (g**2)*C_fn(Delta_0, Delta)
    return t1+t2

def compute_Delta_0(g, x0=None, leak=1):
    def f_opt(Delta_0):
        Delta_0 = np.abs(Delta_0)
        V = compute_V(Delta_0=Delta_0, Delta=0., g=g, leak=leak)
        return V
    res = root(fun=f_opt, x0=g**2 if x0 is None else x0)
    if res.success:
        Delta_0 = res.x.item().__abs__()
        return Delta_0
    else:
        return np.nan

def fix(Delta):
    #takes in NON SYMMETRIZED Delta!
    Delta = Delta.copy()
    #prevent from turning around and/or going negative
    deriv = np.diff(Delta)
    if np.any(deriv > 0):
        i = np.argmax(deriv > 0)
        Delta[i:] = Delta[i]
    if np.any(Delta < 0):
        i = np.argmax(Delta < 0)
        Delta[i:] = Delta[i-1]
    return Delta

def symmetrize(Delta):
    return np.concatenate((Delta, (Delta[-1],), Delta[1:][::-1]))

def integrate_potential(Delta_0, g, leak=1, tau_max=40, N_tau=1000,
                        driving_term=None):
    def dy_dt(y, t):
        Delta, Delta_dot = y[0], y[1]
        V_deriv = compute_V_deriv(
            Delta_0=Delta_0, Delta=Delta, g=g, leak=leak)
        if driving_term is not None:
            y_dot = np.array([Delta_dot, -V_deriv + driving_term[t]])
        else:
            y_dot = np.array([Delta_dot, -V_deriv])
        return y_dot
    t_vals = np.linspace(0, tau_max, N_tau)
    out = odeint(dy_dt, y0=np.array([Delta_0, 0.]), t=t_vals, rtol=1e-12)
    Delta = out[:, 0]
    return t_vals, Delta

def compute_psi_theory(Delta, g, dt):
    #SUBTRACT 1 from M for PRL results!
    w = np.fft.fftfreq(n=len(Delta), d=dt) * 2*np.pi
    df = w[1] - w[0]
    w1 = w.reshape((1, len(w)))
    w2 = w1.T

    Delta_ft = np.fft.fft(Delta, norm='backward') * dt / np.sqrt(2*np.pi)
    C_phi_ft = (1 + w**2) * Delta_ft / g**2
    C_phi = np.fft.ifft(C_phi_ft, norm='forward').real*df/np.sqrt(2*np.pi) #don't actually need
    C_phi_ft_12 = np.outer(C_phi_ft, C_phi_ft)

    alpha = compute_phi_prime_avg(Delta[0])
    S_phi_ft = alpha/(np.sqrt(2*np.pi) * (1 + 1j*w))
    S_phi_ft_12 = np.outer(S_phi_ft, S_phi_ft)

    M = 1./np.abs(1 - 2*np.pi*(g**2)*S_phi_ft_12)**2 - 1
    psi_phi_ft = M*C_phi_ft_12

    psi_phi = (np.fft.ifft2(psi_phi_ft, norm='forward') * df**2 / (2*np.pi)).real

    return psi_phi

