#!/usr/bin/env python3

import numpy as np
import math

def angular_frequencies(N, T):
    dt = T / N
    f = np.fft.fftfreq(N, d=dt)  # cycles per unit time
    w = 2.0 * np.pi * f  # angular frequency
    return w


def erf_gain_from_q(q):
    # phi(x) = erf(√π x / 2) => a = √π/2
    # ⟨ϕ'⟩ = (2a/√π)/sqrt(1 + 2 a^2 q). Here 2a/√π = 1, 2 a^2 = π/2.
    return 1.0 / math.sqrt(1.0 + (math.pi / 2.0) * max(q, 0.0))


def Cphi_from_Cx_time(Cx_tau):
    # Map Cx(τ) -> Cphi(τ) using closed-form arcsin for erf.
    # Cphi = (2/π) * arcsin( ((π/2) q ρ) / (1 + (π/2) q) )
    q = float(Cx_tau[0])
    if q <= 0:
        q = 1e-9
    rho = Cx_tau / q
    num = (math.pi / 2.0) * q * rho
    den = 1.0 + (math.pi / 2.0) * q
    t = np.clip(num / den, -1.0, 1.0)
    return (2.0 / math.pi) * np.arcsin(t)

def solve_spontaneous(
        N=1200, T=60.0,
        alpha=0.5, R=2, D_bulk=2.0, gamma=0.99,
        g=None,  # if provided, g_eff := g; else g_eff := sqrt(alpha*R)*D
        max_iters=500, tol=1e-9, mix=0.5
):
    # grids
    w = angular_frequencies(N, T)
    half = N // 2 + 1

    # decide g_eff
    if g is None:
        g_eff = math.sqrt(alpha * R) * D_bulk

    else:
        g_eff = float(g)

    # initialize Cx(ω): small flat spectrum
    Cx_w = np.full(N, 1e-6, dtype=float)

    last_q = None
    for it in range(max_iters):
        # IFFT to time domain
        Cx_tau = np.fft.ifft(Cx_w).real
        q = float(Cx_tau[0])

        # avg gain
        gain = erf_gain_from_q(q)

        # map to Cphi(τ)
        Cphi_tau = Cphi_from_Cx_time(Cx_tau)

        # FFT back
        Cphi_w = np.fft.fft(Cphi_tau).real

        # update per Eq. (116): Cx(ω) = g_eff^2 * Cphi(ω) / [ 1 - (γ D ⟨ϕ′⟩)^2 + ω^2 ]
        denom_const = 1.0 - (gamma * D_bulk * gain) ** 2
        denom = denom_const + w**2
        denom = np.maximum(denom, 1e-12)  # numeric guard
        Cx_new_w = (g_eff ** 2) * (Cphi_w / denom)

        # simple under-relaxed mixing
        Cx_w = (1.0 - mix) * Cx_w + mix * Cx_new_w

        # convergence check on q and spectrum
        Cx_tau_new = np.fft.ifft(Cx_w).real
        q_new = float(Cx_tau_new[0])
        dq = abs(q_new - q)
        err_spec = np.linalg.norm(Cx_new_w - Cx_w) / (np.linalg.norm(Cx_w) + 1e-12)

        if last_q is not None and max(dq, err_spec) < tol:
            break
        last_q = q_new

    out = {
        "Cx_w": Cx_w,
        "Cphi_w": Cphi_w,
        "Cx_tau": Cx_tau_new,
        "Cphi_tau": Cphi_tau,
        "w": w,
        "q": q_new,
        "gain": gain,
        "g_eff": g_eff,
        "iters": it + 1,
        "denom_const": denom_const,
    }
    return out

def solve_condensed(
        N=4096, T=200.0,
        alpha=1.0, R=2,
        D_bulk=2.0, D0=3.0,
        gamma=0.99, omega_star=0.6,
        g=None, mix=0.25, iters=800, tol=1e-9, plus=True
):
    # grids
    w = angular_frequencies(N, T)
    t = np.arange(N) * (T / N)
    cos_theta = 1.0 / math.sqrt(1.0 + omega_star ** 2)

    res_spont = solve_spontaneous(N=N, T=T,
                                  alpha=alpha, R=R, D_bulk=D_bulk, gamma=gamma,
                                  g=None,  # if provided, g_eff := g; else g_eff := sqrt(alpha*R)*D
                                  max_iters=500, tol=1e-9, mix=0.5)

    # feasibility checks
    if gamma * D0 * cos_theta * res_spont['gain'] <= 1.0:
        # print("No condensed solution: need gamma*D0*cos(theta*) <phi'> > 1.")
        res_spont['spont'] = True
        return res_spont

    # target q from marginality
    q_target = (2.0 / math.pi) * ((gamma * D0 * cos_theta) ** 2 - 1.0)

    # effective bulk coupling
    g_eff = math.sqrt(alpha * R) * D_bulk if g is None else float(g)

    # init
    S_noise = np.full(N, 1e-6, dtype=float)

    last_q = None
    for k in range(iters):
        # noise-only covariance
        Cx_noise = np.fft.ifft(S_noise).real
        q_noise = float(Cx_noise[0])

        # set coherent amplitude to hit marginality exactly
        P = max(0.0, q_target - q_noise)

        #rho = np.sqrt(P * (1 + omega_star ** 2))

        # total covariance in time
        Cx_tau = Cx_noise + P * np.cos(omega_star * t)

        # gain from total q
        q = float(Cx_tau[0])
        gain = erf_gain_from_q(q)

        # nonlinear mapping
        Cphi_tau = Cphi_from_Cx_time(Cx_tau)
        Sphi = np.fft.fft(Cphi_tau).real

        # bulk transfer (uses D_bulk and current gain)
        denom0 = 1.0 - (gamma * D_bulk * gain) ** 2  # should be >0 by feasibility
        denom = np.maximum(denom0 + w**2, 1e-14)

        S_pred = (g_eff ** 2) * (Sphi / denom)

        # remove coherent part in freq (computed numerically)
        Scoh = np.fft.fft(P * np.cos(omega_star * t)).real

        S_noise_new = S_pred

        # conservative mixing (helps conditioning near ω=0)
        S_noise = (1.0 - mix) * S_noise + mix * S_noise_new

        # convergence: spectrum + q
        Cx_noise_new = np.fft.ifft(S_noise).real
        q_noise_new = float(Cx_noise_new[0])
        q_total_new = q_noise_new + P
        dq = abs(q_total_new - q)
        dspec = np.linalg.norm(S_noise_new - S_noise) / (np.linalg.norm(S_noise) + 1e-12)
        if last_q is not None and max(dq, dspec) < tol:
            break
        last_q = q_total_new

    return dict(
        iters=k + 1, w=w, t=t,
        S_noise=S_noise, P=P, S_pred=S_pred, Scoh=Scoh,
        Cx_tau=Cx_noise_new + P * np.cos(omega_star * t),
        q=q_total_new, gain=erf_gain_from_q(q_total_new),
        g_eff=g_eff, q_target=q_target, denom0=denom0,
        spont=False
    )