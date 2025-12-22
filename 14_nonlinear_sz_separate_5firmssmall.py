#!/usr/bin/env python
# nonlinear_RD_model_s_z.py
"""
Nonlinear Cournot with *separate* investments:
  - z_i: process R&D (reduces cost, spills via ω and φ)
  - s_i: product differentiation R&D (reduces substitutability via s_i^(1-φ) in Γ)

Equilibrium solves the block system:
  (2I + Γ(s)) q = (A - c) + z + φ Ω z
  z_i = q_i / κ_i                                      (process FOC)
  κ_i s_i^{3-φ} = (1-φ) q_i (Δ q)_i                    (diff. FOC; closed form)

Stability gate: ρ(Γ(s)) < 2 (Neumann-series sufficiency).
Outputs two figures:
  Fig 1 (4 panels): φ*, W, avg R&D (s+z), avg q
  Fig 2 (2 panels): avg s, avg z
"""

import math
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ---------------------------- 1. PARAMETERS ---------------------------
N          = 5                  # number of firms
A          = 5000              # demand intercept
GRID_RES   = 101                # 0 … 1 in steps of 0.01
PHI_TOL    = 1e-6               # φ convergence (optimizer)
NEWTON_TOL = 1e-8               # q/z/s convergence (∞-norm)
MAX_ITERS  = 60                 # block-iteration limit per φ
SEED       = 42                 # RNG seed for reproducibility
N_CORES    = -1                 # parallel jobs (-1 uses all cores)

rng    = np.random.default_rng(SEED)
c      = rng.normal(25.0, 5.0, N)
kappa  = np.clip(rng.normal(5.0, 1.0, N), 1e-3, None)   # strictly > 0

# ---------------------- 2. HELPERS & BUILDERS -------------------------
def build_similarity(mean: float, size: int) -> np.ndarray:
    """Symmetric dense matrix with off-diagonal mean; zeros on the diagonal."""
    M = np.full((size, size), mean, dtype=float)
    np.fill_diagonal(M, 0.0)
    return M

def spectral_radius(M: np.ndarray) -> float:
    """Return the spectral radius ρ(M) = max |λ_i|."""
    return float(np.max(np.abs(np.linalg.eigvals(M))))

def _solve_s_i(q_i: float, sum_dq_i: float, kappa_i: float, phi: float) -> float:
    """
    Closed form from ∂π_i/∂s_i=0: κ s^{3-φ} = (1-φ) q_i (Σ δ q)_i.
    For φ=1, RHS=0 ⇒ s_i=0.
    """
    rhs = (1.0 - phi) * q_i * max(sum_dq_i, 0.0)
    if rhs <= 0.0:
        return 0.0
    expo = 1.0 / (3.0 - phi)
    return float((rhs / kappa_i) ** expo)

# --------------- 3. INNER EQUILIBRIUM SOLVER: q, z, s -----------------
def equilibrium(delta: np.ndarray,
                omega: np.ndarray,
                phi:   float,
                A: float,
                c: np.ndarray,
                kappa: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """
    Block iteration:
      (1) solve q from (2I + Γ(s)) q = (A - c) + z + φ Ω z  given (s,z)
      (2) update z_i = q_i / κ_i, and s_i from κ s^{3-φ} = (1-φ) q_i (Δ q)_i
      (3) rebuild Γ(s) and re-solve q; check joint convergence (q,z,s)

    Abort cell if ρ(Γ(s))≥2 or any non-positive/NaN solution appears.
    """
    N = len(c)
    # init: positive, simple φ=1 guide for z, and s=1
    z = np.maximum((A - c) / (2.0 * kappa), 1.0)
    s = np.ones(N, dtype=float)

    for _ in range(MAX_ITERS):
        # Γ(s): δ_ij / s_i^{1-φ}  (handle φ≈1 safely)
        if abs(1.0 - phi) < 1e-14:
            denom = np.ones(N)
        else:
            denom = np.maximum(s, 1e-12) ** (1.0 - phi)
        Gamma = delta / denom[:, None]
        np.fill_diagonal(Gamma, 0.0)
        if spectral_radius(Gamma) >= 2.0:
            return False, None, None, None

        # solve q_t
        rhs = (A - c) + z + phi * (omega @ z)
        try:
            q = np.linalg.solve(2.0 * np.eye(N) + Gamma, rhs)
        except np.linalg.LinAlgError:
            return False, None, None, None
        if np.any(q <= 0.0) or not np.all(np.isfinite(q)):
            return False, None, None, None

        # update (z,s)
        z_new = np.maximum(q / kappa, 1e-12)
        sum_dq = delta @ q
        s_new = np.empty_like(s)
        for i in range(N):
            s_new[i] = _solve_s_i(q[i], sum_dq[i], kappa[i], phi)
        s_new = np.maximum(s_new, 0.0)

        # recompute with (z_{t+1}, s_{t+1}) and test convergence
        if abs(1.0 - phi) < 1e-14:
            denom_new = np.ones(N)
        else:
            denom_new = np.maximum(s_new, 1e-12) ** (1.0 - phi)
        Gamma_new = delta / denom_new[:, None]
        np.fill_diagonal(Gamma_new, 0.0)
        if spectral_radius(Gamma_new) >= 2.0:
            return False, None, None, None

        rhs_new = (A - c) + z_new + phi * (omega @ z_new)
        try:
            q_new = np.linalg.solve(2.0 * np.eye(N) + Gamma_new, rhs_new)
        except np.linalg.LinAlgError:
            return False, None, None, None
        if np.any(q_new <= 0.0) or not np.all(np.isfinite(q_new)):
            return False, None, None, None

        # convergence on all three blocks
        if max(np.linalg.norm(q_new - q, ord=np.inf),
               np.linalg.norm(z_new - z, ord=np.inf),
               np.linalg.norm(s_new - s, ord=np.inf)) < NEWTON_TOL:
            return True, q_new, z_new, s_new

        q, z, s = q_new, z_new, s_new

    return False, None, None, None  # no convergence

# ---------------- 4. WELFARE (given q,z,s,φ) ---------------------------
def welfare(q: np.ndarray,
            z: np.ndarray,
            s: np.ndarray,
            delta: np.ndarray,
            omega: np.ndarray,
            phi: float,
            A: float,
            c: np.ndarray,
            kappa: np.ndarray) -> float:
    """
    π_i(q,z,s) + CS(q) for the s–z model.

    p_i = A - q_i - (Σ_j δ_ij q_j)/s_i^{1-φ}
    MC_i = c_i - z_i - φ (ω z)_i
    π_i = (p_i - MC_i) q_i - 0.5 κ_i (z_i^2 + s_i^2)
    CS  = 0.5 * Σ_i q_i^2       (own slope -1 ⇒ area above price)
    """
    if abs(1.0 - phi) < 1e-14:
        denom = 1.0
        subs = (delta @ q) * 1.0   # s^{0} = 1
    else:
        denom = np.maximum(s, 1e-12) ** (1.0 - phi)
        subs = (delta @ q) / denom
    p = A - q - subs
    mc_eff = c - z - phi * (omega @ z)
    profits = (p - mc_eff) * q - 0.5 * kappa * (z**2 + s**2)
    CS = 0.5 * np.sum(q**2)
    return float(np.sum(profits) + CS)

# ------------- 5. OUTER PROBLEM  (δ̄,ω̄) ↦ (Q,R,S,Z,W,φ*) ---------------
def solve_cell(delta_mean: float,
               omega_mean: float) -> tuple[float, float, float, float, float, float]:
    delta = build_similarity(delta_mean, N)
    omega = build_similarity(omega_mean, N)

    def _negW(phi: float) -> float:
        ok, q, z, s = equilibrium(delta, omega, phi, A, c, kappa)
        return np.inf if not ok else -welfare(q, z, s, delta, omega, phi, A, c, kappa)

    try:
        res = opt.minimize_scalar(_negW, bounds=(0, 1), method='bounded',
                                  options={'xatol': PHI_TOL})
    except Exception:
        return (math.nan,)*6

    phi_star = float(res.x)
    if not np.isfinite(res.fun):
        return (math.nan,)*6

    ok, q_star, z_star, s_star = equilibrium(delta, omega, phi_star, A, c, kappa)
    if not ok:
        return (math.nan,)*6

    Q_bar   = float(np.mean(q_star))
    S_bar   = float(np.mean(s_star))
    Z_bar   = float(np.mean(z_star))
    R_bar   = float(np.mean(s_star + z_star))  # total R&D
    W_val   = float(-res.fun)
    return Q_bar, R_bar, S_bar, Z_bar, W_val, phi_star

# --------------------------- 6. MAIN GRID ------------------------------
def main():
    start = time.time()
    grid_vals = np.linspace(0.0, 1.0, GRID_RES, dtype=float)

    flat_results = Parallel(n_jobs=N_CORES, verbose=5, backend='loky')(
        delayed(solve_cell)(d, w)
        for d in grid_vals
        for w in grid_vals
    )

    # unpack results
    Q    = np.full((GRID_RES, GRID_RES), np.nan)
    RND  = np.full_like(Q, np.nan)   # s+z
    S    = np.full_like(Q, np.nan)
    Z    = np.full_like(Q, np.nan)
    W    = np.full_like(Q, np.nan)
    PHI  = np.full_like(Q, np.nan)

    k = 0
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            Q[i, j], RND[i, j], S[i, j], Z[i, j], W[i, j], PHI[i, j] = flat_results[k]
            k += 1

    print(f"Finished in {time.time() - start:6.1f} s")

    # -------- Figure 1: φ*, W, avg R&D (s+z), avg q ------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    mats1   = [PHI, W, RND, Q]
    titles1 = [
        r"Optimal patent transmission $\varphi^*$",
        r"Total welfare $W$",
        r"Average total R\&D $\overline{s+z}$",
        r"Average quantity $\overline{q}$"
    ]
    cmaps1  = ['viridis', 'magma', 'viridis', 'magma']
    extent  = [0, 1, 0, 1]  # ω̄ on x-axis, δ̄ on y-axis

    for ax, mat, title, cmap in zip(axes1.flat, mats1, titles1, cmaps1):
        im = ax.imshow(mat, origin='lower', cmap=cmap, extent=extent, aspect='auto')
        ax.set_xlabel(r"average technology similarity  $\bar{\omega}$")
        ax.set_ylabel(r"average product similarity     $\bar{\delta}$")
        ax.set_title(title, fontsize=11)
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("heatmaps_main.png", dpi=300)
    print("Figure saved as  heatmaps_main.png")

    # -------- Figure 2: avg s, avg z (separate panels) ----------
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    mats2   = [S, Z]
    titles2 = [
        r"Average differentiation investment $\overline{s}$",
        r"Average process R\&D $\overline{z}$"
    ]
    cmaps2  = ['viridis', 'magma']

    for ax, mat, title, cmap in zip(axes2.flat, mats2, titles2, cmaps2):
        im = ax.imshow(mat, origin='lower', cmap=cmap, extent=extent, aspect='auto')
        ax.set_xlabel(r"average technology similarity  $\bar{\omega}$")
        ax.set_ylabel(r"average product similarity     $\bar{\delta}$")
        ax.set_title(title, fontsize=11)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("heatmaps_s_z.png", dpi=300)
    plt.show()
    print("Figure saved as  heatmaps_s_z.png")

# ----------------------------- 7. LAUNCH -------------------------------
if __name__ == "__main__":
    main()
