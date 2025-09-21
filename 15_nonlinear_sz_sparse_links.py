#!/usr/bin/env python
# nonlinear_RD_s_z_sparse_links5.py
"""
Same s+z model, but limit links via a truncated-normal degree target:
- mean degree ~ 5, sd ~ 2 (clamped to [0, N-1]), undirected simple graph.
- Row-normalize weights so each row sum equals the grid mean (δ̄ or ω̄).
This bounds ρ(Δ) ≤ δ̄ and helps satisfy the standard stability gate (ρ(Γ(s)) < 2).  # :contentReference[oaicite:3]{index=3}
"""

import math
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ---------------------------- 1. PARAMETERS ---------------------------
N          = 100               # adjust if you want
A          = 50000             # market size for N≈100
GRID_RES   = 101
PHI_TOL    = 1e-6
NEWTON_TOL = 1e-8
MAX_ITERS  = 60
SEED       = 42
N_CORES    = -1

AVG_LINKS  = 5                 # <— requested average degree
STD_LINKS  = 2

S_MIN      = 1e-8              # floor on s for numerical safety

rng    = np.random.default_rng(SEED)
c      = rng.normal(25.0, 5.0, N)
kappa  = np.clip(rng.normal(5.0, 1.0, N), 1e-3, None)

# ---------------------- 2. SPARSE NETWORK HELPERS --------------------
def draw_sparse_adjacency(size: int, mean_deg: int, sd_deg: int, rng: np.random.Generator) -> np.ndarray:
    """Undirected 0/1 adjacency with approximately Normal degrees (truncated)."""
    A = np.zeros((size, size), dtype=np.float64)
    all_idx = np.arange(size)
    for i in range(size):
        k_i = int(np.clip(np.rint(rng.normal(mean_deg, sd_deg)), 0, size - 1))
        if k_i == 0:
            continue
        choices = rng.choice(all_idx[all_idx != i], size=k_i, replace=False)
        A[i, choices] = 1.0
    A = ((A + A.T) > 0).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    return A

def weights_from_adj(adj: np.ndarray, row_mean: float) -> np.ndarray:
    """Row-normalize so each row sums to row_mean (if deg>0), else 0."""
    deg = adj.sum(axis=1)
    scale = np.divide(row_mean, deg, out=np.zeros_like(deg), where=deg > 0)
    W = adj * scale[:, None]
    np.fill_diagonal(W, 0.0)
    return W

def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))

# Build once; then rescale by (δ̄, ω̄)
ADJ_DELTA = draw_sparse_adjacency(N, AVG_LINKS, STD_LINKS, rng)
ADJ_OMEGA = draw_sparse_adjacency(N, AVG_LINKS, STD_LINKS, rng)

# --------------- 3. INNER EQUILIBRIUM SOLVER: q, z, s -----------------
def _solve_s_i(q_i: float, sum_dq_i: float, kappa_i: float, phi: float) -> float:
    rhs = (1.0 - phi) * q_i * max(sum_dq_i, 0.0)
    if rhs <= 0.0:
        return 0.0
    return float((rhs / kappa_i) ** (1.0 / (3.0 - phi)))

def equilibrium(delta: np.ndarray,
                omega: np.ndarray,
                phi:   float,
                A: float,
                c: np.ndarray,
                kappa: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    N = len(c)
    z = np.maximum((A - c) / (2.0 * kappa), 1.0)
    s = np.ones(N, dtype=float)

    def gamma_from_s(svec):
        denom = np.ones_like(svec) if abs(1.0 - phi) < 1e-14 else np.maximum(svec, S_MIN) ** (1.0 - phi)
        G = delta / denom[:, None]
        np.fill_diagonal(G, 0.0)
        return G

    for _ in range(MAX_ITERS):
        Gamma = gamma_from_s(s)
        if spectral_radius(Gamma) >= 2.0:
            return False, None, None, None
        rhs = (A - c) + z + phi * (omega @ z)
        try:
            q = np.linalg.solve(2.0 * np.eye(N) + Gamma, rhs)
        except np.linalg.LinAlgError:
            return False, None, None, None
        if np.any(q <= 0.0) or not np.all(np.isfinite(q)):
            return False, None, None, None

        z_new = np.maximum(q / kappa, 1e-12)
        sum_dq = delta @ q
        s_new = np.maximum(np.array([_solve_s_i(q[i], sum_dq[i], kappa[i], phi) for i in range(N)]), S_MIN)

        Gamma_new = gamma_from_s(s_new)
        if spectral_radius(Gamma_new) >= 2.0:
            return False, None, None, None
        rhs_new = (A - c) + z_new + phi * (omega @ z_new)
        try:
            q_new = np.linalg.solve(2.0 * np.eye(N) + Gamma_new, rhs_new)
        except np.linalg.LinAlgError:
            return False, None, None, None
        if np.any(q_new <= 0.0) or not np.all(np.isfinite(q_new)):
            return False, None, None, None

        if max(np.linalg.norm(q_new - q, np.inf),
               np.linalg.norm(z_new - z, np.inf),
               np.linalg.norm(s_new - s, np.inf)) < NEWTON_TOL:
            return True, q_new, z_new, s_new
        q, z, s = q_new, z_new, s_new

    return False, None, None, None

# ---------------- 4. WELFARE (given q,z,s,φ) ---------------------------
def welfare(q, z, s, delta, omega, phi, A, c, kappa):
    denom = 1.0 if abs(1.0 - phi) < 1e-14 else np.maximum(s, S_MIN) ** (1.0 - phi)
    subs  = (delta @ q) / denom
    p     = A - q - subs
    mc    = c - z - phi * (omega @ z)
    profits = (p - mc) * q - 0.5 * kappa * (z**2 + s**2)
    CS = 0.5 * np.sum(q**2)
    return float(np.sum(profits) + CS)

# ------------- 5. OUTER PROBLEM  (δ̄,ω̄) ↦ (Q,R,S,Z,W,φ*) ---------------
def solve_cell(delta_mean: float, omega_mean: float):
    delta = weights_from_adj(ADJ_DELTA, delta_mean)
    omega = weights_from_adj(ADJ_OMEGA, omega_mean)

    def _negW(phi: float) -> float:
        ok, q, z, s = equilibrium(delta, omega, phi, A, c, kappa)
        return np.inf if not ok else -welfare(q, z, s, delta, omega, phi, A, c, kappa)

    try:
        res = opt.minimize_scalar(_negW, bounds=(0, 1), method='bounded',
                                  options={'xatol': PHI_TOL})
    except Exception:
        return (math.nan,)*6

    phi_star = float(res.x)
    ok, q_star, z_star, s_star = equilibrium(delta, omega, phi_star, A, c, kappa)
    if not ok:
        return (math.nan,)*6

    Q_bar = float(np.mean(q_star))
    S_bar = float(np.mean(s_star))
    Z_bar = float(np.mean(z_star))
    R_bar = float(np.mean(s_star + z_star))
    W_val = float(-res.fun)
    return Q_bar, R_bar, S_bar, Z_bar, W_val, phi_star

# --------------------------- 6. MAIN GRID ------------------------------
def main():
    start = time.time()
    grid_vals = np.linspace(0.0, 1.0, GRID_RES, dtype=float)

    flat_results = Parallel(n_jobs=N_CORES, verbose=5, backend='loky')(
        delayed(solve_cell)(d, w) for d in grid_vals for w in grid_vals
    )

    Q   = np.full((GRID_RES, GRID_RES), np.nan)
    RND = np.full_like(Q, np.nan)
    S   = np.full_like(Q, np.nan)
    Z   = np.full_like(Q, np.nan)
    W   = np.full_like(Q, np.nan)
    PHI = np.full_like(Q, np.nan)

    k = 0
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            Q[i,j], RND[i,j], S[i,j], Z[i,j], W[i,j], PHI[i,j] = flat_results[k]
            k += 1

    print(f"Finished in {time.time() - start:6.1f} s")

    # ---- Figure 1: φ*, W, avg R&D (s+z), avg q ----
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    mats1   = [PHI, W, RND, Q]
    titles1 = [
        r"Optimal patent transmission $\varphi^*$",
        r"Total welfare $W$",
        r"Average total R\&D $\overline{s+z}$",
        r"Average quantity $\overline{q}$"
    ]
    cmaps1  = ['viridis', 'magma', 'viridis', 'magma']
    extent  = [0, 1, 0, 1]
    for ax, mat, title, cmap in zip(axes1.flat, mats1, titles1, cmaps1):
        im = ax.imshow(mat, origin='lower', cmap=cmap, extent=extent, aspect='auto')
        ax.set_xlabel(r"average technology similarity  $\bar{\omega}$")
        ax.set_ylabel(r"average product similarity     $\bar{\delta}$")
        ax.set_title(title, fontsize=11)
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("heatmaps_main_sparse_links5.png", dpi=300)
    print("Figure saved as  heatmaps_main_sparse_links5.png")

    # ---- Figure 2: avg s, avg z ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    mats2   = [S, Z]
    titles2 = [r"Average differentiation $\overline{s}$", r"Average process R\&D $\overline{z}$"]
    cmaps2  = ['viridis', 'viridis']
    for ax, mat, title, cmap in zip(axes2.flat, mats2, titles2, cmaps2):
        im = ax.imshow(mat, origin='lower', cmap=cmap, extent=extent, aspect='auto')
        ax.set_xlabel(r"average technology similarity  $\bar{\omega}$")
        ax.set_ylabel(r"average product similarity     $\bar{\delta}$")
        ax.set_title(title, fontsize=11)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("heatmaps_s_z_sparse_links5.png", dpi=300)
    plt.show()
    print("Figure saved as  heatmaps_s_z_sparse_links5.png")

# ----------------------------- 7. LAUNCH -------------------------------
if __name__ == "__main__":
    main()
