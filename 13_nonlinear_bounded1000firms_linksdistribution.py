#!/usr/bin/env python
# nonlinear_RD_model_sparse100.py
"""
100 or 1000 firm nonlinear product/process R&D model 
on a (delta, omega) grid with sparse links.
This implements the link-bounding approach described
 in thesis for moving from 10 to 100 or 1000 firms.
"""

import math
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ---------------------------- 1. PARAMETERS ---------------------------
N          = 1000               # number of firms (set to 100 as requested)
A          = 100000             # demand intercept (scales with N
GRID_RES   = 101                # 0 1 in steps of 0.01
PHI_TOL    = 1e-6               # phi convergence (optimizer)
NEWTON_TOL = 1e-8               # tol
MAX_ITERS  = 60                 # block-iteration limit per phi
SEED       = 42                 # RNG seed for reproducibility
N_CORES    = -1                 # parallel jobs (-1 uses all cores)

# Sparse link controls (Normal(AVG_LINKS, STD_LINKS), truncated to [0, N-1])
AVG_LINKS  = 10
STD_LINKS  = 5

rng = np.random.default_rng(SEED)
c      = rng.normal(25.0, 5.0, N)
kappa  = np.clip(rng.normal(5.0, 1.0, N), 1e-3, None)   # strictly > 0

# ---------------------- 2. SPARSE NETWORK HELPERS --------------------
def draw_sparse_adjacency(size: int, mean_deg: int, sd_deg: int, rng: np.random.Generator) -> np.ndarray:
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
    deg = adj.sum(axis=1)
    W = np.zeros_like(adj, dtype=np.float64)
    scale = np.divide(row_mean, deg, out=np.zeros_like(deg), where=deg > 0)
    W = adj * scale[:, None]
    np.fill_diagonal(W, 0.0)
    return W

# Precompute sparse structures
ADJ_DELTA = draw_sparse_adjacency(N, AVG_LINKS, STD_LINKS, rng)
ADJ_OMEGA = draw_sparse_adjacency(N, AVG_LINKS, STD_LINKS, rng)

# ---------------- 3. INNER EQUILIBRIUM SOLVER ----------------
def _newton_z_i(q_i, sum_dq_i, kappa_i, phi, z0):
    s = 2.0 - phi
    a, b, const = kappa_i, q_i, (1 - phi) * q_i * sum_dq_i
    z = max(z0, 1e-8)
    for _ in range(30):
        f  = a * z**(s + 1) - b * z**s - const
        fp = a * (s + 1) * z**s - b * s * z**(s - 1)
        if abs(fp) < 1e-12:
            break
        z_new = z - f / fp
        if z_new <= 0 or not np.isfinite(z_new):
            z_new = z / 2.0
        if abs(z_new - z) < 1e-10:
            z = z_new
            break
        z = z_new
    return max(z, 1e-8)

def equilibrium(delta, omega, phi, A, c, kappa):
    z = np.maximum((A - c) / (2 * kappa), 1.0)
    delta_row_sum = delta.sum(axis=1)

    for _ in range(MAX_ITERS):

        # ---- stability check at z ----
        if np.max(delta_row_sum / np.power(z, 1 - phi)) >= 2.0:
            return False, None, None

        Gamma = delta / np.power(z[:, None], (1 - phi))
        np.fill_diagonal(Gamma, 0.0)

        rhs = (A - c) + z + phi * (omega @ z)
        try:
            q = np.linalg.solve(2 * np.eye(N) + Gamma, rhs)
        except np.linalg.LinAlgError:
            return False, None, None
        if np.any(q <= 0.0) or not np.all(np.isfinite(q)):
            return False, None, None

        sum_dq = delta @ q
        z_new = z.copy()
        for i in range(N):
            z_new[i] = _newton_z_i(q[i], sum_dq[i], kappa[i], phi, z[i])

        # ---- stability check at z_new (ONLY FIX) ----
        if np.max(delta_row_sum / np.power(z_new, 1 - phi)) >= 2.0:
            return False, None, None

        Gamma_new = delta / np.power(z_new[:, None], (1 - phi))
        np.fill_diagonal(Gamma_new, 0.0)

        rhs_new = (A - c) + z_new + phi * (omega @ z_new)
        try:
            q_new = np.linalg.solve(2 * np.eye(N) + Gamma_new, rhs_new)
        except np.linalg.LinAlgError:
            return False, None, None
        if np.any(q_new <= 0.0) or not np.all(np.isfinite(q_new)):
            return False, None, None

        if max(np.linalg.norm(q_new - q, np.inf),
               np.linalg.norm(z_new - z, np.inf)) < NEWTON_TOL:
            return True, q_new, z_new

        q, z = q_new, z_new

    return False, None, None

# ---------------- 4. WELFARE ------------------------------
def welfare(q, z, delta, omega, phi, A, c, kappa):
    subs = (delta @ q) / (np.power(z, (1 - phi)))
    p = A - q - subs
    mc_eff = c - z - phi * (omega @ z)
    profits = (p - mc_eff) * q - 0.5 * kappa * z**2
    CS = 0.5 * np.sum(q**2)
    return float(np.sum(profits) + CS)

# ---------------- 5. OUTER PROBLEM -------------------------
def solve_cell(delta_mean, omega_mean):
    delta = weights_from_adj(ADJ_DELTA, delta_mean)
    omega = weights_from_adj(ADJ_OMEGA, omega_mean)

    def _negW(phi):
        ok, q, z = equilibrium(delta, omega, phi, A, c, kappa)
        return np.inf if not ok else -welfare(q, z, delta, omega, phi, A, c, kappa)

    res = opt.minimize_scalar(_negW, bounds=(0, 1), method='bounded',
                              options={'xatol': PHI_TOL})

    if not np.isfinite(res.fun):
        return math.nan, math.nan, math.nan, math.nan

    ok, q_star, z_star = equilibrium(delta, omega, res.x, A, c, kappa)
    if not ok:
        return math.nan, math.nan, math.nan, math.nan

    return float(np.mean(q_star)), float(np.mean(z_star)), float(-res.fun), float(res.x)

# --------------------------- 6. MAIN GRID ------------------------------
def main():
    grid_vals = np.linspace(0.0, 1.0, GRID_RES)

    flat_results = Parallel(n_jobs=N_CORES, verbose=5, backend='loky')(
        delayed(solve_cell)(d, w) for d in grid_vals for w in grid_vals
    )

    Q = np.full((GRID_RES, GRID_RES), np.nan)
    Z = np.full_like(Q, np.nan)
    W = np.full_like(Q, np.nan)
    PHI = np.full_like(Q, np.nan)

    k = 0
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            Q[i, j], Z[i, j], W[i, j], PHI[i, j] = flat_results[k]
            k += 1

    print("Grid completed")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = [
        r"Optimal patent transmission $\varphi^*$",
        r"Total welfare $W$",
        r"Average R\&D $\bar{z}$",
        r"Average quantity $\bar{q}$"
    ]
    mats = [PHI, W, Z, Q]
    cmaps = ['viridis', 'magma', 'viridis', 'magma']
    extent = [0, 1, 0, 1]

    for ax, mat, title, cmap in zip(axes.flat, mats, titles, cmaps):
        im = ax.imshow(mat, origin='lower', cmap=cmap, extent=extent, aspect='auto')
        ax.set_xlabel(r"average technology similarity  $\bar{\omega}$")
        ax.set_ylabel(r"average product similarity     $\bar{\delta}$")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("heatmaps_sparse_100.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
