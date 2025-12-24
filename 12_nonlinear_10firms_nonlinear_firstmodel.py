#!/usr/bin/env python
# nonlinear_RD_model_fixed.py

import math
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ---------------------------- 1. PARAMETERS ---------------------------
N          = 10
A          = 1000
GRID_RES   = 101
PHI_TOL    = 1e-6
NEWTON_TOL = 1e-8
MAX_ITERS  = 60
SEED       = 42
N_CORES    = -1
PRINT_RESIDUALS = False

rng = np.random.default_rng(SEED)
c      = rng.normal(25.0, 5.0, N)
kappa  = np.clip(rng.normal(5.0, 1.0, N), 1e-3, None)

# ---------------------- 2. HELPER FUNCTIONS ---------------------------
def build_similarity(mean: float, size: int) -> np.ndarray:
    M = np.full((size, size), mean, dtype=float)
    np.fill_diagonal(M, 0.0)
    return M

# --------------- 3. INNER EQUILIBRIUM SOLVER --------------------------
def _newton_z_i(q_i, sum_dq_i, kappa_i, phi, z0):
    s = 2.0 - phi
    const = (1 - phi) * q_i * sum_dq_i
    z = max(z0, 1e-8)
    for _ in range(30):
        f  = kappa_i * z**(s + 1) - q_i * z**s - const
        fp = kappa_i * (s + 1) * z**s - q_i * s * z**(s - 1)
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

        Gamma = delta / z[:, None]**(1 - phi)
        np.fill_diagonal(Gamma, 0.0)

        rhs = (A - c) + z + phi * (omega @ z)
        try:
            q = np.linalg.solve(2 * np.eye(N) + Gamma, rhs)
        except np.linalg.LinAlgError:
            return False, None, None
        if np.any(q <= 0.0) or not np.all(np.isfinite(q)):
            return False, None, None

        # ---- update z ----
        sum_dq = delta @ q
        z_new = np.array([
            _newton_z_i(q[i], sum_dq[i], kappa[i], phi, z[i])
            for i in range(N)
        ])

        # ---- stability check at z_new (### FIX) ----
        if np.max(delta_row_sum / np.power(z_new, 1 - phi)) >= 2.0:
            return False, None, None

        Gamma_new = delta / z_new[:, None]**(1 - phi)
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

# ---------------- 4. WELFARE ------------------------------------------
def welfare(q, z, delta, omega, phi, A, c, kappa):
    subs = (delta @ q) / (z**(1 - phi))
    p = A - q - subs
    mc_eff = c - z - phi * (omega @ z)
    profits = (p - mc_eff) * q - 0.5 * kappa * z**2
    CS = 0.5 * np.sum(q**2)
    return float(np.sum(profits) + CS)

# ---------------- 5. OUTER PROBLEM ------------------------------------
def solve_cell(delta_mean, omega_mean):
    delta = build_similarity(delta_mean, N)
    omega = build_similarity(omega_mean, N)

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

    flat = Parallel(n_jobs=N_CORES, verbose=5)(
        delayed(solve_cell)(d, w) for d in grid_vals for w in grid_vals
    )

    Q = np.full((GRID_RES, GRID_RES), np.nan)
    Z = np.full_like(Q, np.nan)
    W = np.full_like(Q, np.nan)
    PHI = np.full_like(Q, np.nan)

    k = 0
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            Q[i,j], Z[i,j], W[i,j], PHI[i,j] = flat[k]
            k += 1

    print("Grid completed")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = [r"Optimal patent transmission $\varphi^*$",
          r"Total welfare $W$",
          r"Average R\&D $\bar{z}$",
          r"Average quantity $\bar{q}$"]
    mats = [PHI, W, Z, Q]
    cmaps = ["viridis", "magma", "viridis", "magma"]
    extent = [0, 1, 0, 1]

    for ax, mat, title, cmap in zip(axes.flat, mats, titles, cmaps):
     im = ax.imshow(mat, origin="lower", cmap=cmap, extent=extent, aspect="auto")
     ax.set_xlabel(r"average technology similarity  $\bar{\omega}$")
     ax.set_ylabel(r"average product similarity     $\bar{\delta}$")
     ax.set_title(title, fontsize=11)
     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
