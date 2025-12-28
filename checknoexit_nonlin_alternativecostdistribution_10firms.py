#!/usr/bin/env python

import math
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ---------------- PARAMETERS ----------------
N = 10
A = 1000
GRID_RES = 101
PHI_TOL = 1e-6
NEWTON_TOL = 1e-8
MAX_ITERS = 60
SEED = 42
N_CORES = -1

Z_MIN   = 1e-8
PENALTY = 1e30

# Two-type costs
SHARE_LEADERS = 0.2
MU_L, SIG_L = 20.0, 5.0
MU_H, SIG_H = 80.0, 20.0

rng = np.random.default_rng(SEED)
is_leader = rng.random(N) < SHARE_LEADERS
c = np.where(is_leader,
             rng.normal(MU_L, SIG_L, N),
             rng.normal(MU_H, SIG_H, N))
kappa = np.clip(rng.normal(5.0, 1.0, N), 1e-3, None)

# ---------------- HELPERS ----------------
def build_similarity(mean, size):
    M = np.full((size, size), mean)
    np.fill_diagonal(M, 0.0)
    return M

def _newton_z_i(q_i, sum_dq_i, kappa_i, phi, z0):
    s = 2.0 - phi
    const = (1 - phi) * q_i * sum_dq_i
    z = max(z0, Z_MIN)
    for _ in range(40):
        f  = kappa_i * z**(s + 1) - q_i * z**s - const
        fp = kappa_i * (s + 1) * z**s - q_i * s * z**(s - 1)
        if abs(fp) < 1e-14:
            break
        z_new = z - f / fp
        if z_new <= 0 or not np.isfinite(z_new):
            z_new = 0.5 * z
        if abs(z_new - z) < 1e-12:
            z = z_new
            break
        z = z_new
    return max(z, Z_MIN)

# ---------------- EQUILIBRIUM ----------------
def equilibrium(delta, omega, phi):
    z = np.maximum((A - c) / (2 * kappa), 1.0)
    z = np.maximum(z, Z_MIN)
    delta_row_sum = delta.sum(axis=1)

    for _ in range(MAX_ITERS):
        denom = z**(1 - phi)
        if np.max(delta_row_sum / denom) >= 2:
            return False, None, None

        Gamma = delta / denom[:, None]
        rhs = (A - c) + z + phi * (omega @ z)
        q = np.linalg.solve(2*np.eye(N) + Gamma, rhs)
        q = np.maximum(q, 0.0)

        sum_dq = delta @ q
        z_new = np.array([_newton_z_i(q[i], sum_dq[i], kappa[i], phi, z[i])
                          for i in range(N)])
        z_new = np.maximum(z_new, Z_MIN)

        if np.linalg.norm(z_new - z, np.inf) < NEWTON_TOL:
            return True, q, z_new

        z = z_new

    return False, None, None

# ---------------- WELFARE ----------------
def welfare(q, z, delta, omega, phi):
    z = np.maximum(z, Z_MIN)
    subs = (delta @ q) / z**(1 - phi)
    p = A - q - subs
    mc = c - z - phi * (omega @ z)
    profits = (p - mc) * q - 0.5 * kappa * z**2
    return max(float(np.sum(profits) + 0.5*np.sum(q**2)), 0.0)

# ---------------- GRID CELL ----------------
def solve_cell(dbar, wbar):
    delta = build_similarity(dbar, N)
    omega = build_similarity(wbar, N)

    def obj(phi):
        ok, q, z = equilibrium(delta, omega, phi)
        if not ok:
            return PENALTY
        return -welfare(q, z, delta, omega, phi)

    res = opt.minimize_scalar(obj, bounds=(0,1), method="bounded",
                              options={"xatol": PHI_TOL})
    if not np.isfinite(res.fun):
        return math.nan, math.nan, math.nan, math.nan, math.nan

    ok, q, z = equilibrium(delta, omega, res.x)
    if not ok:
        return math.nan, math.nan, math.nan, math.nan, math.nan

    zero_share = np.mean(q <= 1e-12)
    return np.mean(q), np.mean(z), welfare(q, z, delta, omega, res.x), res.x, zero_share

# ---------------- MAIN ----------------
def main():
    grid = np.linspace(0,1,GRID_RES)
    flat = Parallel(n_jobs=N_CORES)(
        delayed(solve_cell)(d,w) for d in grid for w in grid
    )

    Q = np.zeros((GRID_RES, GRID_RES))
    Z = np.zeros_like(Q)
    W = np.zeros_like(Q)
    PHI = np.zeros_like(Q)
    ZEROQ = np.zeros_like(Q)

    k = 0
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            Q[i,j], Z[i,j], W[i,j], PHI[i,j], ZEROQ[i,j] = flat[k]
            k += 1

    fig, axes = plt.subplots(2,3, figsize=(15,9))
    mats = [PHI, W, Z, Q, ZEROQ]
    titles = [r"$\varphi^*$", "Welfare", r"$\bar z$", r"$\bar q$", "Share $q_i=0$"]
    cmaps = ["viridis","magma","viridis","magma","cividis"]

    for ax, mat, title, cmap in zip(axes.flat, mats, titles, cmaps):
        im = ax.imshow(mat, origin="lower", extent=[0,1,0,1], cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
