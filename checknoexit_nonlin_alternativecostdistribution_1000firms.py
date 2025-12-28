#!/usr/bin/env python

import math
import time
import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ---------------------------- 1. PARAMETERS ---------------------------
N          = 1000
A          = 100000
GRID_RES   = 101
PHI_TOL    = 1e-6
NEWTON_TOL = 1e-8
MAX_ITERS  = 60
SEED       = 42
N_CORES    = -1

AVG_LINKS  = 10
STD_LINKS  = 5

# Numerical safety
Z_MIN   = 1e-8
PENALTY = 1e30

# Two-type costs
SHARE_LEADERS = 0.2
MU_L, SIG_L   = 20.0, 5.0
MU_H, SIG_H   = 80.0, 20.0

rng = np.random.default_rng(SEED)

is_leader = rng.random(N) < SHARE_LEADERS
c = np.where(is_leader,
             rng.normal(MU_L, SIG_L, N),
             rng.normal(MU_H, SIG_H, N))

kappa = np.clip(rng.normal(5.0, 1.0, N), 1e-3, None)

# ---------------------- 2. SPARSE NETWORK HELPERS --------------------
def draw_sparse_adjacency(size, mean_deg, sd_deg, rng):
    rows, cols = [], []
    all_idx = np.arange(size)
    for i in range(size):
        k_i = int(np.clip(np.rint(rng.normal(mean_deg, sd_deg)), 0, size - 1))
        if k_i == 0:
            continue
        choices = np.atleast_1d(rng.choice(all_idx[all_idx != i],
                                           size=k_i, replace=False))
        rows.extend([i] * k_i)
        cols.extend(list(choices))
    data = np.ones(len(rows))
    Acoo = sp.coo_matrix((data, (rows, cols)), shape=(size, size))
    A = ((Acoo + Acoo.T) > 0).astype(float).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A

def weights_from_adj(adj, row_mean):
    deg = np.array(adj.sum(axis=1)).ravel()
    scale = np.divide(row_mean, deg, out=np.zeros_like(deg), where=deg > 0)
    W = sp.diags(scale) @ adj
    W.eliminate_zeros()
    return W.tocsr()

ADJ_DELTA = draw_sparse_adjacency(N, AVG_LINKS, STD_LINKS, rng)
ADJ_OMEGA = draw_sparse_adjacency(N, AVG_LINKS, STD_LINKS, rng)

# ---------------------- 3. NEWTON STEP --------------------
def _newton_z_i(q_i, sum_dq_i, kappa_i, phi, z0):
    s = 2.0 - phi
    const = (1 - phi) * q_i * sum_dq_i
    z = max(z0, Z_MIN)
    for _ in range(40):
        f  = kappa_i * z**(s + 1) - q_i * z**s - const
        fp = kappa_i * (s + 1) * z**s - q_i * s * z**(s - 1)
        if not np.isfinite(f) or not np.isfinite(fp) or abs(fp) < 1e-14:
            break
        z_new = z - f / fp
        if (not np.isfinite(z_new)) or (z_new <= 0):
            z_new = 0.5 * z
        if abs(z_new - z) < 1e-12:
            z = z_new
            break
        z = z_new
    return max(z, Z_MIN)

# ---------------------- 4. EQUILIBRIUM --------------------
def equilibrium(delta, omega, phi):
    z = np.maximum((A - c) / (2 * kappa), 1.0)
    z = np.maximum(z, Z_MIN)
    delta_row_sum = np.array(delta.sum(axis=1)).ravel()
    I = sp.eye(N, format="csr")

    for _ in range(MAX_ITERS):
        denom = z**(1 - phi)
        if np.max(delta_row_sum / denom) >= 2:
            return False, None, None

        Gamma = sp.diags(1.0 / denom) @ delta
        rhs = (A - c) + z + phi * (omega @ z)

        q = spla.spsolve(2 * I + Gamma, rhs)
        if not np.all(np.isfinite(q)):
            return False, None, None
        q = np.maximum(q, 0.0)

        sum_dq = delta @ q
        z_new = np.array([_newton_z_i(q[i], sum_dq[i], kappa[i], phi, z[i])
                          for i in range(N)])
        z_new = np.maximum(z_new, Z_MIN)

        if np.linalg.norm(z_new - z, np.inf) < NEWTON_TOL:
            return True, q, z_new

        z = z_new

    return False, None, None

# ---------------------- 5. WELFARE --------------------
def welfare(q, z, delta, omega, phi):
    z = np.maximum(z, Z_MIN)
    subs = (delta @ q) / z**(1 - phi)
    p = A - q - subs
    mc = c - z - phi * (omega @ z)
    profits = (p - mc) * q - 0.5 * kappa * z**2
    return max(float(np.sum(profits) + 0.5 * np.sum(q**2)), 0.0)

# ---------------------- 6. GRID CELL --------------------
def solve_cell(dbar, wbar):
    delta = weights_from_adj(ADJ_DELTA, dbar)
    omega = weights_from_adj(ADJ_OMEGA, wbar)

    def obj(phi):
        phi = min(max(phi, 1e-9), 1 - 1e-9)
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

# ---------------------- 7. MAIN --------------------
def main():
    start = time.time()
    grid = np.linspace(0,1,GRID_RES)

    flat = Parallel(n_jobs=N_CORES, verbose=5)(
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

    print(f"Finished in {time.time() - start:6.1f} s")

    fig, axes = plt.subplots(2,3, figsize=(15,9))
    mats = [PHI, W, Z, Q, ZEROQ]
    titles = [r"$\varphi^*$", "Welfare", r"$\bar z$", r"$\bar q$", "Share $q_i=0$"]
    cmaps = ["viridis","magma","viridis","magma","cividis"]

    for ax, mat, title, cmap in zip(axes.flat, mats, titles, cmaps):
        im = ax.imshow(mat, origin="lower", extent=[0,1,0,1], cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.04)

    plt.tight_layout()
    plt.savefig("heatmaps_sparse1000_twotype_safe_zeroq.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
