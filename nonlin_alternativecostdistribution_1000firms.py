#!/usr/bin/env python
# nonlinear_RD_model_sparse1000_twotype_safe.py

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
def draw_sparse_adjacency(size: int, mean_deg: int, sd_deg: int,
                          rng: np.random.Generator) -> sp.csr_matrix:
    """
    Undirected simple graph adjacency (0/1), sparse CSR.
    Uses np.atleast_1d to avoid scalar-return edge cases when k=1.
    """
    rows = []
    cols = []
    all_idx = np.arange(size)

    for i in range(size):
        k_i = int(np.clip(np.rint(rng.normal(mean_deg, sd_deg)), 0, size - 1))
        if k_i == 0:
            continue
        choices = np.atleast_1d(rng.choice(all_idx[all_idx != i], size=k_i, replace=False))
        rows.extend([i] * k_i)
        cols.extend(list(choices))

    data = np.ones(len(rows), dtype=np.float64)
    Acoo = sp.coo_matrix((data, (np.array(rows), np.array(cols))), shape=(size, size))
    A = ((Acoo + Acoo.T) > 0).astype(np.float64).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A

def weights_from_adj(adj: sp.csr_matrix, row_mean: float) -> sp.csr_matrix:
    """
    Row-normalize so each row sums to `row_mean` (if degree>0), else 0.
    """
    deg = np.array(adj.sum(axis=1)).ravel()
    scale = np.divide(row_mean, deg, out=np.zeros_like(deg), where=deg > 0)
    W = sp.diags(scale) @ adj
    W.eliminate_zeros()
    return W.tocsr()

# Precompute sparse structures once
ADJ_DELTA = draw_sparse_adjacency(N, AVG_LINKS, STD_LINKS, rng)
ADJ_OMEGA = draw_sparse_adjacency(N, AVG_LINKS, STD_LINKS, rng)

# --------------- 3. INNER SOLVER ----------------
def _newton_z_i(q_i, sum_dq_i, kappa_i, phi, z0):
    s = 2.0 - phi
    const = (1 - phi) * q_i * sum_dq_i
    z = max(z0, Z_MIN)
    for _ in range(40):
        z = max(z, Z_MIN)
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

def equilibrium(delta: sp.csr_matrix,
                omega: sp.csr_matrix,
                phi: float,
                A: float,
                c: np.ndarray,
                kappa: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Same block iteration as dense model, but sparse linear algebra.
    Stability via row-sum bound (no eigenvalues):
      max_i (row_sum_delta_i / z_i^{1-φ}) < 2.
    """
    N = len(c)
    z = np.maximum((A - c) / (2 * kappa), 1.0)
    z = np.maximum(z, Z_MIN)

    delta_row_sum = np.array(delta.sum(axis=1)).ravel()

    I = sp.eye(N, format="csr")

    for _ in range(MAX_ITERS):
        z_safe = np.maximum(z, Z_MIN)
        denom = np.power(z_safe, 1 - phi)

        if np.max(delta_row_sum / denom) >= 2.0:
            return False, None, None

        # Gamma = diag(1/denom) @ delta   (sparse, preserves sparsity)
        scale = 1.0 / denom
        Gamma = sp.diags(scale) @ delta

        rhs = (A - c) + z_safe + phi * (omega @ z_safe)

        try:
            q = spla.spsolve(2.0 * I + Gamma, rhs)
        except Exception:
            return False, None, None

        if (not np.all(np.isfinite(q))):
            return False, None, None

        q = np.maximum(q, 0.0)

        sum_dq = delta @ q
        z_new = np.array([_newton_z_i(q[i], sum_dq[i], kappa[i], phi, z_safe[i])
                          for i in range(N)], dtype=float)
        z_new = np.maximum(z_new, Z_MIN)

        # recompute q with z_new
        z_new_safe = np.maximum(z_new, Z_MIN)
        denom_new = np.power(z_new_safe, 1 - phi)

        if np.max(delta_row_sum / denom_new) >= 2.0:
            return False, None, None

        scale_new = 1.0 / denom_new
        Gamma_new = sp.diags(scale_new) @ delta
        rhs_new = (A - c) + z_new_safe + phi * (omega @ z_new_safe)

        try:
            q_new = spla.spsolve(2.0 * I + Gamma_new, rhs_new)
        except Exception:
            return False, None, None

        if (not np.all(np.isfinite(q_new))):
            return False, None, None

        q_new = np.maximum(q_new, 0.0)

        if max(np.linalg.norm(q_new - q, ord=np.inf),
               np.linalg.norm(z_new - z, ord=np.inf)) < NEWTON_TOL:
            return True, q_new, z_new

        q, z = q_new, z_new

    return False, None, None

# ---------------- 4. WELFARE ------------------------------
def welfare(q, z, delta, omega, phi, A, c, kappa) -> float:
    if q is None or z is None:
        return 0.0
    if (not np.all(np.isfinite(q))) or (not np.all(np.isfinite(z))):
        return 0.0

    q = np.maximum(q, 0.0)
    z_safe = np.maximum(z, Z_MIN)

    denom = np.power(z_safe, 1 - phi)
    # subs = (delta @ q) / denom = (1/denom) * (delta @ q)
    subs = (delta @ q) / denom

    p = A - q - subs
    mc_eff = c - z_safe - phi * (omega @ z_safe)

    profits = (p - mc_eff) * q - 0.5 * kappa * z_safe**2
    CS = 0.5 * np.sum(q**2)

    W = float(np.sum(profits) + CS)
    return 0.0 if (not np.isfinite(W)) else W

# ------------- 5. OUTER PROBLEM -------------------
def solve_cell(delta_mean: float, omega_mean: float):
    delta = weights_from_adj(ADJ_DELTA, delta_mean)
    omega = weights_from_adj(ADJ_OMEGA, omega_mean)

    def obj(phi: float) -> float:
        phi = min(max(float(phi), 1e-9), 1.0 - 1e-9)
        ok, q, z = equilibrium(delta, omega, phi, A, c, kappa)
        if not ok:
            return PENALTY
        W = welfare(q, z, delta, omega, phi, A, c, kappa)
        if not np.isfinite(W):
            return PENALTY
        return -W

    try:
        res = opt.minimize_scalar(obj, bounds=(0, 1), method="bounded",
                                  options={"xatol": PHI_TOL})
    except Exception:
        return math.nan, math.nan, math.nan, math.nan

    phi_star = float(res.x)
    if (not np.isfinite(res.fun)) or (not np.isfinite(phi_star)):
        return math.nan, math.nan, math.nan, math.nan

    ok, q_star, z_star = equilibrium(delta, omega, phi_star, A, c, kappa)
    if not ok:
        return math.nan, math.nan, math.nan, math.nan

    W_star = welfare(q_star, z_star, delta, omega, phi_star, A, c, kappa)

    return float(np.mean(q_star)), float(np.mean(z_star)), float(W_star), phi_star

# --------------------------- 6. MAIN GRID + PLOT ------------------------------
def main():
    start = time.time()
    grid_vals = np.linspace(0.0, 1.0, GRID_RES, dtype=float)

    flat = Parallel(n_jobs=N_CORES, verbose=5, backend="loky")(
        delayed(solve_cell)(d, w) for d in grid_vals for w in grid_vals
    )

    Q = np.full((GRID_RES, GRID_RES), np.nan)
    Z = np.full_like(Q, np.nan)
    W = np.full_like(Q, np.nan)
    PHI = np.full_like(Q, np.nan)

    k = 0
    for i in range(GRID_RES):
        for j in range(GRID_RES):
            Q[i, j], Z[i, j], W[i, j], PHI[i, j] = flat[k]
            k += 1

    print(f"Finished in {time.time() - start:6.1f} s")

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
    plt.savefig("heatmaps_sparse1000_twotype_safe.png", dpi=300)
    plt.show()
    print("Figure saved as heatmaps_sparse1000_twotype_safe.png")

if __name__ == "__main__":
    main()
