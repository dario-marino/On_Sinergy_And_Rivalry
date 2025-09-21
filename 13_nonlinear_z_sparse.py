#!/usr/bin/env python
# nonlinear_RD_model_sparse100.py
"""
100-firm nonlinear product/process R&D model on a (δ̄, ω̄) grid with sparse links.

Key ideas:
- Limit links per firm as in the thesis' large-N simulations (degrees ~ N(10,5), no negatives).
- Build δ and ω by row-normalizing the edge weights so each row sums to the grid mean.
  This bounds the spectral radius: ρ(Δ) ≤ δ̄ and therefore ρ(Γ(z)) ≤ δ̄ / z^{1-φ} < 2 for δ̄≤1.
- Use the corrected welfare (CS = 0.5 * sum(q**2)).

This implements the link-bounding approach you describe for moving from 10 to 100/1000 firms.  # See §2.7, pp. 24–25.  # :contentReference[oaicite:1]{index=1}
"""

import math
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ---------------------------- 1. PARAMETERS ---------------------------
N          = 1000                # number of firms (set to 100 as requested)
A          = 100000              # demand intercept (scales with N; cf. thesis grids)  # :contentReference[oaicite:2]{index=2}
GRID_RES   = 101                # 0 … 1 in steps of 0.01
PHI_TOL    = 1e-6               # φ convergence (optimizer)
NEWTON_TOL = 1e-8               # q/z convergence (∞-norm)
MAX_ITERS  = 60                 # block-iteration limit per φ
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
    """
    Build an undirected simple graph adjacency (0/1) with roughly Normal degrees.
    For each node i: sample k_i ~ N(mean_deg, sd_deg), clamp to [0, size-1], choose k_i neighbors.
    Symmetrize at the end.
    """
    A = np.zeros((size, size), dtype=np.float64)
    all_idx = np.arange(size)
    for i in range(size):
        k_i = int(np.clip(np.rint(rng.normal(mean_deg, sd_deg)), 0, size - 1))
        if k_i == 0:
            continue
        choices = rng.choice(all_idx[all_idx != i], size=k_i, replace=False)
        A[i, choices] = 1.0
    # undirected simple graph
    A = ((A + A.T) > 0).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    return A

def weights_from_adj(adj: np.ndarray, row_mean: float) -> np.ndarray:
    """
    Row-normalize the adjacency so each row sums to `row_mean` (if degree>0), else 0.
    This ensures max row sum ≤ row_mean ⇒ ρ(weight matrix) ≤ row_mean.
    """
    deg = adj.sum(axis=1)
    W = np.zeros_like(adj, dtype=np.float64)
    # scale_i = row_mean / deg_i for deg_i>0; 0 otherwise
    scale = np.divide(row_mean, deg, out=np.zeros_like(deg), where=deg > 0)
    W = adj * scale[:, None]
    np.fill_diagonal(W, 0.0)
    return W

def spectral_radius(M: np.ndarray) -> float:
    """Return the spectral radius ρ(M) = max |λ_i|."""
    return float(np.max(np.abs(np.linalg.eigvals(M))))

# Precompute sparse structures ONCE so every grid cell uses the same networks
ADJ_DELTA = draw_sparse_adjacency(N, AVG_LINKS, STD_LINKS, rng)
ADJ_OMEGA = draw_sparse_adjacency(N, AVG_LINKS, STD_LINKS, rng)

# --------------- 3. INNER EQUILIBRIUM SOLVER  q,z | δ,ω,φ --------------
def _newton_z_i(q_i, sum_dq_i, kappa_i, phi, z0):
    """
    Newton on   f(z) = κ z^{3-φ} - q z^{2-φ} - (1-φ) q (Σ δ q) = 0
    Returns z >= 1e-8
    """
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

def equilibrium(delta: np.ndarray,
                omega: np.ndarray,
                phi:   float,
                A: float,
                c: np.ndarray,
                kappa: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Block iteration: (i) solve q given z; (ii) update z given q (per-firm Newton).
    Recompute Γ(z) and the linear system after each z update.
    Stop when max( ||q_{t+1}-q_t||_∞, ||z_{t+1}-z_t||_∞ ) < NEWTON_TOL and ρ(Γ)<2.
    """
    N = len(c)
    # initialization (φ=1 guide; strictly positive)
    z = np.maximum((A - c) / (2 * kappa), 1.0)

    for _ in range(MAX_ITERS):
        # ---- Step 1: compute Γ(z), check stability, and solve q_t ----
        Gamma = delta / np.power(z[:, None], (1 - phi))
        np.fill_diagonal(Gamma, 0.0)
        if spectral_radius(Gamma) >= 2.0:
            return False, None, None

        # RHS: (A - c) + z + φ (ω z)
        rhs = (A - c) + z + phi * (omega @ z)
        try:
            q = np.linalg.solve(2 * np.eye(N) + Gamma, rhs)
        except np.linalg.LinAlgError:
            return False, None, None
        if np.any(q <= 0.0) or not np.all(np.isfinite(q)):
            return False, None, None

        # ---- Step 2: per-firm Newton for z given q_t ----
        sum_dq = delta @ q
        z_new = z.copy()
        for i in range(N):
            z_new[i] = _newton_z_i(q[i], sum_dq[i], kappa[i], phi, z[i])

        # ---- Step 3: recompute with z_{t+1} and test convergence ----
        Gamma_new = delta / np.power(z_new[:, None], (1 - phi))
        np.fill_diagonal(Gamma_new, 0.0)
        if spectral_radius(Gamma_new) >= 2.0:
            return False, None, None

        rhs_new = (A - c) + z_new + phi * (omega @ z_new)
        try:
            q_new = np.linalg.solve(2 * np.eye(N) + Gamma_new, rhs_new)
        except np.linalg.LinAlgError:
            return False, None, None
        if np.any(q_new <= 0.0) or not np.all(np.isfinite(q_new)):
            return False, None, None

        # convergence on both blocks
        if max(np.linalg.norm(q_new - q, ord=np.inf),
               np.linalg.norm(z_new - z, ord=np.inf)) < NEWTON_TOL:
            return True, q_new, z_new

        # iterate
        q, z = q_new, z_new

    # no convergence within MAX_ITERS
    return False, None, None

# ---------------- 4. WELFARE (given q,z,φ) ------------------------------
def welfare(q: np.ndarray,
            z: np.ndarray,
            delta: np.ndarray,
            omega: np.ndarray,
            phi: float,
            A: float,
            c: np.ndarray,
            kappa: np.ndarray) -> float:
    """
    W = Σ_i π_i(q,z) + CS(q)
    - Price: p_i = A - q_i - (Σ_j δ_ij q_j) / z_i^{1-φ}
    - Profit: (p_i - MC_i) q_i - 0.5 κ_i z_i^2,   MC_i = c_i - z_i - φ (ω z)_i
    - CS (correct): 0.5 * Σ_i q_i^2  (since ∂p_i/∂q_i = -1).
    """
    subs = (delta @ q) / (np.power(z, (1 - phi)))
    p = A - q - subs
    mc_eff = c - z - phi * (omega @ z)
    profits = (p - mc_eff) * q - 0.5 * kappa * z**2
    CS = 0.5 * np.sum(q**2)
    return float(np.sum(profits) + CS)

# ------------- 5. OUTER PROBLEM  (δ̄,ω̄) ↦ (Q,Z,W,φ*) -------------------
def solve_cell(delta_mean: float,
               omega_mean: float) -> tuple[float, float, float, float]:
    # Build sparse, normalized similarity matrices for this (δ̄, ω̄)
    delta = weights_from_adj(ADJ_DELTA, delta_mean)
    omega = weights_from_adj(ADJ_OMEGA, omega_mean)

    def _negW(phi: float) -> float:
        ok, q, z = equilibrium(delta, omega, phi, A, c, kappa)
        return np.inf if not ok else -welfare(q, z, delta, omega, phi, A, c, kappa)

    try:
        res = opt.minimize_scalar(_negW, bounds=(0, 1), method='bounded',
                                  options={'xatol': PHI_TOL})
    except Exception:
        return math.nan, math.nan, math.nan, math.nan

    phi_star = float(res.x)
    if not np.isfinite(res.fun):
        return math.nan, math.nan, math.nan, math.nan

    ok, q_star, z_star = equilibrium(delta, omega, phi_star, A, c, kappa)
    if not ok:
        return math.nan, math.nan, math.nan, math.nan

    return float(np.mean(q_star)), float(np.mean(z_star)), float(-res.fun), phi_star

# --------------------------- 6. MAIN GRID ------------------------------
def main():
    start = time.time()
    grid_vals = np.linspace(0.0, 1.0, GRID_RES, dtype=float)

    flat_results = Parallel(n_jobs=N_CORES, verbose=5, backend='loky')(
        delayed(solve_cell)(d, w)
        for d in grid_vals
        for w in grid_vals
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

    print(f"Finished in {time.time() - start:6.1f} s")

    # -------- 7. PLOT: φ*, W, avg z, avg q ------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = [
        r"Optimal patent transmission $\varphi^*$",  # top-left
        r"Total welfare $W$",                        # top-right
        r"Average R\&D $\bar{z}$",                   # bottom-left
        r"Average quantity $\bar{q}$"                # bottom-right
    ]
    mats = [PHI, W, Z, Q]
    cmaps = ['viridis', 'magma', 'viridis', 'magma']
    extent = [0, 1, 0, 1]   # ω̄ on x-axis, δ̄ on y-axis

    for ax, mat, title, cmap in zip(axes.flat, mats, titles, cmaps):
        im = ax.imshow(mat, origin='lower', cmap=cmap, extent=extent, aspect='auto')
        ax.set_xlabel(r"average technology similarity  $\bar{\omega}$")
        ax.set_ylabel(r"average product similarity     $\bar{\delta}$")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("heatmaps_sparse_100.png", dpi=300)
    plt.show()
    print("Figure saved as  heatmaps_sparse_100.png")

# ----------------------------- 8. LAUNCH -------------------------------
if __name__ == "__main__":
    main()
