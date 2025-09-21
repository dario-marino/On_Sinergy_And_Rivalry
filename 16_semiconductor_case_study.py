#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Empirical patent transmission φ_t via two-stage calibration (no smoothing).

Stage 1 (global scales): choose s_q, s_z so that using baseline years and φ_ref=0.7,
the implied κ and c distributions match thesis targets:
  κ ~ N(5,1^2),  SD(c) = 5   (with mean(c)=25 imposed via A_t = 25 + mean(r)).

Stage 2 (year-by-year): with s_q, s_z fixed, estimate φ_t per year by
distribution-matching (no smoothing, no anchors).

Inputs
------
- /home/dariomarino/Thesis/productsim_final.csv
  needs: manufacturer_1, manufacturer_2, release_year,
         product_similarity, patent_similarity,
         sale_start, sale_recv, xrd_start

Outputs -> ./yearwide_outputs_empiricalphi_calibrated2/
  - empirical_phi_two_stage.csv
  - phi_by_year.png
  - avg_sales_by_year.png
  - avg_z_by_year.png
  - num_firms_by_year.png
"""

import os
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize as opt

# ---------------------------- CONFIG ---------------------------------

CSV_PATH = "/home/dariomarino/Thesis/productsim_final.csv"
OUT_DIR  = "./yearwide_outputs_empiricalphi_calibrated2"
FIG_DPI  = 300

# Thesis targets
C_MEAN_TGT = 25.0
C_SD_TGT   = 5.0
K_MEAN_TGT = 5.0
K_SD_TGT   = 1.0

# Stage 1 reference φ and baseline-year selection
PHI_REF          = 0.7
BASE_MIN_FIRMS   = 3       # need at least this many firms with z>0
BASE_MAX_FIRMS   = 10      # prefer thin networks in baseline
BASE_MAX_YEARS   = 8       # use up to this many early baseline years

# Prior on A (very mild, mainly to avoid pathologies)
W_A_BASE   = 0.05          # only used in Stage 1
W_A_STAGE2 = 0.15          # used in Stage 2
A_SIGMA_FRAC = 0.5         # σ_A = A_prior * A_SIGMA_FRAC

# Shape penalties (encourage near-normal κ and c)
W_SKEW  = 0.05
W_KURT  = 0.05

# Robustness / numerics
WINSOR_P = 0.01            # winsorize tails in moment calcs
GRID_RES = 201             # φ grid points in [0,1] for Stage 2
MIN_Q    = 1e-12
MIN_Z    = 1e-12

# ---------------------- BASIC HELPERS --------------------------------

def winsorize(x: np.ndarray, p: float) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0: return x
    lo, hi = np.quantile(x, [p, 1.0 - p])
    return np.clip(x, lo, hi)

def skewness(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3: return np.nan
    m = float(np.mean(x)); s = float(np.std(x, ddof=1))
    if s <= 0: return 0.0
    return float(np.mean(((x - m)/s)**3))

def excess_kurtosis(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 4: return np.nan
    m = float(np.mean(x)); s = float(np.std(x, ddof=1))
    if s <= 0: return 0.0
    return float(np.mean(((x - m)/s)**4) - 3.0)

def edges_for_year(df_y: pd.DataFrame) -> pd.DataFrame:
    tmp = df_y[['manufacturer_1','manufacturer_2',
                'product_similarity','patent_similarity']].dropna()
    tmp = tmp[tmp['manufacturer_1'] != tmp['manufacturer_2']]
    key = tmp.apply(lambda r: tuple(sorted((str(r['manufacturer_1']),
                                            str(r['manufacturer_2'])))), axis=1)
    agg = tmp.assign(pair=key).groupby('pair', as_index=False).agg(
        delta=('product_similarity','mean'),
        omega=('patent_similarity','mean')
    )
    agg[['u','v']] = pd.DataFrame(agg['pair'].tolist(), index=agg.index)
    return agg[['u','v','delta','omega']].copy()

def matrices_from_edges(firms: List[str], edges: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    firms = pd.Index(pd.Series(firms).astype(str)); N = len(firms)
    if N < 2: return np.zeros((N,N)), np.zeros((N,N))
    idx = {f:i for i,f in enumerate(firms)}
    delta = np.zeros((N,N)); omega = np.zeros((N,N)); count = np.zeros((N,N), dtype=int)
    for _, row in edges.iterrows():
        u, v = str(row['u']), str(row['v'])
        if u not in idx or v not in idx: continue
        i, j = idx[u], idx[v]
        d = float(row['delta']); w = float(row['omega'])
        if np.isfinite(d):
            delta[i,j] += d; delta[j,i] += d; count[i,j] += 1; count[j,i] += 1
        if np.isfinite(w):
            omega[i,j] += w; omega[j,i] += w
    with np.errstate(invalid='ignore'):
        delta = np.divide(delta, np.where(count==0, 1, count))
        omega = np.divide(omega, np.where(count==0, 1, count))
    mask = ~np.eye(N, dtype=bool)
    d_mu = np.nanmean(delta[mask]) if np.isfinite(delta[mask]).any() else 0.0
    w_mu = np.nanmean(omega[mask]) if np.isfinite(omega[mask]).any() else 0.0
    delta[mask] = np.where(np.isfinite(delta[mask]), delta[mask], d_mu)
    omega[mask] = np.where(np.isfinite(omega[mask]), omega[mask], w_mu)
    np.fill_diagonal(delta, 0.0); np.fill_diagonal(omega, 0.0)
    return delta, omega

def unique_sales_by_year(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    a = (df[['release_year','manufacturer_1','sale_start']]
         .dropna(subset=['release_year','manufacturer_1'])
         .groupby(['release_year','manufacturer_1'], as_index=False)['sale_start'].median()
         .rename(columns={'manufacturer_1':'firm','sale_start':'sale'}))
    b = (df[['release_year','manufacturer_2','sale_recv']]
         .dropna(subset=['release_year','manufacturer_2'])
         .groupby(['release_year','manufacturer_2'], as_index=False)['sale_recv'].median()
         .rename(columns={'manufacturer_2':'firm','sale_recv':'sale'}))
    ab = pd.concat([a,b], ignore_index=True)
    ab = ab.groupby(['release_year','firm'], as_index=False)['sale'].max()
    # normalize firm to string once to avoid repeated casting later
    ab['firm'] = ab['firm'].astype(str)
    return {y:g[['firm','sale']].reset_index(drop=True) for y,g in ab.groupby('release_year')}

def unique_xrdstart_by_year(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    a = (df[['release_year','manufacturer_1','xrd_start']]
         .dropna(subset=['release_year','manufacturer_1','xrd_start'])
         .groupby(['release_year','manufacturer_1'], as_index=False)['xrd_start'].median()
         .rename(columns={'manufacturer_1':'firm','xrd_start':'z'}))
    return {y:g[['firm','z']].reset_index(drop=True) for y,g in a.groupby('release_year')}

def build_q(year_sales: pd.DataFrame, firms: List[str]) -> np.ndarray:
    s = year_sales.set_index('firm')['sale'] if (year_sales is not None and not year_sales.empty) else pd.Series(dtype=float)
    med = float(pd.to_numeric(s, errors='coerce').dropna().median()) if len(s)>0 else 1.0
    if not np.isfinite(med) or med <= 0: med = 1.0
    q = []
    for f in firms:
        val = s.get(f, np.nan)
        v = float(val) if np.isfinite(val) else med
        q.append(max(v, MIN_Q))
    return np.asarray(q, dtype=float)

def build_z(year_z: pd.DataFrame, firms: List[str]) -> np.ndarray:
    s = year_z.set_index('firm')['z'] if (year_z is not None and not year_z.empty) else pd.Series(dtype=float)
    z = []
    for f in firms:
        val = s.get(f, np.nan)
        z.append(float(val) if np.isfinite(val) and val>0 else np.nan)
    return np.asarray(z, dtype=float)

# ------------------ Implied primitives & objective --------------------

def kappa_hat(phi: float, q1: np.ndarray, z1: np.ndarray, delta: np.ndarray) -> np.ndarray:
    sum_dq = delta @ q1
    with np.errstate(divide='ignore', invalid='ignore'):
        num = q1 * (z1**(2.0 - phi)) + (1.0 - phi) * q1 * sum_dq
        den = z1**(3.0 - phi)
        kap = num / den
    kap[~np.isfinite(kap)] = np.nan
    kap[kap <= 0] = np.nan
    return kap

def r_vector(phi: float, q1: np.ndarray, z1: np.ndarray,
             delta: np.ndarray, omega: np.ndarray) -> np.ndarray:
    zpow = z1**(1.0 - phi)
    zpow[zpow <= 0] = np.nan
    Gamma = delta / zpow[:, None]
    np.fill_diagonal(Gamma, 0.0)
    return (2.0 * np.eye(len(q1)) + Gamma) @ q1 - z1 - phi * (omega @ z1)

# ------------------ Stage 1: global scales (s_q, s_z) ----------------

def baseline_indices(year_objs: List[dict]) -> List[int]:
    idx = [i for i,yo in enumerate(year_objs) if yo['N'] >= BASE_MIN_FIRMS]
    # prefer thin years first (N <= BASE_MAX_FIRMS), and earliest
    thin = [i for i in idx if year_objs[i]['N'] <= BASE_MAX_FIRMS]
    choose = thin[:BASE_MAX_YEARS] if len(thin) >= 1 else idx[:BASE_MAX_YEARS]
    return choose

def stage1_loss(alpha_beta: np.ndarray, year_objs: List[dict]) -> float:
    """
    α=log s_q, β=log s_z. For baseline years and φ_ref, match:
      mean κ ≈ 5, sd κ ≈ 1, sd c ≈ 5; mild A prior.
    """
    s_q = float(np.exp(alpha_beta[0]))
    s_z = float(np.exp(alpha_beta[1]))

    loss = 0.0; count = 0
    for yo in year_objs:
        q = yo['q_raw']; z = yo['z_raw']; delta = yo['delta']; omega = yo['omega']
        sum_sales = yo['sum_sales_raw']
        if q.size < 3: continue

        q1 = np.maximum(q / s_q, MIN_Q)
        z1 = np.maximum(z / s_z, MIN_Z)

        # κ moments at φ_ref
        kap = kappa_hat(PHI_REF, q1, z1, delta)
        kap = winsorize(kap, WINSOR_P)
        if kap.size < 3: continue
        mu_k = float(np.mean(kap))
        sd_k = float(np.std(kap, ddof=1))

        # c moments at φ_ref, with mean(c)=25
        r = r_vector(PHI_REF, q1, z1, delta, omega)
        rv = r[np.isfinite(r)]
        if rv.size < 3: continue
        A_hat = C_MEAN_TGT + float(np.mean(rv))
        c = A_hat - r
        c = winsorize(c, WINSOR_P)
        sd_c = float(np.std(c, ddof=1))

        # A prior (mild)
        A_prior = max(10.0 * sum_sales / s_q, 1.0)
        sigmaA  = max(A_prior * A_SIGMA_FRAC, 1.0)
        term_A  = W_A_BASE * ((A_hat - A_prior) / sigmaA)**2

        loss += ((mu_k - K_MEAN_TGT)/K_SD_TGT)**2 \
              + ((sd_k - K_SD_TGT)/K_SD_TGT)**2 \
              + ((sd_c - C_SD_TGT)/C_SD_TGT)**2 \
              + term_A
        count += 1

    if count == 0:
        return 1e12
    return float(loss / count)

# ------------------ Stage 2: per-year φ (no smoothing) ---------------

def stage2_year_loss(phi: float,
                     q_raw: np.ndarray, z_raw: np.ndarray,
                     delta: np.ndarray, omega: np.ndarray,
                     sum_sales_raw: float,
                     s_q: float, s_z: float,
                     W_A: float) -> float:
    """Distribution-matching loss at φ (no smoothing)."""
    if not (0.0 <= phi <= 1.0) or not np.isfinite(phi):
        return np.inf
    q1 = np.maximum(q_raw / s_q, MIN_Q)
    z1 = np.maximum(z_raw / s_z, MIN_Z)

    kap = kappa_hat(phi, q1, z1, delta)
    kap_ok = winsorize(kap, WINSOR_P)
    if kap_ok.size < 3: return np.inf
    mu_k = float(np.mean(kap_ok))
    sd_k = float(np.std(kap_ok, ddof=1))
    sk_k = float(skewness(kap_ok))
    ku_k = float(excess_kurtosis(kap_ok))

    r = r_vector(phi, q1, z1, delta, omega)
    rv = r[np.isfinite(r)]
    if rv.size < 3: return np.inf
    A_hat = C_MEAN_TGT + float(np.mean(rv))
    c = A_hat - r
    c_ok = winsorize(c, WINSOR_P)
    if c_ok.size < 3: return np.inf
    sd_c = float(np.std(c_ok, ddof=1))
    sk_c = float(skewness(c_ok))
    ku_c = float(excess_kurtosis(c_ok))

    A_prior = max(10.0 * sum_sales_raw / s_q, 1.0)
    sigmaA  = max(A_prior * A_SIGMA_FRAC, 1.0)
    term_A  = W_A * ((A_hat - A_prior) / sigmaA)**2

    j = ((mu_k - K_MEAN_TGT)/K_SD_TGT)**2 \
      + ((sd_k - K_SD_TGT)/K_SD_TGT)**2 \
      + ((sd_c - C_SD_TGT)/C_SD_TGT)**2 \
      + W_SKEW*(sk_k**2 + sk_c**2) \
      + W_KURT*(ku_k**2 + ku_c**2) \
      + term_A
    return float(j)

def estimate_phi_for_year(q_raw: np.ndarray, z_raw: np.ndarray,
                          delta: np.ndarray, omega: np.ndarray,
                          sum_sales_raw: float,
                          s_q: float, s_z: float) -> float:
    grid = np.linspace(0.0, 1.0, GRID_RES)
    vals = np.array([stage2_year_loss(p, q_raw, z_raw, delta, omega, sum_sales_raw, s_q, s_z, W_A_STAGE2)
                     for p in grid])
    if not np.isfinite(vals).any():
        return math.nan
    k0 = int(np.nanargmin(vals)); phi0 = float(grid[k0])
    a = max(0.0, phi0 - 0.2); b = min(1.0, phi0 + 0.2)
    try:
        res = opt.minimize_scalar(lambda p: stage2_year_loss(p, q_raw, z_raw, delta, omega,
                                                             sum_sales_raw, s_q, s_z, W_A_STAGE2),
                                  bounds=(a,b), method='bounded',
                                  options={'xatol':1e-5,'maxiter':500})
        phi_hat = float(np.clip(res.x if np.isfinite(res.x) else phi0, 0.0, 1.0))
    except Exception:
        phi_hat = phi0
    return phi_hat

# ------------------------------ MAIN ----------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    need = ['manufacturer_1','manufacturer_2','release_year',
            'product_similarity','patent_similarity',
            'sale_start','sale_recv','xrd_start']
    header = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
    usecols = [c for c in need if c in header]
    missing = [c for c in need if c not in usecols]
    if missing:
        raise RuntimeError("CSV missing: " + ", ".join(missing))

    df = pd.read_csv(CSV_PATH, usecols=usecols)
    for c in ['product_similarity','patent_similarity','sale_start','sale_recv','xrd_start','release_year']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['product_similarity','patent_similarity','release_year'])
    df['release_year'] = df['release_year'].astype(int)

    years_all = sorted(df['release_year'].unique().tolist())
    SBY = unique_sales_by_year(df)   # year -> DataFrame('firm','sale')
    ZBY = unique_xrdstart_by_year(df)

    # Precompute year objects (ALL links; keep firms with z>0)
    year_objs = []
    for y in years_all:
        df_y = df[df['release_year'] == y]
        edges_y = edges_for_year(df_y)
        if edges_y.empty:
            year_objs.append({'year':y, 'N':0})
            continue
        firms_all = sorted(pd.unique(pd.concat([edges_y['u'], edges_y['v']]).astype(str)))
        q_raw = build_q(SBY.get(y, pd.DataFrame({'firm':[], 'sale':[]})), firms_all)
        z_raw = build_z(ZBY.get(y, pd.DataFrame({'firm':[], 'z':[]})),     firms_all)
        mask  = np.isfinite(z_raw) & (z_raw > 0)
        if mask.sum() < BASE_MIN_FIRMS:
            year_objs.append({'year':y, 'N':int(mask.sum())})
            continue
        firms = [f for f,m in zip(firms_all, mask) if m]
        qv = q_raw[mask]; zv = z_raw[mask]
        delta, omega = matrices_from_edges(firms, edges_y)

        # ---- FIXED: avoid DataFrame.query env and use boolean indexing ----
        sales_df = SBY.get(y, pd.DataFrame({'firm':[], 'sale':[]})).copy()
        # if unique_sales_by_year normalized to str, this is already str; otherwise cast:
        if 'firm' in sales_df.columns:
            sales_df['firm'] = sales_df['firm'].astype(str)
        mask_firms = sales_df['firm'].isin(firms)
        sum_sales_raw = float(
            pd.to_numeric(sales_df.loc[mask_firms, 'sale'], errors='coerce').fillna(0.0).sum()
        )
        # -------------------------------------------------------------------

        year_objs.append({
            'year': y, 'N': int(qv.size),
            'q_raw': qv, 'z_raw': zv,
            'delta': delta, 'omega': omega,
            'sum_sales_raw': sum_sales_raw
        })

    # ----------------- Stage 1: calibrate s_q, s_z -------------------
    base_idx = baseline_indices(year_objs)
    base_years = [year_objs[i]['year'] for i in base_idx if year_objs[i].get('N',0) >= BASE_MIN_FIRMS]
    base_set   = [year_objs[i] for i in base_idx if year_objs[i].get('N',0) >= BASE_MIN_FIRMS]
    if len(base_set) == 0:
        raise RuntimeError("No baseline years with enough firms to calibrate units.")

    print("Baseline years for unit calibration:", base_years)

    # Initial guess: medians of raw sales and xrd_start across whole sample
    sales_all = pd.concat([
        df[['manufacturer_1','sale_start']].rename(columns={'manufacturer_1':'firm','sale_start':'sale'}),
        df[['manufacturer_2','sale_recv']].rename(columns={'manufacturer_2':'firm','sale_recv':'sale'})
    ], ignore_index=True)['sale']
    s_q0 = float(pd.to_numeric(sales_all, errors='coerce').dropna().median())
    s_q0 = s_q0 if (np.isfinite(s_q0) and s_q0>0) else 1.0
    z_all = pd.to_numeric(df['xrd_start'], errors='coerce').dropna()
    s_z0 = float(z_all.median()); s_z0 = s_z0 if (np.isfinite(s_z0) and s_z0>0) else 1.0
    x0 = np.array([np.log(s_q0), np.log(s_z0)], dtype=float)

    def loss_wrapper(x):
        return stage1_loss(x, base_set)

    try:
        res = opt.minimize(loss_wrapper, x0=x0, method='Nelder-Mead',
                           options={'xatol':1e-5,'fatol':1e-5,'maxiter':1000})
        alpha, beta = res.x if res.success else x0
    except Exception:
        alpha, beta = x0

    s_q = float(np.exp(alpha)); s_z = float(np.exp(beta))
    print(f"Calibrated global scales: s_q={s_q:.6g}, s_z={s_z:.6g}")

    # ----------------- Stage 2: estimate φ_t per year -----------------
    rows = []
    for yo in year_objs:
        y = yo['year']
        if yo.get('N',0) < BASE_MIN_FIRMS:
            rows.append({'year': y, 'phi_hat': np.nan, 'N': yo.get('N',0),
                         'avg_sales': np.nan, 'avg_z': np.nan})
            print(f"[{y}] <{BASE_MIN_FIRMS} firms with xrd_start; skipped.")
            continue

        q_raw = yo['q_raw']; z_raw = yo['z_raw']
        delta = yo['delta']; omega = yo['omega']
        sum_sales_raw = yo['sum_sales_raw']

        phi_hat = estimate_phi_for_year(q_raw, z_raw, delta, omega, sum_sales_raw, s_q, s_z)

        rows.append({
            'year': y,
            'phi_hat': float(phi_hat) if np.isfinite(phi_hat) else np.nan,
            'N': yo['N'],
            'avg_sales': float(np.nanmean(q_raw)),
            'avg_z': float(np.nanmean(z_raw))
        })
        print(f"[{y}] φ̂={rows[-1]['phi_hat']:.4f}  N={rows[-1]['N']}")

    # Save results
    res = pd.DataFrame(rows).sort_values('year')
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "empirical_phi_two_stage.csv")
    res.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    if res.empty:
        print("No years to plot."); return

    years_arr = res['year'].to_numpy()

    def plot_with_gaps(series: pd.Series, title: str, ylabel: str, fname: str):
        x = years_arr; y = series.to_numpy(dtype=float)
        plt.figure(figsize=(9,4))
        finite = np.isfinite(y)
        if finite.any():
            idx = np.where(finite)[0]
            breaks = np.where(np.diff(idx) > 1)[0]
            start = 0; segs = []
            for b in breaks:
                segs.append(idx[start:b+1]); start = b+1
            segs.append(idx[start:])
            for s in segs:
                plt.plot(x[s], y[s])
        plt.xlabel("Year"); plt.ylabel(ylabel); plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=FIG_DPI)
        plt.close()

    plot_with_gaps(res['phi_hat'], "Estimated Patent Transmission by Year", "φ̂",
                   "phi_by_year.png")
    plot_with_gaps(res['avg_sales'], "Average Sales per Firm by Year (data)", "Average sales per firm",
                   "avg_sales_by_year.png")
    plot_with_gaps(res['avg_z'], "Average R&D by Year (z = xrd_start)", "Average z",
                   "avg_z_by_year.png")
    plot_with_gaps(res['N'], "Number of Firms by Year", "# Firms",
                   "num_firms_by_year.png")

if __name__ == "__main__":
    main()
