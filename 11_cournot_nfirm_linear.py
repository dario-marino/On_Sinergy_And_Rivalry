#!/usr/bin/env python3
"""
N-firm linear Cournot with cost-reducing R&D and φ; (δ̄, ω̄) heatmaps.
"""
import argparse, numpy as np, matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed

def tnorm(mean, sd, low, high, size=None, rng=None):
    rng = rng or np.random.default_rng()
    a,b=(low-mean)/sd,(high-mean)/sd
    return truncnorm.rvs(a,b, loc=mean, scale=sd, size=size, random_state=rng)

def sim(N, mean, sd, rng):
    M=np.zeros((N,N)); iu=np.triu_indices(N,1)
    off=tnorm(mean,sd,0,1,size=len(iu[0]), rng=rng)
    M[iu]=off; M[(iu[1],iu[0])]=off
    return M

def solve_q(A, c, k, Delta, Omega, phi):
    N=len(c)
    B=Omega / k[None,:]; np.fill_diagonal(B,0.0)
    M=np.diag(2.0 - 1.0/k) + Delta - phi*B
    b=A*np.ones(N) - c
    try: q=np.linalg.solve(M,b)
    except np.linalg.LinAlgError: return None
    if np.any(q<=0) or not np.all(np.isfinite(q)): return None
    return q

def welfare(A,c,k,Delta,Omega,phi):
    q=solve_q(A,c,k,Delta,Omega,phi)
    if q is None: return -np.inf
    z=q/k
    cs=A*q.sum() - 0.5*q.dot(q) - q.dot(Delta.dot(q))
    p=A - q - Delta.dot(q)
    effc=c - z - phi*(Omega*z).sum(axis=1)
    prof=(p - effc)*q - 0.5*k*(z**2)
    return float(cs + prof.sum())

def phi_star(A,c,k,Delta,Omega):
    res=minimize_scalar(lambda ph: -welfare(A,c,k,Delta,Omega,ph), bounds=(1e-12,1-1e-12), method='bounded')
    if res.success and np.isfinite(res.fun): return float(res.x), float(-res.fun)
    P=np.linspace(0,1,101); vals=[welfare(A,c,k,Delta,Omega,p) for p in P]
    i=int(np.nanargmax(vals)); return float(P[i]), float(vals[i])

def simulate_heatmaps(N=10, A=500, cost_mean=25, cost_sd=5, kappa_mean=5, kappa_sd=1, sim_sd=0.1, res=51, seed=42):
    rng=np.random.default_rng(seed)
    d_means=np.linspace(0,1,res); w_means=np.linspace(0,1,res)
    PHI=W=Z=Q=np.full((res,res), np.nan)
    def cell(i,j):
        c=tnorm(cost_mean,cost_sd,1,np.inf,size=N,rng=rng)
        k=tnorm(kappa_mean,kappa_sd,1,np.inf,size=N,rng=rng)
        Delta=sim(N, d_means[i], sim_sd, rng); Omega=sim(N, w_means[j], sim_sd, rng)
        ph, wstar = phi_star(A,c,k,Delta,Omega)
        q = solve_q(A,c,k,Delta,Omega,ph)
        if q is None: return ph, wstar, np.nan, np.nan
        z=q/k; return ph, wstar, float(np.mean(z)), float(np.mean(q))
    results=Parallel(n_jobs=-1)(delayed(cell)(i,j) for i in range(res) for j in range(res))
    k=0
    for i in range(res):
        for j in range(res):
            PHI[i,j],W[i,j],Z[i,j],Q[i,j]=results[k]; k+=1
    return d_means, w_means, PHI, W, Z, Q

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10); ap.add_argument("--A", type=float, default=500)
    ap.add_argument("--cost-mean", type=float, default=25); ap.add_argument("--cost-sd", type=float, default=5)
    ap.add_argument("--kappa-mean", type=float, default=5); ap.add_argument("--kappa-sd", type=float, default=1)
    ap.add_argument("--sim-sd", type=float, default=0.1)
    ap.add_argument("--res", type=int, default=51); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="cournot_nfirm_heatmaps")
    args=ap.parse_args()

    d_means,w_means,PHI,W,Z,Q=simulate_heatmaps(N=args.N, A=args.A, cost_mean=args.cost_mean, cost_sd=args.cost_sd,
                                                kappa_mean=args.kappa_mean, kappa_sd=args.kappa_sd,
                                                sim_sd=args.sim_sd, res=args.res, seed=args.seed)
    fig,ax=plt.subplots(2,2,figsize=(12,10)); ext=[0,1,0,1]
    m=ax[0,0].imshow(PHI,origin='lower',extent=ext,aspect='auto',cmap='viridis'); ax[0,0].set_title("φ*"); plt.colorbar(m,ax=ax[0,0])
    m=ax[0,1].imshow(W,origin='lower',extent=ext,aspect='auto',cmap='plasma'); ax[0,1].set_title("Welfare"); plt.colorbar(m,ax=ax[0,1])
    m=ax[1,0].imshow(Z,origin='lower',extent=ext,aspect='auto',cmap='viridis'); ax[1,0].set_title("Avg R&D"); plt.colorbar(m,ax=ax[1,0])
    m=ax[1,1].imshow(Q,origin='lower',extent=ext,aspect='auto',cmap='magma');  ax[1,1].set_title("Avg q");   plt.colorbar(m,ax=ax[1,1])
    for a in ax.flat: a.set_xlabel("ω"); a.set_ylabel("δ")
    plt.tight_layout(); plt.savefig(f"{args.out}.png", dpi=300); plt.savefig(f"{args.out}.pdf")
    print(f"Saved {args.out}.png/.pdf")

if __name__=="__main__":
    main()
