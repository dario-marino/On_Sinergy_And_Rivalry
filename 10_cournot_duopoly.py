#!/usr/bin/env python3
"""
Duopoly Cournot with cost-reducing R&D and patent transmission φ.
Produces heatmaps for φ*, welfare, avg q, avg z (δ on y, ω on x).
"""
import argparse, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def solve(A,c1,c2,k1,k2,delta,w,phi):
    a1=2-1/k1; a2=2-1/k2
    b1=delta - (phi*w)/k2; b2=delta - (phi*w)/k1
    D=a1*a2 - b1*b2
    if abs(D)<1e-12: return None
    q1=((A-c1)*a2 - b1*(A-c2))/D
    q2=(a1*(A-c2) - b2*(A-c1))/D
    if q1<=0 or q2<=0: return None
    z1=q1/k1; z2=q2/k2
    return q1,q2,z1,z2

def cs(A,delta,q1,q2):
    return (A*q1 - 0.5*q1*q1 - delta*q1*q2) + (A*q2 - 0.5*q2*q2 - delta*q2*q1)

def welfare(A,c1,c2,k1,k2,delta,w,phi):
    sol=solve(A,c1,c2,k1,k2,delta,w,phi)
    if sol is None: return -np.inf
    q1,q2,z1,z2=sol
    p1=A - q1 - delta*q2; p2=A - q2 - delta*q1
    mc1=c1 - z1; mc2=c2 - z2
    prof1=(p1-mc1)*q1 - 0.5*k1*z1*z1
    prof2=(p2-mc2)*q2 - 0.5*k2*z2*z2
    return cs(A,delta,q1,q2) + prof1 + prof2

def phi_star(A,c1,c2,k1,k2,delta,w):
    res=minimize_scalar(lambda ph: -welfare(A,c1,c2,k1,k2,delta,w,ph), bounds=(0,1), method="bounded")
    if not res.success: return np.nan, np.nan
    ph=float(res.x); W=float(-res.fun)
    return ph, W

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--A", type=float, default=100)
    ap.add_argument("--c1", type=float, default=20)
    ap.add_argument("--c2", type=float, default=25)
    ap.add_argument("--k1", type=float, default=5)
    ap.add_argument("--k2", type=float, default=5)
    ap.add_argument("--res", type=int, default=51)
    ap.add_argument("--out", default="cournot_duopoly_heatmaps")
    args=ap.parse_args()

    deltas=np.linspace(0,1,args.res)
    omegas=np.linspace(0,1,args.res)
    PHI=W=Z=Q=np.full((args.res,args.res), np.nan)

    for i,d in enumerate(deltas):
        for j,w in enumerate(omegas):
            ph, wstar = phi_star(args.A,args.c1,args.c2,args.k1,args.k2,d,w)
            PHI[i,j]=ph; W[i,j]=wstar
            sol=solve(args.A,args.c1,args.c2,args.k1,args.k2,d,w,ph)
            if sol:
                q1,q2,z1,z2=sol
                Z[i,j]=(z1+z2)/2; Q[i,j]=(q1+q2)/2

    import os; os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig,ax=plt.subplots(2,2,figsize=(12,10)); ext=[0,1,0,1]
    m=ax[0,0].imshow(PHI,origin='lower',extent=ext,aspect='auto',cmap='viridis'); ax[0,0].set_title("φ*"); plt.colorbar(m,ax=ax[0,0])
    m=ax[0,1].imshow(W,origin='lower',extent=ext,aspect='auto',cmap='plasma'); ax[0,1].set_title("Welfare"); plt.colorbar(m,ax=ax[0,1])
    m=ax[1,0].imshow(Z,origin='lower',extent=ext,aspect='auto',cmap='viridis'); ax[1,0].set_title("Avg R&D"); plt.colorbar(m,ax=ax[1,0])
    m=ax[1,1].imshow(Q,origin='lower',extent=ext,aspect='auto',cmap='magma');  ax[1,1].set_title("Avg q"); plt.colorbar(m,ax=ax[1,1])
    for a in ax.flat: a.set_xlabel("ω"); a.set_ylabel("δ")
    plt.tight_layout(); plt.savefig(f"{args.out}.png", dpi=300); plt.savefig(f"{args.out}.pdf")
    print(f"Saved {args.out}.png/.pdf")

if __name__=="__main__":
    main()
