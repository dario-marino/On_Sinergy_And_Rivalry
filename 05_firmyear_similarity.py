#!/usr/bin/env python3
"""
Firm-year technology similarity for pairs.csv using portfolio centroids (embeddings).
Usage:
  python 05_firmyear_similarity.py --embeddings USPTO_abstracts_embeddings.parquet \
    --company-portfolios companydata_with_portfolio_embedded.csv \
    --pairs pairs.csv --out pairs_simtech.csv
"""
import argparse, ast, gc
import numpy as np, pandas as pd, pyarrow.dataset as ds
from collections import defaultdict
from sklearn.preprocessing import normalize

DIM=768

def diag_cosine(A,B):
    A=normalize(A,axis=1, copy=False); B=normalize(B,axis=1, copy=False)
    return np.sum(A*B,axis=1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)           
    ap.add_argument("--company-portfolios", required=True)   
    ap.add_argument("--pairs", required=True)                
    ap.add_argument("--out", required=True)
    a=ap.parse_args()

    comp=pd.read_csv(a.company_portfolios,
                     converters={"patent_portfolio": lambda s: [] if pd.isna(s) or s.strip()=="[]" else ast.literal_eval(s)},
                     usecols=["gvkey","fyear","patent_portfolio"])
    m=defaultdict(list)
    for _,r in comp.iterrows():
        for pid in r.patent_portfolio:
            m[str(pid)].append((r.gvkey, str(r.fyear)))

    sums=defaultdict(lambda: np.zeros(DIM, dtype=np.float64))
    cnts=defaultdict(int)

    dataset=ds.dataset(a.embeddings, format="parquet")
    cols=["patent_id"]+[f"embedding_{i}" for i in range(DIM)]
    for batch in dataset.to_batches(columns=cols, batch_size=100_000):
        df=batch.to_pandas()
        for row in df.itertuples(index=False):
            pid=str(row.patent_id)
            if pid in m:
                vec=np.asarray(row[1:], dtype=np.float32)
                for key in m[pid]:
                    sums[key]+=vec; cnts[key]+=1
        gc.collect()

    centroids={k:(s/cnts[k]).astype(np.float32) if cnts[k]>0 else np.zeros(DIM, np.float32) for k,s in sums.items()}

    pairs=pd.read_csv(a.pairs)
    N=len(pairs)
    A=np.zeros((N,DIM), np.float32); B=np.zeros_like(A)
    for i,r in pairs.iterrows():
        A[i]=centroids.get((r.gvkey_start, str(r.fyear_start)), np.zeros(DIM, np.float32))
        B[i]=centroids.get((r.gvkey_recv,  str(r.fyear_start)), np.zeros(DIM, np.float32))

    sims=diag_cosine(A,B)
    pairs["sim_tech"]=sims
    pairs.to_csv(a.out, index=False)
    print(f"Wrote {a.out} with {len(pairs):,} rows.")

if __name__=="__main__":
    main()
