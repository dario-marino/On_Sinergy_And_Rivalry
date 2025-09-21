#!/usr/bin/env python3
"""
Compute within- and between-portfolio similarity over time
from product rows with 'patent_portfolio' plus embeddings parquet.
Usage:
  python 04_products_portfolio_similarity.py --products product_factsheets_with_portfolio_embedded.csv \
      --embeddings USPTO_abstracts_embeddings.parquet --outdir ./
"""
import os, argparse, numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

def parse_portfolio(s):
    if not isinstance(s,str) or s.strip() in ("","[]"): return []
    out=[]
    for x in s.strip("[]").split(","):
        x=x.strip().strip("'\""); 
        if x: out.append(x)
    return out

def load_embeddings(path):
    tbl=pq.read_table(path); df=tbl.to_pandas()
    cols=[c for c in df.columns if c.startswith("embedding_")]
    M=df[cols].values.astype(np.float32)
    lut={str(pid):i for i,pid in enumerate(df["patent_id"].astype(str))}
    return M,lut

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--products", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--outdir", required=True)
    a=ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)

    prod=pd.read_csv(a.products)
    prod["portfolio"]=prod["patent_portfolio"].apply(parse_portfolio)

    M,lut=load_embeddings(a.embeddings)

    rows=[]
    valid=prod.dropna(subset=["release_year"])
    for _,r in valid.iterrows():
        firm=r["manufacturer"]; year=int(r["release_year"])
        ids=[p for p in r["portfolio"] if p in lut]
        if len(ids)<2: rows.append((firm,year,np.nan,len(ids))); continue
        X=M[[lut[p] for p in ids]]
        ut=cosine_similarity(X)[np.triu_indices(len(ids),1)]
        rows.append((firm,year,float(np.mean(ut)),len(ids)))
    within=pd.DataFrame(rows, columns=["manufacturer","release_year","within_similarity","patent_count"]).dropna()
    within.to_csv(os.path.join(a.outdir,"within_patent.csv"), index=False)

    yf={}; rows=[]
    for _,r in valid.iterrows():
        firm=r["manufacturer"]; year=int(r["release_year"])
        ids=list({p for p in r["portfolio"] if p in lut})
        if ids: yf.setdefault(year,{}).setdefault(firm,[]).extend(ids)
    for yr, D in yf.items():
        firms=list(D.keys())
        for f1,f2 in combinations(firms,2):
            p1=[p for p in set(D[f1]) if p in lut]
            p2=[p for p in set(D[f2]) if p in lut]
            if not p1 or not p2: rows.append((yr,f1,f2,np.nan,len(p1),len(p2))); continue
            X1=M[[lut[p] for p in p1]].mean(axis=0, keepdims=True)
            X2=M[[lut[p] for p in p2]].mean(axis=0, keepdims=True)
            s=float(cosine_similarity(X1,X2)[0,0])
            rows.append((yr,f1,f2,s,len(p1),len(p2)))
    between=pd.DataFrame(rows, columns=["release_year","manufacturer_1","manufacturer_2","between_similarity","patent_count_1","patent_count_2"]).dropna()
    between.to_csv(os.path.join(a.outdir,"between_patent.csv"), index=False)
    print("Wrote within_patent.csv and between_patent.csv")

if __name__=="__main__":
    main()
