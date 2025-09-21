#!/usr/bin/env python3
"""
Match Compustat 'conm' to patent assignees via SBERT + cosine NN.
Outputs a review CSV for manual tweaks.
Usage:
  python 02_match_companies_to_assignees.py --companies companydata.csv \
    --patents single_patents_with_top.csv \
    --out company_assignee_matching_for_review.csv
"""
import re, os, argparse, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

STOP={'inc','incorporated','corp','corporation','company','co','ltd','limited','llc','lp','plc','sa','nv','bv','gmbh','ag','spa','srl','pty','the','and','of','group','holdings','holding','international','intl','aktiengesellschaft'}

def clean(s:str)->str:
    if not isinstance(s,str): return ''
    s=re.sub(r'[^\w\s]',' ',s.lower().strip())
    for w in STOP: s=re.sub(rf'\b{re.escape(w)}\b','',s)
    return re.sub(r'\s+',' ',s).strip()

def dedupe(series):
    cleaned, m=[], {}
    for orig in series.dropna().astype(str).str.strip().unique().tolist():
        c=clean(orig)
        if c: cleaned.append(c); m[c]=orig
    return cleaned, m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--companies", required=True)
    ap.add_argument("--patents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="all-mpnet-base-v2")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--n-neighbors", type=int, default=100)
    args=ap.parse_args()

    dfc=pd.read_csv(args.companies, dtype=str, low_memory=False)
    dfp=pd.read_csv(args.patents, dtype=str, low_memory=False)

    companies,cmap=dedupe(dfc['conm'])
    assignees,amap=dedupe(dfp['disambig_assignee_organization'])

    enc=SentenceTransformer(args.model)
    C=normalize(enc.encode(companies, convert_to_numpy=True, batch_size=64, show_progress_bar=True), axis=1)
    A=normalize(enc.encode(assignees, convert_to_numpy=True, batch_size=64, show_progress_bar=True), axis=1)

    nn=NearestNeighbors(n_neighbors=min(args.n_neighbors,len(assignees)), metric='cosine', algorithm='brute', n_jobs=-1).fit(A)
    D, I=nn.kneighbors(C); sims=1-D

    rows=[]
    for i, comp in enumerate(companies):
        row=None
        for idx, s in zip(I[i], sims[i]):
            if s >= args.threshold:
                row={"company_original":cmap[comp],"company_processed":comp,
                     "matched_assignee_original":amap[assignees[idx]],
                     "matched_assignee_processed":assignees[idx],
                     "similarity_score":float(s),"valid_match":True,"notes":""}
                break
        if row is None:
            row={"company_original":cmap[comp],"company_processed":comp,
                 "matched_assignee_original":"","matched_assignee_processed":"",
                 "similarity_score":0.0,"valid_match":False,"notes":"No match above threshold"}
        rows.append(row)

    out=pd.DataFrame(rows).sort_values("company_original")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {out['valid_match'].sum()} matches / {len(out)} to {args.out}")

if __name__=="__main__":
    main()
