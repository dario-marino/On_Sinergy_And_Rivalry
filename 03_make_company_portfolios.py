#!/usr/bin/env python3
"""
Build firm-year patent portfolios from company↔assignee matches.
Usage:
  python 03_make_company_portfolios.py \
    --company-data companydataxrd.csv \
    --patents single_patents_with_top.csv \
    --matches company_assignee_matching_for_review.csv \
    --out companydata_with_portfolio_embedded.csv
"""
import argparse, os, pandas as pd

def args_():
    ap=argparse.ArgumentParser()
    ap.add_argument("--company-data", required=True)
    ap.add_argument("--patents", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--out", required=True)
    return ap.parse_args()

def main():
    a=args_()
    dfc=pd.read_csv(a.company_data, dtype=str, low_memory=False)
    dfp=pd.read_csv(a.patents, dtype=str, low_memory=False)
    dfm=pd.read_csv(a.matches)

    dfp['patent_date']=pd.to_datetime(dfp['patent_date'], errors='coerce')
    dfp['patent_year']=dfp['patent_date'].dt.year
    dfp['assignee']=dfp['disambig_assignee_organization'].fillna('')
    dfc['conm_clean']=dfc['conm'].fillna('')
    dfc['fyear_num']=pd.to_numeric(dfc['fyear'], errors='coerce')

    lut={}
    for _,r in dfm.iterrows():
        co=r.get('company_original',''); ao=r.get('matched_assignee_original','')
        if co and ao: lut[co]=ao

    def portfolio(row):
        co=row['conm_clean']; y=row['fyear_num']
        if not co or pd.isna(y): return []
        asg=lut.get(co,''); 
        if not asg: return []
        mask=(dfp['assignee']==asg)&(dfp['patent_year'].between(int(y)-a.window, int(y)))
        return dfp.loc[mask,'patent_id'].dropna().astype(str).unique().tolist()

    dfc['patent_portfolio']=dfc.apply(portfolio, axis=1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    dfc.to_csv(a.out, index=False)
    print(f"Wrote {a.out} (rows={len(dfc):,})")

if __name__=="__main__":
    main()
