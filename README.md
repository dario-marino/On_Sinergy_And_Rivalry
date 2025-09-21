# On_Sinergy_And_Rivalry

### Necessary from server

1) Download these five files to the repo root:
   - `companydata_with_portfolio_embedded.csv`
   - `USPTO_abstracts_embeddings.csv`  ← same schema as the real Parquet
   - `pairs.csv`
   - `pairs_simtech.csv` (expected output to compare)
   - `README_SMOKE.txt` (optional notes)

2) Run step **05 – firm‑year technology similarity** on the tiny inputs.
   - If your script expects Parquet, you can either:
     - `pip install pyarrow` and point to your real Parquet **(for the full run)**, or
     - temporarily switch the loader to read the provided CSV just to verify the math/wiring.

3) Confirm you reproduce a `pairs_simtech.csv` with 4 rows and the same `sim_tech` values
   (order may differ if you re‑sort).

Requirements:

numpy>=1.23
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
scipy>=1.10
pyarrow>=14.0
polars>=1.0
joblib>=1.3
tqdm>=4.66
sentence-transformers>=2.5.0




python 01_embed_patents.py --input g_patent_abstract.tsv --out USPTO_abstracts_embeddings.parquet



