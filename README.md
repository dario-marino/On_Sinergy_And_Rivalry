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

## On Synergy and Rivalry — Full Replication README Start

We study how **technology similarity** (spillovers/synergy) and **product similarity** (substitutability/rivalry) jointly shape firms’ R&D. A regulator chooses **patent transmission** φ to maximize welfare. The theory provides duopoly closed-form solutions and N-firm simulations; the empirics build a firm-pair panel from patent-portfolio embeddings (USPTO abstracts) and product similarity, merged to Compustat, and then estimate FE/GAM/XGB surfaces over the (product, technology) similarity grid.

---

## Files in this repo (flat)

**Python — data & similarity**
- `01_embed_patents.py` — compute **PatentSBERTa_V2** embeddings for USPTO abstracts (chunked, memory-safe) → Parquet `USPTO_abstracts_embeddings.parquet`.
- `02_match_companies_to_assignees.py` — match **Compustat** company names to **PatentsView** assignees (regex clean + SBERT + cosine) → reviewable CSV.
- `03_make_company_portfolios.py` — build **firm-year patent portfolios** (rolling 5-year window) from validated matches → `companydata_with_portfolio_embedded.csv`.
- `04_products_portfolio_similarity.py` — compute **within** and **between** portfolio similarity at the product/release-year level (cosine); writes CSV + PNG.
- `05_firmyear_similarity.py` — compute **firm-year technology similarity** for `pairs.csv` using **centroids** of patent embeddings + **diagonal cosine** → `pairs_simtech.csv`.

**R — empirical analysis**
- `21_build_pair_product.R` — build **pair_product.csv** (merges sales/EBITDA; computes HHI; fixes `gvkey` formats).
- `22_reg_rd.R` — **R&D**: FE + interactions, **binned** FE by similarity deciles, **GAM** heatmaps (with density mask).
- `23_reg_sales.R` — **Sales**: FE + interactions, binned FE, **GAM** heatmaps (with density mask).
- `24_ml_gam_heatmaps.R` — **optional** compact ML/GAM bundle (R&D + Sales heatmaps, plus Tweedie XGBoost maps if `xgboost` is available).

**Python — theory simulations**
- `10_cournot_duopoly.py` — duopoly Cournot with cost-reducing R&D and patent transmission φ; heatmaps for **φ\***, **welfare**, **avg z**, **avg q**; also exports a LaTeX table.
- `11_cournot_nfirm_linear.py` — **N-firm linear** Cournot simulation on a (δ̄, ω̄) grid; parallelized; robust for larger N; heatmaps for **φ\***, **welfare**, **avg z**, **avg q**.
  
> If any script in your local copy uses **constants at the top** (instead of `--flags`), just edit those paths in the file and run. Outputs/flow remain identical.

---

## Environment

### Python (3.10+)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt



