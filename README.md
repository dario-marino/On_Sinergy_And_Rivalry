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



# On Synergy and Rivalry — Full Replication README (flat files, one page)

This is a **single, self-contained README** for a flat GitHub repo (no folders). It explains the **entire pipeline** — theory, data construction, portfolio/technology similarity, regressions, and simulations — and gives **copy-paste commands** for steps **01 → 10**. Put all scripts in the repo root, place inputs under `data/` (or adjust flags), and run from the repo root.

---

## What this project studies (in one paragraph)

The thesis studies how **technology similarity** (ω) and **product similarity** (δ) jointly shape R&D: technology proximity enables **synergy** (spillovers), while product proximity raises **rivalry** (substitutability). A regulator chooses **patent transmission** φ (how much improvements spread) to maximize welfare. The work derives closed-form duopoly results and scalable N-firm simulations, then tests predictions on a firm-pair panel built from patent-portfolio embeddings and product text similarity, merged to Compustat outcomes.

---

## Files in this repo (flat)

**Python — data & similarity**
- `01_embed_patents.py` — Memory-efficient embeddings for USPTO abstracts (PatentSBERTa_V2): chunked, GC-safe, Parquet output.
- `02_match_companies_to_assignees.py` — Compustat company names ↔ PatentsView assignees (regex clean + SBERT + cosine).
- `03_make_company_portfolios.py` — Firm-year patent portfolios (rolling 5-year window) from validated matches.
- `04_products_portfolio_similarity.py` — Within/between portfolio similarity (cosine) at the product level; writes CSV + PNG.
- `05_firmyear_similarity.py` — Firm-year technology similarity for `pairs.csv`: portfolio **centroids** + **diagonal cosine**; writes `pairs_simtech.csv`.

**R — empirical analysis**
- `06_make_pair_product.R` — Builds `pair_product.csv` (Compustat sales/EBITDA merge; HHI by SIC; fixes gvkey formats).
- `07_rd_regressions.R` — R&D FE + interactions; **binned** FE by similarity deciles; **GAM** surfaces; **Tweedie XGBoost** heatmaps with density masking.
- `08_sales_regressions.R` — Sales version of 07 (FE, bins, GAM heatmaps with density mask).
- `09_ml_gam_heatmaps.R` — Optional ML/GAM bundle (organized variants of 07/08).

**Python — theory simulations**
- `10_cournot_duopoly.py` — Duopoly Cournot with R&D and patent transmission φ; heatmaps for φ*, welfare, avg z, avg q; exports figures/LaTeX table.
- `10b_linear_simulation_fixed.py` — N-firm **linear** Cournot on a (δ̄, ω̄) grid; parallel; robust for larger N.

> A few scripts accept CLI flags; if a script in your copy uses **constants** at the top instead, just edit those paths there — everything else stays the same.

---

## Environment

### Python (3.10+)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt




