# On_Sinergy_And_Rivalry

### Necessary from server
0) Files Online Useful for Replication:
   - [The product similarity dataset from Hoberg Philips 2025](https://hobergphillips.tuck.dartmouth.edu/tnic_doc2vec.html). The full version or TNIC 2 works
   - [The abstracts (g_abstract) from patentsview](https://patentsview.org/download/data-download-tables)
   - [The hugging face model to embed patents](https://huggingface.co/AAUBS/PatentSBERTa_V2)
   - [This hugging face model for the name matching process for Compustat and USPTO](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
   - [Compustat Database](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/query.png). You have to access Wharton for this, the image shows how the request looks like.


The first step is to create these files that I am going to give it to you directly in the next section. We have to run [the matching code](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/mymatch.py) that creates a matching table for compustat and USPTO names. This is - `02_match_companies_to_assignees.py` — match **Compustat** company names to **PatentsView** assignees (regex clean + SBERT + cosine) → reviewable CSV. There is also a discern path but it was used to compare the performance of my matching with the DISCERN dataset, it can be deleted. 

If you want to directly download the matching table for replication or because you need a better matching between Compustat names and Assignees in USPTO dataset here is [the table](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/company_assignee_matching_for_review.csv) 

Then with this matching table we can create companydata_with_portfolio_embedded.csv, which is a file I give you directly but it is only what can be created with [the following code](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/mypatentportfolio.py). This is `03_make_company_portfolios.py` — build **firm-year patent portfolios** (rolling 5-year window) from validated matches → `companydata_with_portfolio_embedded.csv`.


You are then going to use `01_embed_patents.py` — compute **PatentSBERTa_V2** embeddings for USPTO abstracts (chunked, memory-safe) → Parquet `USPTO_abstracts_embeddings.parquet`. This is the [code](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/01_embed_patents.py). Remember that for this similarity step you have to download the abstract from patentsview.

Then we are going to create the file that creates a pair for each company every year, this is the [following code](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/pairwisecreation.py). Then we compute the similarity between those companies at the technology level (having already the product level from Hoberg and Phillips). This will give us `pairs_simtech.csv. This is [the code](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/Portfolio_Similarity.py) and its [sbatch](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/Portfolio_Similarity.sbatch) to run it on a server with 128 cores. 


2) Download these files to the repo root:
   - `companydata_with_portfolio_embedded.csv`
   - `pairs_simtech.csv`
   - `pair_product.csv`. This is just `pairs_simtech.csv` attached with the financial data and the hoberg and phillips dataset, in case you want to start directly with the analysis without actually reproducing the dataset creation.

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


Now on to the Results section, all done with R:


---

**R — empirical analysis**

We have to use the dataset from Compustat that we downloaded (companydata) to make it suitable for our analysis. We are going to select the variables, merge with the Hoberg and Phillips data a the SIC 2 digit level (so relationships between firms who share at least the first 2 digits). Here you need `salescompustat.csv` another query from compustat to match sales and ebitda that I didn't download at the first run. This is the [code](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/1%20Dataset%20Preparation.R)
You can skip this part directly and use `pair_product.csv`. I already said that I will make it available.

When you have `pair_product.csv` you can use `Code R&D.R`, this [code](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/2%20Code%20R%26D.R) is for the linear regression without interaction term, with interaction term, the glm, the GAM, the XgBoost, both for levels and logs.

If you want to recreate the data on sales (GAM) available in the appendix you should use this [code](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/4%20Code%20Sales.R).

If you need a version that uses lower memory you can look at [this code](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/GAM%20(lower%20memory%20needed).txt). But again you would still need around 150 GB of RAM for it.

For the semiconductor dataset I do not have the permission to share it for now since it is still under construction for the APTO NSF grant.

## Python — nonlinear & sector Model simulations

- `12_nonlinear_10firms_nonlinear_firstmodel.py` —  Block-iterates on `(q, z, s)`, enforces
  the **spectral-radius gate** `ρ(Γ(s)) < 2`, and exports heatmaps for **φ\***, **W**, **avg(s+z)**, **avg(q)**.
  Here 10 firms

- `13_nonlinear_bounded1000firms_linksdistribution.py` - Here 1000 firms, for this reason we have a distribution of firm links (normal mean 10 se 5)

- `14_nonlinear_sz_separate_5firmssmall` — Nonlinear Cournot with **separate** R&D channels:
  process R&D `z` (cost reduction, spills via ω and φ) and product-differentiation R&D `s`
  (reduces substitutability via `s_i^(1-φ)` in Γ). 
  `(2I + Γ(s)) q = (A − c) + z + φ Ω z`, `z_i = q_i/κ_i`,
  and `κ_i s_i^{3−φ} = (1−φ) q_i (Δ q)_i` (closed-form update for `s_i`), with full heatmaps.

- `15_nonlinear_sz_100firms_boundedlinks` — Separate `s`+`z` **with bounded links** (mean degree ≈ 5),
  row-normalized so each row equals the grid mean for δ̄ or ω̄; this helps ensure `ρ(Γ(s)) < 2`
  while scaling to larger **N**. Exports main (φ*, W, avg(s+z), avg q) and split (avg s, avg z) maps.

- `16_semiconductor_case_study.py` — estimates yearly patent transmission **φ̂_t** via a **two-stage** distribution-matching approach.  
  **Stage 1:** choose global scales (s_q, s_z) so that, using baseline years at φ_ref=0.7, the implied R&D cost **κ** and marginal cost **c** match thesis targets (κ~N(5,1²), SD(c)=5; mean(c)=25 enforced via A_t = 25 + mean(r)).  
  **Stage 2:** with (s_q, s_z) fixed, estimate **φ̂_t** for each year by matching the distributions of κ and c with a mild prior on **A**, then save **empirical_phi_two_stage.csv** and the figures (φ̂_t path, avg sales, avg z, #firms).  :contentReference[oaicite:0]{index=0}

- You can also see the alternative cost distribution simulations in [this code for 10 firms](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/nonlin_alternativecostdistribution_10firms.py) and [this code for 1000 firms](https://github.com/dario-marino/On_Sinergy_And_Rivalry/blob/main/nonlin_alternativecostdistribution_1000firms.py) 

  
**Python - Old theory simulations (available in Appendix)**
- `10_cournot_duopoly.py` — duopoly Cournot with cost-reducing R&D and patent transmission φ; heatmaps for **φ\***, **welfare**, **avg z**, **avg q**; also exports a LaTeX table.
- `11_cournot_nfirm_linear.py` — **N-firm linear** Cournot simulation on a (δ̄, ω̄) grid; parallelized; robust for larger N; heatmaps for **φ\***, **welfare**, **avg z**, **avg q**.

---

## Environment

### Python (3.10+)

# R (4.2+)
install.packages(c(
  "fixest","mgcv","xgboost","Matrix",
  "dplyr","ggplot2","tidyr","viridis"

  I used 40 cores and 250 GB of RAM on the remote UChicago Acropolis Server
))
