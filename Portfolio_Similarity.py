#!/usr/bin/env python3
import ast
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.preprocessing import normalize
from collections import defaultdict
import gc
import time

# Paths and constants
PARQUET = "/scratch/midway3/dariomarino/USPTO_abstracts_embeddings.parquet"
COMP_CSV = "/scratch/midway3/dariomarino/companydata_with_portfolio_embedded.csv"
PAIRS_CSV = "/scratch/midway3/dariomarino/pairs.csv"
OUTPUT_CSV = "/scratch/midway3/dariomarino/pairs_simtech.csv"
DIM = 768

def memory_efficient_cosine_similarity(A, B):
    """
    Compute pairwise cosine similarity between corresponding rows of A and B.
    Memory efficient - only computes diagonal elements, not full cross-product.
    """
    # Normalize vectors to unit length
    A_normalized = normalize(A, norm='l2', axis=1, copy=False)
    B_normalized = normalize(B, norm='l2', axis=1, copy=False)
    
    # Compute element-wise dot product (cosine similarity for normalized vectors)
    similarities = np.sum(A_normalized * B_normalized, axis=1)
    
    return similarities

def chunked_cosine_similarity(A, B, chunk_size=50000):
    """
    Alternative: Compute similarities in chunks for very large datasets.
    """
    n_samples = A.shape[0]
    similarities = np.zeros(n_samples, dtype=np.float32)
    
    print(f"Computing similarities in chunks of {chunk_size}...")
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_A = A[start_idx:end_idx]
        chunk_B = B[start_idx:end_idx]
        
        similarities[start_idx:end_idx] = memory_efficient_cosine_similarity(chunk_A, chunk_B)
        
        if start_idx % (chunk_size * 10) == 0:
            print(f"Processed {end_idx}/{n_samples} pairs ({100*end_idx/n_samples:.1f}%)")
    
    return similarities

def main():
    start_time = time.time()
    
    # 1️⃣ Load company metadata
    print("Loading company metadata...")
    print(f"Reading {COMP_CSV}")
    
    # Read company data with patent portfolios
    comp = pd.read_csv(COMP_CSV, 
                       converters={"patent_portfolio": ast.literal_eval},
                       usecols=["gvkey", "fyear", "patent_portfolio"])
    
    print(f"Loaded {len(comp)} company-year records")
    
    # Build mapping: patent_id → list[(gvkey, fyear)]
    print("Building patent-to-company mapping...")
    mapping = defaultdict(list)
    total_patents = 0
    
    for _, r in comp.iterrows():
        if r.patent_portfolio:  # Skip empty portfolios
            for pid in r.patent_portfolio:
                # Keep patent IDs as strings for consistency
                patent_str = str(pid)
                mapping[patent_str].append((r.gvkey, r.fyear))
                total_patents += 1
    
    print(f"Mapped {len(mapping)} unique patents across {total_patents} patent-company relationships")
    
    # 2️⃣ Aggregate embeddings into firm-year sums and counts
    print("Aggregating embeddings by firm-year...")
    sums = defaultdict(lambda: np.zeros(DIM, dtype=np.float64))  # Use float64 for accumulation
    counts = defaultdict(int)
    
    # Read parquet in batches
    dataset = ds.dataset(PARQUET, format="parquet")
    cols = ["patent_id"] + [f"embedding_{i}" for i in range(DIM)]
    
    batch_count = 0
    processed_patents = 0
    
    for batch in dataset.to_batches(columns=cols, batch_size=100_000):  # Reduced batch size
        batch_count += 1
        df = batch.to_pandas()
        
        print(f"Processing batch {batch_count}, size: {len(df)}")
        
        # Process each row in the batch
        for row in df.itertuples(index=False):
            pid = str(row.patent_id)
            
            if pid in mapping:
                # Extract embedding vector
                vec = np.array(row[1:], dtype=np.float32)
                
                # Add to all companies that have this patent
                for company_key in mapping[pid]:
                    sums[company_key] += vec.astype(np.float64)
                    counts[company_key] += 1
                
                processed_patents += 1
        
        # Periodic memory cleanup
        if batch_count % 10 == 0:
            gc.collect()
            print(f"Processed {processed_patents} relevant patents so far...")
    
    print(f"Finished processing embeddings. Total relevant patents: {processed_patents}")
    
    # 3️⃣ Compute centroids (mean vectors)
    print("Computing firm-year centroids...")
    centroids = {}
    
    for company_key in sums:
        if counts[company_key] > 0:
            # Convert back to float32 after averaging
            centroids[company_key] = (sums[company_key] / counts[company_key]).astype(np.float32)
        else:
            centroids[company_key] = np.zeros(DIM, dtype=np.float32)
    
    print(f"Computed centroids for {len(centroids)} firm-year combinations")
    
    # Clear memory
    del sums, mapping
    gc.collect()
    
    # 4️⃣ Load pairs and build matrices
    print("Loading pairs and building similarity matrices...")
    dfp = pd.read_csv(PAIRS_CSV)
    N = len(dfp)
    
    print(f"Loaded {N} firm pairs to process")
    
    # Pre-allocate matrices
    A = np.zeros((N, DIM), dtype=np.float32)
    B = np.zeros((N, DIM), dtype=np.float32)
    
    missing_firms = []
    
    for idx, r in dfp.iterrows():
        key_start = (r.gvkey_start, r.fyear_start)
        key_recv = (r.gvkey_recv, r.fyear_start)  # Same year for both firms
        
        if key_start in centroids:
            A[idx] = centroids[key_start]
        else:
            missing_firms.append(f"gvkey_start {r.gvkey_start} in year {r.fyear_start}")
            A[idx] = np.zeros(DIM, dtype=np.float32)  # Zero vector for missing firms
        
        if key_recv in centroids:
            B[idx] = centroids[key_recv]
        else:
            missing_firms.append(f"gvkey_recv {r.gvkey_recv} in year {r.fyear_start}")
            B[idx] = np.zeros(DIM, dtype=np.float32)  # Zero vector for missing firms
        
        if idx % 100000 == 0 and idx > 0:
            print(f"Built matrices for {idx}/{N} pairs...")
    
    if missing_firms:
        print(f"Warning: {len(missing_firms)} firm-year combinations not found in embeddings")
        # Print first few missing firms for debugging
        for i, firm in enumerate(missing_firms[:10]):
            print(f"  Missing: {firm}")
        if len(missing_firms) > 10:
            print(f"  ... and {len(missing_firms) - 10} more")
    
    # Clear centroids to save memory
    del centroids
    gc.collect()
    
    # 5️⃣ Compute cosine similarities
    print("Computing cosine similarities...")
    print(f"Matrix A shape: {A.shape}, Matrix B shape: {B.shape}")
    
    # Choose method based on dataset size
    if N > 1000000:  # Use chunked approach for very large datasets
        similarities = chunked_cosine_similarity(A, B, chunk_size=50000)
    else:
        similarities = memory_efficient_cosine_similarity(A, B)
    
    print(f"Computed {len(similarities)} similarities")
    print(f"Similarity stats: min={similarities.min():.4f}, max={similarities.max():.4f}, mean={similarities.mean():.4f}")
    
    # 6️⃣ Save results
    print("Saving results...")
    dfp["sim_tech"] = similarities
    dfp.to_csv(OUTPUT_CSV, index=False)
    
    # Summary statistics
    end_time = time.time()
    print(f"\n✅ Done! Results saved to {OUTPUT_CSV}")
    print(f"Total processing time: {(end_time - start_time)/60:.1f} minutes")
    print(f"Output file contains {len(dfp)} pairs with technology similarities")

if __name__ == "__main__":
    main()