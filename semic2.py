import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import time

def create_lookup_dicts(valid_matches, companydata, salescompustat):
    """
    Create optimized lookup dictionaries for faster data retrieval
    """
    print("Creating optimized lookup dictionaries...")
    
    # Create manufacturer to (conm, gvkey) mapping
    manufacturer_lookup = {}
    for _, row in valid_matches.iterrows():
        manufacturer_lookup[row['manufacturer']] = {
            'conm': row['conm'],
            'gvkey': row['gvkey']
        }
    
    # Create (conm, fyear) -> company data mapping
    company_lookup = {}
    for _, row in companydata.iterrows():
        key = (row['conm'], row['fyear'])
        company_lookup[key] = {
            'at': row.get('at', np.nan),
            'emp': row.get('emp', np.nan),
            'xrd': row.get('xrd', np.nan),
            'sic': row.get('sic', np.nan)
        }
    
    # Create (gvkey, fyear) -> sales data mapping
    sales_lookup = {}
    for _, row in salescompustat.iterrows():
        key = (row['gvkey'], row['fyear'])
        sales_lookup[key] = {
            'sale': row.get('sale', np.nan),
            'ebitda': row.get('ebitda', np.nan)
        }
    
    return manufacturer_lookup, company_lookup, sales_lookup

def process_manufacturer_batch(args):
    """
    Process a batch of manufacturers using vectorized operations
    """
    chunk_data, manufacturer_col, suffix, manufacturer_lookup, company_lookup, sales_lookup = args
    chunk, start_idx = chunk_data
    
    # Initialize result columns
    result_cols = {
        f'at{suffix}': [],
        f'emp{suffix}': [],
        f'xrd{suffix}': [],
        f'sale{suffix}': [],
        f'ebitda{suffix}': [],
        f'sic{suffix}': []
    }
    
    for _, row in chunk.iterrows():
        manufacturer = row[manufacturer_col]
        release_year = row['release_year']
        
        # Default values
        values = {col: np.nan for col in result_cols.keys()}
        
        if pd.notna(manufacturer) and pd.notna(release_year):
            # Get manufacturer info
            if manufacturer in manufacturer_lookup:
                mfg_info = manufacturer_lookup[manufacturer]
                conm = mfg_info['conm']
                gvkey = mfg_info['gvkey']
                
                # Get company data
                company_key = (conm, release_year)
                if company_key in company_lookup:
                    company_data = company_lookup[company_key]
                    values[f'at{suffix}'] = company_data['at']
                    values[f'emp{suffix}'] = company_data['emp']
                    values[f'xrd{suffix}'] = company_data['xrd']
                    values[f'sic{suffix}'] = company_data['sic']
                
                # Get sales data
                sales_key = (gvkey, release_year)
                if sales_key in sales_lookup:
                    sales_data = sales_lookup[sales_key]
                    values[f'sale{suffix}'] = sales_data['sale']
                    values[f'ebitda{suffix}'] = sales_data['ebitda']
        
        # Append values to result columns
        for col in result_cols.keys():
            result_cols[col].append(values[col])
    
    return chunk.index.tolist(), result_cols

def process_chunk_optimized(chunk_data, manufacturer_lookup, company_lookup, sales_lookup):
    """
    Process a chunk using optimized lookup dictionaries
    """
    chunk, start_idx = chunk_data
    
    # Process manufacturer_1 (_start columns)
    mfg1_args = (chunk_data, 'manufacturer_1', '_start', 
                 manufacturer_lookup, company_lookup, sales_lookup)
    indices1, results1 = process_manufacturer_batch(mfg1_args)
    
    # Process manufacturer_2 (_recv columns)  
    mfg2_args = (chunk_data, 'manufacturer_2', '_recv',
                 manufacturer_lookup, company_lookup, sales_lookup)
    indices2, results2 = process_manufacturer_batch(mfg2_args)
    
    # Add all new columns to the chunk
    for col, values in results1.items():
        chunk[col] = values
    
    for col, values in results2.items():
        chunk[col] = values
    
    return chunk

def main():
    print("Loading datasets...")
    start_time = time.time()
    
    # Load all datasets
    product_similarities = pd.read_csv('/home/dariomarino/Thesis/product_similarities_with_patents.csv')
    companydata = pd.read_csv('/home/dariomarino/Thesis/companydata.csv')
    salescompustat = pd.read_csv('/home/dariomarino/Thesis/salescompustat.csv')
    matching_table = pd.read_csv('/home/dariomarino/Thesis/company_matching_table.csv')
    
    # Extract year from datadate in salescompustat (first 4 characters)
    salescompustat['fyear'] = salescompustat['datadate'].astype(str).str[:4].astype(int)
    
    print(f"Product similarities shape: {product_similarities.shape}")
    print(f"Company data shape: {companydata.shape}")
    print(f"Sales compustat shape: {salescompustat.shape}")
    print(f"Matching table shape: {matching_table.shape}")
    
    # Filter matching table to only valid matches
    valid_matches = matching_table[matching_table['valid'] == True].copy()
    print(f"Valid matches: {len(valid_matches)}")
    
    # Create optimized lookup dictionaries
    manufacturer_lookup, company_lookup, sales_lookup = create_lookup_dicts(
        valid_matches, companydata, salescompustat)
    
    print(f"Created lookup dictionaries:")
    print(f"  - Manufacturer lookup: {len(manufacturer_lookup)} entries")
    print(f"  - Company lookup: {len(company_lookup)} entries") 
    print(f"  - Sales lookup: {len(sales_lookup)} entries")
    
    # Set up parallel processing
    n_cores = min(40, cpu_count())
    print(f"Using {n_cores} cores for processing")
    
    # Create chunks
    chunk_size = len(product_similarities) // n_cores + 1
    chunks = []
    
    for i in range(0, len(product_similarities), chunk_size):
        chunk = product_similarities.iloc[i:i + chunk_size].copy()
        chunks.append((chunk, i))
    
    print(f"Created {len(chunks)} chunks (avg size: {chunk_size})")
    
    # Create partial function with lookup dictionaries
    process_func = partial(process_chunk_optimized,
                          manufacturer_lookup=manufacturer_lookup,
                          company_lookup=company_lookup,
                          sales_lookup=sales_lookup)
    
    # Process chunks in parallel
    print("Processing chunks in parallel...")
    processing_start = time.time()
    
    with Pool(processes=n_cores) as pool:
        processed_chunks = pool.map(process_func, chunks)
    
    processing_time = time.time() - processing_start
    print(f"Parallel processing completed in {processing_time:.2f} seconds")
    
    # Combine all processed chunks
    print("Combining processed chunks...")
    combine_start = time.time()
    final_df = pd.concat(processed_chunks, ignore_index=True)
    combine_time = time.time() - combine_start
    print(f"Combining completed in {combine_time:.2f} seconds")
    
    # Print statistics about the matching
    print("\nMatching Statistics:")
    print("===================")
    
    columns_to_check = ['at_start', 'emp_start', 'xrd_start', 'sale_start', 'ebitda_start', 'sic_start',
                       'at_recv', 'emp_recv', 'xrd_recv', 'sale_recv', 'ebitda_recv', 'sic_recv']
    
    for col in columns_to_check:
        non_null_count = final_df[col].notna().sum()
        total_count = len(final_df)
        percentage = (non_null_count / total_count) * 100
        print(f"  {col}: {non_null_count}/{total_count} ({percentage:.1f}%) non-null values")
    
    # Save the final dataset
    print("Saving final dataset...")
    save_start = time.time()
    output_file = '/home/dariomarino/Thesis/productsim_final.csv'
    final_df.to_csv(output_file, index=False)
    save_time = time.time() - save_start
    print(f"Saving completed in {save_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\nPerformance Summary:")
    print(f"  - Data loading: {processing_start - start_time:.2f}s")
    print(f"  - Parallel processing: {processing_time:.2f}s")
    print(f"  - Combining chunks: {combine_time:.2f}s")
    print(f"  - Saving results: {save_time:.2f}s")
    print(f"  - Total execution time: {total_time:.2f}s")
    
    print(f"\nFinal dataset saved to: {output_file}")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"New columns added: {columns_to_check}")
    
    # Show sample of the new data
    print("\nSample of new columns:")
    sample_cols = ['manufacturer_1', 'manufacturer_2', 'release_year'] + columns_to_check[:6]  # Show first 6 new cols
    print(final_df[sample_cols].head())
    
    print("\nSample of remaining new columns:")
    sample_cols2 = ['manufacturer_1', 'manufacturer_2', 'release_year'] + columns_to_check[6:]  # Show last 6 new cols
    print(final_df[sample_cols2].head())

if __name__ == "__main__":
    main()