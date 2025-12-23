import pandas as pd
import numpy as np
import gc
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def calculate_herfindahl_index(df):
    """
    Calculate Herfindahl-Hirschman Index (HHI) for market concentration
    using assets (at) within each SIC code for each fyear.
    """
    df_assets = df[(df['at'].notna()) & (df['at'] > 0) & (df['sic'].notna())].copy()
    hhi_results = []
    for fyear in df_assets['fyear'].unique():
        fyear_data = df_assets[df_assets['fyear'] == fyear]
        for sic in fyear_data['sic'].unique():
            sic_data = fyear_data[fyear_data['sic'] == sic]
            total_assets = sic_data['at'].sum()
            if total_assets > 0:
                shares = sic_data['at'] / total_assets
                hhi = (shares ** 2).sum()
                hhi_results.append({
                    'fyear': fyear,
                    'sic': sic,
                    'concentration_hhi': hhi,
                    'num_firms_in_sic': len(sic_data),
                    'total_assets_in_sic': total_assets
                })
    hhi_df = pd.DataFrame(hhi_results)
    return hhi_df


def process_fyear_pairs(fyear_firms, fyear, hhi_lookup):
    """
    Create all directed pairs among firms in a given year.
    """
    # Efficient cross-join minus self
    left = fyear_firms.add_suffix('_start')
    right = fyear_firms.add_suffix('_recv')
    # full cross
    pairs = left.merge(right, how='cross')
    # drop self-pairs
    pairs = pairs[pairs['gvkey_start'] != pairs['gvkey_recv']]
    # lookup concentrations
    pairs['concentration_start'] = pairs.apply(
        lambda row: hhi_lookup.get((fyear, row['sic_start']), np.nan), axis=1)
    pairs['concentration_recv'] = pairs.apply(
        lambda row: hhi_lookup.get((fyear, row['sic_recv']), np.nan), axis=1)
    pairs['fyear'] = fyear
    return pairs


def create_pairwise_network_sequential(input_file, output_file):
    """
    Stream each year's pairwise DataFrame to a single Parquet file using row-groups.
    """
    # prepare output
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # read and preprocess
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df):,} rows from CSV.")
    hhi_df = calculate_herfindahl_index(df)
    hhi_lookup = {(r.fyear, r.sic): r.concentration_hhi for r in hhi_df.itertuples()}
    del hhi_df
    gc.collect()

    df = df[(df['patent_portfolio'].notna()) \
            & (df['patent_portfolio'] != '[]') \
            & (df['patent_portfolio'].astype(str).str.strip() != '')]
    print(f"After filtering patents: {len(df):,} rows.")

    fyears = sorted(df['fyear'].unique())
    print(f"Processing years: {fyears}")

    writer = None
    for idx, year in enumerate(fyears, 1):
        sub = df[df['fyear'] == year].copy()
        if len(sub) < 2:
            print(f"Skipping {year}, only {len(sub)} firms.")
            continue
        print(f"Year {year}: {len(sub)} firms -> computing pairs...")
        pairs_df = process_fyear_pairs(sub, year, hhi_lookup)
        # convert to Arrow
        table = pa.Table.from_pandas(pairs_df)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema)
        writer.write_table(table)
        print(f"Wrote row-group for year {year} ({len(pairs_df):,} rows)")
        del sub, pairs_df, table
        gc.collect()

    if writer:
        writer.close()
        print(f"Completed writing parquet: {output_file}")

    # final load and summary
    final = pq.read_table(str(out_path)).to_pandas()
    print(f"Final table: {final.shape[0]:,} rows, {final.shape[1]} columns")
    return final


if __name__ == '__main__':

    inp = '/home/dariomarino/Thesis/companydata_with_portfolio_embedded.csv'
    outp = '/home/dariomarino/Thesis/pairwise_network_final.parquet'

    df_final = create_pairwise_network_sequential(inp, outp)
    print(df_final.head())

    # --------------------------------------------------
    # Extract minimal pairs.csv for similarity stage
    # --------------------------------------------------

    print("Creating minimal pairs.csv for similarity step...")

    out_path = Path(outp)

    pairs_min = (
        pq.read_table(out_path)
          .to_pandas()[["gvkey_start", "gvkey_recv", "fyear_start"]]
    )

    pairs_csv_path = out_path.with_name("pairs.csv")
    pairs_min.to_csv(pairs_csv_path, index=False)

    print(f"Saved minimal pairs file to: {pairs_csv_path}")
    print(f"Rows: {len(pairs_min):,}")

