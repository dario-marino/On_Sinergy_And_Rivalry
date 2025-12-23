import os
import pandas as pd
import numpy as np

# File paths
COMPANY_DATA_PATH = '/home/dariomarino/Thesis/companydataxrd.csv'
PAT_PATH = '/home/dariomarino/single_patents_with_top.csv'
MATCHING_PATH = 'company_assignee_matching_for_review.csv'  # From step 1

# Columns to keep from company data
company_cols = [
    "gvkey", "fyear", "conm", "aco", "act", "ao", "at",
    "ceql", "ceqt", "emp", "xrd", "exchg", "cik",
    "sich", "sic"
]

# Ensure files exist
for path in (COMPANY_DATA_PATH, PAT_PATH, MATCHING_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# Load datasets
print("Loading datasets...")
df_company = pd.read_csv(COMPANY_DATA_PATH, dtype=str, low_memory=False)
df_pat = pd.read_csv(PAT_PATH, dtype=str, low_memory=False)
df_matching = pd.read_csv(MATCHING_PATH)

print(f"Company data shape: {df_company.shape}")
print(f"Patent data shape: {df_pat.shape}")
print(f"Matching data shape: {df_matching.shape}")

# Debug: Check column names in matching file
print(f"Matching file columns: {list(df_matching.columns)}")

# Check if all required columns exist in company data
missing_cols = [col for col in company_cols if col not in df_company.columns]
if missing_cols:
    print(f"WARNING: Missing columns in company data: {missing_cols}")
    # Use available columns only
    available_cols = [col for col in company_cols if col in df_company.columns]
    print(f"Using available columns: {available_cols}")
    company_cols = available_cols

# Prepare the lookup table from validated matches
print("Building lookup table from validated matches...")
df_valid_matches = df_matching

# Create the lookup dictionary - using correct column names
lookup = {}
for _, row in df_valid_matches.iterrows():
    # Use the actual column names from the CSV
    company = row['company_original']  # This is the same as 'conm' from company data
    assignee = row['matched_assignee_original']  # This is the same as 'disambig_assignee_organization' from patent data
    if company and assignee:
        # Direct lookup - no case conversion needed since these are exact matches from the same sources
        lookup[company] = assignee

# Print lookup statistics
print(f"Valid matches loaded: {len(lookup)}")
print(f"Sample matches:")
for i, (company, assignee) in enumerate(list(lookup.items())[:5]):
    print(f"  {company} → {assignee}")

# Validation: Check that all lookup entries exist in their respective datasets
print(f"\n=== VALIDATION CHECKS ===")
company_names_in_lookup = set(lookup.keys())
assignee_names_in_lookup = set(lookup.values())

# Check company names
company_names_in_data = set(df_company['conm'].fillna('').unique())
missing_companies = company_names_in_lookup - company_names_in_data
if missing_companies:
    print(f"WARNING: {len(missing_companies)} company names from lookup table NOT found in company data:")
    for company in list(missing_companies)[:10]:  # Show first 10
        print(f"  '{company}'")
    if len(missing_companies) > 10:
        print(f"  ... and {len(missing_companies) - 10} more")
else:
    print(f"✓ All {len(company_names_in_lookup)} company names from lookup table found in company data")

# Check assignee names
assignee_names_in_data = set(df_pat['disambig_assignee_organization'].fillna('').unique())
missing_assignees = assignee_names_in_lookup - assignee_names_in_data
if missing_assignees:
    print(f"WARNING: {len(missing_assignees)} assignee names from lookup table NOT found in patent data:")
    for assignee in list(missing_assignees)[:10]:  # Show first 10
        print(f"  '{assignee}'")
    if len(missing_assignees) > 10:
        print(f"  ... and {len(missing_assignees) - 10} more")
else:
    print(f"✓ All {len(assignee_names_in_lookup)} assignee names from lookup table found in patent data")

# Summary of validation
total_issues = len(missing_companies) + len(missing_assignees)
if total_issues == 0:
    print(f"✓ VALIDATION PASSED: All lookup entries have matches in source datasets")
else:
    print(f"⚠ VALIDATION ISSUES: {total_issues} lookup entries missing from source datasets")
    print(f"  This may indicate data inconsistencies or changes in source files")

# Prepare patent data
print("Preparing patent data...")
df_pat['patent_date'] = pd.to_datetime(df_pat['patent_date'], errors='coerce')
df_pat['patent_year'] = df_pat['patent_date'].dt.year
df_pat['assignee_clean'] = df_pat['disambig_assignee_organization'].fillna('')

# Prepare company data
df_company['conm_clean'] = df_company['conm'].fillna('')

# Convert fyear to numeric for date comparisons
df_company['fyear_numeric'] = pd.to_numeric(df_company['fyear'], errors='coerce')

# Function to assemble patent portfolio
def get_portfolio(row):
    """
    Get patent portfolio for a company based on:
    - Company name matching to assignee organization
    - Patents filed 5 years before to year of company's fiscal year
    """
    company_name = row['conm_clean']
    company_year = row['fyear_numeric']
    
    if not company_name or pd.isna(company_year):
        return []
    
    try:
        company_year = int(company_year)
    except (ValueError, TypeError):
        return []
    
    # Look up the matched assignee
    assignee = lookup.get(company_name)
    if not assignee:
        return []
    
    # Filter patents by assignee and date range
    mask = (
        (df_pat['assignee_clean'] == assignee) &
        (df_pat['patent_year'] >= company_year - 5) &
        (df_pat['patent_year'] <= company_year)
    )
    
    patents = df_pat.loc[mask, 'patent_id'].dropna().unique().tolist()
    return patents

# Apply portfolio generation
print("Generating patent portfolios...")
df_company['patent_portfolio'] = df_company.apply(get_portfolio, axis=1)

# Calculate portfolio statistics
df_company['portfolio_size'] = df_company['patent_portfolio'].apply(len)
companies_with_portfolios = (df_company['portfolio_size'] > 0).sum()
total_companies = len(df_company)
avg_portfolio_size = df_company['portfolio_size'].mean()

print(f"\n=== PORTFOLIO GENERATION SUMMARY ===")
print(f"Total company-year observations: {total_companies}")
print(f"Company-year observations with patent portfolios: {companies_with_portfolios}")
print(f"Coverage: {companies_with_portfolios/total_companies*100:.2f}%")
print(f"Average portfolio size: {avg_portfolio_size:.2f}")
print(f"Total unique patents assigned: {df_company['patent_portfolio'].apply(len).sum()}")

# Show distribution of portfolio sizes
portfolio_dist = df_company['portfolio_size'].value_counts().sort_index()
print(f"\nPortfolio size distribution:")
for size, count in portfolio_dist.head(10).items():
    print(f"  {size} patents: {count} company-year observations")

# Show companies with largest portfolios
print(f"\nTop 10 companies by portfolio size:")
top_portfolios = df_company.nlargest(10, 'portfolio_size')[['conm', 'fyear', 'portfolio_size']]
for _, row in top_portfolios.iterrows():
    print(f"  {row['conm']} ({row['fyear']}): {row['portfolio_size']} patents")

# Analyze by year
print(f"\nPortfolio statistics by year:")
yearly_stats = df_company.groupby('fyear_numeric').agg({
    'portfolio_size': ['count', 'sum', 'mean'],
    'conm': 'count'
}).round(2)
yearly_stats.columns = ['companies_with_portfolios', 'total_patents', 'avg_portfolio_size', 'total_companies']
yearly_stats['coverage_pct'] = (yearly_stats['companies_with_portfolios'] / yearly_stats['total_companies'] * 100).round(2)

# Show recent years
recent_years = yearly_stats.tail(10)
print(recent_years.to_string())

# Keep only specified columns plus portfolio information
final_cols = company_cols + ['patent_portfolio', 'portfolio_size']
df_company_final = df_company[final_cols].copy()

# Clean up temporary columns
df_company_final.drop(columns=['portfolio_size'], inplace=True)

# Save final results
OUTPUT_PATH = 'companydata_with_portfolio_embedded.csv'
df_company_final.to_csv(OUTPUT_PATH, index=False)

print(f"\nFinal results saved to: {OUTPUT_PATH}")
print(f"Final dataset shape: {df_company_final.shape}")
print(f"Columns in final dataset: {list(df_company_final.columns)}")

# Optional: Save a summary report
print("\nGenerating summary report...")
summary_data = []
for _, row in df_company_final.iterrows():
    if len(row['patent_portfolio']) > 0:
        summary_data.append({
            'gvkey': row.get('gvkey', ''),
            'fyear': row.get('fyear', ''),
            'conm': row.get('conm', ''),
            'portfolio_size': len(row['patent_portfolio']),
            'patent_ids': ', '.join(row['patent_portfolio'][:10]),  # First 10 patents
            'xrd': row.get('xrd', ''),
            'emp': row.get('emp', ''),
            'at': row.get('at', '')
        })

if summary_data:
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv('company_portfolio_summary.csv', index=False)
    print(f"Portfolio summary saved to: company_portfolio_summary.csv")
    print(f"Summary contains {len(df_summary)} company-year observations with patents")

# Additional analysis: R&D spending vs portfolio size
print(f"\n=== R&D ANALYSIS ===")
if 'xrd' in df_company_final.columns:
    df_company_final['xrd_numeric'] = pd.to_numeric(df_company_final['xrd'], errors='coerce')
    df_company_final['has_rd'] = df_company_final['xrd_numeric'] > 0
    
    rd_stats = df_company_final.groupby('has_rd').agg({
        'patent_portfolio': lambda x: x.apply(len).mean(),
        'conm': 'count'
    }).round(2)
    rd_stats.columns = ['avg_portfolio_size', 'count']
    print("R&D spending vs portfolio size:")
    print(rd_stats.to_string())

# Industry analysis
print(f"\n=== INDUSTRY ANALYSIS ===")
if 'sich' in df_company_final.columns:
    # Get top industries by number of companies
    industry_stats = df_company_final.groupby('sich').agg({
        'patent_portfolio': lambda x: x.apply(len).mean(),
        'conm': 'count'
    }).round(2)
    industry_stats.columns = ['avg_portfolio_size', 'count']
    industry_stats = industry_stats.sort_values('count', ascending=False)
    
    print("Top 10 industries by number of companies:")
    print(industry_stats.head(10).to_string())

print("\n=== PROCESS COMPLETED SUCCESSFULLY ===")
print("Files created:")
print(f"1. {OUTPUT_PATH} - Main dataset with patent portfolios")
print(f"2. company_portfolio_summary.csv - Summary of companies with patents")
print(f"3. company_assignee_matching_for_review.csv - Matching results for review")