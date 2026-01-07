import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Stop words to remove (complete words only)
STOP_WORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'company', 'co', 'ltd', 'limited',
    'llc', 'lp', 'plc', 'sa', 'nv', 'bv', 'gmbh', 'ag', 'spa', 'srl', 'pty',
    'the', 'and', 'of', 'group', 'holdings', 'holding', 'international', 'intl',
    'aktiengesellschaft', 'llp', 'agency', 'association', 'enterprise', 'enterprises',
    'partnership', 'partners', 'solutions', 'services', 'system', 'systems', 'tech',
    'technology', 'technologies', 'industries', 'industry', 'global', 'worldwide',
    'usa', 'us', 'americas', 'na', 'north', 'america'
}

def clean_company_name(name):
    """Clean company name by removing stop words and punctuation"""
    if pd.isna(name):
        return ""
    
    # Convert to lowercase
    name = str(name).lower()
    
    # Remove punctuation
    name = name.translate(str.maketrans('', '', string.punctuation))
    
    # Split into words
    words = name.split()
    
    # Remove stop words (complete words only)
    cleaned_words = [word for word in words if word not in STOP_WORDS]
    
    # Join back
    return ' '.join(cleaned_words)

def main():
    print("Loading datasets...")
    
    # Load datasets
    companydata = pd.read_csv('/home/dariomarino/Thesis/companydata.csv')
    product_similarities = pd.read_csv('/home/dariomarino/Thesis/product_similarities_with_patents.csv')
    
    print(f"Company data shape: {companydata.shape}")
    print(f"Product similarities shape: {product_similarities.shape}")
    
    # Get unique company names from companydata (conm + gvkey)
    company_names_df = companydata[['conm', 'gvkey']].drop_duplicates()
    print(f"Unique companies in companydata: {len(company_names_df)}")
    
    # Get unique manufacturer names from both manufacturer_1 and manufacturer_2
    manufacturer_1_names = product_similarities['manufacturer_1'].dropna().unique()
    manufacturer_2_names = product_similarities['manufacturer_2'].dropna().unique()
    
    # Combine and get unique manufacturer names
    all_manufacturers = pd.Series(np.concatenate([manufacturer_1_names, manufacturer_2_names])).unique()
    print(f"Unique manufacturers: {len(all_manufacturers)}")
    
    # Clean company names
    print("Cleaning company names...")
    company_names_df['conm_cleaned'] = company_names_df['conm'].apply(clean_company_name)
    
    # Clean manufacturer names
    manufacturers_df = pd.DataFrame({'manufacturer': all_manufacturers})
    manufacturers_df['manufacturer_cleaned'] = manufacturers_df['manufacturer'].apply(clean_company_name)
    
    # Remove empty cleaned names
    company_names_df = company_names_df[company_names_df['conm_cleaned'] != '']
    manufacturers_df = manufacturers_df[manufacturers_df['manufacturer_cleaned'] != '']
    
    print(f"After cleaning - Companies: {len(company_names_df)}, Manufacturers: {len(manufacturers_df)}")
    
    # Load sentence transformer model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Create embeddings
    print("Creating embeddings for company names...")
    company_embeddings = model.encode(company_names_df['conm_cleaned'].tolist())
    
    print("Creating embeddings for manufacturer names...")
    manufacturer_embeddings = model.encode(manufacturers_df['manufacturer_cleaned'].tolist())
    
    # Calculate cosine similarities and find best matches
    print("Finding best matches...")
    similarity_matrix = cosine_similarity(manufacturer_embeddings, company_embeddings)
    
    # Create matching table
    matching_results = []
    
    for i, manufacturer in enumerate(manufacturers_df['manufacturer']):
        # Find the index of the best match
        best_match_idx = np.argmax(similarity_matrix[i])
        best_similarity = similarity_matrix[i][best_match_idx]
        
        # Get the corresponding company info
        best_match_company = company_names_df.iloc[best_match_idx]
        
        matching_results.append({
            'manufacturer': manufacturer,
            'manufacturer_cleaned': manufacturers_df.iloc[i]['manufacturer_cleaned'],
            'conm': best_match_company['conm'],
            'conm_cleaned': best_match_company['conm_cleaned'],
            'gvkey': best_match_company['gvkey'],
            'cosine_similarity': best_similarity,
            'valid': True
        })
    
    # Create matching table DataFrame
    matching_table = pd.DataFrame(matching_results)
    
    # Sort by cosine similarity (descending) to see best matches first
    matching_table = matching_table.sort_values('cosine_similarity', ascending=False)
    
    # Save matching table
    output_file = '/home/dariomarino/Thesis/company_matching_table.csv'
    matching_table.to_csv(output_file, index=False)
    
    print(f"\nMatching table saved to: {output_file}")
    print(f"Total matches: {len(matching_table)}")
    print(f"Average cosine similarity: {matching_table['cosine_similarity'].mean():.3f}")
    print(f"Min cosine similarity: {matching_table['cosine_similarity'].min():.3f}")
    print(f"Max cosine similarity: {matching_table['cosine_similarity'].max():.3f}")
    
    print("\nSample of best matches:")
    print(matching_table[['manufacturer', 'conm', 'cosine_similarity']].head(10))
    
    print("\nSample of worst matches:")
    print(matching_table[['manufacturer', 'conm', 'cosine_similarity']].tail(10))
    
    print(f"\nPlease review and edit the 'valid' column in {output_file}")
    print("Set 'valid' to False for incorrect matches, then run the second script.")

if __name__ == "__main__":
    main()