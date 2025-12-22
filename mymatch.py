import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import re

# Configuration
MODEL_NAME = 'all-mpnet-base-v2'  # Embedding model
NUM_WORKERS = 30                   # Parallel encoding processes
REGEX_THRESHOLD = 0.95            # Cosine similarity cutoff for regex-based matching
N_NEIGHBORS = 100                 # Number of neighbors to consider in matching

# File paths
DISCERN_COMPUSTAT_PATH = '/home/dariomarino/Thesis/discern_compustat.csv'
COMPANY_DATA_PATH       = '/home/dariomarino/Thesis/companydata.csv'
PAT_PATH                = '/home/dariomarino/single_patents_with_top.csv'

# Stop words to remove (complete words only)
STOP_WORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'company', 'co', 'ltd', 'limited',
    'llc', 'lp', 'plc', 'sa', 'nv', 'bv', 'gmbh', 'ag', 'spa', 'srl', 'pty',
    'the', 'and', 'of', 'group', 'holdings', 'holding', 'international', 'intl',
    'aktiengesellschaft'
}

def clean_company_name(name: str) -> str:
    """
    Clean company or assignee names by removing punctuation and defined stop words.
    Does NOT remove single letters or numbers.
    """
    if not name or pd.isna(name):
        return ''
    # Lowercase and strip whitespace
    name = str(name).lower().strip()
    # Remove punctuation (retain letters, numbers, and whitespace)
    name = re.sub(r'[^\w\s]', ' ', name)
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    # Remove stop words (full-word matches)
    for sw in STOP_WORDS:
        pattern = r"\b" + re.escape(sw) + r"\b"
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    # Collapse spaces again after removals
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def preprocess_names(names: list, label: str):
    """
    Preprocess a list of names (companies or assignees) using regex cleaning.
    Returns cleaned list and mapping to originals.
    """
    cleaned = []
    mapping = {}
    for orig in names:
        cleaned_name = clean_company_name(orig)
        if cleaned_name:
            cleaned.append(cleaned_name)
            mapping[cleaned_name] = orig
    return cleaned, mapping


def create_matches_dataframe(companies, assignees, comp_emb, asg_emb, threshold, comp_map, asg_map):
    """
    Match each company embedding to nearest assignee embeddings above a similarity threshold.
    """
    nn = NearestNeighbors(
        n_neighbors=min(N_NEIGHBORS, len(assignees)),
        metric='cosine', algorithm='brute', n_jobs=NUM_WORKERS
    ).fit(asg_emb)

    distances, indices = nn.kneighbors(comp_emb)
    sims = 1 - distances

    results = []
    for i, comp in enumerate(companies):
        orig_comp = comp_map.get(comp, comp)
        matched = False
        for idx, sim in zip(indices[i], sims[i]):
            if sim >= threshold:
                asg = assignees[idx]
                orig_asg = asg_map.get(asg, asg)
                results.append({
                    'company_original': orig_comp,
                    'company_processed': comp,
                    'matched_assignee_original': orig_asg,
                    'matched_assignee_processed': asg,
                    'similarity_score': sim,
                    'valid_match': True,
                    'notes': ''
                })
                matched = True
                break
        if not matched:
            results.append({
                'company_original': orig_comp,
                'company_processed': comp,
                'matched_assignee_original': '',
                'matched_assignee_processed': '',
                'similarity_score': 0.0,
                'valid_match': False,
                'notes': 'No match above threshold'
            })

    df = pd.DataFrame(results).sort_values('company_original')
    return df

# Verify inputs exist
for p in (DISCERN_COMPUSTAT_PATH, COMPANY_DATA_PATH, PAT_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing file: {p}")

# Load data
import pandas as pd

df_discern = pd.read_csv(DISCERN_COMPUSTAT_PATH, dtype=str, low_memory=False)

df_comp = pd.read_csv(COMPANY_DATA_PATH, dtype=str, low_memory=False)
df_pat = pd.read_csv(PAT_PATH, dtype=str, low_memory=False)

companies_orig = df_comp['conm'].dropna().str.strip().unique().tolist()
assignees_orig = df_pat['disambig_assignee_organization'].dropna().str.strip().unique().tolist()

# Encode and match
model = SentenceTransformer(MODEL_NAME)

comp_clean, comp_map = preprocess_names(companies_orig, 'company')
asg_clean, asg_map = preprocess_names(assignees_orig, 'assignee')

comp_emb = normalize(
    model.encode(comp_clean, convert_to_numpy=True, show_progress_bar=True,
                 batch_size=64, num_workers=NUM_WORKERS),
    norm='l2', axis=1
)
asg_emb = normalize(
    model.encode(asg_clean, convert_to_numpy=True, show_progress_bar=True,
                 batch_size=64, num_workers=NUM_WORKERS),
    norm='l2', axis=1
)

df_matches = create_matches_dataframe(
    comp_clean, asg_clean, comp_emb, asg_emb,
    REGEX_THRESHOLD, comp_map, asg_map
)

# Save valid matches
output_file = 'company_assignee_matching_for_review.csv'
df_valid = df_matches[df_matches['valid_match']]
df_valid.to_csv(output_file, index=False)
print(f"Saved {len(df_valid)} matches to {output_file}")
