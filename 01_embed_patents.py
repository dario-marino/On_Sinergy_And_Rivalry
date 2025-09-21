#!/usr/bin/env python3
"""
Create SBERT embeddings for USPTO abstracts -> Parquet with columns:
    patent_id, embedding_0 ... embedding_767
Usage:
  python 01_embed_patents.py --input g_patent_abstract.tsv \
      --out USPTO_abstracts_embeddings.parquet \
      --model AAUBS/PatentSBERTa_V2 --batch-size 800 --sub-batch 100
"""
import os, gc, time, glob, argparse
import numpy as np, pandas as pd, polars as pl
import pyarrow as pa, pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TSV with columns [patent_id, patent_abstract]")
    ap.add_argument("--out", required=True, help="Output Parquet file")
    ap.add_argument("--tmpdir", default="tmp_embeddings", help="Temp dir for chunk files")
    ap.add_argument("--model", default="AAUBS/PatentSBERTa_V2")
    ap.add_argument("--batch-size", type=int, default=800)
    ap.add_argument("--sub-batch", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=200)
    return ap.parse_args()

def save_chunk(df: pl.DataFrame, idx: int, tmpdir: str) -> str:
    arr = np.vstack(df["embedding"].to_list())
    out = {"patent_id": df["patent_id"].to_list()}
    for j in range(arr.shape[1]): out[f"embedding_{j}"] = arr[:, j].tolist()
    tbl = pa.Table.from_pandas(pl.DataFrame(out).to_pandas(), preserve_index=False)
    fn = os.path.join(tmpdir, f"chunk_{idx:04d}.parquet")
    pq.write_table(tbl, fn)
    return fn

def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(a.tmpdir, exist_ok=True)

    try:
        df = pl.read_csv(a.input, separator="\t", schema_overrides={"patent_id": pl.Utf8})
    except Exception:
        df = pl.from_pandas(pd.read_csv(a.input, sep="\t", dtype={"patent_id": str}))
    df = df.filter(pl.col("patent_abstract").is_not_null() & (pl.col("patent_abstract").str.strip_chars() != ""))

    model = SentenceTransformer(a.model); model.eval()

    batches=[]; B=a.batch_size
    for s in range(0, len(df), B):
        sub=df.slice(s, min(B, len(df)-s))
        batches.append((list(range(s, s+len(sub))), sub["patent_id"].to_list(), sub["patent_abstract"].to_list()))

    chunk_idx=0
    for s in range(0, len(batches), a.save_every):
        e=min(s+a.save_every, len(batches)); work=batches[s:e]
        out_rows=[]
        for idxs, pids, texts in tqdm(work, desc=f"Chunk {chunk_idx+1}"):
            embs=[]
            for i in range(0, len(texts), a.sub_batch):
                embs.extend(model.encode(texts[i:i+a.sub_batch], convert_to_tensor=False, show_progress_bar=False, batch_size=a.sub_batch))
            out_rows.extend([{"index": i, "patent_id": pid, "embedding": emb} for i, pid, emb in zip(idxs, pids, embs)])
        chunk=pl.DataFrame(out_rows).sort("index")
        save_chunk(chunk, chunk_idx, a.tmpdir)
        del chunk, out_rows; gc.collect(); chunk_idx+=1

    all_parts=[pl.read_parquet(fn) for fn in sorted(glob.glob(os.path.join(a.tmpdir, "chunk_*.parquet")))]
    final_df=pl.concat(all_parts)
    pq.write_table(pa.Table.from_pandas(final_df.to_pandas(), preserve_index=False), a.out)
    print(f"Done. Wrote {len(final_df):,} embeddings to {a.out}")

if __name__ == "__main__":
    main()
