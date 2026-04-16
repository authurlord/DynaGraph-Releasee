"""
Extract and persist all embeddings from Phase 1 pipeline.

Skips: KG extraction, QA generation.
Reads: existing rewrite_query_4B.npy, retrieve_result_4B_*.npy.
Requires: embedding server on port 8001 (Qwen3-Embedding-0.6B).

Output per dataset in tmp_data/{dataset}/embeddings/:
  corpus_1hop.npy          (N_corpus, 1024)   — doc-level 1-hop embeddings
  corpus_1hop_index.json   [{doc_id, title}]
  corpus_2hop.npy          (N_q * top_k, 1024) — augmented query embeddings (corpus level)
  corpus_2hop_index.json   [{qid, source_doc_id, title, augmented_query_preview}]
  chunk_1hop.npy           (N_chunks, 512)    — chunk-level 1-hop embeddings
  chunk_1hop_index.json    [{chunk_id, doc_id, title}]
  chunk_2hop.npy           (N_q * top_k, 512) — augmented query embeddings (chunk level)
  chunk_2hop_index.json    [{qid, source_chunk_id, doc_id, title}]

Usage:
  conda run -n vllm-qwen3.5 python3 src/extract_embeddings.py --dataset 2wikimultihopqa
  conda run -n vllm-qwen3.5 python3 src/extract_embeddings.py --all
"""
import argparse
import json
import os
import sys
import time

import httpx
import numpy as np
import pandas as pd
import semchunk
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

# ============================================================
# Config
# ============================================================
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_API_KEY = "embed-key-m3"
EMBED_API_BASE = "http://localhost:8001/v1"

# Same as pipeline_baseline.py
LLM_PATH = os.environ.get("TOKENIZER_PATH", "Qwen/Qwen3.5-4B")  # only for tokenizer (chunking); override with env var
CHUNK_SIZE = 128
CHUNK_OVERLAP = 0.5

SOURCE_EXP_IDS = {
    "2wikimultihopqa": "baseline_v3",
    "hotpotqa": "baseline_full_1000q",
    "musique": "1",
}
TOP_K_HOP_1 = 5  # same across all datasets

BATCH_SIZE = 256  # embedding batch size (tune for GPU memory)


# ============================================================
# Helpers
# ============================================================
def get_client():
    return OpenAI(
        api_key=EMBED_API_KEY,
        base_url=EMBED_API_BASE,
        http_client=httpx.Client(trust_env=False),
    )


def batch_embed(client, texts, dimensions=None, desc="Embedding"):
    """Embed texts in batches, return (N, dim) float32 array."""
    all_embeddings = []
    model_name = client.models.list().data[0].id

    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch = texts[start : start + BATCH_SIZE]
        kwargs = {"input": batch, "model": model_name}
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        try:
            resp = client.embeddings.create(**kwargs)
        except Exception as e:
            if "matryoshka" in str(e).lower() or "dimensions" in str(e).lower():
                kwargs.pop("dimensions", None)
                resp = client.embeddings.create(**kwargs)
            else:
                raise

        # Reconstruct in correct order
        embs = [None] * len(batch)
        for d in resp.data:
            embs[d.index] = d.embedding
        all_embeddings.extend(embs)

    arr = np.array(all_embeddings, dtype=np.float32)
    return arr


def save_npy_and_json(out_dir, name, array, index_list):
    np.save(os.path.join(out_dir, f"{name}.npy"), array)
    with open(os.path.join(out_dir, f"{name}_index.json"), "w") as f:
        json.dump(index_list, f, ensure_ascii=False, indent=1)
    print(f"  Saved {name}.npy  shape={array.shape}  ({array.nbytes / 1e6:.1f} MB)")
    print(f"  Saved {name}_index.json  entries={len(index_list)}")


# ============================================================
# Main extraction for one dataset
# ============================================================
def extract_dataset(dataset_name, client):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")
    t0 = time.time()

    # --- Load data ---
    corpus_path = os.path.join(REPO_ROOT, "data", f"{dataset_name}_corpus.json")
    data_path = os.path.join(REPO_ROOT, "data", f"{dataset_name}.json")
    corpus_df = pd.read_json(corpus_path)
    data_df = pd.read_json(data_path)
    n_corpus = len(corpus_df)
    n_queries = min(1000, len(data_df))
    print(f"  Corpus docs: {n_corpus}, Queries: {n_queries}")

    src_id = SOURCE_EXP_IDS[dataset_name]
    rr_path = os.path.join(REPO_ROOT, "tmp_data", dataset_name,
                           f"retrieve_result_4B_{src_id}.npy")
    rr = np.load(rr_path, allow_pickle=True).item()

    rw_path = os.path.join(REPO_ROOT, "tmp_data", dataset_name, "rewrite_query_4B.npy")
    rewrite_query = np.load(rw_path, allow_pickle=True).item()
    print(f"  Loaded retrieve_result and rewrite_query")

    # Output directory
    out_dir = os.path.join(REPO_ROOT, "tmp_data", dataset_name, "embeddings")
    os.makedirs(out_dir, exist_ok=True)

    # ============================================================
    # 1. CORPUS 1-HOP EMBEDDINGS (1024-dim)
    # ============================================================
    print(f"\n--- Corpus 1-hop embeddings (1024-dim) ---")
    corpus_texts = [f"Title: {row['title']}\n{row['text']}"
                    for _, row in corpus_df.iterrows()]
    corpus_1hop = batch_embed(client, corpus_texts, dimensions=None,
                              desc=f"[{dataset_name}] corpus 1-hop")
    corpus_1hop_index = [
        {"doc_id": i, "title": corpus_df.iloc[i]["title"]}
        for i in range(n_corpus)
    ]
    save_npy_and_json(out_dir, "corpus_1hop", corpus_1hop, corpus_1hop_index)

    # ============================================================
    # 2. CORPUS 2-HOP EMBEDDINGS (1024-dim)
    #    "Query:{rewrite}\nBackground Information:\n{doc_text}"
    #    for each query × top-5 1-hop docs (from FAISS, not reranker)
    # ============================================================
    print(f"\n--- Corpus 2-hop embeddings (1024-dim) ---")

    # We need 1-hop FAISS indices. Recompute from corpus embeddings.
    from utils.retrieve import faiss_cosine_search_cpu

    questions = data_df["question"].to_list()[:n_queries]
    q_emb = batch_embed(client, questions, dimensions=None,
                        desc=f"[{dataset_name}] query embed")
    # FAISS search for 1-hop top-5
    indices_1hop, _ = faiss_cosine_search_cpu(q_emb, corpus_1hop, TOP_K_HOP_1)

    augmented_corpus_queries = []
    corpus_2hop_index = []
    for qid in range(n_queries):
        rw_q = rewrite_query.get(qid, questions[qid])
        hop1_docs = indices_1hop[qid][:TOP_K_HOP_1]
        for doc_id in hop1_docs:
            doc_id = int(doc_id)
            context = corpus_df.iloc[doc_id]["text"]
            aug_q = f"Query:{rw_q}\nBackground Information:\n{context}"
            augmented_corpus_queries.append(aug_q)
            corpus_2hop_index.append({
                "qid": qid,
                "source_doc_id": doc_id,
                "title": corpus_df.iloc[doc_id]["title"],
                "augmented_query_preview": aug_q[:120],
            })

    corpus_2hop = batch_embed(client, augmented_corpus_queries, dimensions=None,
                              desc=f"[{dataset_name}] corpus 2-hop")
    save_npy_and_json(out_dir, "corpus_2hop", corpus_2hop, corpus_2hop_index)

    # ============================================================
    # 3. CHUNK 1-HOP EMBEDDINGS (512-dim)
    # ============================================================
    print(f"\n--- Chunking corpus (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ---")
    chunker = semchunk.chunkerify(AutoTokenizer.from_pretrained(LLM_PATH), CHUNK_SIZE)
    all_chunks = []  # [(title, chunk_text, doc_id)]
    for i in tqdm(range(n_corpus), desc=f"[{dataset_name}] chunking"):
        text = corpus_df.iloc[i]["text"]
        title = corpus_df.iloc[i]["title"]
        chunks = chunker(text, offsets=False, overlap=CHUNK_OVERLAP)
        for chunk in chunks:
            all_chunks.append((title, chunk, i))

    chunk_df = pd.DataFrame(all_chunks, columns=["title", "text", "doc_id"])
    n_chunks = len(chunk_df)
    print(f"  Total chunks: {n_chunks} (avg {n_chunks/n_corpus:.1f} per doc)")

    chunk_texts = [f"Title: {row['title']}\n{row['text']}"
                   for _, row in chunk_df.iterrows()]
    chunk_1hop = batch_embed(client, chunk_texts, dimensions=512,
                             desc=f"[{dataset_name}] chunk 1-hop")
    chunk_1hop_index = [
        {"chunk_id": i, "doc_id": int(chunk_df.iloc[i]["doc_id"]),
         "title": chunk_df.iloc[i]["title"]}
        for i in range(n_chunks)
    ]
    save_npy_and_json(out_dir, "chunk_1hop", chunk_1hop, chunk_1hop_index)

    # ============================================================
    # 4. CHUNK 2-HOP EMBEDDINGS (512-dim)
    #    "Query:{rewrite}\nBackground Information:\n{chunk_text}"
    #    for each query × top-5 1-hop chunks (from FAISS chunk search)
    # ============================================================
    print(f"\n--- Chunk 2-hop embeddings (512-dim) ---")

    q_emb_512 = batch_embed(client, questions, dimensions=512,
                            desc=f"[{dataset_name}] query embed 512")
    indices_1hop_chunk, _ = faiss_cosine_search_cpu(q_emb_512, chunk_1hop, TOP_K_HOP_1)

    augmented_chunk_queries = []
    chunk_2hop_index = []
    for qid in range(n_queries):
        rw_q = rewrite_query.get(qid, questions[qid])
        hop1_chunks = indices_1hop_chunk[qid][:TOP_K_HOP_1]
        for chunk_id in hop1_chunks:
            chunk_id = int(chunk_id)
            context = chunk_df.iloc[chunk_id]["text"]
            aug_q = f"Query:{rw_q}\nBackground Information:\n{context}"
            augmented_chunk_queries.append(aug_q)
            chunk_2hop_index.append({
                "qid": qid,
                "source_chunk_id": chunk_id,
                "doc_id": int(chunk_df.iloc[chunk_id]["doc_id"]),
                "title": chunk_df.iloc[chunk_id]["title"],
                "augmented_query_preview": aug_q[:120],
            })

    chunk_2hop = batch_embed(client, augmented_chunk_queries, dimensions=512,
                             desc=f"[{dataset_name}] chunk 2-hop")
    save_npy_and_json(out_dir, "chunk_2hop", chunk_2hop, chunk_2hop_index)

    # ============================================================
    # Also save query embeddings for convenience
    # ============================================================
    print(f"\n--- Query embeddings ---")
    q_index = [{"qid": i, "question": questions[i][:100]} for i in range(n_queries)]
    save_npy_and_json(out_dir, "query_1024", q_emb, q_index)
    save_npy_and_json(out_dir, "query_512", q_emb_512, q_index)

    elapsed = time.time() - t0
    print(f"\n  {dataset_name} DONE in {elapsed:.0f}s")
    print(f"  Output dir: {out_dir}")

    # Summary
    summary = {
        "dataset": dataset_name,
        "n_corpus": n_corpus,
        "n_chunks": n_chunks,
        "n_queries": n_queries,
        "top_k_hop_1": TOP_K_HOP_1,
        "corpus_1hop_shape": list(corpus_1hop.shape),
        "corpus_2hop_shape": list(corpus_2hop.shape),
        "chunk_1hop_shape": list(chunk_1hop.shape),
        "chunk_2hop_shape": list(chunk_2hop.shape),
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Extract & persist Phase 1 embeddings")
    parser.add_argument("--dataset", type=str,
                        choices=["2wikimultihopqa", "hotpotqa", "musique"],
                        default=None, help="Single dataset")
    parser.add_argument("--all", action="store_true", help="Run all 3 datasets")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Embedding batch size")
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    datasets = ["2wikimultihopqa", "hotpotqa", "musique"] if args.all else [args.dataset]
    if not args.all and args.dataset is None:
        parser.error("Specify --dataset or --all")

    # Test connection
    client = get_client()
    try:
        models = client.models.list()
        print(f"Embedding server OK: {models.data[0].id}")
    except Exception as e:
        print(f"ERROR: Cannot connect to embedding server at {EMBED_API_BASE}")
        print(f"  Start it with: bash script/embedding_server_pipeline.sh")
        print(f"  Error: {e}")
        sys.exit(1)

    results = []
    for ds in datasets:
        summary = extract_dataset(ds, client)
        results.append(summary)

    print(f"\n{'='*60}")
    print(f"  ALL DONE")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['dataset']:20s}: corpus={r['corpus_1hop_shape']}, "
              f"chunks={r['chunk_1hop_shape']}, "
              f"2hop_corpus={r['corpus_2hop_shape']}, "
              f"2hop_chunk={r['chunk_2hop_shape']}, "
              f"time={r['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
