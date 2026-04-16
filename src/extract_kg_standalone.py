"""
Standalone KG extraction script — runs triple extraction on a dataset corpus
using an LLM server (Qwen3.5-35B or others) and saves results.

Usage:
    conda run -n vllm-qwen3.5 python src/extract_kg_standalone.py \
        --dataset 2wikimultihopqa --model_name qwen3.5-35b

Custom full-corpus usage:
    conda run -n vllm-qwen3.5 python src/extract_kg_standalone.py \
        --dataset 2wikimultihopqa \
        --model_name qwen3.5-35b \
        --corpus_path fullset/data/2wikimultihopqa_dev_corpus.json \
        --output_dir fullset/kg/2wikimultihopqa_dev \
        --output_name triple_extraction_35B.npy
"""
import argparse
import ast
import os
import sys
import time
from typing import List

import numpy as np
import pandas as pd
from json_repair import repair_json
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.async_llm import infer_prompts
from utils.triple_prompt import generate_triple_extraction_prompt, parse_triple_response

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model registry. `path` is only used as a hint for `--model` if you prefer
# passing a full HF path; vLLM/SGLang servers look up by `served_name`.
# Override with env var MODEL_PATH_ROOT=/path/to/your/models (default ./models).
_MODEL_ROOT = os.environ.get("MODEL_PATH_ROOT", os.path.join(REPO_ROOT, "models"))
MODEL_CONFIG = {
    "qwen3-4b": {
        "path": os.path.join(_MODEL_ROOT, "Qwen3-4B-Instruct-2507"),
        "served_name": "qwen3-4b",
        "size": "4B",
    },
    "qwen3-30b": {
        "path": os.path.join(_MODEL_ROOT, "Qwen3-30B-A3B-Instruct-2507-FP8"),
        "served_name": "qwen3-30b",
        "size": "30B",
    },
    "qwen3.5-35b": {
        "path": os.path.join(_MODEL_ROOT, "Qwen3.5-35B-A3B-FP8"),
        "served_name": "qwen3.5-35b",
        "size": "35B",
    },
    "qwen3.5-4b": {
        "path": os.path.join(_MODEL_ROOT, "Qwen3.5-4B"),
        "served_name": "qwen3.5-4b",
        "size": "4B",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Standalone KG Triple Extraction")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["2wikimultihopqa", "hotpotqa", "musique"])
    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Concurrent LLM requests (35B on 4090x2 should be low)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (default: REPO_ROOT/tmp_data/{dataset}/)")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output filename (default: triple_extraction_{size}.npy)")
    parser.add_argument("--corpus_path", type=str, default=None,
                        help="Override corpus JSON path (default: data/{dataset}_corpus.json)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start offset in corpus for sharded extraction")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional number of docs to extract from --start")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. Default 0.0 for reproducible extraction.")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Sampling top_p. Default 1.0 for reproducible extraction.")
    args = parser.parse_args()

    cfg = MODEL_CONFIG[args.model_name]
    model_size = cfg["size"]
    served_name = cfg["served_name"]
    dataset = args.dataset

    # Setup logging
    log_dir = os.path.join(REPO_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(f"{log_dir}/kg_extract_{dataset}_{model_size}.log", rotation="500 MB", level="INFO", mode="w")

    # Output path
    output_dir = args.output_dir or os.path.join(REPO_ROOT, "tmp_data", dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_name = args.output_name or f"triple_extraction_{model_size}.npy"
    output_path = os.path.join(output_dir, output_name)

    logger.info(f"KG Extraction: dataset={dataset}, model={args.model_name} ({model_size})")
    logger.info(f"Server: localhost:{args.port}, served_name={served_name}")
    logger.info(f"Output: {output_path}")

    # Load corpus
    corpus_path = args.corpus_path or os.path.join(REPO_ROOT, "data", f"{dataset}_corpus.json")
    corpus = pd.read_json(corpus_path)
    total_docs = len(corpus)
    if "title" not in corpus.columns or "text" not in corpus.columns:
        raise ValueError(f"Corpus at {corpus_path} must contain 'title' and 'text' columns")

    if "idx" in corpus.columns:
        doc_ids: List[int] = [int(v) for v in corpus["idx"].tolist()]
    else:
        doc_ids = [int(i) for i in corpus.index.tolist()]

    start = max(args.start, 0)
    end = len(corpus) if args.limit is None else min(len(corpus), start + args.limit)
    corpus = corpus.iloc[start:end].reset_index(drop=True)
    doc_ids = doc_ids[start:end]

    logger.info(f"Loaded corpus: total={total_docs}, slice=[{start}:{end}] => {len(corpus)} documents")
    logger.info(
        f"Extraction sampling: temperature={args.temperature}, top_p={args.top_p}, "
        f"concurrency={args.concurrency}"
    )

    # Generate prompts
    prompt_list = [
        generate_triple_extraction_prompt(
            title=corpus.iloc[i]["title"],
            text=corpus.iloc[i]["text"]
        )
        for i in tqdm(range(len(corpus)), desc="Generating prompts")
    ]
    logger.info(f"Generated {len(prompt_list)} extraction prompts")

    # Run inference
    t0 = time.time()
    qa_final = infer_prompts(
        prompt_list,
        llm_path=cfg["path"],
        llm_name=served_name,
        api_base=f"http://127.0.0.1:{args.port}/v1",
        api_key="api-key-qwen3",
        temperature=args.temperature,
        top_p=args.top_p,
        sample_num=1,
        concurrency=args.concurrency,
    )
    t1 = time.time()
    logger.info(f"Inference done in {t1-t0:.1f}s, tokens: {qa_final[2]}")

    # Parse results
    return_str_list = {}
    parse_errors = 0
    for i in tqdm(range(len(corpus)), desc="Parsing triples"):
        doc_id = int(doc_ids[i])
        try:
            return_str = eval(qa_final[0][i][0])
        except Exception:
            try:
                return_str = ast.literal_eval(repair_json(qa_final[0][i][0]))
            except Exception:
                return_str = parse_triple_response(qa_final[0][i][0])
        try:
            return_str_list[doc_id] = return_str["triplets"]
            assert isinstance(return_str, dict)
        except Exception:
            if isinstance(return_str, list):
                return_str_list[doc_id] = return_str
            else:
                return_str_list[doc_id] = []
                parse_errors += 1

    logger.info(f"Parsed {len(return_str_list)} documents, {parse_errors} parse errors")

    # Save
    np.save(output_path, return_str_list)
    logger.info(f"Saved to {output_path}")

    # Summary
    total_triples = sum(len(v) for v in return_str_list.values() if isinstance(v, list))
    logger.info(f"Total triples extracted: {total_triples}")
    logger.info(f"Avg triples per doc: {total_triples / len(corpus):.1f}")


if __name__ == "__main__":
    main()
