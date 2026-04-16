"""
Phase 2 Standalone Evaluation Script.

Provides a self-contained evaluation harness for the Adaptive Aggregation module (Phase 2).
Coauthor can modify adaptive_aggregator.py (select_docs / DifficultyScorer) and re-run
this script to measure the impact on:

  1. Retrieval quality  — NDCG@K (no LLM required)
  2. QA quality         — F1 / ACC (requires running LLM server)
  3. Serving latency    — per-query TTFT, p50/p99 TTFT, SLO attainment (streaming)

Usage:
  # Offline only (NDCG, no LLM needed):
  python eval_phase2_standalone.py --dataset 2wikimultihopqa --questions 100 --offline_only

  # Full evaluation (NDCG + F1/ACC + TTFT):
  python eval_phase2_standalone.py --dataset 2wikimultihopqa --questions 100 \
      --host localhost --port 8000

  # Cross-dataset sweep:
  for ds in 2wikimultihopqa hotpotqa musique; do
      python eval_phase2_standalone.py --dataset $ds --questions 1000 --exp_id P2_v1_$ds
  done

Inputs (all pre-computed, no additional servers needed for NDCG):
  - retrieve_result_4B_{source_exp_id}.npy  (Phase 1 retrieval output)
  - graph_features_1000q.json               (Phase 1 topological features)
  - {dataset}_corpus.json                   (corpus)
  - {dataset}.json                          (queries + ground-truth answers)

Outputs:
  - JSON report: {output_dir}/{exp_id}_phase2_report.json
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from adaptive_aggregator import AdaptiveAggregator, AggregationConfig, EnhancedDifficultyScorer
from utils.evaluate import calculate_metric_scores


# ═══════════════════════════════════════════════════════════════════
#  NDCG computation (pure retrieval quality, no LLM needed)
# ═══════════════════════════════════════════════════════════════════

def _dcg(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def ndcg_at_k(relevances: List[float], k: int) -> float:
    """Normalized DCG@K. relevances[i] = relevance of doc at rank i."""
    dcg = _dcg(relevances, k)
    ideal = _dcg(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def compute_ndcg_for_query(
    selected_doc_indices: List[int],
    gold_doc_indices: List[int],
    k: int,
) -> float:
    """Compute NDCG@K for a single query.

    Args:
        selected_doc_indices: doc indices chosen by the aggregator (ordered)
        gold_doc_indices: ground-truth relevant doc indices
        k: cutoff rank
    """
    gold_set = set(gold_doc_indices)
    relevances = [1.0 if idx in gold_set else 0.0 for idx in selected_doc_indices]
    return ndcg_at_k(relevances, k)


def get_gold_doc_indices(
    qid: int,
    data_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    dataset: str = "2wikimultihopqa",
) -> List[int]:
    """Extract ground-truth relevant document indices for a query.

    Uses the 'supporting_facts' field (2wiki, hotpotqa) or 'paragraphs'
    field with is_supporting flag (musique).
    """
    row = data_df.iloc[qid]

    # MuSiQue: uses 'paragraphs' with 'is_supporting' flag
    if dataset == "musique":
        paragraphs = row.get("paragraphs", [])
        if not paragraphs:
            return []
        gold = []
        for p in paragraphs:
            if p.get("is_supporting"):
                title = p.get("title", "")
                text = p.get("paragraph_text", "")
                matches = corpus_df[(corpus_df["title"] == title) & (corpus_df["text"] == text)]
                if len(matches) > 0:
                    gold.append(matches.index[0])
        return gold

    # 2Wiki / HotpotQA: supporting_facts field (list of [title, sent_idx])
    if "supporting_facts" in row and row["supporting_facts"]:
        sf = row["supporting_facts"]
        if isinstance(sf, list) and len(sf) > 0:
            if isinstance(sf, dict):
                titles = sf.get("title", [])
            elif isinstance(sf[0], (list, tuple)):
                titles = [item[0] for item in sf]
            else:
                titles = sf
            if titles:
                title_set = set(str(t) for t in titles)
                gold_indices = corpus_df.index[
                    corpus_df["title"].astype(str).isin(title_set)
                ].tolist()
                if gold_indices:
                    return gold_indices

    return []


# ═══════════════════════════════════════════════════════════════════
#  Prompt builders (faithful to KGQA format, same as benchmark_adaptive_agg.py)
# ═══════════════════════════════════════════════════════════════════

def _build_prompt_2wiki_hotpot(question: str, current_contexts: list) -> str:
    """Exact reproduction of KGQA row_to_prompt_cot_fixed_demos (for 2wiki & hotpotqa)."""
    fixed_demos_text = """
--- Demonstration 1 ---
Example Question: Are Quaqtaq Airport and Fort Nelson/Gordon Field Airport located in the same country?
Given Context: Title: Quaqtaq Airport
Content: "Quaqtaq Airport,, is located near Quaqtaq, Quebec, Canada."
Given Context: Title: Fort Nelson/Gordon Field Airport
Content: "Fort Nelson/ Gordon Field Airport,, is located near to Fort Nelson, British Columbia, Canada."

Step-by-Step Reasoning:
1. Quaqtaq Airport is located in Quebec, Canada.
2. Fort Nelson/Gordon Field Airport is located in British Columbia, Canada.
3. Both locations are in Canada.
ANSWER: yes

--- Demonstration 2 ---
Example Question: Who was born first out of Erick Chipeta and Praveen Agrawal?
Given Context: Title: Erick Chipeta
Content: "Erick Chipeta( born 28 June 1990) is a Zimbabwean professional footballer who plays as a defender and midfielder for the Zimbabwe national team."
Given Context: Title: Praveen Agrawal
Content: "Praveen Agrawal( born 20 November 1969) is an Indian politician and a member of the Bharatiya Janata Party."

Step-by-Step Reasoning:
1. Erick Chipeta was born on 28 June 1990.
2. Praveen Agrawal was born on 20 November 1969.
3. 1969 is earlier than 1990, so Praveen Agrawal was born first.
ANSWER: Praveen Agrawal

--- Demonstration 3 ---
Example Question: Who lived longer, Ludovico Sabbatini or Marcelino Manuel Da Gra\u00e7a?
Given Context: Title: Ludovico Sabbatini
Content: "Venerable Ludovico Sabbatini( 30 August 1650 \u2013 11 June 1724) was an Italian priest and religious educator, who was beatified by the Catholic Church in 1765."
Given Context: Title: Marcelino Manuel da Gra\u00e7a
Content: "Marcelino Manuel da Gra\u00e7a( January 25, 1881 or 1884 \u2014 January 12, 1960), better known as Charles Manuel\\" Sweet Daddy\\" Grace, or Daddy Grace, was the founder and first bishop of the predominantly African- American denomination the United House of Prayer For All People."

Step-by-Step Reasoning:
1. Ludovico Sabbatini lived from 1650 to 1724 (approx 74 years).
2. Marcelino Manuel da Gra\u00e7a lived from 1881/1884 to 1960 (approx 76-79 years).
3. Marcelino lived longer than Ludovico.
ANSWER: Marcelino Manuel Da Gra\u00e7a
"""
    parts = []
    parts.append("You are an expert multi-hop reasoning engine. Your goal is to answer complex questions by connecting facts from multiple provided context snippets.")
    parts.append("\n### INSTRUCTIONS:")
    parts.append("1. **Analyze the Question:** Identify the key entity and what relationship is being queried.")
    parts.append("2. **Search & Bridge:** Find the initial entity, identify the connecting link (bridge entity), and locate the final information.")
    parts.append("3. **Filter Noise:** Ignore context snippets that do not contribute to the logical chain.")
    parts.append("4. **Derive Answer:** Combine the facts to form the final answer.")
    parts.append("\n### RESPONSE FORMAT:")
    parts.append("- **Step-by-Step Reasoning:** Briefly explain the logical steps (e.g., 1. Found Entity X... 2. Linked to Y... 3. Concluded Z).")
    parts.append("- **ANSWER:** Output ONLY the final answer line starting with 'ANSWER: '.")
    parts.append("\nDEMONSTRATIONS:")
    parts.append(fixed_demos_text)
    parts.append("\nCURRENT TASK:")
    parts.append(f"Question: {question}")
    parts.append("Given Contexts:")
    for title, content in current_contexts:
        quoted_content = json.dumps(content, ensure_ascii=False)
        if title:
            parts.append(f"Title: {title}\nContent: {quoted_content}\n")
        else:
            parts.append(f"Content: {quoted_content}\n")
    parts.append("\n### YOUR OUTPUT:")
    parts.append("Step-by-Step Reasoning:")
    return "\n".join(parts)


def _build_prompt_musique(question: str, current_contexts: list) -> str:
    """Exact reproduction of KGQA row_to_prompt_cot_fixed_demos_musique."""
    fixed_demos_text = """
--- Demonstration 1 ---
Example Question: When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?

Given Contexts:
Title: FC Barcelona
Content: "Barcelona advanced to the semi-finals of the Copa del Rey, winning the first leg against Getafe 5\u20132, with a goal from Messi bringing comparison to Diego Maradona's goal of the century, but then lost the second leg 4\u20130."
Title: FC Barcelona
Content: "In June 1982, Diego Maradona was signed for a world record fee of \u00a35 million from Boca Juniors. In the following season, under coach Luis, Barcelona won the Copa del Rey..."

Step-by-Step Reasoning:
1. The question asks about the person whose goals Messi's Copa del Rey goals were compared to.
2. According to the context, Messi's goal brought comparison to **Diego Maradona** ("Diego Maradona's goal of the century").
3. The question then asks when this person (Diego Maradona) was signed by Barcelona.
4. The context states that **Diego Maradona** was signed in **June 1982**.
ANSWER: June 1982

--- Demonstration 2 ---
Example Question: What was the name of the star of Forced Vengeance in Walter, Texas Ranger?

Given Contexts:
Title: Forced Vengeance
Content: "Dragon's chief of security, Josh Randall (Chuck Norris) goes looking for the head of the syndicate to exact revenge for the murder of his employer, friend and mentor."
Title: Walker, Texas Ranger
Content: "Name Portrayed by ... Cordell Walker Chuck Norris Texas Ranger Main ..."

Step-by-Step Reasoning:
1. First, identify the star of the movie "Forced Vengeance". The context mentions **Chuck Norris** as the star (Josh Randall).
2. Next, find the name of the character played by **Chuck Norris** in the show "Walker, Texas Ranger".
3. The table/context for "Walker, Texas Ranger" lists **Chuck Norris** portraying **Cordell Walker**.
ANSWER: Cordell Walker
"""
    parts = []
    parts.append("You are an expert multi-hop reasoning engine. Your goal is to answer complex questions by connecting facts from multiple provided context snippets.")
    parts.append("\n### INSTRUCTIONS:")
    parts.append("1. **Analyze the Question:** Identify the key entity and what relationship is being queried.")
    parts.append("2. **Search & Bridge:** Find the initial entity, identify the connecting link (bridge entity), and locate the final information.")
    parts.append("3. **Filter Noise:** Ignore context snippets that do not contribute to the logical chain.")
    parts.append("4. **Derive Answer:** Combine the facts to form the final answer.")
    parts.append("\n### RESPONSE FORMAT:")
    parts.append("- **Step-by-Step Reasoning:** Briefly explain the logical steps (e.g., 1. Found Entity X... 2. Linked to Y... 3. Concluded Z).")
    parts.append("- **ANSWER:** Output ONLY the final answer line starting with 'ANSWER: '.")
    parts.append("\nDEMONSTRATIONS:")
    parts.append(fixed_demos_text)
    parts.append("\n### CURRENT TASK:")
    parts.append(f"Question: {question}")
    formatted = []
    for title, content in current_contexts:
        quoted_content = json.dumps(str(content), ensure_ascii=False)
        if title:
            formatted.append(f"Title: {title}\nContent: {quoted_content}")
        else:
            formatted.append(f"Content: {quoted_content}")
    parts.append("\nGiven Contexts:")
    parts.append("\n\n".join(formatted))
    parts.append("\n### YOUR OUTPUT:")
    parts.append("Step-by-Step Reasoning:")
    return "\n".join(parts)


def build_prompt(
    qid: int,
    corpus_df: pd.DataFrame,
    data_df: pd.DataFrame,
    doc_indices: list,
    dataset: str = "2wikimultihopqa",
) -> Tuple[str, int]:
    """Build a prompt from selected doc indices using KGQA-faithful format."""
    question = str(data_df.iloc[qid].get("question", ""))
    current_contexts = []
    for didx in doc_indices:
        didx = int(didx)
        title = str(corpus_df.iloc[didx]["title"])
        content = str(corpus_df.iloc[didx]["text"])
        current_contexts.append((title, content))

    if dataset == "musique":
        prompt = _build_prompt_musique(question, current_contexts)
    else:
        prompt = _build_prompt_2wiki_hotpot(question, current_contexts)

    token_est = len(prompt) // 4
    return prompt, token_est


# ═══════════════════════════════════════════════════════════════════
#  LLM inference with streaming TTFT measurement
# ═══════════════════════════════════════════════════════════════════

async def send_request_streaming(
    session,
    url: str,
    model_name: str,
    prompt: str,
    qid: int,
    max_tokens: int = 2048,
) -> dict:
    """Send a single chat completion request with streaming to measure TTFT."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t_send = time.time()
    ttft = None
    full_content = []

    try:
        async with session.post(url, json=payload, timeout=300) as resp:
            if resp.status != 200:
                err = await resp.text()
                return {
                    "qid": qid, "generation": "", "ttft_ms": 0,
                    "e2e_latency_ms": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "error": f"HTTP {resp.status}: {err}",
                }

            usage = {}
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:]  # strip "data: " prefix
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices") or []
                    delta = choices[0].get("delta", {}) if choices else {}
                    # Qwen3 reasoning parser: output in 'reasoning' field
                    content_piece = delta.get("content", "") or ""
                    reasoning_piece = delta.get("reasoning", "") or ""
                    piece = content_piece + reasoning_piece
                    if piece and ttft is None:
                        ttft = (time.time() - t_send) * 1000  # ms
                    if piece:
                        full_content.append(piece)
                    # Extract usage from final chunk if available
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                except json.JSONDecodeError:
                    continue

        e2e_latency = (time.time() - t_send) * 1000  # ms
        generation = "".join(full_content)

        return {
            "qid": qid,
            "generation": generation,
            "ttft_ms": round(ttft, 2) if ttft else 0,
            "e2e_latency_ms": round(e2e_latency, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "error": None,
        }
    except Exception as e:
        return {
            "qid": qid, "generation": "", "ttft_ms": 0,
            "e2e_latency_ms": (time.time() - t_send) * 1000,
            "prompt_tokens": 0, "completion_tokens": 0,
            "error": str(e),
        }


async def send_request_non_streaming(
    session,
    url: str,
    model_name: str,
    prompt: str,
    qid: int,
    max_tokens: int = 2048,
) -> dict:
    """Send a single chat completion request (non-streaming) for F1/ACC only."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    try:
        async with session.post(url, json=payload, timeout=300) as resp:
            data = await resp.json()
            e2e = (time.time() - t0) * 1000
            if "choices" in data and data["choices"]:
                msg = data["choices"][0].get("message", {})
                # Qwen3 reasoning parser: output in 'reasoning' field
                generation = msg.get("content") or msg.get("reasoning") or ""
            else:
                generation = ""
                if "error" in data:
                    return {
                        "qid": qid, "generation": "", "ttft_ms": 0,
                        "e2e_latency_ms": e2e, "prompt_tokens": 0,
                        "completion_tokens": 0, "error": str(data["error"]),
                    }
            return {
                "qid": qid,
                "generation": generation,
                "ttft_ms": 0,  # not measured in non-streaming mode
                "e2e_latency_ms": round(e2e, 2),
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                "error": None,
            }
    except Exception as e:
        return {
            "qid": qid, "generation": "", "ttft_ms": 0,
            "e2e_latency_ms": (time.time() - t0) * 1000,
            "prompt_tokens": 0, "completion_tokens": 0, "error": str(e),
        }


async def run_llm_eval(
    prompts: List[str],
    qids: List[int],
    host: str,
    port: int,
    model_name: str,
    concurrency: int = 8,
    max_tokens: int = 2048,
    streaming: bool = True,
    api_key: Optional[str] = None,
) -> List[dict]:
    """Send all prompts to LLM server, collect generations + TTFT."""
    import aiohttp

    url = f"http://{host}:{port}/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    send_fn = send_request_streaming if streaming else send_request_non_streaming

    async with aiohttp.ClientSession(headers=headers, trust_env=False) as session:
        async def bounded_request(idx):
            async with sem:
                return await send_fn(
                    session, url, model_name, prompts[idx], qids[idx], max_tokens
                )

        tasks = [bounded_request(i) for i in range(len(prompts))]
        results = await asyncio.gather(*tasks)

    return list(results)


def get_model_name(host: str, port: int, api_key: Optional[str] = None) -> str:
    """Get model name from server /v1/models endpoint."""
    try:
        req = urllib.request.Request(f"http://{host}:{port}/v1/models")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        resp = opener.open(req, timeout=5)
        data = json.loads(resp.read())
        return data["data"][0]["id"]
    except Exception:
        return "qwen3-4b"


# ═══════════════════════════════════════════════════════════════════
#  Data loading helpers
# ═══════════════════════════════════════════════════════════════════

# Per-dataset source_exp_id mapping (the canonical retrieval results)
DATASET_SOURCE_EXP = {
    "2wikimultihopqa": "baseline_v3",
    "hotpotqa": "baseline_full_1000q",
    "musique": "1",
}

# Per-dataset default top_k_final from the paper config
DATASET_TOP_K = {
    "2wikimultihopqa": 20,
    "hotpotqa": 10,
    "musique": 20,
}


def load_data(
    dataset: str,
    data_base: str,
    tmp_base: str,
    questions: int,
    source_exp_id: Optional[str] = None,
    use_expanded_pool: bool = False,
    graph_features_name: Optional[str] = None,
):
    """Load corpus, dataset, retrieval results, and graph features.

    Args:
        use_expanded_pool: if True, use corpus_list (30 docs) instead of
            retrieved_docs (20 docs) when available.
    """
    corpus_df = pd.read_json(f"{data_base}/{dataset}_corpus.json")
    data_df = pd.read_json(f"{data_base}/{dataset}.json")

    exp_id = source_exp_id or DATASET_SOURCE_EXP.get(dataset, "baseline_v3")
    rr_path = f"{tmp_base}/{dataset}/retrieve_result_4B_{exp_id}.npy"
    if not os.path.exists(rr_path):
        logger.error(f"Retrieve result not found: {rr_path}")
        logger.info(f"Available files: {os.listdir(f'{tmp_base}/{dataset}/')}")
        sys.exit(1)

    rr = np.load(rr_path, allow_pickle=True).item()
    retrieved_docs = rr["retrieved_docs"]
    corpus_score = rr.get("corpus_score", None)
    corpus_list = rr.get("corpus_list", None)

    # Use expanded pool (30 candidates) if available and requested
    if use_expanded_pool and corpus_list is not None and len(corpus_list) > 0:
        if len(corpus_list[0]) > len(retrieved_docs[0]):
            logger.info(f"Using expanded pool: corpus_list ({len(corpus_list[0])} docs/q)")
            retrieved_docs = corpus_list

    gf_name = graph_features_name or "graph_features_1000q.json"
    gf_path = f"{tmp_base}/{dataset}/{gf_name}"
    if not os.path.exists(gf_path):
        logger.error(f"Graph features not found: {gf_path}")
        sys.exit(1)

    n = min(questions, len(retrieved_docs), len(data_df))
    logger.info(f"Loaded {dataset}: corpus={len(corpus_df)}, queries={len(data_df)}, "
                f"retrieved={len(retrieved_docs)}, pool_size={len(retrieved_docs[0])}, "
                f"using n={n}")
    return corpus_df, data_df, retrieved_docs, gf_path, n, corpus_score


# ═══════════════════════════════════════════════════════════════════
#  Main evaluation pipeline
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 Standalone Evaluation: NDCG@K + F1/ACC + TTFT"
    )
    # Data args
    parser.add_argument("--dataset", default="2wikimultihopqa",
                        choices=["2wikimultihopqa", "hotpotqa", "musique"])
    parser.add_argument("--questions", type=int, default=100)
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--data_base",
                        default=os.path.join(_REPO_ROOT, "data"))
    parser.add_argument("--tmp_base",
                        default=os.path.join(_REPO_ROOT, "tmp_data"))
    parser.add_argument("--source_exp_id", default=None,
                        help="Override source retrieval exp_id. Auto-detected per dataset.")
    parser.add_argument("--graph_features_name", default=None,
                        help="Graph features filename (default: graph_features_1000q.json)")

    # Aggregation args (passed to AdaptiveAggregator)
    parser.add_argument("--agg_mode", default="full",
                        choices=["full", "fixed_k", "difficulty", "backpressure",
                                 "joint", "score_gap", "enhanced_joint", "binary_adaptive",
                                 "adaptive_k", "uniform_budget"])
    parser.add_argument("--fixed_k", type=int, default=20)
    parser.add_argument("--k_min", type=int, default=5)
    parser.add_argument("--k_max", type=int, default=20)
    parser.add_argument("--diff_alpha", type=float, default=0.4)
    parser.add_argument("--diff_beta", type=float, default=0.3)
    parser.add_argument("--diff_gamma", type=float, default=0.3)
    parser.add_argument("--k_mapping", default="linear",
                        choices=["linear", "sigmoid"])
    parser.add_argument("--backpressure_threshold", type=float, default=0.7,
                        help="Queue utilization threshold for backpressure activation")
    parser.add_argument("--backpressure_k_max", type=int, default=8,
                        help="Reduced K ceiling under backpressure (for joint mode)")
    # Per-difficulty K ranges (for score_gap / enhanced_joint)
    parser.add_argument("--easy_k_min", type=int, default=5)
    parser.add_argument("--easy_k_max", type=int, default=12)
    parser.add_argument("--medium_k_min", type=int, default=10)
    parser.add_argument("--medium_k_max", type=int, default=18)
    parser.add_argument("--hard_k_min", type=int, default=15)
    parser.add_argument("--hard_k_max", type=int, default=25)
    parser.add_argument("--score_weight", type=float, default=0.5,
                        help="Weight for reranker score features in enhanced scorer (0=graph only, 1=score only)")
    parser.add_argument("--use_expanded_pool", action="store_true",
                        help="Use corpus_list (30 docs) instead of retrieved_docs (20)")
    # Binary adaptive mode args
    parser.add_argument("--binary_easy_k", type=int, default=5,
                        help="K for easy queries in binary_adaptive mode")
    parser.add_argument("--binary_hard_k", type=int, default=20,
                        help="K for hard queries in binary_adaptive mode")
    parser.add_argument("--binary_medium_k", type=int, default=12,
                        help="K for hard queries under backpressure in binary_adaptive mode")
    parser.add_argument("--binary_threshold", type=float, default=0.5,
                        help="Classifier probability threshold for hard classification")
    parser.add_argument("--binary_classifier_type", default="score_top1",
                        choices=["score_top1", "n_triples", "lr"],
                        help="Binary classifier signal type")
    parser.add_argument("--binary_feature_threshold", type=float, default=4.0,
                        help="Feature threshold for score_top1 or n_triples classifier")

    # Adaptive-k baseline (RQ4)
    parser.add_argument("--adaptive_k_ratio", type=float, default=0.3,
                        help="Score ratio threshold for adaptive_k mode (keep docs with score >= top*ratio)")
    # Uniform budget baseline (RQ4)
    parser.add_argument("--uniform_budget_ratio", type=float, default=0.5,
                        help="Budget ratio for uniform_budget mode (K = k_max * ratio)")

    # Bridge-aware assembly
    parser.add_argument("--bridge_aware", action="store_true",
                        help="Enable bridge-aware doc assembly (promote bridge docs into top-K)")
    parser.add_argument("--bridge_max_promote", type=int, default=3,
                        help="Max bridge docs to promote from outside top-K")

    # NDCG args
    parser.add_argument("--ndcg_ks", default="5,10,15,20",
                        help="Comma-separated K values for NDCG computation")

    # Server args (for F1/ACC + TTFT)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api_key", default="api-key-qwen3",
                        help="vLLM API key (default: api-key-qwen3)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--streaming", action="store_true", default=True,
                        help="Use streaming for TTFT measurement (default: True)")
    parser.add_argument("--no_streaming", action="store_true",
                        help="Disable streaming (no TTFT measurement)")
    parser.add_argument("--slo_ttft_ms", type=float, default=2000,
                        help="SLO threshold for TTFT in ms")

    # Mode
    parser.add_argument("--offline_only", action="store_true",
                        help="Only compute NDCG (no LLM server needed)")

    # Output
    parser.add_argument("--exp_id", default="P2_eval_v1")
    parser.add_argument("--output_dir", default=None)

    args = parser.parse_args()
    if args.no_streaming:
        args.streaming = False
    ndcg_ks = [int(k) for k in args.ndcg_ks.split(",")]

    # ── Load data ───────────────────────────────────────────────────
    corpus_df, data_df, retrieved_docs, gf_path, n_queries, corpus_score = load_data(
        args.dataset, args.data_base, args.tmp_base, args.questions,
        args.source_exp_id, args.use_expanded_pool,
        args.graph_features_name,
    )

    # ── Create aggregator ───────────────────────────────────────────
    agg_config = AggregationConfig(
        mode=args.agg_mode,
        fixed_k=args.fixed_k,
        k_min=args.k_min,
        k_max=args.k_max,
        diff_alpha=args.diff_alpha,
        diff_beta=args.diff_beta,
        diff_gamma=args.diff_gamma,
        k_mapping=args.k_mapping,
        backpressure_threshold=args.backpressure_threshold,
        backpressure_k_max=args.backpressure_k_max,
        easy_k_min=args.easy_k_min,
        easy_k_max=args.easy_k_max,
        medium_k_min=args.medium_k_min,
        medium_k_max=args.medium_k_max,
        hard_k_min=args.hard_k_min,
        hard_k_max=args.hard_k_max,
        score_weight=args.score_weight,
        binary_easy_k=args.binary_easy_k,
        binary_hard_k=args.binary_hard_k,
        binary_medium_k=args.binary_medium_k,
        binary_threshold=args.binary_threshold,
        binary_classifier_type=args.binary_classifier_type,
        binary_feature_threshold=args.binary_feature_threshold,
        bridge_aware=args.bridge_aware,
        bridge_max_promote=args.bridge_max_promote,
        adaptive_k_ratio=args.adaptive_k_ratio,
        uniform_budget_ratio=args.uniform_budget_ratio,
    )
    aggregator = AdaptiveAggregator(
        gf_path, agg_config,
        corpus_score=corpus_score,
        doc_pool=retrieved_docs,
        dataset=args.dataset,
    )

    # ── Run aggregation + build prompts ─────────────────────────────
    prompts = []
    answers = []
    per_query = []
    k_values = []
    token_counts = []
    ndcg_results = {k_val: [] for k_val in ndcg_ks}

    for qid in range(n_queries):
        doc_indices = list(np.asarray(retrieved_docs[qid]))
        # Get per-doc reranker scores for score_gap mode
        doc_scores = None
        if corpus_score is not None and qid < len(corpus_score):
            cs_q = np.array(corpus_score[qid])
            doc_scores = [float(cs_q[int(d)]) for d in doc_indices]
        selected_docs, meta = aggregator.select_docs(qid, doc_indices,
                                                     doc_scores=doc_scores)

        # Build prompt
        prompt, token_est = build_prompt(
            qid, corpus_df, data_df, selected_docs, dataset=args.dataset,
        )
        prompts.append(prompt)
        answer = data_df.iloc[qid].get("answer", "")
        answers.append(str(answer))

        # NDCG: measure how well selected docs cover ground-truth
        gold_indices = get_gold_doc_indices(qid, data_df, corpus_df, args.dataset)
        q_ndcg = {}
        for k_val in ndcg_ks:
            if gold_indices:
                n = compute_ndcg_for_query(selected_docs, gold_indices, k_val)
            else:
                n = float("nan")
            ndcg_results[k_val].append(n)
            q_ndcg[f"ndcg@{k_val}"] = round(n, 4) if not math.isnan(n) else None

        meta["qid"] = qid
        meta["token_estimate"] = token_est
        meta["question"] = str(data_df.iloc[qid].get("question", ""))[:200]
        meta["n_gold_docs"] = len(gold_indices)
        meta.update(q_ndcg)
        per_query.append(meta)
        k_values.append(meta["k_decided"])
        token_counts.append(token_est)

    # ── Aggregate NDCG stats ────────────────────────────────────────
    ndcg_summary = {}
    for k_val in ndcg_ks:
        vals = [v for v in ndcg_results[k_val] if not math.isnan(v)]
        if vals:
            ndcg_summary[f"ndcg@{k_val}"] = round(float(np.mean(vals)), 4)
            ndcg_summary[f"ndcg@{k_val}_std"] = round(float(np.std(vals)), 4)
            ndcg_summary[f"ndcg@{k_val}_n"] = len(vals)
        else:
            ndcg_summary[f"ndcg@{k_val}"] = None

    avg_k = float(np.mean(k_values))
    avg_tokens = float(np.mean(token_counts))
    k_dist = dict(Counter(k_values).most_common())

    logger.info(f"Aggregation done: n={n_queries}, avg_K={avg_k:.1f}, "
                f"avg_tokens={avg_tokens:.0f}")
    logger.info(f"K distribution: {k_dist}")
    logger.info(f"NDCG summary: {ndcg_summary}")

    # ── LLM evaluation (F1/ACC + TTFT) ─────────────────────────────
    qa_summary = {}
    ttft_summary = {}
    if not args.offline_only:
        model_name = get_model_name(args.host, args.port, args.api_key)
        logger.info(f"Sending {n_queries} requests to {args.host}:{args.port} "
                    f"(model={model_name}, concurrency={args.concurrency}, "
                    f"streaming={args.streaming})")

        results = asyncio.run(run_llm_eval(
            prompts, list(range(n_queries)),
            args.host, args.port, model_name,
            args.concurrency, args.max_tokens, args.streaming,
            args.api_key,
        ))

        # Extract predictions
        predictions = [""] * n_queries
        for r in results:
            predictions[r["qid"]] = r["generation"]

        errors = sum(1 for r in results if r["error"])
        if errors:
            logger.warning(f"{errors}/{n_queries} requests had errors")
            for r in results:
                if r["error"]:
                    logger.debug(f"  qid={r['qid']}: {r['error'][:200]}")

        # F1 / ACC
        metrics = calculate_metric_scores(answers, predictions)
        f1 = metrics["pooled_results"]["F1"]
        acc = metrics["pooled_results"]["ACC"]
        qa_summary = {"f1": round(f1, 4), "acc": round(acc, 4), "errors": errors}

        # Per-query merge
        for i, sr in enumerate(metrics["specific_results"]):
            per_query[i]["f1"] = sr["F1"]
            per_query[i]["acc"] = sr["ACC"]
            per_query[i]["generation"] = sr["generation"][:500]

        for r in results:
            per_query[r["qid"]]["ttft_ms"] = r["ttft_ms"]
            per_query[r["qid"]]["e2e_latency_ms"] = r["e2e_latency_ms"]
            if r["error"]:
                per_query[r["qid"]]["error"] = r["error"][:300]

        # TTFT stats
        ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] > 0 and not r["error"]]
        e2es = [r["e2e_latency_ms"] for r in results if not r["error"]]
        if ttfts:
            ttft_arr = np.array(ttfts)
            slo_met = sum(1 for t in ttfts if t <= args.slo_ttft_ms)
            ttft_summary = {
                "ttft_p50_ms": round(float(np.percentile(ttft_arr, 50)), 2),
                "ttft_p90_ms": round(float(np.percentile(ttft_arr, 90)), 2),
                "ttft_p99_ms": round(float(np.percentile(ttft_arr, 99)), 2),
                "ttft_mean_ms": round(float(np.mean(ttft_arr)), 2),
                f"slo@{int(args.slo_ttft_ms)}ms": round(slo_met / len(ttfts), 4),
                "n_ttft_measured": len(ttfts),
            }
        if e2es:
            e2e_arr = np.array(e2es)
            ttft_summary["e2e_p50_ms"] = round(float(np.percentile(e2e_arr, 50)), 2)
            ttft_summary["e2e_p99_ms"] = round(float(np.percentile(e2e_arr, 99)), 2)

        qa_correct_count = sum(1 for sr in metrics["specific_results"] if sr["F1"] > 0)
        qa_goodput_count = 0
        for r in results:
            if r["error"] or r["ttft_ms"] <= 0:
                continue
            qf1 = metrics["specific_results"][r["qid"]]["F1"]
            if qf1 > 0 and r["ttft_ms"] <= args.slo_ttft_ms:
                qa_goodput_count += 1
        ok_results = [r for r in results if not r["error"] and r["ttft_ms"] > 0]
        if ok_results:
            qa_summary["qa_correct_count"] = qa_correct_count
            qa_summary["qa_goodput_count"] = qa_goodput_count
            qa_summary["qa_goodput_pct"] = round(100.0 * qa_goodput_count / len(ok_results), 2)
            qa_summary["qa_correct_pct"] = round(100.0 * qa_correct_count / len(ok_results), 2)

        logger.info(f"QA: F1={f1:.4f}, ACC={acc:.4f}, errors={errors}")
        logger.info(f"TTFT: {ttft_summary}")

    # ── Build report ────────────────────────────────────────────────
    report = {
        "config": {
            "dataset": args.dataset,
            "questions": n_queries,
            "agg_mode": args.agg_mode,
            "agg_config": {
                "k_min": agg_config.k_min,
                "k_max": agg_config.k_max,
                "fixed_k": agg_config.fixed_k,
                "diff_alpha": agg_config.diff_alpha,
                "diff_beta": agg_config.diff_beta,
                "diff_gamma": agg_config.diff_gamma,
                "k_mapping": agg_config.k_mapping,
                "backpressure_threshold": agg_config.backpressure_threshold,
                "backpressure_k_max": agg_config.backpressure_k_max,
                "easy_k_range": [agg_config.easy_k_min, agg_config.easy_k_max],
                "medium_k_range": [agg_config.medium_k_min, agg_config.medium_k_max],
                "hard_k_range": [agg_config.hard_k_min, agg_config.hard_k_max],
                "score_weight": agg_config.score_weight,
                "binary_easy_k": agg_config.binary_easy_k,
                "binary_hard_k": agg_config.binary_hard_k,
                "binary_medium_k": agg_config.binary_medium_k,
                "binary_threshold": agg_config.binary_threshold,
                "binary_classifier_type": agg_config.binary_classifier_type,
                "binary_feature_threshold": agg_config.binary_feature_threshold,
            },
            "exp_id": args.exp_id,
            "host": args.host if not args.offline_only else None,
            "port": args.port if not args.offline_only else None,
            "model_name": model_name if not args.offline_only else None,
            "streaming": args.streaming if not args.offline_only else None,
            "slo_ttft_ms": args.slo_ttft_ms,
            "thinking_enabled": False if not args.offline_only else None,
            "offline_only": args.offline_only,
        },
        "summary": {
            "avg_k": round(avg_k, 2),
            "avg_tokens": round(avg_tokens, 0),
            "k_distribution": k_dist,
            "difficulty_distribution": (
                aggregator.binary_classifier.get_distribution(agg_config)
                if aggregator.binary_classifier is not None
                else aggregator.scorer.get_distribution(agg_config)
            ),
            **ndcg_summary,
            **qa_summary,
            **ttft_summary,
        },
        "per_query": per_query,
    }

    # ── Save ────────────────────────────────────────────────────────
    out_dir = args.output_dir or f"{args.tmp_base}/{args.dataset}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{out_dir}/{args.exp_id}_phase2_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved to {out_path}")

    # ── Console summary ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Phase 2 Eval | {args.exp_id} | {args.dataset} | mode={args.agg_mode}")
    print(f"{'='*70}")
    print(f"  avg K:        {avg_k:.1f}")
    print(f"  avg tokens:   {avg_tokens:.0f}")
    print(f"  K dist:       {k_dist}")
    for k_val in ndcg_ks:
        val = ndcg_summary.get(f"ndcg@{k_val}")
        if val is not None:
            print(f"  NDCG@{k_val}:     {val:.4f}")
    if qa_summary:
        print(f"  F1:           {qa_summary['f1']:.4f}")
        print(f"  ACC:          {qa_summary['acc']:.4f}")
        print(f"  errors:       {qa_summary['errors']}")
    if ttft_summary:
        print(f"  TTFT p50:     {ttft_summary.get('ttft_p50_ms', 'N/A')} ms")
        print(f"  TTFT p99:     {ttft_summary.get('ttft_p99_ms', 'N/A')} ms")
        slo_key = f"slo@{int(args.slo_ttft_ms)}ms"
        if slo_key in ttft_summary:
            print(f"  SLO@{int(args.slo_ttft_ms)}ms:  {ttft_summary[slo_key]*100:.1f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
