"""
End-to-End Joint Benchmark: P1 (Graph Features) → P2 (Adaptive Aggregation) → P3 (SLO Priority).

Implements the full feedback loop:
  P1 → P2: graph_features feed difficulty scoring → adaptive K
  P2 → P3: difficulty + prompt_tokens feed priority scheduling
  P3 → P2: queue_utilization (backpressure) feeds back to K decision

The main config is ``DynaGraph`` (full Phase 1 + Phase 2 + Phase 3 with
bidirectional feedback). Additional ablation configs are kept for
reference.

Usage (200q quick test):
  python src/benchmark_e2e_joint.py --dataset 2wikimultihopqa --questions 200 \
    --arrival poisson --rate 5 --config DynaGraph --exp_id demo
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
from loguru import logger

# Module 2
from adaptive_aggregator import AdaptiveAggregator, AggregationConfig, DifficultyScorer
# Module 3
from utils.joint_priority import (
    JointPriorityConfig,
    GraphEnhancedPriorityCalculator,
    estimate_prompt_tokens,
)
from utils.evaluate import calculate_metric_scores


# ===================== KGQA-faithful prompt builders =====================

def _build_prompt_2wiki_hotpot(question: str, contexts: list) -> str:
    fixed_demos = """
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
Example Question: Who lived longer, Ludovico Sabbatini or Marcelino Manuel Da Graça?
Given Context: Title: Ludovico Sabbatini
Content: "Venerable Ludovico Sabbatini( 30 August 1650 – 11 June 1724) was an Italian priest and religious educator, who was beatified by the Catholic Church in 1765."
Given Context: Title: Marcelino Manuel da Graça
Content: "Marcelino Manuel da Graça( January 25, 1881 or 1884 — January 12, 1960), better known as Charles Manuel\\" Sweet Daddy\\" Grace, or Daddy Grace, was the founder and first bishop of the predominantly African- American denomination the United House of Prayer For All People."

Step-by-Step Reasoning:
1. Ludovico Sabbatini lived from 1650 to 1724 (approx 74 years).
2. Marcelino Manuel da Graça lived from 1881/1884 to 1960 (approx 76-79 years).
3. Marcelino lived longer than Ludovico.
ANSWER: Marcelino Manuel Da Graça
"""
    parts = []
    parts.append("You are an expert multi-hop reasoning engine. Your goal is to answer complex questions by connecting facts from multiple provided context snippets.")
    parts.append("\n### INSTRUCTIONS:")
    parts.append("1. **Analyze the Question:** Identify the key entity and what relationship is being queried.")
    parts.append("2. **Search & Bridge:** Find the initial entity, identify the connecting link (bridge entity), and locate the final information.")
    parts.append("3. **Filter Noise:** Ignore context snippets that do not contribute to the logical chain.")
    parts.append("4. **Derive Answer:** Combine the facts to form the final answer.")
    parts.append("\n### RESPONSE FORMAT:")
    parts.append("- **Step-by-Step Reasoning:** Briefly explain the logical steps.")
    parts.append("- **ANSWER:** Output ONLY the final answer line starting with 'ANSWER: '.")
    parts.append("\nDEMONSTRATIONS:")
    parts.append(fixed_demos)
    parts.append("\nCURRENT TASK:")
    parts.append(f"Question: {question}")
    parts.append("Given Contexts:")
    for title, content in contexts:
        quoted = json.dumps(content, ensure_ascii=False)
        if title:
            parts.append(f"Title: {title}\nContent: {quoted}\n")
        else:
            parts.append(f"Content: {quoted}\n")
    parts.append("\n### YOUR OUTPUT:")
    parts.append("Step-by-Step Reasoning:")
    return "\n".join(parts)


def _build_prompt_musique(question: str, contexts: list) -> str:
    fixed_demos = """
--- Demonstration 1 ---
Example Question: When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?

Given Contexts:
Title: FC Barcelona
Content: "Barcelona advanced to the semi-finals of the Copa del Rey, winning the first leg against Getafe 5–2, with a goal from Messi bringing comparison to Diego Maradona's goal of the century, but then lost the second leg 4–0."
Title: FC Barcelona
Content: "In June 1982, Diego Maradona was signed for a world record fee of £5 million from Boca Juniors. In the following season, under coach Luis, Barcelona won the Copa del Rey..."

Step-by-Step Reasoning:
1. Messi's goal brought comparison to **Diego Maradona**.
2. Diego Maradona was signed by Barcelona in **June 1982**.
ANSWER: June 1982

--- Demonstration 2 ---
Example Question: What was the name of the star of Forced Vengeance in Walter, Texas Ranger?

Given Contexts:
Title: Forced Vengeance
Content: "Dragon's chief of security, Josh Randall (Chuck Norris) goes looking for the head of the syndicate to exact revenge for the murder of his employer, friend and mentor."
Title: Walker, Texas Ranger
Content: "Name Portrayed by ... Cordell Walker Chuck Norris Texas Ranger Main ..."

Step-by-Step Reasoning:
1. The star of Forced Vengeance is **Chuck Norris**.
2. Chuck Norris portrays **Cordell Walker** in Walker, Texas Ranger.
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
    parts.append("- **Step-by-Step Reasoning:** Briefly explain the logical steps.")
    parts.append("- **ANSWER:** Output ONLY the final answer line starting with 'ANSWER: '.")
    parts.append("\nDEMONSTRATIONS:")
    parts.append(fixed_demos)
    parts.append("\n### CURRENT TASK:")
    parts.append(f"Question: {question}")
    formatted = []
    for title, content in contexts:
        quoted = json.dumps(str(content), ensure_ascii=False)
        if title:
            formatted.append(f"Title: {title}\nContent: {quoted}")
        else:
            formatted.append(f"Content: {quoted}")
    parts.append("\nGiven Contexts:")
    parts.append("\n\n".join(formatted))
    parts.append("\n### YOUR OUTPUT:")
    parts.append("Step-by-Step Reasoning:")
    return "\n".join(parts)


def build_prompt(question, contexts, dataset):
    if dataset == "musique":
        return _build_prompt_musique(question, contexts)
    return _build_prompt_2wiki_hotpot(question, contexts)


# ===================== Experiment Configs =====================

EXPERIMENT_CONFIGS = {
    "DynaGraph": {  # Full method: Phase 1 + Phase 2 + Phase 3 with bidirectional feedback
        "agg_mode": "joint", "strategy": "graph_slo",
        "graph_weight": 1.0, "slo_weight": 1.0, "cost_weight": 0.2,
        "use_difficulty_signal": True,
    },
    "Baseline": {  # Isolated baseline: no signals, K=K_max, FCFS
        "agg_mode": "full", "strategy": "none",
        "graph_weight": 0, "slo_weight": 0, "cost_weight": 0,
        "use_difficulty_signal": False,
    },
    "B4": {  # Joint graph+SLO priority (no adaptive K)
        "agg_mode": "full", "strategy": "graph_slo",
        "graph_weight": 1.0, "slo_weight": 1.0, "cost_weight": 0.2,
        "use_difficulty_signal": True,
    },
    "B5": {  # Dynamic weights: cache-state-aware
        "agg_mode": "joint", "strategy": "graph_slo",
        "graph_weight": 1.0, "slo_weight": 1.0, "cost_weight": 0.2,
        "use_difficulty_signal": True,
        "dynamic_weights": True,  # cold phase: high graph_weight, warm phase: lower
    },
    # --- Phase 2 additions ---
    "SLO_PRIORITY": {  # K=20 + SLO urgency priority (pure P3, no P2)
        "agg_mode": "full", "strategy": "graph_slo",
        "graph_weight": 0, "slo_weight": 1.0, "cost_weight": 0.2,
        "use_difficulty_signal": False,
    },
    "NAIVE_JOINT": {  # Difficulty K + FCFS (no priority signal)
        "agg_mode": "difficulty", "strategy": "none",
        "graph_weight": 0, "slo_weight": 0, "cost_weight": 0,
        "use_difficulty_signal": False,
    },
    "P2_AGG": {  # Fixed K=5 aggressive pruning
        "agg_mode": "fixed_k", "strategy": "none",
        "graph_weight": 0, "slo_weight": 0, "cost_weight": 0,
        "use_difficulty_signal": False,
        "fixed_k": 5,
    },
    "P2_CON": {  # Fixed K=15 conservative pruning
        "agg_mode": "fixed_k", "strategy": "none",
        "graph_weight": 0, "slo_weight": 0, "cost_weight": 0,
        "use_difficulty_signal": False,
        "fixed_k": 15,
    },
    "ADAPTIVE_K": {  # [RQ4] Taguchi et al. score-ratio thresholding, no scheduling
        "agg_mode": "adaptive_k", "strategy": "none",
        "graph_weight": 0, "slo_weight": 0, "cost_weight": 0,
        "use_difficulty_signal": False,
    },
    "UNIFORM_BUDGET": {  # [RQ4] Uniform budget reduction, no scheduling
        "agg_mode": "uniform_budget", "strategy": "none",
        "graph_weight": 0, "slo_weight": 0, "cost_weight": 0,
        "use_difficulty_signal": False,
    },
    "RANDOM_K": {  # [EXP-3] Random K baseline, no scheduling
        "agg_mode": "random", "strategy": "none",
        "graph_weight": 0, "slo_weight": 0, "cost_weight": 0,
        "use_difficulty_signal": False,
    },
}


# ===================== Queue Tracker (P3→P2 feedback) =====================

class QueueTracker:
    """Simulates queue utilization for backpressure feedback."""
    def __init__(self, concurrency: int):
        self.concurrency = concurrency
        self._active = 0
        self._lock = asyncio.Lock() if asyncio is not None else None

    @property
    def utilization(self) -> float:
        return self._active / max(self.concurrency, 1)

    async def acquire(self):
        async with self._lock:
            self._active += 1

    async def release(self):
        async with self._lock:
            self._active = max(0, self._active - 1)


# ===================== Request sender =====================

@dataclass
class E2EResult:
    query_id: int
    send_time: float
    ttft: float
    e2e_latency: float
    prompt_tokens: int
    completion_tokens: int
    generation: str
    priority: int
    k_used: int
    difficulty_score: float
    queue_util_at_send: float
    slo_met: bool = True
    slo_deadline_ms: float = 2000.0
    slo_weight: float = 1.0
    slo_class: str = "default"
    f1: float = 0.0
    error: Optional[str] = None


async def send_request(
    session, url, model_name, prompt, query_id, priority, k_used,
    difficulty_score, queue_util, slo_ttft_ms, max_tokens=2048,
    api_key: Optional[str] = None,
) -> E2EResult:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}
    }
    # Only include priority when non-zero: vLLM ≥0.17 supports it for priority scheduling;
    # vLLM 0.8.5.post1 (CacheBlend env) raises ValueError if field is present at all.
    if int(priority) != 0:
        payload["priority"] = int(priority)
    send_time = time.perf_counter()
    ttft = None
    full_text = ""
    pt = ct = 0

    try:
        headers = {"Content-Type": "application/json"}
        auth_key = api_key or os.environ.get("OPENAI_API_KEY")
        if auth_key:
            headers["Authorization"] = f"Bearer {auth_key}"
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                err = await resp.text()
                return E2EResult(query_id=query_id, send_time=send_time, ttft=0, e2e_latency=0,
                    prompt_tokens=0, completion_tokens=0, generation="", priority=priority,
                    k_used=k_used, difficulty_score=difficulty_score, queue_util_at_send=queue_util,
                    slo_met=False, error=f"HTTP {resp.status}: {err[:200]}")

            async for line in resp.content:
                s = line.decode("utf-8").strip()
                if not s or not s.startswith("data: "):
                    continue
                d = s[6:]
                if d == "[DONE]":
                    break
                try:
                    chunk = json.loads(d)
                except json.JSONDecodeError:
                    continue
                if ttft is None:
                    choices = chunk.get("choices", [])
                    if choices:
                        d0 = choices[0].get("delta", {})
                        if d0.get("content") is not None or d0.get("reasoning_content") is not None or d0.get("reasoning") is not None:
                            ttft = time.perf_counter() - send_time
                
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    c = delta.get("content", "") or ""
                    r = delta.get("reasoning_content", "") or delta.get("reasoning", "") or ""
                    if c or r:
                        full_text += r + c
                usage = chunk.get("usage")
                if usage:
                    pt = usage.get("prompt_tokens", 0)
                    ct = usage.get("completion_tokens", 0)

        e2e = time.perf_counter() - send_time
        if ttft is None:
            ttft = e2e
            
        return E2EResult(query_id=query_id, send_time=send_time, ttft=ttft, e2e_latency=e2e,
            prompt_tokens=pt, completion_tokens=ct, generation=full_text, priority=priority,
            k_used=k_used, difficulty_score=difficulty_score, queue_util_at_send=queue_util,
            slo_met=(ttft * 1000) <= slo_ttft_ms)
    except Exception as e:
        return E2EResult(query_id=query_id, send_time=send_time, ttft=0,
            e2e_latency=time.perf_counter() - send_time, prompt_tokens=0, completion_tokens=0,
            generation="", priority=priority, k_used=k_used, difficulty_score=difficulty_score,
            queue_util_at_send=queue_util, slo_met=False, error=str(e))


# ===================== Main =====================

def main():
    parser = argparse.ArgumentParser(description="E2E Joint Benchmark: P1→P2→P3")
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--source_exp_id", default="baseline_v3")
    parser.add_argument("--questions", type=int, default=200)
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--data_base", default=os.path.join(_REPO_ROOT, "data"))
    parser.add_argument("--tmp_base", default=os.path.join(_REPO_ROOT, "tmp_data"))
    parser.add_argument("--graph_features", default=None)

    parser.add_argument("--config", default="DynaGraph")
    parser.add_argument("--arrival", default="poisson", choices=["sequential", "poisson", "zipf"])
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--total_requests", type=int, default=0)
    parser.add_argument("--zipf_alpha", type=float, default=1.0)

    parser.add_argument("--slo_ttft_ms", type=float, default=2000.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api_key", default="api-key-qwen3")
    parser.add_argument("--engine", default="sglang", choices=["sglang", "vllm"])

    parser.add_argument("--k_min", type=int, default=5)
    parser.add_argument("--k_max", type=int, default=20)
    parser.add_argument("--bp_threshold", type=float, default=0.7)
    parser.add_argument("--bp_k_max", type=int, default=8)
    parser.add_argument("--k_mapping", default="linear", choices=["linear", "sigmoid"])
    parser.add_argument("--sigmoid_steepness", type=float, default=10.0)
    parser.add_argument("--bp_decay", default="fixed",
                        choices=["fixed", "linear", "exponential", "step3"],
                        help="Backpressure K_bp decay shape (E3)")
    parser.add_argument("--diff_alpha", type=float, default=None,
                        help="S_diff weight for bridge_doc_ratio (T1.1)")
    parser.add_argument("--diff_beta", type=float, default=None,
                        help="S_diff weight for entity_overlap_matrix_density")
    parser.add_argument("--diff_gamma", type=float, default=None,
                        help="S_diff weight for n_unique_entities")

    parser.add_argument("--heterogeneous_slo", action="store_true",
                        help="Enable per-query heterogeneous SLO deadlines")
    parser.add_argument("--slo_classes", default="500:0.3:3,1500:0.5:2,5000:0.2:1",
                        help="SLO classes: 'deadline_ms:proportion:weight,...'")
    parser.add_argument("--signal_mask", default="all",
                        choices=["all", "slo_only", "cost_only", "graph_only", "slo_cost", "none"],
                        help="Which scheduling signals to use in P3 priority")
    parser.add_argument("--model_name_tag", default="",
                        help="Tag for the model used (for report metadata)")

    parser.add_argument("--exp_id", default="e2e_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_priority", action="store_true",
                        help="Do not include 'priority' field in requests "
                             "(needed for vLLM 0.8.x which rejects the field)")
    parser.add_argument("--query_subset", default="all",
                        choices=["all", "2wiki_bridge_comparison", "oracle_k_gt_10"],
                        help="Optional query subset for stress tests; keeps original dataset indices.")
    parser.add_argument("--custom_k_policy", default="none",
                        choices=["none", "random75_k3_k20", "random40_k3_k20", "inverted_hard_k3_easy_k20"],
                        help="Override selected K for counterfactual S_diff stress tests.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    logger.info("  Loading dataset files...")
    # Load data
    corpus_df = pd.read_json(f"{args.data_base}/{args.dataset}_corpus.json")
    data_df = pd.read_json(f"{args.data_base}/{args.dataset}.json")
    
    logger.info("  Loading retrieved docs...")
    rr_path = f"{args.tmp_base}/{args.dataset}/retrieve_result_4B_{args.source_exp_id}.npy"
    rr = np.load(rr_path, allow_pickle=True).item()
    retrieved_docs_raw = rr["retrieved_docs"]

    logger.info("  Formatting retrieved docs...")
    # Handle list vs dict formats
    corpus_scores_raw = rr.get("corpus_score", None)  # per-query full-corpus scores
    if isinstance(retrieved_docs_raw, list):
        id_col = "_id" if "_id" in data_df.columns else "id"
        qids_to_run = data_df[id_col].tolist()
        retrieved_docs = {}
        for i, doc_ids in enumerate(retrieved_docs_raw):
            if i < len(qids_to_run):
                entry = {"ids": doc_ids}
                # Extract reranker scores for the retrieved docs
                if corpus_scores_raw is not None and i < len(corpus_scores_raw):
                    all_scores = np.asarray(corpus_scores_raw[i])
                    entry["scores"] = [float(all_scores[int(d)]) for d in doc_ids]
                retrieved_docs[qids_to_run[i]] = entry
    else:
        retrieved_docs = retrieved_docs_raw
        qids_to_run = list(retrieved_docs.keys())
    
    selected_indices = list(range(min(len(retrieved_docs), len(data_df))))
    hard_label_by_idx = {}
    if args.query_subset == "2wiki_bridge_comparison":
        if args.dataset != "2wikimultihopqa":
            logger.warning("--query_subset=2wiki_bridge_comparison requested for non-2Wiki dataset; using all queries.")
        else:
            selected_indices = [
                i for i in selected_indices
                if str(data_df.iloc[i].get("type", "")) == "bridge_comparison"
            ]
            hard_label_by_idx = {i: True for i in selected_indices}
    elif args.query_subset == "oracle_k_gt_10":
        oracle_path = Path(os.path.join(_REPO_ROOT, "tmp_data", "oracle_k_2wiki.json"))
        if args.dataset != "2wikimultihopqa" or not oracle_path.exists():
            logger.warning("--query_subset=oracle_k_gt_10 unavailable; using all queries.")
        else:
            oracle_k = json.load(open(oracle_path)).get("oracle_k", [])
            selected_indices = [
                i for i in selected_indices
                if i < len(oracle_k) and float(oracle_k[i]) > 10
            ]
            hard_label_by_idx = {i: True for i in selected_indices}
    if args.custom_k_policy == "inverted_hard_k3_easy_k20" and not hard_label_by_idx:
        if args.dataset == "2wikimultihopqa":
            hard_label_by_idx = {
                i: str(data_df.iloc[i].get("type", "")) == "bridge_comparison"
                for i in range(len(data_df))
            }
        else:
            hard_label_by_idx = {i: False for i in range(len(data_df))}

    n_unique = min(args.questions, len(selected_indices))

    logger.info("  Loading graph features...")
    # Graph features
    gf_path = args.graph_features or f"{args.tmp_base}/{args.dataset}/graph_features_1000q.json"
    with open(gf_path) as f:
        graph_features = json.load(f)

    logger.info("  Initializing aggregator...")
    # Get experiment config
    if args.config not in EXPERIMENT_CONFIGS:
        logger.error(f"Unknown config '{args.config}'. Available: {list(EXPERIMENT_CONFIGS.keys())}")
        return
    exp_cfg = EXPERIMENT_CONFIGS[args.config]
    agg_mode = exp_cfg["agg_mode"]

    # P2: Adaptive Aggregator
    agg_config = AggregationConfig(
        mode=agg_mode, k_min=args.k_min, k_max=args.k_max,
        backpressure_threshold=args.bp_threshold, backpressure_k_max=args.bp_k_max,
        k_mapping=args.k_mapping, sigmoid_steepness=args.sigmoid_steepness,
        bp_decay=args.bp_decay,
    )
    if args.diff_alpha is not None:
        agg_config.diff_alpha = args.diff_alpha
    if args.diff_beta is not None:
        agg_config.diff_beta = args.diff_beta
    if args.diff_gamma is not None:
        agg_config.diff_gamma = args.diff_gamma
    if "fixed_k" in exp_cfg:
        agg_config.fixed_k = exp_cfg["fixed_k"]
    aggregator = AdaptiveAggregator(gf_path, agg_config)

    # P3: Priority Calculator
    priority_cfg = JointPriorityConfig(
        strategy=exp_cfg["strategy"],
        slo_ttft_ms=args.slo_ttft_ms,
        graph_weight=exp_cfg["graph_weight"],
        slo_weight=exp_cfg["slo_weight"],
        cost_weight=exp_cfg["cost_weight"],
        use_difficulty_signal=exp_cfg["use_difficulty_signal"],
    )
    priority_calc = GraphEnhancedPriorityCalculator(
        config=priority_cfg,
        graph_features=graph_features,
        difficulty_scores=aggregator.difficulty_scores,
    )

    # Generate query order and arrival times
    total_requests = args.total_requests if args.total_requests > 0 else n_unique
    total_requests = max(total_requests, n_unique)

    if args.arrival == "zipf":
        ranks = np.arange(1, n_unique + 1)
        weights = 1.0 / np.power(ranks, args.zipf_alpha)
        weights /= weights.sum()
        query_order = np.random.choice(n_unique, size=total_requests, p=weights)
    else:
        if total_requests <= n_unique:
            query_order = np.arange(total_requests)
        else:
            query_order = np.tile(np.arange(n_unique), (total_requests // n_unique) + 1)[:total_requests]

    if args.arrival == "sequential":
        arrival_times = np.zeros(total_requests)
    else:
        inter = np.random.exponential(1.0 / args.rate, size=total_requests)
        inter[0] = 0
        arrival_times = np.cumsum(inter)

    # Get model name & server metadata (P0.2)
    server_meta = {}
    try:
        req = urllib.request.Request(f"http://{args.host}:{args.port}/v1/models")
        auth_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if auth_key:
            req.add_header("Authorization", f"Bearer {auth_key}")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        resp = opener.open(req, timeout=5)
        model_name = json.loads(resp.read())["data"][0]["id"]
    except Exception:
        model_name = "default"
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        resp = opener.open(f"http://{args.host}:{args.port}/get_server_info", timeout=5)
        sinfo = json.loads(resp.read())
        server_meta["schedule_policy"] = sinfo.get("schedule_policy", "unknown")
        server_meta["enable_priority_scheduling"] = sinfo.get("enable_priority_scheduling", "unknown")
        server_meta["server_version"] = sinfo.get("version", "unknown")
    except Exception:
        server_meta["schedule_policy"] = "unavailable"
        server_meta["enable_priority_scheduling"] = "unavailable"
    # Git hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"
    server_meta["git_hash"] = git_hash
    # Retrieval file hash
    try:
        h = hashlib.md5(open(rr_path, "rb").read()).hexdigest()
        server_meta["retrieval_file_hash"] = h
        server_meta["retrieval_file_path"] = rr_path
    except Exception:
        server_meta["retrieval_file_hash"] = "unknown"

    # Parse heterogeneous SLO classes
    slo_classes = []
    if args.heterogeneous_slo:
        for cls_str in args.slo_classes.split(","):
            parts = cls_str.strip().split(":")
            slo_classes.append({
                "deadline_ms": float(parts[0]),
                "proportion": float(parts[1]),
                "weight": float(parts[2]),
                "class_name": f"slo_{parts[0]}ms",
            })
        logger.info(f"  Heterogeneous SLO classes: {slo_classes}")

    # Apply signal_mask to priority calculator weights
    if args.signal_mask != "all":
        mask = args.signal_mask
        if mask == "slo_only":
            priority_cfg.graph_weight = 0
            priority_cfg.cost_weight = 0
        elif mask == "cost_only":
            priority_cfg.graph_weight = 0
            priority_cfg.slo_weight = 0
        elif mask == "graph_only":
            priority_cfg.slo_weight = 0
            priority_cfg.cost_weight = 0
        elif mask == "slo_cost":
            priority_cfg.graph_weight = 0
        elif mask == "none":
            priority_cfg.graph_weight = 0
            priority_cfg.slo_weight = 0
            priority_cfg.cost_weight = 0
        logger.info(f"  signal_mask={mask} → g={priority_cfg.graph_weight} "
                     f"s={priority_cfg.slo_weight} c={priority_cfg.cost_weight}")

    logger.info(f"E2E Joint Benchmark: config={args.config} dataset={args.dataset}")
    logger.info(f"  P2 mode={agg_mode}, P3 strategy={exp_cfg['strategy']}")
    logger.info(f"  arrival={args.arrival}@{args.rate}qps, unique={n_unique}, total={total_requests}")
    logger.info(f"  model={model_name}, concurrency={args.concurrency}")
    if args.model_name_tag:
        logger.info(f"  model_tag={args.model_name_tag}")

    # ===================== Run benchmark =====================

    async def run():
        url = f"http://{args.host}:{args.port}/v1/chat/completions"
        queue = QueueTracker(args.concurrency)
        sem = asyncio.Semaphore(args.concurrency)
        connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
        timeout = aiohttp.ClientTimeout(total=300)
        results = []

        # Pre-build prompts for each unique query (with K=20 initial, will be re-pruned per request)
        # We cache the doc indices per query
        all_doc_indices = {}
        all_doc_scores = {}
        for local_idx in range(n_unique):
            data_idx = selected_indices[local_idx]
            actual_qid = qids_to_run[data_idx]
            all_doc_indices[local_idx] = list(np.asarray(retrieved_docs[actual_qid]["ids"]))
            all_doc_scores[local_idx] = retrieved_docs[actual_qid].get("scores", None)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=False,
        ) as session:
            async def send_one(idx):
                local_qid = int(query_order[idx])
                qid = int(selected_indices[local_qid])
                scheduled = arrival_times[idx]

                # Wait for scheduled time
                now = time.perf_counter()
                wait = scheduled - (now - t0)
                if wait > 0:
                    await asyncio.sleep(wait)

                # Heterogeneous SLO assignment
                query_slo_ms = args.slo_ttft_ms
                query_slo_weight = 1.0
                query_slo_class = "default"
                if args.heterogeneous_slo and slo_classes:
                    rng = np.random.RandomState(args.seed + idx)
                    r_val = rng.random()
                    cumulative = 0.0
                    for cls in slo_classes:
                        cumulative += cls["proportion"]
                        if r_val < cumulative:
                            query_slo_ms = cls["deadline_ms"]
                            query_slo_weight = cls["weight"]
                            query_slo_class = cls["class_name"]
                            break

                # P3→P2 feedback: current queue utilization
                queue_util = queue.utilization

                # P2: Adaptive K with backpressure
                doc_indices = all_doc_indices[local_qid]
                doc_scores = all_doc_scores.get(local_qid, None)
                selected_docs, agg_meta = aggregator.select_docs(qid, doc_indices, queue_util, doc_scores=doc_scores)
                k_used = agg_meta["k_decided"]
                diff_score = agg_meta["difficulty_score"]
                if args.custom_k_policy == "random75_k3_k20":
                    rng = np.random.RandomState(args.seed + idx)
                    k_used = 3 if rng.random() < 0.75 else 20
                    selected_docs = doc_indices[:k_used]
                    agg_meta["k_decided"] = k_used
                    agg_meta["mode_used"] = "custom_random75_k3_k20"
                elif args.custom_k_policy == "random40_k3_k20":
                    rng = np.random.RandomState(args.seed + idx)
                    k_used = 3 if rng.random() < 0.40 else 20
                    selected_docs = doc_indices[:k_used]
                    agg_meta["k_decided"] = k_used
                    agg_meta["mode_used"] = "custom_random40_k3_k20"
                elif args.custom_k_policy == "inverted_hard_k3_easy_k20":
                    is_hard = bool(hard_label_by_idx.get(qid, False))
                    k_used = 3 if is_hard else 20
                    selected_docs = doc_indices[:k_used]
                    agg_meta["k_decided"] = k_used
                    agg_meta["mode_used"] = "custom_inverted_hard_k3_easy_k20"

                # Build prompt
                question = str(data_df.iloc[qid].get("question", ""))
                contexts = [(str(corpus_df.iloc[int(d)]["title"]), str(corpus_df.iloc[int(d)]["text"]))
                            for d in selected_docs]
                prompt = build_prompt(question, contexts, args.dataset)
                pt = estimate_prompt_tokens(prompt)

                # P3: Compute priority (with optional dynamic weights for B5)
                use_dynamic = exp_cfg.get("dynamic_weights", False)
                if use_dynamic:
                    progress = idx / max(total_requests, 1)
                    if progress < 0.3:
                        priority_calc.config.graph_weight = 2.0
                        priority_calc.config.slo_weight = 0.5
                    else:
                        priority_calc.config.graph_weight = 0.5
                        priority_calc.config.slo_weight = 2.0

                # Override SLO in priority calc for heterogeneous mode
                orig_slo = priority_calc.config.slo_ttft_ms
                if args.heterogeneous_slo:
                    priority_calc.config.slo_ttft_ms = query_slo_ms

                priority, _ = priority_calc.compute_priority(qid, pt)

                if args.heterogeneous_slo:
                    priority_calc.config.slo_ttft_ms = orig_slo

                if args.engine == "vllm":
                    if not args.skip_priority:
                        priority = max(0, 20 - priority)
                    else:
                        priority = 0

                async with sem:
                    await queue.acquire()
                    try:
                        r = await send_request(
                            session, url, model_name, prompt, qid, priority, k_used,
                            diff_score, queue_util, query_slo_ms, args.max_tokens,
                            args.api_key,
                        )
                        r.slo_deadline_ms = query_slo_ms
                        r.slo_weight = query_slo_weight
                        r.slo_class = query_slo_class
                    finally:
                        await queue.release()
                    return r

            t0 = time.perf_counter()
            tasks = [asyncio.create_task(send_one(i)) for i in range(total_requests)]

            done = 0
            for coro in asyncio.as_completed(tasks):
                r = await coro
                results.append(r)
                done += 1
                if done % 50 == 0 or done == total_requests:
                    ok = [x for x in results if x.error is None and x.ttft > 0]
                    avg_ttft = np.mean([x.ttft for x in ok]) * 1000 if ok else 0
                    slo_met = sum(1 for x in ok if x.slo_met)
                    logger.info(f"  [{done}/{total_requests}] ok={len(ok)} "
                                f"avg_ttft={avg_ttft:.0f}ms SLO={slo_met}/{len(ok)}")

        results.sort(key=lambda r: r.send_time)
        return results, time.perf_counter() - t0

    results, wall_clock = asyncio.run(run())

    # ===================== Evaluate =====================

    ok = [r for r in results if r.error is None and r.ttft > 0]
    errors = len(results) - len(ok)

    if not ok:
        logger.error("No successful requests!")
        return

    # Dedup predictions per query
    pred_map = {}
    for r in ok:
        if r.query_id not in pred_map:
            pred_map[r.query_id] = r.generation
    eval_qids = sorted(pred_map.keys())
    if eval_qids:
        metrics = calculate_metric_scores(
            answers=[str(data_df.iloc[q].get("answer", "")) for q in eval_qids],
            predictions=[pred_map[q] for q in eval_qids],
        )
        f1 = metrics["pooled_results"]["F1"]
        acc = metrics["pooled_results"]["ACC"]
    else:
        f1 = acc = 0.0

    ttfts = np.array([r.ttft for r in ok])
    e2es = np.array([r.e2e_latency for r in ok])
    slo_met = sum(1 for r in ok if r.slo_met)
    slo_rate = 100.0 * slo_met / len(ok)
    k_values = [r.k_used for r in ok]
    priorities = [r.priority for r in ok]

    # Per-request F1 for QA-Goodput — reuse specific_results from calculate_metric_scores
    # to ensure per-request F1 is consistent with summary F1 (same extraction logic)
    per_request_f1 = {}
    if eval_qids:
        for i, qid in enumerate(eval_qids):
            per_request_f1[qid] = metrics["specific_results"][i]["F1"]

    # QA-Goodput: soft continuous form (paper Eq. qa_goodput)
    #   J = (1/N) * Σ F1_i * I(TTFT_i ≤ τ)
    # Also report hard-gate version with F1 > 0.5 threshold for reference.
    qa_goodput_soft_sum = 0.0
    goodput_count = 0        # hard-gate: F1 > 0.5 AND SLO met
    qa_correct_count = 0     # F1 > 0.5 (regardless of SLO)
    for r in ok:
        qf1 = per_request_f1.get(r.query_id, 0.0)
        if qf1 > 0.5:
            qa_correct_count += 1
        if r.slo_met:
            qa_goodput_soft_sum += qf1
            if qf1 > 0.5:
                goodput_count += 1
    qa_goodput = 100.0 * qa_goodput_soft_sum / len(ok) if ok else 0  # soft form (%)
    qa_goodput_hard = 100.0 * goodput_count / len(ok) if ok else 0   # hard gate F1>0.5

    # Write per-request F1 back into result objects for report
    for r in results:
        r.f1 = per_request_f1.get(r.query_id, 0.0)

    # Heterogeneous SLO metrics
    weighted_slo_pct = None
    per_class_slo = {}
    if args.heterogeneous_slo:
        total_w = sum(r.slo_weight for r in ok)
        weighted_hit = sum(r.slo_weight for r in ok if r.slo_met)
        weighted_slo_pct = round(100.0 * weighted_hit / total_w, 2) if total_w > 0 else 0

        class_stats = {}
        for r in ok:
            cls = r.slo_class
            if cls not in class_stats:
                class_stats[cls] = {"total": 0, "met": 0, "deadline_ms": r.slo_deadline_ms}
            class_stats[cls]["total"] += 1
            if r.slo_met:
                class_stats[cls]["met"] += 1
        for cls, st in class_stats.items():
            st["rate_pct"] = round(100.0 * st["met"] / st["total"], 2) if st["total"] > 0 else 0
        per_class_slo = class_stats

    # Print summary
    print(f"\n{'='*70}")
    print(f"  {args.exp_id} | {args.config} | {args.dataset}")
    print(f"  P2={agg_mode} P3={exp_cfg['strategy']} arrival={args.arrival}@{args.rate}qps")
    print(f"{'='*70}")
    print(f"  TTFT p50:    {np.percentile(ttfts, 50)*1000:.0f} ms")
    print(f"  TTFT p95:    {np.percentile(ttfts, 95)*1000:.0f} ms")
    print(f"  TTFT p99:    {np.percentile(ttfts, 99)*1000:.0f} ms")
    print(f"  SLO@{args.slo_ttft_ms:.0f}ms: {slo_met}/{len(ok)} ({slo_rate:.1f}%)")
    if weighted_slo_pct is not None:
        print(f"  Weighted-SLO: {weighted_slo_pct:.1f}%")
        for cls, st in sorted(per_class_slo.items()):
            print(f"    {cls}: {st['met']}/{st['total']} ({st['rate_pct']:.1f}%) [τ={st['deadline_ms']:.0f}ms]")
    print(f"  QA-Goodput:  {qa_goodput:.1f}% (soft F1-weighted)")
    print(f"  QA-GP-Hard:  {qa_goodput_hard:.1f}% (F1>0.5 gate)")
    print(f"  F1:          {f1:.4f}")
    print(f"  ACC:         {acc:.4f}")
    print(f"  avg K:       {np.mean(k_values):.1f}")
    print(f"  avg priority:{np.mean(priorities):.1f}")
    print(f"  QPS:         {len(ok)/wall_clock:.1f}")
    print(f"  errors:      {errors}")
    print(f"{'='*70}\n")

    # Save report
    out_dir = f"{args.tmp_base}/{args.dataset}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    summary_dict = {
        "ttft_p50_ms": round(np.percentile(ttfts, 50) * 1000, 1),
        "ttft_p95_ms": round(np.percentile(ttfts, 95) * 1000, 1),
        "ttft_p99_ms": round(np.percentile(ttfts, 99) * 1000, 1),
        "ttft_mean_ms": round(np.mean(ttfts) * 1000, 1),
        "slo_met": slo_met,
        "slo_total": len(ok),
        "slo_rate_pct": round(slo_rate, 2),
        "qa_goodput_pct": round(qa_goodput, 2),  # soft F1-weighted
        "qa_goodput_hard_pct": round(qa_goodput_hard, 2),  # hard gate F1>0.5
        "f1": round(f1, 4),
        "acc": round(acc, 4),
        "avg_k": round(np.mean(k_values), 2),
        "avg_priority": round(np.mean(priorities), 2),
        "k_distribution": dict(Counter(k_values).most_common()),
        "qps": round(len(ok) / wall_clock, 2),
        "wall_clock_s": round(wall_clock, 1),
        "errors": errors,
        "total_requests": total_requests,
        "unique_queries": n_unique,
        "qa_correct_count": qa_correct_count,
        "qa_goodput_count": goodput_count,
    }
    if weighted_slo_pct is not None:
        summary_dict["weighted_slo_pct"] = weighted_slo_pct
        summary_dict["per_class_slo"] = per_class_slo

    report = {
        "config_name": args.config,
        "exp_id": args.exp_id,
        "experiment_config": exp_cfg,
        "args": {k: v for k, v in vars(args).items()},
        "model_tag": args.model_name_tag or model_name,
        "signal_mask": args.signal_mask,
        "server_meta": server_meta,
        "summary": summary_dict,
        "per_request": [asdict(r) for r in results],
    }
    report_path = f"{out_dir}/e2e_{args.exp_id}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
