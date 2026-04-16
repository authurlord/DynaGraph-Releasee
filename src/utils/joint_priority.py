"""
Joint priority utilities for online multi-hop QA serving.

This module unifies three signals:
1) Graph signal: hot docs + bridge docs from retrieval traces.
2) SLO signal: urgency based on estimated prefill cost vs TTFT budget.
3) Cost signal: prompt length proxy for prefill workload.

The output is an integer priority per request, ready to be sent through
the OpenAI-compatible `priority` field.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class JointPriorityConfig:
    strategy: str = "none"
    slo_ttft_ms: float = 2000.0
    tokens_per_ms: float = 4.0
    bucket_size: int = 500
    hot_top_k: int = 100
    hot_bonus: int = 1
    bridge_bonus: int = 3
    bridge_min_freq: int = 2
    graph_weight: float = 1.0
    slo_weight: float = 1.0
    cost_weight: float = 0.0
    # Module 1/2 integration
    graph_features_path: str = None
    use_difficulty_signal: bool = False


def estimate_prompt_tokens(prompt: str) -> int:
    # A practical approximation used in existing scripts.
    return max(len(prompt) // 4, 1)


def _iter_query_ids(n_queries: int, query_ids: Optional[Sequence[int]]) -> Sequence[int]:
    return query_ids if query_ids is not None else range(n_queries)


def identify_hot_docs(
    retrieved_docs: Sequence[Sequence[int]],
    n_queries: int,
    top_k: int = 100,
    query_ids: Optional[Sequence[int]] = None,
) -> Set[int]:
    """Identify hot docs by frequency, weighted by replay order if provided."""
    freq: Counter = Counter()
    for qid in _iter_query_ids(n_queries, query_ids):
        freq.update(np.asarray(retrieved_docs[int(qid)]).tolist())
    return set(doc_id for doc_id, _ in freq.most_common(top_k))


def identify_bridge_docs(
    data_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    n_queries: int,
    min_freq: int = 2,
    query_ids: Optional[Sequence[int]] = None,
) -> Set[int]:
    """Map supporting_facts titles to corpus doc indices and keep frequent ones."""
    if "supporting_facts" not in data_df.columns:
        return set()

    title_to_doc = {row["title"]: idx for idx, row in corpus_df.iterrows()}
    sf_freq: Counter = Counter()

    for qid in _iter_query_ids(n_queries, query_ids):
        qid = int(qid)
        if qid >= len(data_df):
            continue
        sf = data_df.iloc[qid].get("supporting_facts")
        if sf is None:
            continue

        sf_titles = set()
        if isinstance(sf, list):
            for item in sf:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    sf_titles.add(str(item[0]))

        for title in sf_titles:
            doc_id = title_to_doc.get(title)
            if doc_id is not None:
                sf_freq[doc_id] += 1

    return {doc_id for doc_id, cnt in sf_freq.items() if cnt >= min_freq}


def _graph_score_for_query(
    query_docs: Sequence[int],
    hot_docs: Set[int],
    bridge_docs: Set[int],
    hot_bonus: int,
    bridge_bonus: int,
) -> int:
    score = 0
    for doc_id in query_docs:
        if doc_id in bridge_docs:
            score += bridge_bonus
        elif doc_id in hot_docs:
            score += hot_bonus
    return score


def _cost_bucket(tokens: int, bucket_size: int) -> int:
    return tokens // max(bucket_size, 1)


def _slo_urgency(tokens: int, slo_ttft_ms: float, tokens_per_ms: float) -> float:
    est_prefill_ms = tokens / max(tokens_per_ms, 1e-6)
    return min(est_prefill_ms / max(slo_ttft_ms, 1.0) * 10.0, 10.0)


def build_joint_priorities(
    prompts: Sequence[str],
    query_ids: Sequence[int],
    config: JointPriorityConfig,
    retrieved_docs: Optional[Sequence[Sequence[int]]] = None,
    data_df: Optional[pd.DataFrame] = None,
    corpus_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[int], List[int], Dict[str, object]]:
    """
    Build integer priorities for each request.

    Returns:
      priorities: integer priority per request (higher value = more urgent)
      estimated_tokens: estimated prompt tokens per request
      aux: debug stats for reporting
    """
    n_reqs = len(query_ids)
    prompt_tokens = [estimate_prompt_tokens(prompts[int(qid)]) for qid in query_ids]

    strategy = config.strategy
    if strategy == "none":
        priorities = [0] * n_reqs
        return priorities, prompt_tokens, {"strategy": strategy}

    if strategy == "cost_aware":
        priorities = [_cost_bucket(t, config.bucket_size) for t in prompt_tokens]
        return priorities, prompt_tokens, {"strategy": strategy}

    if strategy == "inverse_cost":
        max_tok = max(prompt_tokens) if prompt_tokens else 0
        priorities = [max((max_tok - t) // max(config.bucket_size, 1), 0) for t in prompt_tokens]
        return priorities, prompt_tokens, {"strategy": strategy}

    if strategy == "slo_deadline":
        priorities = [
            int(_slo_urgency(t, config.slo_ttft_ms, config.tokens_per_ms))
            for t in prompt_tokens
        ]
        return priorities, prompt_tokens, {"strategy": strategy}

    if strategy != "graph_slo":
        raise ValueError(f"Unsupported strategy: {strategy}")

    if retrieved_docs is None or data_df is None or corpus_df is None:
        raise ValueError("graph_slo requires retrieved_docs, data_df and corpus_df")

    n_queries = len(prompts)
    hot_docs = identify_hot_docs(
        retrieved_docs=retrieved_docs,
        n_queries=n_queries,
        top_k=config.hot_top_k,
        query_ids=query_ids,
    )
    bridge_docs = identify_bridge_docs(
        data_df=data_df,
        corpus_df=corpus_df,
        n_queries=n_queries,
        min_freq=config.bridge_min_freq,
        query_ids=query_ids,
    )

    raw_graph_scores = [
        _graph_score_for_query(
            query_docs=np.asarray(retrieved_docs[int(qid)]).tolist(),
            hot_docs=hot_docs,
            bridge_docs=bridge_docs,
            hot_bonus=config.hot_bonus,
            bridge_bonus=config.bridge_bonus,
        )
        for qid in query_ids
    ]
    graph_max = max(raw_graph_scores) if raw_graph_scores else 1
    graph_norm = [10.0 * s / max(graph_max, 1) for s in raw_graph_scores]
    slo_scores = [_slo_urgency(t, config.slo_ttft_ms, config.tokens_per_ms) for t in prompt_tokens]
    cost_scores = [_cost_bucket(t, config.bucket_size) for t in prompt_tokens]

    raw_joint = [
        config.graph_weight * graph_norm[i]
        + config.slo_weight * slo_scores[i]
        + config.cost_weight * cost_scores[i]
        for i in range(n_reqs)
    ]
    priorities = [int(round(x)) for x in raw_joint]

    aux = {
        "strategy": strategy,
        "hot_docs": len(hot_docs),
        "bridge_docs": len(bridge_docs),
        "hot_bridge_overlap": len(hot_docs & bridge_docs),
        "graph_score_max": graph_max,
        "graph_weight": config.graph_weight,
        "slo_weight": config.slo_weight,
        "cost_weight": config.cost_weight,
        "hot_bonus": config.hot_bonus,
        "bridge_bonus": config.bridge_bonus,
        "bridge_min_freq": config.bridge_min_freq,
    }
    return priorities, prompt_tokens, aux


class GraphEnhancedPriorityCalculator:
    """Priority calculator integrating Module 1 graph features + Module 2 difficulty.

    P(q) = w_slo * U_slo + w_graph * G_graph - w_cost * C_prefill
    where G_graph uses bridge_doc_ratio + entity_density from graph_features.json
    """

    def __init__(self, config: JointPriorityConfig,
                 graph_features: Optional[Dict] = None,
                 difficulty_scores: Optional[Dict] = None):
        self.config = config
        self.graph_features = graph_features or {}
        self.difficulty_scores = difficulty_scores or {}

    def _graph_signal(self, qid) -> float:
        """Graph signal from Module 1 features."""
        gf = self.graph_features.get(str(qid), {})
        if not gf:
            return 0.5
        bridge_ratio = gf.get("bridge_doc_ratio", 0)
        entity_density = min(gf.get("entity_density", 0) / 5.0, 1.0)
        n_bridge = min(gf.get("n_bridge_docs", 0) / 5.0, 1.0)
        return bridge_ratio * 0.5 + entity_density * 0.3 + n_bridge * 0.2

    def _slo_urgency(self, prompt_tokens: int) -> float:
        """SLO urgency: higher = closer to deadline."""
        est_ms = prompt_tokens / max(self.config.tokens_per_ms, 1e-6)
        return min(est_ms / max(self.config.slo_ttft_ms, 1.0), 1.0)

    def compute_priority(self, qid, prompt_tokens: int) -> Tuple[int, dict]:
        """Compute priority integrating graph + SLO + difficulty signals."""
        g = self._graph_signal(qid)
        u = self._slo_urgency(prompt_tokens)
        c = prompt_tokens / 5000.0
        d = self.difficulty_scores.get(str(qid), 0.5)

        # Difficulty amplifies SLO urgency: hard queries need faster service
        if self.config.use_difficulty_signal and d > 0.5:
            u *= (1.0 + (d - 0.5))  # up to 1.5x amplification

        raw = (
            self.config.graph_weight * g * 10
            + self.config.slo_weight * u * 10
            - self.config.cost_weight * c * 10
        )
        priority = max(0, int(round(raw)))
        return priority, {
            "graph_signal": round(g, 4),
            "slo_urgency": round(u, 4),
            "cost_signal": round(c, 4),
            "difficulty_score": round(d, 4),
            "priority_raw": priority,
        }

    def compute_batch(self, query_ids, prompt_tokens_list) -> Tuple[List[int], List[dict]]:
        priorities = []
        metas = []
        for qid, pt in zip(query_ids, prompt_tokens_list):
            p, m = self.compute_priority(qid, pt)
            priorities.append(p)
            metas.append(m)
        return priorities, metas
