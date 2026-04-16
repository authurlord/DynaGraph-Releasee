"""
Module 2: P2 Difficulty-Aware Adaptive Aggregation.

Implements:
  - DifficultyScorer: graph-topo-based query difficulty estimation
  - EnhancedDifficultyScorer: graph + reranker-score-based difficulty estimation
  - BinaryDifficultyClassifier: binary easy/hard classifier using 18 features
  - AdaptiveAggregator: dynamic top-K doc selection per query

Modes:
  full            — keep all K=k_max docs (baseline)
  fixed_k         — fixed top-K by reranker score
  difficulty      — K = k_min + (k_max - k_min) * S_diff
  backpressure    — degrade K when queue utilization is high
  joint           — difficulty + backpressure
  score_gap       — per-difficulty K range + largest reranker score gap
  enhanced_joint  — enhanced difficulty (graph+score) + joint with per-class K ranges
  binary_adaptive — binary easy/hard classifier: easy→binary_easy_k, hard→binary_hard_k,
                    with optional backpressure degradation of hard queries to binary_medium_k
  adaptive_k      — [RQ4 baseline] Adaptive-k Retrieval (Taguchi et al., EMNLP'25):
                    training-free score-ratio thresholding. Keep docs while
                    score[i]/score[0] >= ratio_threshold. No graph signals.
  uniform_budget  — [RQ4 baseline] uniform budget reduction: K = round(k_max * budget_ratio)
                    for ALL queries (no per-query adaptation)

Reference: docs/M3_GraphRAG_分模块实验方案.md §Module 2
"""

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class AggregationConfig:
    """Aggregation strategy configuration."""
    mode: str = "full"          # full / fixed_k / difficulty / backpressure / joint / score_gap / enhanced_joint
    fixed_k: int = 20           # fixed_k mode K value
    k_min: int = 5              # difficulty mode minimum K
    k_max: int = 20             # difficulty mode maximum K
    diff_alpha: float = 0.4     # weight for bridge_doc_ratio
    diff_beta: float = 0.3      # weight for entity_overlap_matrix_density
    diff_gamma: float = 0.3     # weight for n_unique_entities (normalized)
    backpressure_threshold: float = 0.7   # queue util threshold for degradation
    backpressure_k_max: int = 8           # max K in degraded mode
    k_mapping: str = "linear"             # "linear" / "sigmoid"
    sigmoid_steepness: float = 10.0       # sigmoid curve steepness
    # Per-difficulty K ranges (for score_gap / enhanced_joint modes)
    easy_k_min: int = 5
    easy_k_max: int = 12
    medium_k_min: int = 10
    medium_k_max: int = 18
    hard_k_min: int = 15
    hard_k_max: int = 25
    # Enhanced scorer: weight for reranker score features vs graph features
    score_weight: float = 0.5
    # Binary adaptive mode (binary_adaptive)
    binary_easy_k: int = 5          # K for easy queries
    binary_hard_k: int = 20         # K for hard queries
    binary_medium_k: int = 12       # K for hard queries under backpressure
    binary_threshold: float = 0.5   # classifier score threshold: >threshold → hard
    # binary_classifier_type: "score_top1" (threshold on reranker top-1 score),
    #   "n_triples" (threshold on graph n_triples), or "lr" (logistic regression)
    binary_classifier_type: str = "score_top1"
    # Per-dataset best thresholds from oracle analysis (used by score_top1 / n_triples)
    binary_feature_threshold: float = 4.0  # dataset-specific threshold value
    # Bridge-aware document assembly: promote bridge docs into top-K
    bridge_aware: bool = False
    bridge_max_promote: int = 3  # max bridge docs to promote from outside top-K
    # Adaptive-k baseline (RQ4): score-ratio thresholding
    adaptive_k_ratio: float = 0.3   # keep docs with score >= top_score * ratio
    # Uniform budget baseline (RQ4): budget_ratio * k_max for all queries
    uniform_budget_ratio: float = 0.5
    # Dynamic backpressure decay shape (E3): fixed / linear / exponential / step3
    bp_decay: str = "fixed"


class DifficultyScorer:
    """Graph-topology-based query difficulty scorer.

    S_diff ∈ [0, 1], higher = harder query = needs more context.
    """

    def __init__(self, graph_features: Dict[str, dict]):
        self.features = graph_features
        self._norm_keys = [
            "n_unique_entities",
            "entity_overlap_matrix_density",
            "bridge_doc_ratio",
            "n_bridge_docs",
            "estimated_prompt_tokens",
        ]
        self.norm_params: Dict[str, Dict[str, float]] = {}
        self._compute_normalization()

    def _compute_normalization(self) -> None:
        """Compute min-max normalization parameters across all queries."""
        for key in self._norm_keys:
            vals = [f[key] for f in self.features.values() if key in f]
            if not vals:
                self.norm_params[key] = {"min": 0, "max": 1, "mean": 0.5, "std": 0}
                continue
            self.norm_params[key] = {
                "min": min(vals),
                "max": max(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    def _norm(self, key: str, value: float) -> float:
        p = self.norm_params[key]
        if p["max"] == p["min"]:
            return 0.5
        return (value - p["min"]) / (p["max"] - p["min"])

    def score(self, qid: str, config: Optional[AggregationConfig] = None) -> float:
        """Compute difficulty score for a single query."""
        if config is None:
            config = AggregationConfig()
        f = self.features.get(str(qid), self.features.get(int(qid), {}))
        if not f:
            return 0.5  # fallback for missing queries

        s = (
            config.diff_alpha * self._norm("bridge_doc_ratio", f.get("bridge_doc_ratio", 0))
            + config.diff_beta * self._norm("entity_overlap_matrix_density", f.get("entity_overlap_matrix_density", 0))
            + config.diff_gamma * self._norm("n_unique_entities", f.get("n_unique_entities", 0))
        )
        return min(max(s, 0.0), 1.0)

    def score_all(self, config: Optional[AggregationConfig] = None) -> Dict[str, float]:
        return {qid: self.score(qid, config) for qid in self.features}

    def get_distribution(self, config: Optional[AggregationConfig] = None) -> Dict[str, int]:
        """Get difficulty score distribution in 3 bins: easy/medium/hard."""
        scores = self.score_all(config)
        dist = {"easy(<0.3)": 0, "medium(0.3-0.7)": 0, "hard(>0.7)": 0}
        for s in scores.values():
            if s < 0.3:
                dist["easy(<0.3)"] += 1
            elif s > 0.7:
                dist["hard(>0.7)"] += 1
            else:
                dist["medium(0.3-0.7)"] += 1
        return dist


class EnhancedDifficultyScorer:
    """Graph + reranker-score-based difficulty scorer.

    Combines graph topology features with reranker score distribution features
    for more accurate difficulty estimation. Oracle analysis showed reranker
    score features (score_gap_max ρ=-0.37, score_top1 ρ=-0.26) are stronger
    predictors of oracle K than graph features alone (best ρ=0.25).

    S_enhanced ∈ [0, 1], higher = harder query = needs more context.
    """

    def __init__(self, graph_features: Dict[str, dict],
                 corpus_score: Optional[np.ndarray] = None,
                 doc_pool: Optional[list] = None):
        """
        Args:
            graph_features: per-query graph topology features
            corpus_score: [n_queries, n_corpus] reranker scores (optional)
            doc_pool: [n_queries] list of doc index arrays (optional)
        """
        self.graph_features = graph_features
        self.corpus_score = corpus_score
        self.doc_pool = doc_pool
        self._graph_scorer = DifficultyScorer(graph_features)

        # Pre-compute reranker score features for all queries
        self._score_features: Dict[str, Dict[str, float]] = {}
        if corpus_score is not None and doc_pool is not None:
            self._compute_score_features()

        # Normalization params for score features
        self._score_norm: Dict[str, Dict[str, float]] = {}
        if self._score_features:
            for key in ["score_gap_max", "score_top1", "score_gap_max_pos"]:
                vals = [f[key] for f in self._score_features.values() if key in f]
                if vals:
                    self._score_norm[key] = {
                        "min": min(vals), "max": max(vals),
                    }
                else:
                    self._score_norm[key] = {"min": 0, "max": 1}

    def _compute_score_features(self) -> None:
        """Extract reranker score distribution features per query."""
        n = min(len(self.corpus_score), len(self.doc_pool))
        for qid in range(n):
            doc_indices = np.asarray(self.doc_pool[qid])
            cs_q = np.array(self.corpus_score[qid])
            doc_scores = np.array([cs_q[int(d)] for d in doc_indices])
            sorted_scores = np.sort(doc_scores)[::-1]
            ns = len(sorted_scores)

            feats = {}
            feats["score_top1"] = float(sorted_scores[0]) if ns > 0 else 0.0
            if ns >= 2:
                gaps = [float(sorted_scores[i] - sorted_scores[i + 1])
                        for i in range(ns - 1)]
                feats["score_gap_max"] = max(gaps)
                feats["score_gap_max_pos"] = int(np.argmax(gaps)) + 1
            else:
                feats["score_gap_max"] = 0.0
                feats["score_gap_max_pos"] = 1
            self._score_features[str(qid)] = feats

    def _norm_score(self, key: str, value: float) -> float:
        p = self._score_norm.get(key, {"min": 0, "max": 1})
        if p["max"] == p["min"]:
            return 0.5
        return max(0.0, min(1.0, (value - p["min"]) / (p["max"] - p["min"])))

    def score(self, qid: str, config: Optional[AggregationConfig] = None) -> float:
        """Compute enhanced difficulty score combining graph and score features."""
        if config is None:
            config = AggregationConfig()

        s_graph = self._graph_scorer.score(qid, config)

        sf = self._score_features.get(str(qid))
        if sf is None:
            return s_graph  # fallback to graph-only

        # Score-based difficulty: higher = harder
        # Large gap = easy (good separation), so invert
        # High top1 = easy (strong evidence), so invert
        # High gap_pos = hard (gap far from top), keep as-is
        s_score = (
            0.4 * (1.0 - self._norm_score("score_gap_max", sf["score_gap_max"]))
            + 0.3 * (1.0 - self._norm_score("score_top1", sf["score_top1"]))
            + 0.3 * self._norm_score("score_gap_max_pos", sf["score_gap_max_pos"])
        )

        # Blend graph and score features
        s_enhanced = (1.0 - config.score_weight) * s_graph + config.score_weight * s_score
        return min(max(s_enhanced, 0.0), 1.0)

    def score_all(self, config: Optional[AggregationConfig] = None) -> Dict[str, float]:
        all_qids = set(self.graph_features.keys())
        all_qids.update(self._score_features.keys())
        return {qid: self.score(qid, config) for qid in all_qids}

    def get_distribution(self, config: Optional[AggregationConfig] = None) -> Dict[str, int]:
        scores = self.score_all(config)
        dist = {"easy(<0.3)": 0, "medium(0.3-0.7)": 0, "hard(>0.7)": 0}
        for s in scores.values():
            if s < 0.3:
                dist["easy(<0.3)"] += 1
            elif s > 0.7:
                dist["hard(>0.7)"] += 1
            else:
                dist["medium(0.3-0.7)"] += 1
        return dist


class BinaryDifficultyClassifier:
    """Binary easy/hard classifier for Phase 2 adaptive K selection.

    Classification strategy (from oracle analysis of 100-query QA oracle data):
    - 2Wiki: best single feature = score_top1 (threshold ~4.85, F1=0.444)
      80% easy / 20% hard; features are weak overall.
    - MuSiQue: best single feature = n_triples (threshold ~196, F1=0.603)
      59% easy / 41% hard; graph features are more informative.

    Supported classifier_type values:
      "score_top1"  — threshold on reranker top-1 score (inverted: low → hard)
      "n_triples"   — threshold on graph n_triples count (high → hard)
      "lr"          — logistic regression on 18 features (requires fitting first)

    The LR classifier is trained offline (binary_classifier_analysis.py) and
    stored as coefficient vectors, applied here at inference time without sklearn.
    """

    # LR coefficients fit on 100-query oracle data (binary_classifier_analysis.py)
    # Standardized features: z = (x - mean) / std, then logit = sum(coef * z) + intercept
    # Predict hard (1) if sigmoid(logit) > binary_threshold
    _LR_PARAMS: Dict[str, Dict] = {
        "2wikimultihopqa": {
            "coef": {
                "bridge_doc_ratio": 0.237948,
                "entity_overlap_matrix_density": 0.186481,
                "n_unique_entities": -0.136571,
                "n_bridge_docs": 0.12899,
                "avg_path_hops": -0.147682,
                "n_triples": -0.767616,
                "entity_density": -0.136571,
                "max_entity_degree": -0.293498,
                "score_top1": -0.588802,
                "score_top3_mean": -0.856249,
                "score_top5_mean": -0.268044,
                "score_top10_mean": 0.216044,
                "score_mean_all": 0.274956,
                "score_std": 1.020106,
                "score_gap_max": -0.310048,
                "score_gap_max_pos": -0.653921,
                "score_gap_top5_max": 0.166726,
                "score_decay_rate": 0.153905,
            },
            "intercept": -1.464029,
            "scaler_mean": {
                "bridge_doc_ratio": 0.2277,
                "entity_overlap_matrix_density": 0.0785,
                "n_unique_entities": 174.84,
                "n_bridge_docs": 4.42,
                "avg_path_hops": 2.7248,
                "n_triples": 185.19,
                "entity_density": 8.742,
                "max_entity_degree": 4.05,
                "score_top1": 4.4908,
                "score_top3_mean": 4.0523,
                "score_top5_mean": 3.2969,
                "score_top10_mean": 2.5584,
                "score_mean_all": 1.4538,
                "score_std": 1.1481,
                "score_gap_max": 1.5253,
                "score_gap_max_pos": 2.78,
                "score_gap_top5_max": 1.3005,
                "score_decay_rate": 0.4904,
            },
            "scaler_std": {
                "bridge_doc_ratio": 0.0948,
                "entity_overlap_matrix_density": 0.0267,
                "n_unique_entities": 71.79,
                "n_bridge_docs": 1.896,
                "avg_path_hops": 0.2286,
                "n_triples": 75.03,
                "entity_density": 3.589,
                "max_entity_degree": 1.584,
                "score_top1": 0.7085,
                "score_top3_mean": 0.7037,
                "score_top5_mean": 0.6218,
                "score_top10_mean": 0.5459,
                "score_mean_all": 0.3393,
                "score_std": 0.2011,
                "score_gap_max": 0.7613,
                "score_gap_max_pos": 2.128,
                "score_gap_top5_max": 0.6754,
                "score_decay_rate": 0.1234,
            },
        },
        "musique": {
            "coef": {
                "bridge_doc_ratio": -0.089266,
                "entity_overlap_matrix_density": -0.003696,
                "n_unique_entities": 0.174459,
                "n_bridge_docs": -0.089266,
                "avg_path_hops": -0.513882,
                "n_triples": 0.342017,
                "entity_density": 0.174459,
                "max_entity_degree": -0.859238,
                "score_top1": -0.36523,
                "score_top3_mean": -0.514363,
                "score_top5_mean": 0.180289,
                "score_top10_mean": 0.163007,
                "score_mean_all": 0.737384,
                "score_std": 0.277742,
                "score_gap_max": 0.054285,
                "score_gap_max_pos": 0.107284,
                "score_gap_top5_max": 0.515628,
                "score_decay_rate": 0.001716,
            },
            "intercept": -0.229804,
            "scaler_mean": {
                "bridge_doc_ratio": 0.1523,
                "entity_overlap_matrix_density": 0.0578,
                "n_unique_entities": 213.24,
                "n_bridge_docs": 3.047,
                "avg_path_hops": 2.5957,
                "n_triples": 221.21,
                "entity_density": 10.662,
                "max_entity_degree": 7.578,
                "score_top1": 3.5895,
                "score_top3_mean": 3.0483,
                "score_top5_mean": 2.3742,
                "score_top10_mean": 1.7286,
                "score_mean_all": 0.9026,
                "score_std": 0.9395,
                "score_gap_max": 1.0857,
                "score_gap_max_pos": 3.647,
                "score_gap_top5_max": 0.8729,
                "score_decay_rate": 0.3825,
            },
            "scaler_std": {
                "bridge_doc_ratio": 0.0783,
                "entity_overlap_matrix_density": 0.027,
                "n_unique_entities": 102.65,
                "n_bridge_docs": 1.567,
                "avg_path_hops": 0.4086,
                "n_triples": 107.64,
                "entity_density": 5.133,
                "max_entity_degree": 3.887,
                "score_top1": 0.8629,
                "score_top3_mean": 0.7927,
                "score_top5_mean": 0.7135,
                "score_top10_mean": 0.6168,
                "score_mean_all": 0.376,
                "score_std": 0.2705,
                "score_gap_max": 0.8046,
                "score_gap_max_pos": 2.777,
                "score_gap_top5_max": 0.6685,
                "score_decay_rate": 0.1283,
            },
        },
    }

    # Default LR params for unknown datasets (use 2wiki coefficients as fallback)
    _LR_DEFAULT_DATASET = "2wikimultihopqa"

    def __init__(
        self,
        graph_features: Dict[str, dict],
        corpus_score: Optional[np.ndarray] = None,
        doc_pool: Optional[list] = None,
        dataset: str = "2wikimultihopqa",
    ):
        self.graph_features = graph_features
        self.corpus_score = corpus_score
        self.doc_pool = doc_pool
        self.dataset = dataset

        # Pre-compute reranker score features (same as EnhancedDifficultyScorer)
        self._score_features: Dict[str, Dict[str, float]] = {}
        if corpus_score is not None and doc_pool is not None:
            self._compute_score_features()

    def _compute_score_features(self) -> None:
        """Extract reranker score distribution features per query."""
        n = min(len(self.corpus_score), len(self.doc_pool))
        for qid in range(n):
            doc_indices = np.asarray(self.doc_pool[qid])
            cs_q = np.array(self.corpus_score[qid])
            doc_scores = np.array([cs_q[int(d)] for d in doc_indices])
            sorted_scores = np.sort(doc_scores)[::-1]
            ns = len(sorted_scores)

            feats: Dict[str, float] = {}
            feats["score_top1"] = float(sorted_scores[0]) if ns > 0 else 0.0
            feats["score_top3_mean"] = float(np.mean(sorted_scores[:3])) if ns >= 3 else feats["score_top1"]
            feats["score_top5_mean"] = float(np.mean(sorted_scores[:5])) if ns >= 5 else feats["score_top1"]
            feats["score_top10_mean"] = float(np.mean(sorted_scores[:10])) if ns >= 10 else feats["score_top1"]
            feats["score_mean_all"] = float(np.mean(sorted_scores)) if ns > 0 else 0.0
            feats["score_std"] = float(np.std(sorted_scores)) if ns > 1 else 0.0
            if ns >= 2:
                gaps = [float(sorted_scores[i] - sorted_scores[i + 1]) for i in range(ns - 1)]
                feats["score_gap_max"] = max(gaps)
                feats["score_gap_max_pos"] = int(np.argmax(gaps)) + 1
                feats["score_gap_top5_max"] = max(gaps[:min(4, len(gaps))])
            else:
                feats["score_gap_max"] = 0.0
                feats["score_gap_max_pos"] = 1
                feats["score_gap_top5_max"] = 0.0
            if ns >= 2:
                feats["score_decay_rate"] = (feats["score_top1"] - float(np.mean(sorted_scores))) / max(feats["score_top1"], 1e-9)
            else:
                feats["score_decay_rate"] = 0.0
            self._score_features[str(qid)] = feats

    def _get_all_features(self, qid) -> Dict[str, float]:
        """Merge graph and score features for a query."""
        qid_str = str(qid)
        gf = self.graph_features.get(qid_str, self.graph_features.get(qid, {}))
        sf = self._score_features.get(qid_str, {})
        merged = {}
        for k in ["bridge_doc_ratio", "entity_overlap_matrix_density", "n_unique_entities",
                  "n_bridge_docs", "avg_path_hops", "n_triples", "entity_density", "max_entity_degree"]:
            merged[k] = float(gf.get(k, 0.0))
        for k in ["score_top1", "score_top3_mean", "score_top5_mean", "score_top10_mean",
                  "score_mean_all", "score_std", "score_gap_max", "score_gap_max_pos",
                  "score_gap_top5_max", "score_decay_rate"]:
            merged[k] = float(sf.get(k, 0.0))
        return merged

    def classify(self, qid, config: AggregationConfig) -> Tuple[bool, float]:
        """Classify query as hard (True) or easy (False).

        Returns:
            is_hard: True if query is classified as hard
            score: classifier confidence that query is hard [0, 1]
        """
        feats = self._get_all_features(qid)

        if config.binary_classifier_type == "score_top1":
            # Low score_top1 → hard (inverted signal)
            # Threshold is applied on raw score; hard if score_top1 <= threshold
            s = feats["score_top1"]
            is_hard = s <= config.binary_feature_threshold
            # Normalize to [0,1] probability: low score → high hard probability
            score = max(0.0, min(1.0, 1.0 - s / max(config.binary_feature_threshold * 2, 1e-9)))
            return is_hard, score

        elif config.binary_classifier_type == "n_triples":
            # High n_triples → hard
            s = feats["n_triples"]
            is_hard = s > config.binary_feature_threshold
            score = max(0.0, min(1.0, s / max(config.binary_feature_threshold * 2, 1e-9)))
            return is_hard, score

        elif config.binary_classifier_type == "lr":
            # Logistic regression on standardized 18 features
            lr_key = self.dataset if self.dataset in self._LR_PARAMS else self._LR_DEFAULT_DATASET
            lr = self._LR_PARAMS[lr_key]
            logit = lr["intercept"]
            for fname, coef in lr["coef"].items():
                raw_val = feats.get(fname, 0.0)
                mean = lr["scaler_mean"].get(fname, 0.0)
                std = lr["scaler_std"].get(fname, 1.0)
                if std < 1e-9:
                    std = 1.0
                z = (raw_val - mean) / std
                logit += coef * z
            import math
            prob_hard = 1.0 / (1.0 + math.exp(-logit))
            is_hard = prob_hard > config.binary_threshold
            return is_hard, prob_hard

        else:
            # Unknown type: fallback to hard (conservative)
            logger.warning(f"Unknown binary_classifier_type: {config.binary_classifier_type}, defaulting to hard")
            return True, 1.0

    def classify_all(self, config: AggregationConfig) -> Dict[str, Tuple[bool, float]]:
        """Classify all queries in graph_features."""
        results = {}
        all_qids = set(self.graph_features.keys())
        all_qids.update(self._score_features.keys())
        for qid in all_qids:
            results[str(qid)] = self.classify(qid, config)
        return results

    def get_distribution(self, config: Optional[AggregationConfig] = None) -> Dict[str, int]:
        """Get easy/hard distribution across all queries."""
        if config is None:
            config = AggregationConfig()
        classifications = self.classify_all(config)
        n_hard = sum(1 for is_hard, _ in classifications.values() if is_hard)
        n_easy = len(classifications) - n_hard
        return {"easy": n_easy, "hard": n_hard}


def _get_difficulty_range(s_diff: float, config: AggregationConfig) -> Tuple[int, int]:
    """Map difficulty score to per-class [k_min, k_max] range."""
    if s_diff < 0.3:
        return config.easy_k_min, config.easy_k_max
    elif s_diff > 0.7:
        return config.hard_k_min, config.hard_k_max
    else:
        return config.medium_k_min, config.medium_k_max


class AdaptiveAggregator:
    """Adaptive aggregator: decides how many docs to keep per query."""

    def __init__(
        self,
        graph_features_path: str,
        config: Optional[AggregationConfig] = None,
        corpus_score: Optional[np.ndarray] = None,
        doc_pool: Optional[list] = None,
        dataset: str = "2wikimultihopqa",
    ):
        with open(graph_features_path) as f:
            self.graph_features = json.load(f)
        self.config = config or AggregationConfig()
        self.corpus_score = corpus_score
        self.doc_pool = doc_pool
        self.dataset = dataset

        # Choose scorer based on mode
        if self.config.mode in ("score_gap", "enhanced_joint") and corpus_score is not None:
            self.scorer = EnhancedDifficultyScorer(
                self.graph_features, corpus_score, doc_pool,
            )
        else:
            self.scorer = DifficultyScorer(self.graph_features)
        self.difficulty_scores = self.scorer.score_all(self.config)

        # Binary classifier (for binary_adaptive mode)
        self.binary_classifier: Optional[BinaryDifficultyClassifier] = None
        if self.config.mode == "binary_adaptive":
            self.binary_classifier = BinaryDifficultyClassifier(
                self.graph_features, corpus_score, doc_pool, dataset=dataset,
            )
            bin_dist = self.binary_classifier.get_distribution(self.config)
            logger.info(
                f"AdaptiveAggregator: mode={self.config.mode}, "
                f"classifier_type={self.config.binary_classifier_type}, "
                f"threshold={self.config.binary_feature_threshold}, "
                f"{len(self.graph_features)} queries, "
                f"binary_dist={bin_dist}"
            )
        else:
            logger.info(
                f"AdaptiveAggregator: mode={self.config.mode}, "
                f"scorer={type(self.scorer).__name__}, "
                f"{len(self.graph_features)} queries, "
                f"difficulty dist={self.scorer.get_distribution(self.config)}"
            )

    def decide_k(self, qid, queue_utilization: float = 0.0,
                 doc_scores: Optional[List[float]] = None) -> Tuple[int, dict]:
        """Decide how many docs to keep for this query.

        Args:
            qid: query ID
            queue_utilization: current system queue utilization [0, 1]
            doc_scores: per-doc reranker scores in retrieval order (for score_gap mode)

        Returns:
            k: number of docs to keep
            meta: decision metadata
        """
        c = self.config
        qid_str = str(qid)
        s_diff = self.difficulty_scores.get(qid_str, self.difficulty_scores.get(qid, 0.5))

        if c.mode == "full":
            k = c.k_max
            mode_used = "full"

        elif c.mode == "fixed_k":
            k = c.fixed_k
            mode_used = "fixed_k"

        elif c.mode == "difficulty":
            if c.k_mapping == "sigmoid":
                import math
                s_stretched = 1 / (1 + math.exp(-c.sigmoid_steepness * (s_diff - 0.5)))
                k = int(round(c.k_min + (c.k_max - c.k_min) * s_stretched))
            else:
                k = int(round(c.k_min + (c.k_max - c.k_min) * s_diff))
            mode_used = f"difficulty_{c.k_mapping}"

        elif c.mode == "backpressure":
            if queue_utilization > c.backpressure_threshold:
                k = c.backpressure_k_max
                mode_used = "backpressure_frugal"
            else:
                k = c.k_max
                mode_used = "backpressure_normal"

        elif c.mode == "joint":
            import math
            if c.k_mapping == "sigmoid":
                s_stretched = 1 / (1 + math.exp(-c.sigmoid_steepness * (s_diff - 0.5)))
                s_mapped = s_stretched
            else:
                s_mapped = s_diff
            k_diff = int(round(c.k_min + (c.k_max - c.k_min) * s_mapped))
            if queue_utilization > c.backpressure_threshold:
                # Dynamic K_bp based on decay shape (E3)
                u_normalized = min(
                    (queue_utilization - c.backpressure_threshold)
                    / (1.0 - c.backpressure_threshold),
                    1.0,
                )
                if c.bp_decay == "fixed":
                    k_bp = float(c.backpressure_k_max)
                elif c.bp_decay == "linear":
                    k_bp = c.k_min + (c.k_max - c.k_min) * (1.0 - u_normalized)
                elif c.bp_decay == "exponential":
                    alpha = math.log(c.k_max / max(c.k_min, 1))
                    k_bp = c.k_min + (c.k_max - c.k_min) * math.exp(-alpha * u_normalized)
                elif c.bp_decay == "step3":
                    if queue_utilization < 0.8:
                        k_bp = float(c.k_max)
                    elif queue_utilization < 0.9:
                        k_bp = (c.k_min + c.k_max) / 2.0
                    else:
                        k_bp = float(c.k_min)
                else:
                    raise ValueError(f"Unknown bp_decay: {c.bp_decay}")
                k_bp = max(float(c.k_min), min(k_bp, float(c.k_max)))
                k = int(round(c.k_min + (k_bp - c.k_min) * s_mapped))
                mode_used = f"joint_frugal_{c.bp_decay}"
                k_bp_value = round(k_bp, 2)
            else:
                k = k_diff
                mode_used = "joint_normal"
                k_bp_value = float(c.k_max)

        elif c.mode == "score_gap":
            # Per-difficulty K range + largest reranker score gap
            k_lo, k_hi = _get_difficulty_range(s_diff, c)
            k = self._score_gap_k(doc_scores, k_lo, k_hi)
            mode_used = f"score_gap_{self._difficulty_class(s_diff)}"

        elif c.mode == "enhanced_joint":
            # Enhanced difficulty scorer + per-difficulty K ranges + backpressure
            k_lo, k_hi = _get_difficulty_range(s_diff, c)
            if c.k_mapping == "sigmoid":
                import math
                s_stretched = 1 / (1 + math.exp(-c.sigmoid_steepness * (s_diff - 0.5)))
                s_mapped = s_stretched
            else:
                s_mapped = s_diff
            k_diff = int(round(k_lo + (k_hi - k_lo) * s_mapped))
            if queue_utilization > c.backpressure_threshold:
                bp_k_hi = min(k_hi, c.backpressure_k_max)
                k = int(round(k_lo + (bp_k_hi - k_lo) * s_mapped))
                mode_used = f"enhanced_joint_frugal_{self._difficulty_class(s_diff)}"
            else:
                k = k_diff
                mode_used = f"enhanced_joint_{self._difficulty_class(s_diff)}"

        elif c.mode == "adaptive_k":
            # RQ4 baseline: Adaptive-k Retrieval (Taguchi et al., EMNLP'25)
            # Score-ratio thresholding: keep docs while score[i]/score[0] >= ratio
            k = self._adaptive_k_ratio(doc_scores, c.k_min, c.k_max, c.adaptive_k_ratio)
            mode_used = "adaptive_k_ratio"

        elif c.mode == "uniform_budget":
            # RQ4 baseline: uniform budget reduction for ALL queries
            k = max(c.k_min, min(int(round(c.k_max * c.uniform_budget_ratio)), c.k_max))
            mode_used = "uniform_budget"

        elif c.mode == "random":
            # EXP-3 baseline: random K per query (prove S_diff is not random)
            import random as _rnd
            k = _rnd.randint(c.k_min, c.k_max)
            mode_used = "random"

        elif c.mode == "binary_adaptive":
            # Binary classifier: easy → binary_easy_k, hard → binary_hard_k
            # Under backpressure: hard queries are degraded to binary_medium_k
            if self.binary_classifier is None:
                # Fallback if classifier was not initialized
                is_hard, bin_score = True, 1.0
            else:
                is_hard, bin_score = self.binary_classifier.classify(qid, c)

            if is_hard:
                if queue_utilization > c.backpressure_threshold:
                    k = c.binary_medium_k
                    mode_used = "binary_hard_bp"
                else:
                    k = c.binary_hard_k
                    mode_used = "binary_hard"
            else:
                k = c.binary_easy_k
                mode_used = "binary_easy"

            # Override the difficulty_score / class with binary classifier result
            s_diff = bin_score
            return max(c.k_min, min(k, c.k_max)), {
                "difficulty_score": round(bin_score, 4),
                "difficulty_class": "hard" if is_hard else "easy",
                "binary_is_hard": is_hard,
                "mode_used": mode_used,
                "k_decided": max(c.k_min, min(k, c.k_max)),
                "queue_utilization": queue_utilization,
            }

        else:
            raise ValueError(f"Unknown mode: {c.mode}")

        k = max(c.k_min, min(k, c.k_max))

        meta = {
            "difficulty_score": round(s_diff, 4),
            "difficulty_class": self._difficulty_class(s_diff),
            "mode_used": mode_used,
            "k_decided": k,
            "queue_utilization": queue_utilization,
        }
        if c.mode == "joint":
            meta["k_bp_value"] = k_bp_value
        return k, meta

    @staticmethod
    def _difficulty_class(s_diff: float) -> str:
        if s_diff < 0.3:
            return "easy"
        elif s_diff > 0.7:
            return "hard"
        return "medium"

    @staticmethod
    def _adaptive_k_ratio(
        doc_scores: Optional[List[float]],
        k_min: int, k_max: int,
        ratio: float = 0.3,
    ) -> int:
        """Adaptive-k baseline: keep docs with score >= top_score * ratio.

        Implements the core idea from Taguchi et al. (EMNLP'25):
        score-ratio thresholding without any graph or difficulty signals.
        """
        if doc_scores is None or len(doc_scores) < 1:
            return k_max  # fallback
        top_score = doc_scores[0]
        if top_score <= 0:
            return k_min
        threshold = top_score * ratio
        k = 0
        for i, s in enumerate(doc_scores):
            if s >= threshold:
                k = i + 1
            else:
                break
        return max(k_min, min(k, k_max))

    @staticmethod
    def _score_gap_k(doc_scores: Optional[List[float]],
                     k_lo: int, k_hi: int) -> int:
        """Find K by largest reranker score gap within [k_lo, k_hi]."""
        if doc_scores is None or len(doc_scores) < 2:
            return k_hi  # fallback

        # Look at gaps between positions k_lo-1 and k_hi-1
        # (positions are 0-indexed, K is 1-indexed)
        search_start = max(0, k_lo - 1)
        search_end = min(len(doc_scores) - 1, k_hi)

        if search_start >= search_end:
            return k_lo

        gaps = []
        for i in range(search_start, search_end):
            gap = doc_scores[i] - doc_scores[i + 1] if i + 1 < len(doc_scores) else 0.0
            gaps.append((gap, i + 1))  # (gap_value, cut_position as K)

        if not gaps:
            return k_hi

        # Find largest gap → cut there
        best_gap, best_k = max(gaps, key=lambda x: x[0])
        return max(k_lo, min(best_k, k_hi))

    def select_docs(
        self,
        qid,
        doc_indices: list,
        queue_utilization: float = 0.0,
        doc_scores: Optional[List[float]] = None,
    ) -> Tuple[list, dict]:
        """Select top-K docs from the retrieved list.

        Args:
            qid: query ID
            doc_indices: retrieved doc indices (assumed sorted by reranker score)
            queue_utilization: current system queue utilization [0, 1]
            doc_scores: per-doc reranker scores in same order as doc_indices

        Returns:
            selected_indices: top-K doc indices
            meta: decision metadata
        """
        k, meta = self.decide_k(qid, queue_utilization, doc_scores)
        if self.config.mode == "random":
            # Random mode: shuffle docs before selecting top-K
            import random as _rnd
            shuffled = list(doc_indices)
            _rnd.shuffle(shuffled)
            selected = shuffled[:k]
        else:
            selected = doc_indices[:k]

        # Bridge-aware assembly: promote bridge docs from outside top-K
        if self.config.bridge_aware and k < len(doc_indices):
            bridge_ids = set(
                self.graph_features.get(str(qid), {}).get("bridge_doc_ids", [])
            )
            if bridge_ids:
                topk_set = set(int(d) for d in selected)
                # Bridge docs in positions K..N that would be missed
                missed_bridges = [
                    d for d in doc_indices[k:]
                    if int(d) in bridge_ids and int(d) not in topk_set
                ]
                # Non-bridge docs in top-K that can be swapped out (from the tail)
                non_bridge_in_topk = [
                    (i, d) for i, d in enumerate(selected)
                    if int(d) not in bridge_ids
                ]
                n_promote = min(
                    len(missed_bridges),
                    len(non_bridge_in_topk),
                    self.config.bridge_max_promote,
                )
                if n_promote > 0:
                    # Remove the lowest-ranked non-bridge docs, append promoted bridge docs
                    remove_indices = set(
                        i for i, _ in non_bridge_in_topk[-n_promote:]
                    )
                    selected = [
                        d for i, d in enumerate(selected) if i not in remove_indices
                    ] + missed_bridges[:n_promote]
                    meta["bridge_promoted"] = n_promote

        meta["n_docs_before"] = len(doc_indices)
        meta["n_docs_after"] = len(selected)
        meta["token_reduction_pct"] = round(
            1 - len(selected) / max(len(doc_indices), 1), 4
        )
        return selected, meta
