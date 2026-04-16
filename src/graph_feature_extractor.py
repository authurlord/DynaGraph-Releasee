"""
Module 1: P1 Graph Topology Feature Extraction.

Extracts per-query graph topology feature vectors (V_topo) from:
  - retrieve_result_4B_{exp_id}.npy  (P1-P7 pipeline output)
  - triple_extraction_4B.npy         (KG triple extraction)
  - dataset JSON + corpus JSON       (for supporting_facts & text)

Output: graph_features_{dataset}.json — one entry per query with features
usable by Module 2 (difficulty scoring) and Module 3 (joint priority).

Reference: docs/M3_GraphRAG_分模块实验方案.md §Module 1
"""

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class GraphFeatureExtractor:
    """Extract graph topology features from retrieval results + KG triples."""

    def __init__(
        self,
        retrieve_result_path: str,
        triple_extraction_path: str,
        corpus_df: pd.DataFrame,
        data_df: pd.DataFrame,
    ):
        """
        Args:
            retrieve_result_path: .npy with dict containing 'retrieved_docs' etc.
            triple_extraction_path: .npy with dict keyed by corpus doc index (int),
                                    values are lists of [head, relation, tail].
            corpus_df: DataFrame with columns ['title', 'text'].
            data_df: DataFrame with query info including 'supporting_facts'.
        """
        rr = np.load(retrieve_result_path, allow_pickle=True).item()
        self.retrieved_docs: List[np.ndarray] = rr["retrieved_docs"]
        self.corpus_scores: List = rr.get("corpus_score", [])
        self.kg_lists: List = rr.get("kg_list", [])

        self.triples: Dict[int, List[List[str]]] = np.load(
            triple_extraction_path, allow_pickle=True
        ).item()

        self.corpus_df = corpus_df
        self.data_df = data_df

        # Build title -> corpus index mapping for supporting_facts resolution
        self.title_to_idx: Dict[str, int] = {
            row["title"]: idx for idx, row in corpus_df.iterrows()
        }

        logger.info(
            f"Loaded {len(self.retrieved_docs)} queries, "
            f"{len(self.triples)} doc triple entries, "
            f"{len(self.corpus_df)} corpus docs"
        )

    # ------------------------------------------------------------------
    # Per-doc entity extraction
    # ------------------------------------------------------------------

    def _get_doc_entities(self, doc_idx: int) -> Set[str]:
        """Get the set of entity names appearing in triples for a corpus doc."""
        triples = self.triples.get(doc_idx, [])
        entities: Set[str] = set()
        for triple in triples:
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                entities.add(str(triple[0]))  # head
                entities.add(str(triple[2]))  # tail
        return entities

    def _get_doc_triples_count(self, doc_idx: int) -> int:
        return len(self.triples.get(doc_idx, []))

    # ------------------------------------------------------------------
    # Supporting facts resolution
    # ------------------------------------------------------------------

    def _get_supporting_doc_indices(self, qid: int) -> Set[int]:
        """Resolve supporting_facts titles to corpus doc indices."""
        if qid >= len(self.data_df):
            return set()
        sf = self.data_df.iloc[qid].get("supporting_facts")
        if sf is None or not isinstance(sf, list):
            return set()
        indices: Set[int] = set()
        for item in sf:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                title = str(item[0])
                idx = self.title_to_idx.get(title)
                if idx is not None:
                    indices.add(idx)
        return indices

    # ------------------------------------------------------------------
    # Core: extract features for one query
    # ------------------------------------------------------------------

    def extract_for_query(self, qid: int) -> Dict[str, Any]:
        """Extract V_topo for a single query.

        Returns dict with all fields specified in the plan §1.2.
        """
        doc_indices: np.ndarray = np.asarray(self.retrieved_docs[qid])
        n_docs = len(doc_indices)

        # --- 1. Per-doc entity sets ---
        doc_entity_map: Dict[int, Set[str]] = {}
        for didx in doc_indices:
            didx = int(didx)
            doc_entity_map[didx] = self._get_doc_entities(didx)

        all_entities: Set[str] = set()
        for eset in doc_entity_map.values():
            all_entities.update(eset)
        n_unique_entities = len(all_entities)

        # --- 2. Triple count ---
        n_triples = sum(
            self._get_doc_triples_count(int(d)) for d in doc_indices
        )

        # --- 3. Entity density ---
        entity_density = n_unique_entities / max(n_docs, 1)

        # --- 4. Entity degree (across how many docs each entity appears) ---
        entity_doc_count: Counter = Counter()
        for eset in doc_entity_map.values():
            for e in eset:
                entity_doc_count[e] += 1
        max_entity_degree = (
            max(entity_doc_count.values()) if entity_doc_count else 0
        )

        # --- 5. Bridge docs ---
        # A bridge doc has ≥2 of its entities also appearing in other docs.
        # (Aligned with plan §1.3: shared_with_others is the set of entities
        #  in doc_i that also appear in at least one other doc.)
        bridge_doc_indices: List[int] = []
        doc_idx_list = [int(d) for d in doc_indices]
        for i, didx in enumerate(doc_idx_list):
            my_entities = doc_entity_map[didx]
            if not my_entities:
                continue
            shared_with_others: Set[str] = set()
            for j, other_didx in enumerate(doc_idx_list):
                if i != j:
                    shared = my_entities & doc_entity_map[other_didx]
                    shared_with_others.update(shared)
            if len(shared_with_others) >= 2:
                bridge_doc_indices.append(didx)
        n_bridge_docs = len(bridge_doc_indices)
        bridge_doc_ratio = n_bridge_docs / max(n_docs, 1)

        # --- 6. Entity overlap matrix density ---
        # Fraction of doc-pairs that share at least one entity
        n_pairs = n_docs * (n_docs - 1) // 2
        overlap_pairs = 0
        for i in range(len(doc_idx_list)):
            for j in range(i + 1, len(doc_idx_list)):
                if doc_entity_map[doc_idx_list[i]] & doc_entity_map[doc_idx_list[j]]:
                    overlap_pairs += 1
        entity_overlap_matrix_density = (
            overlap_pairs / n_pairs if n_pairs > 0 else 0.0
        )

        # --- 7. Average path hops (approximate) ---
        # Use BFS on the doc-entity bipartite graph to estimate avg shortest
        # path between doc pairs. For efficiency, approximate via connected
        # component analysis: if two docs share an entity, distance=1;
        # otherwise check 2-hop through shared entities.
        avg_path_hops = self._estimate_avg_path_hops(
            doc_idx_list, doc_entity_map
        )

        # --- 8. Supporting facts count ---
        sf_indices = self._get_supporting_doc_indices(qid)
        supporting_facts_count = len(sf_indices)

        # --- 9. Estimated prompt tokens ---
        estimated_prompt_tokens = self._estimate_tokens(qid, doc_indices)

        return {
            "qid": qid,
            "n_docs": n_docs,
            "n_unique_entities": n_unique_entities,
            "n_triples": n_triples,
            "entity_density": round(entity_density, 4),
            "max_entity_degree": max_entity_degree,
            "n_bridge_docs": n_bridge_docs,
            "bridge_doc_ratio": round(bridge_doc_ratio, 4),
            "entity_overlap_matrix_density": round(
                entity_overlap_matrix_density, 4
            ),
            "avg_path_hops": round(avg_path_hops, 4),
            "supporting_facts_count": supporting_facts_count,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "bridge_doc_ids": bridge_doc_indices,
            "hot_entity_ids": [
                e for e, _ in entity_doc_count.most_common(5)
            ],
            "difficulty_score": 0.0,  # placeholder, Module 2 fills this
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_avg_path_hops(
        self,
        doc_idx_list: List[int],
        doc_entity_map: Dict[int, Set[str]],
    ) -> float:
        """Estimate average shortest path between doc pairs via shared entities.

        Heuristic:
          - distance(d_i, d_j) = 1 if they share an entity directly
          - distance(d_i, d_j) = 2 if they share entities with a common
            intermediary doc
          - distance(d_i, d_j) = inf (treated as 3) otherwise
        """
        n = len(doc_idx_list)
        if n < 2:
            return 0.0

        # Build adjacency: doc_i <-> doc_j if they share an entity
        adj: Dict[int, Set[int]] = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                di, dj = doc_idx_list[i], doc_idx_list[j]
                if doc_entity_map[di] & doc_entity_map[dj]:
                    adj[i].add(j)
                    adj[j].add(i)

        total_hops = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    total_hops += 1  # direct connection
                elif adj[i] & adj[j]:
                    total_hops += 2  # 2-hop via intermediary
                else:
                    total_hops += 3  # disconnected, use sentinel
                count += 1

        return total_hops / max(count, 1)

    def _estimate_tokens(self, qid: int, doc_indices: np.ndarray) -> int:
        """Rough token count estimate: sum of doc text lengths / 4 + overhead."""
        total_chars = 0
        for didx in doc_indices:
            didx = int(didx)
            if didx < len(self.corpus_df):
                text = self.corpus_df.iloc[didx].get("text", "")
                total_chars += len(str(text))
        # ~4 chars per token + system/demo overhead (~200 tokens)
        return total_chars // 4 + 200

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_all(self, n_queries: Optional[int] = None) -> Dict[int, Dict]:
        """Extract features for all (or first n) queries."""
        total = len(self.retrieved_docs)
        if n_queries is not None:
            total = min(n_queries, total)
        features: Dict[int, Dict] = {}
        for qid in range(total):
            features[qid] = self.extract_for_query(qid)
            if (qid + 1) % 100 == 0:
                logger.info(f"Extracted {qid + 1}/{total} queries")
        logger.info(f"Extraction complete: {total} queries")
        return features

    @staticmethod
    def save(features: Dict[int, Dict], output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Convert int keys to str for JSON compatibility
        serializable = {str(k): v for k, v in features.items()}
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(features)} query features to {output_path}")

    @staticmethod
    def load(path: str) -> Dict[int, Dict]:
        with open(path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}


# ======================================================================
# CLI
# ======================================================================

def print_summary(features: Dict[int, Dict]) -> None:
    """Print statistical summary matching plan §1.5 verification."""
    summary_keys = [
        "n_bridge_docs",
        "bridge_doc_ratio",
        "entity_density",
        "entity_overlap_matrix_density",
        "estimated_prompt_tokens",
        "n_unique_entities",
        "max_entity_degree",
        "avg_path_hops",
        "n_triples",
        "supporting_facts_count",
    ]
    print(f"\n{'='*60}")
    print(f"  Summary for {len(features)} queries")
    print(f"{'='*60}")
    for key in summary_keys:
        vals = [f[key] for f in features.values() if key in f]
        if not vals:
            continue
        print(
            f"  {key:40s}: "
            f"mean={statistics.mean(vals):8.2f}  "
            f"std={statistics.stdev(vals) if len(vals) > 1 else 0:8.2f}  "
            f"min={min(vals):8.2f}  max={max(vals):8.2f}"
        )
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Module 1: Extract graph topology features (V_topo)"
    )
    parser.add_argument("--dataset", default="2wikimultihopqa")
    parser.add_argument("--questions", type=int, default=1000)
    parser.add_argument("--source_exp_id", default="baseline_v3")
    parser.add_argument("--triple_size", default="35B", help="Triple extraction model size (4B or 35B)")
    parser.add_argument(
        "--data_base",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        help="Base directory for dataset JSON files",
    )
    parser.add_argument(
        "--tmp_base",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp_data"),
        help="Base directory for pipeline intermediate files",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_dir = f"{args.tmp_base}/{args.dataset}"
    retrieve_path = f"{dataset_dir}/retrieve_result_4B_{args.source_exp_id}.npy"
    triple_path = f"{dataset_dir}/triple_extraction_{args.triple_size}.npy"
    corpus_path = f"{args.data_base}/{args.dataset}_corpus.json"
    data_path = f"{args.data_base}/{args.dataset}.json"

    logger.info(f"Dataset: {args.dataset}, questions: {args.questions}")
    logger.info(f"Retrieve result: {retrieve_path}")
    logger.info(f"Triple extraction: {triple_path}")

    corpus_df = pd.read_json(corpus_path)
    data_df = pd.read_json(data_path)

    extractor = GraphFeatureExtractor(
        retrieve_result_path=retrieve_path,
        triple_extraction_path=triple_path,
        corpus_df=corpus_df,
        data_df=data_df,
    )

    features = extractor.extract_all(n_queries=args.questions)

    out_path = args.output or f"{dataset_dir}/graph_features_{args.questions}q.json"
    extractor.save(features, out_path)

    print_summary(features)


if __name__ == "__main__":
    main()
