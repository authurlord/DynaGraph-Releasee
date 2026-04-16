# DynaGraph — KG-Enhanced Multi-Hop QA Serving

DynaGraph is a KG-guided multi-hop QA serving system that jointly
adapts retrieval budget and request scheduling under an end-to-end
latency SLO. The accompanying paper (full version in
[`paper_full_version.pdf`](./paper_full_version.pdf)) introduces three
coupled modules that form the C5 configuration reported in the main
tables:

- **P1 — Graph topology features.** For every retrieved top-K set we
  extract `bridge_doc_ratio`, `entity_overlap_density`, and
  `n_unique_entities` from the LLM-extracted KG.
- **P2 — Adaptive-K aggregator.** A training-free linear scorer maps
  the graph features to a per-query K budget (`src/adaptive_aggregator.py`).
  Queue pressure from P3 decays K exponentially as load builds.
- **P3 — SLO-priority scheduler.** Graph-enhanced priority
  (`src/utils/joint_priority.py`) combines `S_diff` with estimated
  prompt tokens so that easy-but-short requests do not block hard
  queries near their SLO deadline.

The online benchmark (`src/benchmark_e2e_joint.py`) replays a Zipf
arrival process of 1000 unique questions × 3000 total requests at 30
QPS against a running vLLM / SGLang server and reports TTFT, SLO
attainment, F1, accuracy, and QA-Goodput.

> **Anonymous submission.** This repository is released for
> double-blind review. Please report any residual identifying strings
> via the submission portal.

---

## Contents

```
DynaGraph-Release/
├── src/                  # method + online benchmark + pipeline
├── scripts/              # server launchers + run wrappers
├── data/                 # raw 1000q corpora & queries (3 datasets)
├── tmp_data/             # graph features (committed); retrieval results via external zip
├── paper_full_version.pdf
├── downloads_manifest.md # external artifact list
└── requirements.txt
```

Three public datasets are bundled (`2wikimultihopqa`, `hotpotqa`,
`musique`). For each dataset `data/{name}.json` holds the 1000
questions and `data/{name}_corpus.json` holds the associated passages.

---

## Reproducibility modes

There are two paths. **We recommend the pure-QA path** for reviewing:
it skips Phase-1 retrieval entirely and reproduces the benchmark
numbers using bundled intermediate artifacts.

### Mode A — Pure-QA (recommended)

1. Clone the repo.
2. Unpack the bundled intermediate artifacts at the repo root:

   ```bash
   unzip dynagraph_intermediate_artifacts.zip -d .
   sha256sum -c SHA256SUMS.txt
   ```
3. Install dependencies: `pip install -r requirements.txt` plus one
   serving backend (vLLM 0.17+ **or** SGLang 0.4.5+).
4. Start an LLM server — see §"LLM server configurations" below.
5. Warm the server once (a single `curl /v1/models` is enough).
6. Run the benchmark:

   ```bash
   bash scripts/run_c5_all_datasets.sh qwen3.5-4b 8000 64 30 3000
   ```

This goes straight from the cached retrieval arrays to the online
benchmark. Expected wall-clock: ~3 minutes per dataset on dual
RTX 4090 with a warmed Qwen3.5-4B TP2 server.

### Mode B — End-to-End from scratch

If you want to regenerate every intermediate artifact, you also need an
embedding server (Qwen3-Embedding-0.6B) and a reranker server
(bge-reranker-v2-m3) next to the LLM server.

```bash
# Each in its own terminal:
MODEL_PATH=/path/to/Qwen3-Embedding-0.6B  bash scripts/start_embedding_server.sh
MODEL_PATH=/path/to/bge-reranker-v2-m3    bash scripts/start_reranker_server.sh
MODEL_PATH=/path/to/Qwen3.5-4B            bash scripts/start_vllm_qwen35_4b_tuned.sh

# Then rebuild KG triples, retrieval, and graph features for all 3 datasets:
bash scripts/run_pipeline_from_scratch.sh qwen3.5-4b 35B

# Finally run the benchmark:
bash scripts/run_c5_all_datasets.sh qwen3.5-4b 8000 64 30 3000
```

Wall-clock for Mode B on dual 4090 (all three datasets): several hours,
dominated by KG extraction and full-corpus retrieval.

---

## LLM server configurations

The released scripts expose the parameters we tuned for the paper.
Override paths with `MODEL_PATH=/absolute/path/to/model` — no model
weights live in this repo.

### vLLM (recommended)

#### Qwen3.5-4B — C5 anchor (main paper setting)

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
  bash scripts/start_vllm_qwen35_4b_tuned.sh
```

Key flags this script sets for you:

- tensor-parallel 2, FLASHINFER attention
- `gpu-memory-utilization 0.85`
- `max-model-len 23000`, `max-num-batched-tokens 8192`, `max-num-seqs 64`
- `kv-cache-dtype fp8`, `enable-prefix-caching`, `enable-chunked-prefill`
- `performance-mode throughput`
- `reasoning-parser qwen3` (required — Qwen3.5 defaults to thinking mode)

> **Warm the server once before any TTFT-sensitive measurement**, or
> the first request's CUDA-graph compile will dominate p99.

#### Qwen3.5-9B / 35B (characterisation sweeps)

```bash
MODEL_PATH=/path/to/Qwen3.5-9B \
  bash scripts/start_vllm_qwen35_tp2.sh 9B

MODEL_PATH=/path/to/Qwen3.5-35B-A3B-FP8 \
  bash scripts/start_vllm_qwen35_tp2.sh 35B
```

35B is a FP8 MoE model and uses `max-num-seqs=8`,
`performance-mode=interactivity`, `--enable-expert-parallel`,
`--all2all-backend allgather_reducescatter`, `kv-cache-dtype=auto`
for stability on dual 4090. This is a low-QPS characterisation config,
not the paper main setting.

#### Qwen3-4B-Instruct-2507 (historical baseline / LMCache-compatible)

```bash
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
  bash scripts/start_vllm_qwen3_4b.sh
```

### SGLang (alternative)

SGLang support is kept as a fallback; the main-paper numbers were
collected under vLLM because Qwen3.5's GDN / hybrid attention path is
most mature there.

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
  bash scripts/start_sglang_qwen35_4b.sh

MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 \
  bash scripts/start_sglang_qwen3_4b.sh
```

### Client-side note

Every local OpenAI-compatible client must bypass proxy env vars; the
code already sets `httpx.Client(trust_env=False)`. For Qwen3.5 models
the client must also pass
`extra_body={"chat_template_kwargs": {"enable_thinking": False}}` on
every request — our benchmark script does this automatically.

---

## Running the benchmark

### Full paper main setting (all 3 datasets, C5)

```bash
bash scripts/run_c5_all_datasets.sh qwen3.5-4b 8000 64 30 3000
```

### Single dataset, single config

```bash
python src/benchmark_e2e_joint.py \
  --dataset 2wikimultihopqa \
  --source_exp_id baseline_v3 \
  --config C5 \
  --arrival zipf --zipf_alpha 1.0 --rate 30 \
  --questions 1000 --total_requests 3000 \
  --concurrency 64 --slo_ttft_ms 2000 \
  --k_min 10 --k_max 20 \
  --bp_k_max 19 --bp_threshold 0.7 \
  --k_mapping linear --bp_decay exponential \
  --engine vllm --port 8000 --api_key api-key-qwen3 \
  --model_name_tag qwen3.5-4b \
  --exp_id C5_demo
```

`source_exp_id` must match the retrieval filename stored under
`tmp_data/{dataset}/retrieve_result_4B_{source_exp_id}.npy`. The three
defaults are:

| Dataset | `source_exp_id` |
|---|---|
| 2WikiMultihopQA | `baseline_v3` |
| HotpotQA | `baseline_full_1000q` |
| MuSiQue | `1` |

### Standalone P2 evaluation (no server needed)

For fast offline iteration of the adaptive-K logic:

```bash
bash scripts/run_standalone_eval.sh 2wikimultihopqa joint C5_standalone
```

---

## Expected results (Qwen3.5-4B TP2, Zipf@30 QPS, 3000 req)

These are the C5 numbers reported in the paper; small variation (±1 pp
F1, ±2 pp QA-GP) across server instances is normal and usually
traceable to cache-warmth variation.

| Dataset | SLO@2s | QA-Goodput | F1 | avg K | TTFT p50 |
|---|---|---|---|---|---|
| 2Wiki   | 98.5 % | 87.4 % | 0.796 | 14.3 | 255 ms |
| HotpotQA| 99.0 % | 86.5 % | 0.770 | 14.5 | 225 ms |
| MuSiQue | 98.4 % | 63.6 % | 0.492 | 14.2 | 190 ms |

---

## Scope of this release

This release focuses on the DynaGraph main method and the three-dataset
benchmark that drives Table 1–3 of the paper. The following are
**intentionally omitted** and documented only in the paper:

- Baselines (HippoRAG, SiReRAG, KET-RAG, CacheBlend/LMCache, DepCache,
  ClueRAG, CompSelect, RAG-CSM, Adaptive-k)
- RQ5 full-dev AUROC scalability (12 K queries)
- RQ6 QPS sweep and RQ7 model-size scaling tables
- Phase-2 alternatives study (binary classifier, two-pass, bridge-aware)

All of these are reproducible from the same codebase with additional
scripts and data; we omit them here to keep the release focused on the
method itself.

---

## Directory pointers

- `src/adaptive_aggregator.py` — P2 adaptive K logic (full `joint`,
  `binary_adaptive`, baseline `adaptive_k` / `uniform_budget` modes).
- `src/utils/joint_priority.py` — P3 priority scheduler.
- `src/graph_feature_extractor.py` — P1 graph feature extraction.
- `src/benchmark_e2e_joint.py` — online benchmark driver.
- `src/eval_phase2_standalone.py` — offline P2 evaluator.
- `src/pipeline_baseline.py` — Phase-1 retrieval pipeline (from
  scratch).

## Citation

Anonymous for submission.
