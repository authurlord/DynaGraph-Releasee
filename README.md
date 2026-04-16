# DynaGraph: Holistic Optimization for Latency-Critical GraphRAG Serving via Bidirectional Feedback

Anonymous code release accompanying the submission. The full paper is
at [`paper_full_version.pdf`](./paper_full_version.pdf).

DynaGraph is a serving framework for knowledge-graph-enhanced
multi-hop question answering under a time-to-first-token (TTFT)
service level objective. The framework establishes a bidirectional
feedback loop across three phases:

- **Phase 1 — Sensor.** Builds a query-induced subgraph and extracts a
  topological feature vector that proxies query complexity.
- **Phase 2 — Controller.** Allocates a per-query evidence budget from
  the Phase-1 features and a live backpressure signal from Phase 3,
  and assembles the prompt.
- **Phase 3 — Actuator.** Runs an SLO-aware priority queue that orders
  requests by urgency and estimated prefill cost, and reports queue
  pressure back to Phase 2.

The objective is *QA-Goodput*: the rate of responses that are both
correct and delivered within the TTFT deadline.

---

## Repository layout

```
DynaGraph-Release/
├── src/                                     # method + benchmark + pipeline
├── scripts/                                 # server launchers & runners
├── data/                                    # 1000-query corpora & queries
├── tmp_data/                                # graph features (committed)
├── dynagraph_intermediate_artifacts.zip     # Phase-1 outputs (unzip before use)
├── paper_full_version.pdf
└── requirements.txt
```

Models used (download once from Hugging Face):

| Role | Hugging Face |
|---|---|
| Default LLM | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| Alternative LLM | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Embedding | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| Reranker | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) |

Default results are reported on **Qwen3.5-4B** (2× RTX 4090, TP=2).

---

## Environment

Python 3.11, CUDA 12.8.

```bash
pip install -r requirements.txt
pip install vllm==0.17.1           # recommended; or:
pip install "sglang[all]>=0.5.9"   # alternative
```

---

## Main setting (reviewer fast path)

The benchmark expects the Phase-1 artifacts to be present in
`tmp_data/`. The default LLM is Qwen3.5-4B.

```bash
# 1. Unpack Phase-1 artifacts
unzip dynagraph_intermediate_artifacts.zip -d .

# 2. Start the LLM server on two GPUs (priority scheduling enabled)
MODEL_PATH=/path/to/Qwen3.5-4B bash scripts/start_vllm_qwen35_4b.sh

# 3. Warm up the server (REQUIRED — see note below)
bash scripts/warmup_server.sh

# 4. Run the benchmark on all three datasets
bash scripts/run_dynagraph.sh
```

The benchmark replays Zipf α=1.0, 1000 unique queries, 3000 total
requests, 30 QPS, concurrency 64, TTFT SLO 2000 ms. Per-request JSON
reports land under `tmp_data/{dataset}/`.

> **Why warmup matters.** Without warmup the first ~200 requests of
> each run hit cold CUDA graphs, an empty prefix cache, and unreached
> priority-queue steady state. TTFT for those requests can reach 8+ s,
> which distorts SLO attainment and QA-Goodput. The provided script
> sends 300 realistic QA-formatted requests before measurement.

### Using Qwen3-4B-Instruct-2507 instead

```bash
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 bash scripts/start_vllm_qwen3_4b.sh
bash scripts/warmup_server.sh qwen3-4b
bash scripts/run_dynagraph.sh qwen3-4b
```

### Using SGLang (>= 0.5.9) instead

```bash
MODEL_PATH=/path/to/Qwen3.5-4B bash scripts/start_sglang_qwen35_4b.sh
# warmup + run_dynagraph as above
```

---

## End-to-end pipeline

The end-to-end path regenerates every Phase-1 artifact from only the
committed corpus and query JSONs. It requires three servers co-located
on two GPUs:

- **GPU 0** — LLM server.
- **GPU 1** — embedding server and reranker server (they share GPU 1).

```bash
# GPU 0: LLM
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path/to/Qwen3.5-4B \
  bash scripts/start_vllm_qwen35_4b.sh

# GPU 1: embedding server
CUDA_VISIBLE_DEVICES=1 MODEL_PATH=/path/to/Qwen3-Embedding-0.6B \
  bash scripts/start_embedding_server.sh

# GPU 1: reranker server (shares GPU 1 with the embedding server)
CUDA_VISIBLE_DEVICES=1 MODEL_PATH=/path/to/bge-reranker-v2-m3 \
  bash scripts/start_reranker_server.sh

# Regenerate retrieval + KG + graph features, then run the benchmark:
bash scripts/run_pipeline_from_scratch.sh qwen3.5-4b 4B
bash scripts/warmup_server.sh
bash scripts/run_dynagraph.sh
```

When the LLM uses TP=2 across both GPUs, lower `GPU_MEM` in the
LLM launcher so the embedding + reranker servers can share the memory.

The from-scratch path takes several hours on dual 4090s; use the main
setting above for quick verification.

---

## Scope of this release

DynaGraph main method on 2WikiMultihopQA, HotpotQA, and MuSiQue.

Baselines (HippoRAG, SiReRAG, KET-RAG, CacheBlend, DepCache, etc.),
full-dev scalability, QPS sweeps, and additional ablations appear only
in the paper.

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
