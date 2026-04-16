# Intermediate artifacts bundle

The 1000-query corpora are distributed as `data/{dataset}.json` +
`data/{dataset}_corpus.json`. Phase-1 outputs (retrieval arrays and
KG triples) are bundled in the repo as
`dynagraph_intermediate_artifacts.zip` because their uncompressed
sizes (144–260 MB per retrieval array) exceed GitHub's single-file
limit but compress to under 10 MB combined.

## Contents

| Dataset | Target path | Uncompressed | Produced by |
|---|---|---|---|
| 2WikiMultihopQA | `tmp_data/2wikimultihopqa/retrieve_result_4B_baseline_v3.npy` | 144 MB | `src/pipeline_baseline.py` stage 2 |
| 2WikiMultihopQA | `tmp_data/2wikimultihopqa/triple_extraction_35B.npy` | 2.7 MB | `src/extract_kg_standalone.py` |
| HotpotQA | `tmp_data/hotpotqa/retrieve_result_4B_baseline_full_1000q.npy` | 238 MB | pipeline stage 2 |
| HotpotQA | `tmp_data/hotpotqa/triple_extraction_35B.npy` | 5.1 MB | KG extraction |
| MuSiQue | `tmp_data/musique/retrieve_result_4B_1.npy` | 260 MB | pipeline stage 2 |
| MuSiQue | `tmp_data/musique/triple_extraction_35B.npy` | 5.4 MB | KG extraction |

Compressed zip size: ~8 MB. Integrity:

```bash
unzip -t dynagraph_intermediate_artifacts.zip
sha256sum -c SHA256SUMS.txt
```

## Usage

Pure-QA mode (recommended for reviewing):

```bash
unzip dynagraph_intermediate_artifacts.zip -d .
```

That lays the files directly into `tmp_data/<dataset>/`. The
`graph_features_1000q.json` file for each dataset is already committed
alongside them, so after the unzip the pure-QA pipeline has all inputs
it needs.

From-scratch mode: you do not need this zip. `scripts/run_pipeline_from_scratch.sh`
rebuilds every file listed above.
