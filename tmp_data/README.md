# Intermediate artifacts

Structure after unpacking `dynagraph_intermediate_artifacts.zip`:

```
tmp_data/
├── 2wikimultihopqa/
│   ├── graph_features_1000q.json           (committed)
│   ├── retrieve_result_4B_baseline_v3.npy  (from zip)
│   └── triple_extraction_35B.npy           (from zip)
├── hotpotqa/
│   ├── graph_features_1000q.json
│   ├── retrieve_result_4B_baseline_full_1000q.npy
│   └── triple_extraction_35B.npy
└── musique/
    ├── graph_features_1000q.json
    ├── retrieve_result_4B_1.npy
    └── triple_extraction_35B.npy
```

The end-to-end pipeline (`scripts/run_pipeline_from_scratch.sh`)
regenerates every file listed above from `data/{dataset}.json` and
`data/{dataset}_corpus.json`.
