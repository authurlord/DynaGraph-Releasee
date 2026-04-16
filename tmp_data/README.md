# Intermediate artifacts

This directory holds Phase-1 outputs that are consumed by the online
benchmark (`src/benchmark_e2e_joint.py`). The directory layout is:

```
tmp_data/
├── 2wikimultihopqa/
│   ├── graph_features_1000q.json          (committed to git)
│   ├── retrieve_result_4B_baseline_v3.npy (external — see below)
│   └── triple_extraction_*.npy            (external — optional)
├── hotpotqa/
│   ├── graph_features_1000q.json
│   ├── retrieve_result_4B_baseline_full_1000q.npy
│   └── triple_extraction_*.npy
└── musique/
    ├── graph_features_1000q.json
    ├── retrieve_result_4B_1.npy
    └── triple_extraction_*.npy
```

The small `graph_features_1000q.json` files (~700 KB each) are committed
directly. The large retrieval-result arrays (144–260 MB each) exceed
GitHub's 100 MB single-file cap and are distributed separately.

## Option A — download the artifacts bundle (recommended)

Download `dynagraph_intermediate_artifacts.zip` from the link in the
top-level `README.md` and unpack it at the repo root. The zip is
structured so it lays directly onto this directory.

```
unzip dynagraph_intermediate_artifacts.zip -d .
```

After unpacking you should have the three `retrieve_result_*.npy` files
listed above.

## Option B — regenerate from scratch

Follow `README.md` §"Reproduce from scratch". The pipeline script
rebuilds every file listed here from `data/{dataset}.json` and the
corresponding corpus.
