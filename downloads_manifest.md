# Bundled intermediate artifacts

Phase-1 outputs (retrieval arrays and KG triples) are shipped as
`dynagraph_intermediate_artifacts.zip` at the repo root. The
uncompressed retrieval arrays are 144–260 MB each; the zip compresses
them to ~8 MB using deflate on the repeated document text.

Unpack at the repo root:

```bash
unzip dynagraph_intermediate_artifacts.zip -d .
sha256sum -c SHA256SUMS.txt   # optional integrity check
```

| Dataset | Target path | Uncompressed |
|---|---|---|
| 2WikiMultihopQA | `tmp_data/2wikimultihopqa/retrieve_result_4B_baseline_v3.npy` | 144 MB |
| 2WikiMultihopQA | `tmp_data/2wikimultihopqa/triple_extraction_35B.npy` | 2.7 MB |
| HotpotQA | `tmp_data/hotpotqa/retrieve_result_4B_baseline_full_1000q.npy` | 238 MB |
| HotpotQA | `tmp_data/hotpotqa/triple_extraction_35B.npy` | 5.1 MB |
| MuSiQue | `tmp_data/musique/retrieve_result_4B_1.npy` | 260 MB |
| MuSiQue | `tmp_data/musique/triple_extraction_35B.npy` | 5.4 MB |

The three `graph_features_1000q.json` files (under 1 MB each) are
committed directly to the repo; the zip is not required for the
end-to-end pipeline, only for the main-setting reviewer fast path.
