# HH Goa retrieval-stack recommendation

## Selected stack

**Best embedding model → Best chunking strategy → Best index/storage system**

**`BAAI/bge-m3` → `fixed_size` → `faiss_hnsw`**

Selection used only the development split. The upstream validation-derived test split remained
sealed until all three winners and their configurations were fixed.

## Exact reproducibility configuration

- Dataset: `ai4bharat/MSMARCO-XI`, balanced languages `as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, te, ur`, Hub revision
  `bf5cdc1f26e581e519018e434db14edd1b77602b`
- Processed dataset artifact: `data\processed\2c0dcd7c6bc9f61e`
- Random seed: `20260818`
- Embedding checkpoint: `BAAI/bge-m3` revision `5617a9f61b028005a4858fdac845db406aefb181`
- Embedding dimension: `1024`
- Normalization: final float32 L2 (`float32_l2_v1`); similarity is cosine-equivalent inner product
- Chunking: punctuation-aware sentence packing, maximum 128 whitespace-delimited words,
  parent-level qrels
- Index configuration: `{"ef_construction": 200, "ef_search": 128, "engine": "faiss", "index_type": "hnsw", "m": 32, "name": "faiss_hnsw"}`
- Retrieval: top 10 unique parents, search oversampling 20×, 20 warm-up queries
- Evaluation: `1000` balanced development queries and
  `1000` sealed test queries exactly once

## Embedding ablation (development)

| model | recall_at_10 | mrr_at_10 | ndcg_at_10 | query_embedding_p50_ms |
| --- | --- | --- | --- | --- |
| BAAI/bge-m3 | 0.7697 | 0.4318 | 0.5079 | 9.0403 |
| intfloat/multilingual-e5-base | 0.7620 | 0.4146 | 0.4932 | 5.2177 |
| Alibaba-NLP/gte-multilingual-base | 0.6789 | 0.3571 | 0.4268 | 5.8934 |
| jinaai/jina-embeddings-v3 | 0.1898 | 0.0846 | 0.1071 | 57.5605 |

The winner used the predeclared lexicographic quality priority
`['ndcg_at_10', 'mrr_at_10', 'recall_at_10']`; latency was used only after quality
ties. Each row also records per-language metrics.

## Chunking ablation (development)

| strategy | recall_at_10 | mrr_at_10 | ndcg_at_10 | retrieval_p50_ms |
| --- | --- | --- | --- | --- |
| fixed_size | 0.7697 | 0.4318 | 0.5079 | 1.6149 |
| overlapping | 0.7662 | 0.4244 | 0.5016 | 2.2136 |
| sentence_based | 0.7697 | 0.4305 | 0.5069 | 1.6656 |
| semantic | 0.7625 | 0.4197 | 0.4974 | 2.6062 |
| parent_child | 0.7412 | 0.4106 | 0.4847 | 5.8295 |

Chunking was selected on the same multilingual development artifact with the embedding model
held fixed; the table reports its quality/latency trade-off.

## Index/storage ablation (development)

| backend | recall_at_10 | mrr_at_10 | ndcg_at_10 | retrieval_p50_ms | retrieval_p95_ms | indexing_time_ms |
| --- | --- | --- | --- | --- | --- | --- |
| faiss_flat_ip | 0.7697 | 0.4318 | 0.5079 | 1.5989 | 4.3760 | 56.6680 |
| faiss_hnsw | 0.7697 | 0.4323 | 0.5082 | 0.3174 | 0.5303 | 314.0737 |
| faiss_ivf_flat | 0.7497 | 0.4213 | 0.4949 | 0.5009 | 0.7306 | 101.7897 |
| faiss_ivf_pq | 0.5993 | 0.2766 | 0.3486 | 0.0635 | 0.1890 | 887.1873 |

Index candidates were evaluated with the same normalized vectors and query set; the table reports
FAISS algorithm quality, build cost, serialized size, and p50/p95/p100 retrieval latency.

## Sealed test result

| recall_at_1 | recall_at_3 | recall_at_5 | recall_at_10 | mrr_at_10 | ndcg_at_10 | retrieval_p50_ms | retrieval_p95_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2803 | 0.5294 | 0.6669 | 0.8037 | 0.4546 | 0.5343 | 0.5974 | 0.9270 |

- Test chunks: 10220
- Indexing time: 327.2067 ms
- Index disk size: 42.57 MiB
- Query embedding P50: 8.9462 ms
- Final index artifact: `cache\indexes\final\afd021d4b0104558\retriever.faiss`
- Chunk-to-parent artifact: `data\processed\2c0dcd7c6bc9f61e\chunks\final-test-afd021d4b0104558.jsonl`

`hh_goa_rag.retriever.ParentFaissRetriever` loads these two artifacts and accepts an already encoded
query vector, so it can be placed directly after the future STT/query-embedding boundary without
adding generation, speech, or frontend code here.

## Project-local model cleanup

The winning `BAAI/bge-m3` directory was preserved. Only direct children of `cache\models`
carrying the exact `.hh_goa_model.json` ownership marker for this experiment were eligible for
deletion. Removed:



Freed 0.00 GiB. No global Hugging Face cache, model outside
this repository, processed dataset, embeddings, index, or winning model was deleted.

## Scope and caveats

These findings are retrieval-only and do not evaluate STT errors, RAG generation, frontend behavior,
or a production Goa/Konkani document corpus. The current Hub revision has a validation-only Telugu
file, which is treated as a zero-shot holdout. Re-run the same leakage-safe protocol when the target
corpus and real voice-derived queries become available.
