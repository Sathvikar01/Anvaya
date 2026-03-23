# Small embedding and index ablation

Run the standalone study with:

```powershell
python -m hh_goa_rag.cli --config configs/small_ablation.yaml small-ablation run
```

The study uses the fixed development split and fixed 128-word chunks. It first compares small
multilingual embedding models with a FAISS FlatIP index. It then selects the best embedding by
the predeclared order MRR@10, Recall@10, Recall@5, Recall@1, with query-embedding P50 latency as
the tie-breaker, and selects only `faiss_flat_ip2`. The repository has no native FAISS `FlatIP2`;
`faiss_flat_ip2` is the exact `IndexFlatL2` baseline. Because vectors are L2-normalized, FlatL2
and FlatIP have equivalent rankings.

The requested metrics and latency measurements are written to:

- `results/small_embedding_ablation.csv`
- `results/small_index_ablation.csv`
- `results/small_ablation_summary.json`

Embedding rows include model size, dimension, corpus/query embedding latency, retrieval latency,
Recall@1, Recall@5, Recall@10, and MRR@10. Index rows include indexing time, retrieval P50/P95,
serialized index size, Recall@1, Recall@5, Recall@10, and MRR@10. The index comparison is run on
the selected small embedding so that index quality and latency are compared at fixed vectors.

For a CPU-only index rerun using the existing vectors:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m hh_goa_rag.cli --config configs/small_ablation.yaml small-ablation run --index-only
```

The CPU-only output is written to `results/small_index_ablation_cpu.csv` and its device check to
`results/small_index_ablation_cpu_summary.json`.

## Measured results

Development split: 1,000 queries, 10,389 fixed-word chunks, and 20 warm-up queries. The index
numbers below are from the explicit CPU-only rerun; the GPU visibility check was `false` and the
installed FAISS package exposes no GPU index API.

### Small embedding models

| Model | Dim. | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Query P50 | Search P50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `intfloat/multilingual-e5-small` | 384 | 0.3144 | 0.7158 | 0.8366 | 0.4921 | 6.8144 ms | 0.3453 ms |
| `l3cube-pune/indic-sentence-bert-nli` | 768 | 0.1581 | 0.4409 | 0.5902 | 0.2884 | 11.9136 ms | 1.3205 ms |
| `l3cube-pune/indic-sentence-similarity-sbert` | 768 | 0.1553 | 0.4606 | 0.5804 | 0.2862 | 11.8408 ms | 1.0484 ms |
| `paraphrase-MiniLM-L3-v2` (English-only control) | 384 | 0.0030 | 0.0050 | 0.0090 | 0.0040 | 2.0121 ms | 0.3587 ms |

### CPU index on `multilingual-e5-small`

| Index | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Build time | Search P50 / P95 | Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FAISS FlatIP2 (`IndexFlatL2`) | 0.3144 | 0.7158 | 0.8366 | 0.4921 | 10.51 ms | 0.3248 / 0.3738 ms | 15.22 MiB |

### Chunking used

The small-model study uses `fixed_words`, 128 words per chunk, with zero overlap. This is the
same fixed chunking used for the BGE-M3 embedding-model ablation, so the model comparison is
controlled. It is not the final production chunking: the separate BGE-M3 chunking ablation selected
punctuation-aware `sentence` chunks with `max_words=128`.

The expanded completed study compares `multilingual-e5-small`, two Indic SBERT variants, and
the English-only `sentence-transformers/paraphrase-MiniLM-L3-v2` speed control. A CPU pre-screen
of `paraphrase-multilingual-mpnet-base-v2` was attempted but did not finish within the bounded
run window, so it is not included in the measured table. The selected index remains
`faiss_flat_ip2` (`IndexFlatL2`).
This is a development comparison on the repository’s Hindi MSMARCO-XI artifact; re-run it on the
target Goa/Konkani corpus before treating the ranking as production evidence.
