# Paired BGE-M3 and multilingual E5-small index ablation

Development split: 1,000 balanced queries across 13 train languages, fixed 128-word chunks,
20 warm-up queries, normalized vectors, and parent-level scoring. FlatIP2 is the repository name
for FAISS `IndexFlatL2`; with L2-normalized vectors, its rankings are cosine-equivalent to
`IndexFlatIP`.

## Embedding comparison

Both models use the same fixed chunking and exact FlatIP baseline.

| Model | Dim. | Recall@10 | MRR@10 | nDCG@10 | Query p50 ms | Query p95 ms | Retrieval p50 ms | Retrieval p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BAAI/bge-m3 | 1024 | 0.7697 | 0.4318 | 0.5079 | 8.8926 | 16.4383 | 2.3841 | 4.7818 |
| intfloat/multilingual-e5-small | 384 | 0.6908 | 0.3635 | 0.4366 | 5.1675 | 10.7342 | 0.4004 | 0.4940 |

## Paired FAISS index comparison

| Model | Index | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms | p100 ms | Build ms | Size MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE-M3 | FlatIP | 0.7697 | 0.4318 | 0.5079 | 2.1944 | 4.9624 | 5.3735 | 48.1859 | 39.77 |
| BGE-M3 | FlatIP2 / FlatL2 | 0.7697 | 0.4323 | 0.5082 | 1.6570 | 4.9414 | 5.3628 | 38.7366 | 39.77 |
| BGE-M3 | HNSW | 0.7697 | 0.4323 | 0.5082 | 0.3261 | 0.3973 | 0.4889 | 332.9872 | 42.41 |
| BGE-M3 | IVF-Flat | 0.7497 | 0.4213 | 0.4949 | 0.4158 | 0.7357 | 0.9731 | 98.1074 | 40.35 |
| BGE-M3 | IVF-PQ | 0.5996 | 0.2758 | 0.3477 | 0.0409 | 0.0704 | 0.1145 | 932.6245 | 1.73 |
| E5-small | FlatIP | 0.6908 | 0.3635 | 0.4366 | 0.3221 | 0.4360 | 1.0913 | 16.4127 | 14.92 |
| E5-small | FlatIP2 / FlatL2 | 0.6908 | 0.3636 | 0.4367 | 0.4081 | 0.9854 | 1.3755 | 12.1342 | 14.92 |
| E5-small | HNSW | 0.6908 | 0.3636 | 0.4367 | 0.1590 | 0.1888 | 0.2809 | 153.8336 | 17.55 |
| E5-small | IVF-Flat | 0.6611 | 0.3531 | 0.4218 | 0.0655 | 0.1236 | 0.3222 | 63.8980 | 15.18 |
| E5-small | IVF-PQ | 0.4137 | 0.1772 | 0.2274 | 0.0423 | 0.0482 | 0.0992 | 775.7543 | 0.80 |

## Per-language Recall@10 / nDCG@10 using HNSW

| Language | BGE Recall | BGE nDCG | E5 Recall | E5 nDCG |
| --- | ---: | ---: | ---: | ---: |
| Assamese | 0.7781 | 0.5171 | 0.6494 | 0.4372 |
| Bengali | 0.7500 | 0.5588 | 0.7565 | 0.4548 |
| Gujarati | 0.8377 | 0.5626 | 0.6970 | 0.4206 |
| Hindi | 0.8074 | 0.5117 | 0.9026 | 0.5488 |
| Kannada | 0.8182 | 0.5293 | 0.7532 | 0.4499 |
| Malayalam | 0.6991 | 0.4740 | 0.7013 | 0.4643 |
| Marathi | 0.7359 | 0.5420 | 0.6840 | 0.4332 |
| Nepali | 0.7814 | 0.4642 | 0.7532 | 0.4849 |
| Odia | 0.7814 | 0.4816 | 0.6515 | 0.4287 |
| Punjabi | 0.7792 | 0.5595 | 0.6688 | 0.4037 |
| Sanskrit | 0.7078 | 0.4467 | 0.4740 | 0.2964 |
| Tamil | 0.7857 | 0.5000 | 0.6050 | 0.3829 |
| Urdu | 0.7434 | 0.4590 | 0.6842 | 0.4668 |

## Interpretation

- BGE-M3 is the quality winner on this dev artifact: nDCG@10 `0.5079` versus E5-small `0.4366`.
- HNSW is the best quality/latency choice for both models.
- FlatIP2 has effectively identical quality to FlatIP, as expected from normalized vectors.
- IVF-PQ is smallest and fastest, but its quality loss is substantial for both embeddings.
- These are development comparisons; the validation split remains reserved for one final locked-stack evaluation.
