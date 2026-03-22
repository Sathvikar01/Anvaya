# Multilingual retrieval ablation protocol

The retrieval study is pinned to the Hub revision recorded by the downloader. The
full MSMARCO-XI train/validation parquet inventory is retained under
`cache/huggingface/full/<revision>/` and verified against the Hub-declared byte sizes.
The raw snapshot is not sampled or deleted during acquisition.

## Evaluation design

- Every language with an available source split is included. The current revision has
  13 train languages and 14 validation languages; Telugu is validation-only here.
- Development and sealed-test artifacts use deterministic hash sampling with equal
  query budgets per language. Query IDs are language-prefixed so IDs cannot collide
  across translated files.
- All retrieval scores are computed at the parent-passage level. Chunks are mapped
  back to their parent before Recall, MRR, and nDCG are calculated.
- Embedding ablation holds the baseline fixed-word chunker and FAISS FlatIP fixed.
- Chunking ablation uses the embedding winner only and holds the index fixed to
  FAISS FlatIP.
- FAISS index ablation uses the embedding and chunking winners only and compares
  FlatIP, HNSW, IVF-Flat, and IVF-PQ with the same normalized vectors.
- The development split selects winners. The validation split is read only once by
  `finalize run` after the winner is frozen; it is not used for tuning.

Each CSV row contains overall metrics, a JSON `language_metrics` field, corpus/index
build time, query embedding latency, and retrieval mean/p50/p95/p100 latency. This
makes quality/latency trade-offs inspectable instead of hiding them in a single score.

## Commands

```powershell
python -m hh_goa_rag.cli dataset download-full
python -m hh_goa_rag.cli dataset prepare --json
python -m hh_goa_rag.cli embedding run --data-dir <prepared-artifact>
python -m hh_goa_rag.cli chunking run --data-dir <prepared-artifact>
python -m hh_goa_rag.cli index run --data-dir <prepared-artifact>
python -m hh_goa_rag.cli finalize run --data-dir <prepared-artifact>
```

The download command is resumable. Set `HF_HUB_DISABLE_XET=1` in environments where
the Hub Xet transport stalls; the same pinned files and verification are used.
