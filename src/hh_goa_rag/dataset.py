"""MSMARCO-XI discovery and leakage-safe evaluation artifact creation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from hh_goa_rag.config import stable_fingerprint
from hh_goa_rag.io import write_json, write_jsonl

LANGUAGE_PREFIXES: dict[str, str] = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}
DEFAULT_LANGUAGE = "hi"
ALL_LANGUAGES = "all"


@dataclass(frozen=True)
class DatasetResolution:
    repository: str
    revision: str
    language: str
    train_file: str | None
    validation_file: str | None
    available_languages: tuple[str, ...]


@dataclass(frozen=True)
class FullDatasetManifest:
    repository: str
    revision: str
    root: str
    parquet_files: tuple[dict[str, Any], ...]
    total_bytes: int


def _discover_language_files(
    api: HfApi, repository: str, revision: str | None
) -> tuple[str, dict[str, dict[str, str]]]:
    info = api.dataset_info(repository, revision=revision, files_metadata=True)
    files = {sibling.rfilename for sibling in info.siblings}
    by_language: dict[str, dict[str, str]] = {}
    for language, prefix in LANGUAGE_PREFIXES.items():
        for split, folder, suffix in (
            ("train", "train", "train"),
            ("validation", "validation", "val"),
        ):
            expected = f"{folder}/{prefix}{suffix}.parquet"
            if expected in files:
                by_language.setdefault(language, {})[split] = expected
    return info.sha, by_language


def discover_resolutions(
    repository: str,
    requested_language: str = ALL_LANGUAGES,
    revision: str | None = None,
    *,
    api: HfApi | None = None,
) -> dict[str, DatasetResolution]:
    """Discover all language files and return immutable per-language resolutions."""
    resolved_revision, by_language = _discover_language_files(
        api or HfApi(), repository, revision
    )
    available = tuple(sorted(by_language))
    if not available:
        raise RuntimeError(f"No recognized MSMARCO-XI parquet files found in {repository}")
    if (
        requested_language not in (ALL_LANGUAGES, "auto")
        and requested_language not in by_language
    ):
        raise ValueError(
            f"Language {requested_language!r} unavailable; discovered {list(available)}"
        )
    selected = available if requested_language == ALL_LANGUAGES else (
        DEFAULT_LANGUAGE if requested_language == "auto" else requested_language,
    )
    return {
        language: DatasetResolution(
            repository=repository,
            revision=resolved_revision,
            language=language,
            train_file=by_language[language].get("train"),
            validation_file=by_language[language].get("validation"),
            available_languages=available,
        )
        for language in selected
    }


def discover_resolution(
    repository: str,
    requested_language: str = "auto",
    revision: str | None = None,
    *,
    api: HfApi | None = None,
) -> DatasetResolution:
    """Resolve current Hub files to an immutable revision and language.

    The dataset's legacy loading script is no longer honored by recent versions of
    ``datasets``. File discovery is therefore based on the actual parquet inventory.
    """
    resolutions = discover_resolutions(
        repository, requested_language, revision, api=api
    )
    if len(resolutions) != 1:
        raise ValueError(
            "discover_resolution accepts one language; use discover_resolutions for all"
        )
    return next(iter(resolutions.values()))


def download_full_dataset(
    repository: str,
    cache_dir: str | Path,
    *,
    revision: str | None = None,
    max_workers: int = 8,
    force: bool = False,
    api: HfApi | None = None,
) -> FullDatasetManifest:
    """Download and verify every train/validation parquet in a pinned snapshot.

    The Hub download is resumable and parallelized. This phase never samples or
    deletes raw data; evaluation materialization is deliberately separate.
    """
    hub = api or HfApi()
    info = hub.dataset_info(repository, revision=revision, files_metadata=True)
    parquet = sorted(
        (
            sibling
            for sibling in info.siblings
            if re.fullmatch(r"(?:train|validation)/[^/]+\.parquet", sibling.rfilename)
        ),
        key=lambda sibling: sibling.rfilename,
    )
    if not parquet:
        raise RuntimeError(f"No train/validation parquet files found in {repository}")
    root = Path(cache_dir) / "full" / info.sha
    manifest_path = root / "full_dataset_manifest.json"
    expected = [
        {"path": item.rfilename, "size_bytes": int(item.size or 0)}
        for item in parquet
    ]
    if not force and manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached.get("revision") == info.sha and all(
            (root / item["path"]).is_file()
            and (root / item["path"]).stat().st_size == item["size_bytes"]
            for item in expected
        ):
            return FullDatasetManifest(
                repository=repository,
                revision=info.sha,
                root=str(root),
                parquet_files=tuple(expected),
                total_bytes=sum(item["size_bytes"] for item in expected),
            )
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        repo_type="dataset",
        revision=info.sha,
        local_dir=root,
        allow_patterns=[item["path"] for item in expected],
        max_workers=max(1, int(max_workers)),
    )
    missing = [
        item
        for item in expected
        if not (root / item["path"]).is_file()
        or (root / item["path"]).stat().st_size != item["size_bytes"]
    ]
    if missing:
        raise RuntimeError(
            f"Full dataset verification failed for {len(missing)} files: {missing[:3]}"
        )
    record = {
        "format_version": 1,
        "repository": repository,
        "revision": info.sha,
        "root": str(root),
        "max_workers": max(1, int(max_workers)),
        "parquet_files": expected,
        "total_bytes": sum(item["size_bytes"] for item in expected),
        "verified": True,
    }
    write_json(manifest_path, record)
    return FullDatasetManifest(
        repository=repository,
        revision=info.sha,
        root=str(root),
        parquet_files=tuple(expected),
        total_bytes=int(record["total_bytes"]),
    )


def hf_parquet_uri(resolution: DatasetResolution, split: str) -> str:
    filename = resolution.train_file if split == "train" else resolution.validation_file
    if filename is None:
        raise ValueError(f"{resolution.language!r} has no {split!r} parquet file")
    return f"hf://datasets/{resolution.repository}@{resolution.revision}/{filename}"


def download_split(
    resolution: DatasetResolution, split: str, cache_dir: str | Path
) -> Path:
    """Download one pinned parquet to a caller-managed local directory."""
    filename = resolution.train_file if split == "train" else resolution.validation_file
    if filename is None:
        raise ValueError(f"{resolution.language!r} has no {split!r} parquet file")
    local_dir = Path(cache_dir) / "raw" / resolution.revision
    return Path(
        hf_hub_download(
            repo_id=resolution.repository,
            filename=filename,
            repo_type="dataset",
            revision=resolution.revision,
            local_dir=local_dir,
        )
    )


def iter_parquet_rows(path: str | Path, batch_size: int = 1024) -> Iterator[dict[str, Any]]:
    """Read only retrieval-relevant columns, bounded by an Arrow batch."""
    columns = ["source_lang", "target_lang", "query_id", "query", "passages"]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def _stable_unit_interval(seed: int, split: str, query_id: Any) -> float:
    digest = hashlib.sha256(f"{seed}:{split}:{query_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def select_rows(
    rows: Iterable[dict[str, Any]],
    *,
    count: int,
    seed: int,
    split: str,
    acceptance_rate: float,
    max_rows_scanned: int,
) -> tuple[list[dict[str, Any]], int]:
    """Choose a deterministic hash sample without relying on stream shuffle buffers."""
    if not 0 < acceptance_rate <= 1:
        raise ValueError("acceptance_rate must be in (0, 1]")
    selected: list[dict[str, Any]] = []
    scanned = 0
    for row in rows:
        scanned += 1
        labels = row.get("passages", {}).get("is_selected") or []
        usable = bool(row.get("query")) and any(int(label) == 1 for label in labels)
        score = _stable_unit_interval(seed, split, row.get("query_id"))
        if usable and score < acceptance_rate:
            selected.append(row)
            if len(selected) == count:
                break
        if scanned >= max_rows_scanned:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"Only selected {len(selected)}/{count} usable {split} rows after scanning "
            f"{scanned}; raise acceptance_rate or max_rows_scanned_multiplier"
        )
    return selected, scanned


def _passage_id(language: str, text: str) -> str:
    digest = hashlib.sha256(f"{language}\0{text}".encode()).hexdigest()[:24]
    return f"p-{digest}"


def materialize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    language: str,
    passage_field: str,
    query_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Pool passages and retain query-to-parent-passage relevance labels."""
    corpus_by_id: dict[str, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    seen_queries: set[str] = set()

    for row in rows:
        source_query_id = str(row["query_id"])
        query_id = f"{language}:{source_query_id}"
        if query_id in seen_queries:
            continue
        passages = row["passages"]
        texts = passages[passage_field]
        labels = passages["is_selected"]
        if len(texts) != len(labels):
            raise ValueError(f"Passage/label length mismatch for query {query_id}")
        relevant: set[str] = set()
        candidate_ids: list[str] = []
        for text, label in zip(texts, labels, strict=True):
            normalized_text = " ".join(str(text).split())
            if not normalized_text:
                continue
            passage_id = _passage_id(language, normalized_text)
            candidate_ids.append(passage_id)
            corpus_by_id.setdefault(
                passage_id,
                {"passage_id": passage_id, "language": language, "text": normalized_text},
            )
            if int(label) == 1:
                relevant.add(passage_id)
        if not relevant:
            continue
        seen_queries.add(query_id)
        queries.append(
            {
                "query_id": query_id,
                "language": language,
                "text": " ".join(str(row[query_field]).split()),
                "candidate_parent_ids": sorted(set(candidate_ids)),
            }
        )
        for passage_id in sorted(relevant):
            qrels.append({"query_id": query_id, "passage_id": passage_id, "relevance": 1})

    corpus = sorted(corpus_by_id.values(), key=lambda item: item["passage_id"])
    queries.sort(key=lambda item: item["query_id"])
    qrels.sort(key=lambda item: (item["query_id"], item["passage_id"]))
    return corpus, queries, qrels


def inspect_schema(row: dict[str, Any]) -> dict[str, Any]:
    passages = row.get("passages") or {}
    return {
        "top_level_fields": sorted(row),
        "passage_fields": sorted(passages),
        "passage_count": len(passages.get("is_selected") or []),
        "source_lang": row.get("source_lang"),
        "target_lang": row.get("target_lang"),
    }


def prepare_dataset(config: dict[str, Any], *, force: bool = False) -> Path:
    """Download the full snapshot, then materialize balanced multilingual eval artifacts."""
    dataset_config = config["dataset"]
    seed = int(config["experiment"]["seed"])
    requested_language = dataset_config.get("language", ALL_LANGUAGES)
    resolutions = discover_resolutions(
        dataset_config["repository"], requested_language, dataset_config.get("revision")
    )
    full_download = download_full_dataset(
        dataset_config["repository"],
        config["cache"]["huggingface"],
        revision=next(iter(resolutions.values())).revision,
        max_workers=int(dataset_config.get("download_workers", 8)),
    )
    artifact_identity = {
        "repository": dataset_config["repository"],
        "revision": full_download.revision,
        "languages": sorted(resolutions),
        "seed": seed,
        "dev_queries": int(dataset_config["dev_queries"]),
        "test_queries": int(dataset_config["test_queries"]),
        "sampling_acceptance_rate": float(dataset_config["sampling_acceptance_rate"]),
        "max_rows_scanned_multiplier": int(dataset_config["max_rows_scanned_multiplier"]),
        "passage_field": dataset_config["passage_field"],
        "query_field": dataset_config["query_field"],
        "balanced_by_language": requested_language == ALL_LANGUAGES,
    }
    output_dir = Path(dataset_config["output_dir"]) / stable_fingerprint(artifact_identity)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return output_dir

    split_specs = {
        "dev": (dataset_config["dev_source_split"], int(dataset_config["dev_queries"])),
        "test": (dataset_config["test_source_split"], int(dataset_config["test_queries"])),
    }
    split_stats: dict[str, Any] = {}
    schemas: dict[str, dict[str, Any]] = {}
    raw_root = Path(full_download.root)
    for output_split, (source_split, total_count) in split_specs.items():
        split_resolutions = {
            language: resolution
            for language, resolution in resolutions.items()
            if (resolution.train_file if source_split == "train" else resolution.validation_file)
        }
        if not split_resolutions:
            raise RuntimeError(f"No {source_split} files available for selected languages")
        base, remainder = divmod(total_count, len(split_resolutions))
        all_corpus: dict[str, dict[str, Any]] = {}
        all_queries: list[dict[str, Any]] = []
        all_qrels: list[dict[str, Any]] = []
        language_stats: dict[str, Any] = {}
        for offset, (language, resolution) in enumerate(sorted(split_resolutions.items())):
            count = base + int(offset < remainder)
            filename = (
                resolution.train_file
                if source_split == "train"
                else resolution.validation_file
            )
            assert filename is not None
            parquet_path = raw_root / filename
            max_scanned = count * int(dataset_config["max_rows_scanned_multiplier"])
            selected, scanned = select_rows(
                iter_parquet_rows(parquet_path),
                count=count,
                seed=seed,
                split=f"{source_split}:{language}",
                acceptance_rate=float(dataset_config["sampling_acceptance_rate"]),
                max_rows_scanned=max_scanned,
            )
            schemas[language] = inspect_schema(selected[0])
            corpus, queries, qrels = materialize_rows(
                selected,
                language=language,
                passage_field=dataset_config["passage_field"],
                query_field=dataset_config["query_field"],
            )
            all_corpus.update({item["passage_id"]: item for item in corpus})
            all_queries.extend(queries)
            all_qrels.extend(qrels)
            language_stats[language] = {
                "requested_queries": count,
                "rows_scanned": scanned,
                "queries": len(queries),
                "unique_parent_passages": len(corpus),
                "qrels": len(qrels),
            }
        write_jsonl(
            output_dir / f"{output_split}_corpus.jsonl",
            sorted(all_corpus.values(), key=lambda item: item["passage_id"]),
        )
        write_jsonl(
            output_dir / f"{output_split}_queries.jsonl",
            sorted(all_queries, key=lambda item: item["query_id"]),
        )
        write_jsonl(
            output_dir / f"{output_split}_qrels.jsonl",
            sorted(all_qrels, key=lambda item: (item["query_id"], item["passage_id"])),
        )
        split_stats[output_split] = {
            "source_split": source_split,
            "languages": sorted(language_stats),
            "rows_scanned": sum(item["rows_scanned"] for item in language_stats.values()),
            "queries": len(all_queries),
            "unique_parent_passages": len(all_corpus),
            "qrels": len(all_qrels),
            "by_language": language_stats,
        }

    manifest = {
        "format_version": 2,
        "artifact_fingerprint": output_dir.name,
        "artifact_identity": artifact_identity,
        "full_download": asdict(full_download),
        "dataset_schema_by_language": schemas,
        "splits": split_stats,
        "leakage_policy": {
            "development": (
                "A deterministic, language-balanced sample of every available upstream train file."
            ),
            "test": (
                "A deterministic, language-balanced sample of every available upstream validation "
                "file; final winner only."
            ),
            "gold_unit": (
                "Parent passage ID; retrieved chunks are mapped to their parent before scoring."
            ),
        },
    }
    write_json(manifest_path, manifest)
    return output_dir


def read_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
