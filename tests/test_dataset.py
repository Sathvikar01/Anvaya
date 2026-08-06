from pathlib import Path
from types import SimpleNamespace

import pytest

import hh_goa_rag.dataset as dataset_module
from hh_goa_rag.dataset import (
    discover_resolution,
    discover_resolutions,
    download_full_dataset,
    materialize_rows,
    select_rows,
)


class FakeApi:
    def dataset_info(self, *_args, **_kwargs):
        return SimpleNamespace(
            sha="abc123",
            siblings=[
                SimpleNamespace(rfilename="train/hintrain.parquet"),
                SimpleNamespace(rfilename="validation/hinval.parquet"),
                SimpleNamespace(rfilename="validation/telval.parquet"),
            ],
        )


def test_discovers_default_language_and_missing_train_split() -> None:
    resolution = discover_resolution("repo", api=FakeApi())
    assert resolution.language == "hi"
    assert resolution.revision == "abc123"
    assert resolution.train_file == "train/hintrain.parquet"
    telugu = discover_resolution("repo", "te", api=FakeApi())
    assert telugu.train_file is None
    assert telugu.validation_file == "validation/telval.parquet"


def test_unknown_language_lists_discovered_options() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        discover_resolution("repo", "xx", api=FakeApi())


def test_discovers_all_available_languages() -> None:
    resolutions = discover_resolutions("repo", "all", api=FakeApi())
    assert sorted(resolutions) == ["hi", "te"]
    assert resolutions["te"].train_file is None


def test_full_download_is_resumable_and_size_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    siblings = [
        SimpleNamespace(rfilename="train/hintrain.parquet", size=3),
        SimpleNamespace(rfilename="validation/hinval.parquet", size=4),
        SimpleNamespace(rfilename="README.md", size=1),
    ]

    class DownloadApi:
        def dataset_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="rev", siblings=siblings)

    def fake_snapshot_download(*_args, local_dir: Path, allow_patterns: list[str], **_kwargs):
        for relative in allow_patterns:
            path = local_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"123" if path.name.endswith("train.parquet") else b"1234")
        return str(local_dir)

    monkeypatch.setattr(dataset_module, "snapshot_download", fake_snapshot_download)
    manifest = download_full_dataset("repo", tmp_path, api=DownloadApi())
    assert manifest.revision == "rev"
    assert manifest.total_bytes == 7
    assert (tmp_path / "full" / "rev" / "full_dataset_manifest.json").exists()


def test_select_rows_is_deterministic_and_requires_positive() -> None:
    rows = [
        {
            "query_id": index,
            "query": f"q{index}",
            "passages": {"is_selected": [1 if index % 2 else 0]},
        }
        for index in range(20)
    ]
    first, scanned = select_rows(
        rows, count=3, seed=7, split="train", acceptance_rate=1, max_rows_scanned=20
    )
    second, _ = select_rows(
        rows, count=3, seed=7, split="train", acceptance_rate=1, max_rows_scanned=20
    )
    assert [row["query_id"] for row in first] == [1, 3, 5]
    assert first == second
    assert scanned == 6


def test_materialize_rows_deduplicates_and_preserves_parent_qrels() -> None:
    rows = [
        {
            "query_id": 11,
            "query": " प्रश्न ",
            "passages": {
                "Translated_passages": ["उत्तर गद्य", " दूसरा  गद्य "],
                "is_selected": [1, 0],
            },
        },
        {
            "query_id": 12,
            "query": "दूसरा प्रश्न",
            "passages": {
                "Translated_passages": ["उत्तर गद्य", "तीसरा गद्य"],
                "is_selected": [1, 0],
            },
        },
    ]
    corpus, queries, qrels = materialize_rows(
        rows, language="hi", passage_field="Translated_passages", query_field="query"
    )
    assert len(corpus) == 3
    assert len(queries) == 2
    assert len(qrels) == 2
    assert qrels[0]["passage_id"] == qrels[1]["passage_id"]
    assert queries[0]["text"] == "प्रश्न"
