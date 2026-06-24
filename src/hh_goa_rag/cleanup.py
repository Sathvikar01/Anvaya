"""Marker-guarded cleanup for model directories owned by this experiment only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from hh_goa_rag.index_backends import directory_size

OWNER = "hh-goa-retrieval-ablation"
MARKER = ".hh_goa_model.json"


def cleanup_losing_models(
    model_root: str | Path,
    *,
    winner: str,
    candidates: list[str],
) -> dict[str, Any]:
    """Delete only direct child directories with this project's exact ownership marker."""
    configured_root = Path(model_root)
    root = configured_root.resolve(strict=True)
    candidate_set = set(candidates)
    removed: list[dict[str, Any]] = []
    preserved: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for child in sorted(root.iterdir()):
        marker = child / MARKER
        if not child.is_dir() or not marker.is_file():
            continue
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        repository = str(metadata.get("repository", ""))
        if metadata.get("owned_by") != OWNER or repository not in candidate_set:
            skipped.append(
                {
                    "path": str(configured_root / child.name),
                    "reason": "unowned_or_not_a_candidate",
                }
            )
            continue
        resolved = child.resolve(strict=True)
        if child.is_symlink() or resolved.parent != root:
            raise RuntimeError(f"Refusing cleanup target outside direct model cache: {resolved}")
        if repository == winner:
            preserved.append(
                {"repository": repository, "path": str(configured_root / child.name)}
            )
            continue
        size = directory_size(resolved)
        shutil.rmtree(resolved)
        removed.append(
            {
                "repository": repository,
                "path": str(configured_root / child.name),
                "bytes": size,
            }
        )
    if not any(item["repository"] == winner for item in preserved):
        raise RuntimeError("Winning model was not found and preserved in the project model cache")
    return {
        "root": str(configured_root),
        "winner": winner,
        "candidates": sorted(candidate_set),
        "removed": removed,
        "preserved": preserved,
        "skipped": skipped,
        "removed_bytes": sum(int(item["bytes"]) for item in removed),
    }
