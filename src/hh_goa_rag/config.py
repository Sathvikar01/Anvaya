"""Configuration loading and reproducible fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the experiment YAML and validate the invariant-bearing fields."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    required = {"experiment", "dataset", "cache"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if int(config["experiment"]["seed"]) < 0:
        raise ValueError("experiment.seed must be non-negative")
    for field in ("dev_queries", "test_queries"):
        if int(config["dataset"][field]) <= 0:
            raise ValueError(f"dataset.{field} must be positive")
    retrieval = config.get("retrieval", {})
    if retrieval.get("normalize_embeddings") is not True:
        raise ValueError("retrieval.normalize_embeddings must remain true for cosine retrieval")
    if retrieval.get("normalization_method") != "float32_l2_v1":
        raise ValueError("retrieval.normalization_method must remain float32_l2_v1")
    return config


def stable_fingerprint(value: Any, length: int = 16) -> str:
    """Return a stable, short SHA-256 fingerprint for JSON-compatible data."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
