import json
from pathlib import Path

from hh_goa_rag.cleanup import MARKER, cleanup_losing_models


def _model(path: Path, repository: str, owner: str = "hh-goa-retrieval-ablation") -> None:
    path.mkdir()
    (path / "weights.bin").write_bytes(b"weights")
    (path / MARKER).write_text(
        json.dumps({"owned_by": owner, "repository": repository}), encoding="utf-8"
    )


def test_cleanup_removes_only_owned_losers_and_preserves_winner(tmp_path: Path) -> None:
    _model(tmp_path / "winner", "winner/model")
    _model(tmp_path / "loser", "loser/model")
    _model(tmp_path / "other-project", "loser/model", owner="another-project")
    report = cleanup_losing_models(
        tmp_path,
        winner="winner/model",
        candidates=["winner/model", "loser/model"],
    )
    assert (tmp_path / "winner").exists()
    assert not (tmp_path / "loser").exists()
    assert (tmp_path / "other-project").exists()
    assert [item["repository"] for item in report["removed"]] == ["loser/model"]
