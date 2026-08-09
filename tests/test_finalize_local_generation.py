from __future__ import annotations

import pytest

from eval.finalize_local_generation import ANSWER_BUDGET_MS, FIXED_PRE_ANSWER_MS, percentile_summary


def test_sub_200_answer_budget_uses_all_measured_fixed_stages() -> None:
    assert pytest.approx(45.92) == FIXED_PRE_ANSWER_MS
    assert pytest.approx(154.08) == ANSWER_BUDGET_MS


def test_percentiles_include_p70_and_max() -> None:
    summary = percentile_summary([10.0, 20.0, 30.0, 40.0])

    assert summary["p50_ms"] == pytest.approx(25.0)
    assert summary["p70_ms"] == pytest.approx(31.0)
    assert summary["p95_ms"] == pytest.approx(38.5)
    assert summary["p100_ms"] == 40.0
