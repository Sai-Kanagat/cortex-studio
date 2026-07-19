"""M5: golden-set evaluation thresholds as regression gates."""
from __future__ import annotations

from app.eval.harness import run_all, heuristic_score
from app.llm import client


def test_golden_set_passes_thresholds():
    client._CACHE.clear()
    report = run_all()
    assert report["pass_rate"] == 1.0, report
    assert report["avg_faithfulness"] >= 0.5, report


def test_heuristic_catches_missing_channels():
    case = {"must_include_any": ["x"], "must_not_include": [], "min_channels": 5, "expect_status": "approved"}
    pkg = {"copy": {"headline": "x"}, "strategy": {"channels": ["a"]}, "status": "approved"}
    res = heuristic_score(case, pkg)
    assert res["checks"]["has_channels"] is False
    assert res["passed"] is False


def test_heuristic_catches_banned_terms():
    case = {"must_include_any": ["safe"], "must_not_include": ["deadly"], "min_channels": 0, "expect_status": "approved"}
    pkg = {"copy": {"headline": "safe but deadly"}, "strategy": {"channels": []}, "status": "approved"}
    res = heuristic_score(case, pkg)
    assert res["checks"]["no_banned_terms"] is False
