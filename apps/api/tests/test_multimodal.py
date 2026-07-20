"""v2: brand intake, image generation, vision-QA regen loop, gating, kit."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.graph import agents
from app.graph.build import build_graph
from app.llm import client
from app.media import store as media


def _run(brief="New Year campaign for Vespa Elettrica in Italy", **extra):
    client._CACHE.clear()
    rid = extra.pop("run_id", "mm")
    agents.reset_run(rid)
    return build_graph().invoke({"run_id": rid, "brief": brief, **extra})


def test_run_produces_visuals_and_storyboard():
    pkg = _run(run_id="mm1")["package"]
    kinds = {v["kind"] for v in pkg["visuals"]}
    assert {"hero", "poster"} <= kinds
    assert len(pkg["storyboard"]) >= 4
    for v in pkg["visuals"]:
        assert v["url"].startswith("/api/assets/")
        assert v["qa_score"] > 0


def test_brand_intake_extracts_profile():
    pkg = _run(run_id="mm2")["package"]
    assert "brand_profile" in pkg
    assert isinstance(pkg["brand_profile"], dict)


def test_generate_image_mock_is_svg():
    img = client.generate_image("a vespa in rome")
    assert img.mock and img.mime == "image/svg+xml"
    assert b"<svg" in img.data


def test_vision_qa_regen_loop_fires(monkeypatch):
    # Force the vision judge to reject once, then the loop must regenerate.
    calls = {"n": 0}
    def bad_then_good(image, mime, criteria):
        calls["n"] += 1
        return '{"approved": false, "score": 0.2, "issues": ["off-brand"]}'
    monkeypatch.setattr(agents, "vision_judge", bad_then_good)
    agents.reset_run("mm3")
    asset = agents._gen_and_qa("mm3", "hero", "test", "cap", "palette #fff", max_regen=1)
    assert asset["regenerated"] == 1  # regenerated once after the failed QA


def test_passcode_gate():
    from app.core.ratelimit import require_passcode
    from fastapi import HTTPException
    ok = SimpleNamespace(headers={"x-cortex-passcode": "0000"}, query_params={})
    bad = SimpleNamespace(headers={}, query_params={})
    require_passcode(ok)  # no raise
    with pytest.raises(HTTPException):
        require_passcode(bad)


def test_kit_zip_builds():
    _run(run_id="mm4")
    data = media.zip_kit("mm4", '{"x":1}', "# kit")
    assert data[:2] == b"PK"  # zip magic
    assert len(data) > 100
