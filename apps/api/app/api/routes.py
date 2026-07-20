"""HTTP surface: submit a brief, stream the multi-agent run as it happens."""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.ratelimit import check_daily_cap, rate_limit, require_passcode
from app.graph import agents
from app.graph.build import build_graph
from app.media import store as media

_UPLOADS = Path(os.environ.get("CORTEX_DATA_DIR", Path(__file__).resolve().parents[4] / "data")) / "uploads"
_UPLOADS.mkdir(parents=True, exist_ok=True)

router = APIRouter()

# In-memory checkpointer + graph. The Postgres-backed checkpointer (production) is a
# drop-in swap here; the interface is identical.
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

_CHECKPOINTER = MemorySaver()
_GRAPH = build_graph(checkpointer=_CHECKPOINTER)

# In-memory job registry for async runs. Production: a Redis queue + worker pool so
# jobs survive restarts and scale horizontally (see docs/scaling.md).
_JOBS: dict[str, dict] = {}


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Auth is opt-in: disabled when CORTEX_API_KEY is unset (local dev)."""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class RunRequest(BaseModel):
    brief: str


@router.post("/runs", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def create_run(req: RunRequest):
    """Non-streaming: run to completion and return the package."""
    run_id = str(uuid.uuid4())
    agents.reset_run(run_id)  # reset per-run cost meter + tracer
    config = {"configurable": {"thread_id": run_id}}
    final = _GRAPH.invoke({"run_id": run_id, "brief": req.brief}, config)
    return {
        "run_id": run_id,
        "package": final.get("package"),
        "cost": final.get("cost"),
        "trace": final.get("trace"),
    }


@router.post("/runs/stream", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def create_run_stream(req: RunRequest):
    """Server-sent-events stream of agent events as the graph executes."""
    run_id = str(uuid.uuid4())
    agents.reset_run(run_id)
    config = {"configurable": {"thread_id": run_id}}

    def gen():
        yield _sse({"type": "run_started", "run_id": run_id})
        for chunk in _GRAPH.stream(
            {"run_id": run_id, "brief": req.brief}, config, stream_mode="updates"
        ):
            for node, update in chunk.items():
                for ev in (update or {}).get("events", []):
                    yield _sse({"type": "agent_event", "node": node, **ev})
        state = _GRAPH.get_state(config).values
        yield _sse({"type": "run_finished", "package": state.get("package"),
                    "cost": state.get("cost"), "trace": state.get("trace")})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/runs/{run_id}/trace")
def get_trace(run_id: str):
    """Fetch the persisted span trace for a run (for the UI trace viewer)."""
    from app.obs.tracing import load_trace

    tr = load_trace(run_id)
    if tr is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return tr


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# --- Human-in-the-loop (M6) ---

class ResumeRequest(BaseModel):
    action: str = "approve"          # approve | edit | reject
    edits: dict | None = None


@router.post("/runs/hitl", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def create_hitl_run(req: RunRequest):
    """Start a run that pauses at the human gate. Returns the draft for review."""
    run_id = str(uuid.uuid4())
    agents.reset_run(run_id)
    cfg = {"configurable": {"thread_id": run_id}}
    _GRAPH.invoke({"run_id": run_id, "brief": req.brief, "hitl_enabled": True}, cfg)
    snap = _GRAPH.get_state(cfg)
    if not snap.next:  # completed without pausing (shouldn't happen with hitl on)
        return {"run_id": run_id, "status": "completed", "package": snap.values.get("package")}
    return {
        "run_id": run_id,
        "status": "awaiting_approval",
        "draft": {
            "copy": snap.values.get("copy"),
            "creative": snap.values.get("creative"),
            "critique": snap.values.get("critique"),
        },
    }


@router.post("/runs/{run_id}/resume", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def resume_run(run_id: str, req: ResumeRequest):
    """Resume a paused run with a human decision."""
    cfg = {"configurable": {"thread_id": run_id}}
    snap = _GRAPH.get_state(cfg)
    if not snap.next:
        raise HTTPException(status_code=409, detail="run is not awaiting a decision")
    final = _GRAPH.invoke(Command(resume={"action": req.action, "edits": req.edits}), cfg)
    return {"run_id": run_id, "status": "completed", "decision": req.action,
            "package": final.get("package"), "cost": final.get("cost")}


# --- Async / scaling (M6) ---

def _run_job(run_id: str, brief: str) -> None:
    agents.reset_run(run_id)
    cfg = {"configurable": {"thread_id": run_id}}
    try:
        final = _GRAPH.invoke({"run_id": run_id, "brief": brief}, cfg)
        _JOBS[run_id] = {"status": "completed", "package": final.get("package"), "cost": final.get("cost")}
    except Exception as e:  # pragma: no cover
        _JOBS[run_id] = {"status": "failed", "error": repr(e)}


@router.post("/runs/async", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def create_async_run(req: RunRequest):
    """Enqueue a run and return immediately. Poll GET /runs/{id} for the result.
    Demonstrates the non-blocking, queue-based execution model used for scaling."""
    run_id = str(uuid.uuid4())
    _JOBS[run_id] = {"status": "queued"}
    # Offload the blocking graph run to a worker thread so the event loop stays free.
    asyncio.get_event_loop().run_in_executor(None, _run_job, run_id, req.brief)
    return {"run_id": run_id, "status": "queued"}


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    """Poll an async run's status/result."""
    if run_id not in _JOBS:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, **_JOBS[run_id]}


# --- v2: passcode gate, brand-book upload, assets, gated live run, kit download ---

@router.get("/gate")
def gate_check(request: Request):
    """Front-end calls this to validate the microsite passcode before showing the app."""
    try:
        require_passcode(request)
        return {"ok": True}
    except HTTPException:
        return {"ok": False}


@router.post("/uploads", dependencies=[Depends(require_passcode)])
async def upload_brand_book(file: UploadFile = File(...)):
    """Accept a brand book (PDF/image). Returns an upload_id the run can reference."""
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=415, detail=f"unsupported type {ext}")
    upload_id = str(uuid.uuid4())
    dest = _UPLOADS / f"{upload_id}{ext}"
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="file too large (max 15MB)")
    dest.write_bytes(data)
    return {"upload_id": upload_id, "filename": file.filename, "path": str(dest)}


class LiveRunRequest(BaseModel):
    brief: str
    upload_id: str | None = None


@router.post("/live/stream", dependencies=[Depends(require_passcode), Depends(check_daily_cap), Depends(rate_limit)])
def live_run_stream(req: LiveRunRequest):
    """Gated live run for the microsite: streams the full multimodal campaign build."""
    run_id = str(uuid.uuid4())
    agents.reset_run(run_id)
    cfg = {"configurable": {"thread_id": run_id}}
    upload_path = ""
    if req.upload_id:
        hits = list(_UPLOADS.glob(f"{req.upload_id}.*"))
        upload_path = str(hits[0]) if hits else ""

    def gen():
        yield _sse({"type": "run_started", "run_id": run_id})
        state_in = {"run_id": run_id, "brief": req.brief, "upload_path": upload_path}
        for chunk in _GRAPH.stream(state_in, cfg, stream_mode="updates"):
            for node, update in chunk.items():
                for ev in (update or {}).get("events", []):
                    yield _sse({"type": "agent_event", "node": node, **ev})
        st = _GRAPH.get_state(cfg).values
        yield _sse({"type": "run_finished", "run_id": run_id, "package": st.get("package"),
                    "cost": st.get("cost"), "trace": st.get("trace")})

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/assets/{run_id}/{fname}")
def get_asset(run_id: str, fname: str):
    """Serve a generated asset file (image / video)."""
    p = media.asset_path(run_id, fname)
    if not p:
        raise HTTPException(status_code=404, detail="asset not found")
    mime = {"svg": "image/svg+xml", "png": "image/png", "jpg": "image/jpeg",
            "webp": "image/webp", "mp4": "video/mp4"}.get(p.suffix.lstrip("."), "application/octet-stream")
    return Response(content=p.read_bytes(), media_type=mime)


@router.get("/runs/{run_id}/kit")
def download_kit(run_id: str):
    """Download the whole campaign kit (assets + manifests) as a zip."""
    from app.media.store import _ASSETS  # noqa
    if not (_ASSETS / run_id).exists():
        raise HTTPException(status_code=404, detail="run not found")
    data = media.zip_kit(run_id, json.dumps({"run_id": run_id}), f"# Campaign kit {run_id}\n")
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="cortex-kit-{run_id}.zip"'})
