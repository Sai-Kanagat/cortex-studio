"""FastAPI entrypoint for Cortex Studio."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings

_WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="Cortex Studio API", version="0.1.0")

_origins = ["http://localhost:3000", "http://localhost:8000"] + [
    o.strip() for o in settings.cors_origins.split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
def _ingest_brand_docs() -> None:
    """Auto-ingest brand docs into the vector store on boot so the RAG-grounded
    research agent works on a fresh deploy without a manual ingestion step."""
    try:
        from app.memory.vector_store import get_store
        from app.tools.registry import BRAND_STORE

        store = get_store(BRAND_STORE)
        if store.count() > 0:
            return
        brand_dir = Path(__file__).resolve().parents[3] / "data" / "brand"
        for md in sorted(brand_dir.glob("*.md")):
            parts = [p.strip() for p in md.read_text().split("\n\n") if len(p.strip()) > 40]
            for i, ch in enumerate(parts):
                store.add(f"{md.stem}:{i}", ch, {"source": md.name})
    except Exception:
        pass  # never block startup on ingestion


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "using_real_llm": settings.use_real_llm,
        "models": {"cheap": settings.tier_cheap, "mid": settings.tier_mid, "heavy": settings.tier_heavy},
    }


# Serve the static dashboard at / (mounted last so it doesn't shadow /api or /health).
if _WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
