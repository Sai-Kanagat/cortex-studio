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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


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
