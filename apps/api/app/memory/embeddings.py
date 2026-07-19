"""Embeddings with a deterministic offline mock and a real-provider hook.

The mock uses the hashing trick (bag-of-tokens hashed into a fixed-dim vector,
L2-normalized) so cosine similarity tracks keyword overlap. That makes RAG
retrieval demonstrably relevant with no API key. Swap `_real_embed` for Voyage/
OpenAI embeddings in production; the `embed()` signature stays put."""
from __future__ import annotations

import hashlib
import math
import re

from app.core.config import settings

DIM = 256


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _mock_embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for tok in _tokens(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % DIM
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _real_embed(text: str) -> list[float]:  # pragma: no cover - needs provider key
    # Placeholder for a real embedding provider (e.g. Voyage). Anthropic has no
    # embeddings endpoint, so production wires a dedicated embeddings model here.
    raise NotImplementedError("configure a real embeddings provider for production")


def embed(text: str) -> list[float]:
    if settings.llm_provider == "anthropic" and False:  # flip when a provider is wired
        return _real_embed(text)
    return _mock_embed(text)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are already normalized
