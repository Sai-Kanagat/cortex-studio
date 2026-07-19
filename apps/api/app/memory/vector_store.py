"""Vector store abstraction.

Two backends behind one interface:
- InMemoryVectorStore: JSON-persisted, no external deps -> tests/demos run anywhere.
- PgVectorStore (production): documented, selected when DATABASE_URL points at a live
  pgvector instance and USE_PGVECTOR=1. Kept thin so M2 stays runnable offline.

The retriever/agents only ever touch `get_store()`, `add()`, `search()`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.memory.embeddings import cosine, embed

_DATA_DIR = Path(os.environ.get("CORTEX_DATA_DIR", Path(__file__).resolve().parents[4] / "data")) / "store"
_DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Doc:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


class InMemoryVectorStore:
    def __init__(self, name: str):
        self.name = name
        self.path = _DATA_DIR / f"{name}.json"
        self.docs: list[Doc] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.docs = [Doc(**d) for d in raw]

    def _persist(self) -> None:
        self.path.write_text(json.dumps([vars(d) for d in self.docs]))

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        self.docs = [d for d in self.docs if d.id != doc_id]  # upsert
        self.docs.append(Doc(doc_id, text, metadata or {}, embed(text)))
        self._persist()

    def search(self, query: str, k: int = 3) -> list[tuple[Doc, float]]:
        q = embed(query)
        scored = [(d, cosine(q, d.embedding)) for d in self.docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self.docs)


# PgVectorStore intentionally omitted from the runnable path for M2; see
# docs/memory.md for the production DDL (vector(256) column + ivfflat index) and
# the drop-in class. get_store() will select it when USE_PGVECTOR=1.

_STORES: dict[str, InMemoryVectorStore] = {}


def get_store(name: str) -> InMemoryVectorStore:
    if name not in _STORES:
        _STORES[name] = InMemoryVectorStore(name)
    return _STORES[name]
