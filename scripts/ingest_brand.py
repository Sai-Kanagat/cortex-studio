"""Ingest brand documents into the vector store for RAG.

    python scripts/ingest_brand.py [data/brand]

Chunks each markdown file on blank lines and upserts chunks into the "brand" store.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from app.tools.registry import BRAND_STORE  # noqa: E402
from app.memory.vector_store import get_store  # noqa: E402


def chunk(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if len(p) > 40]


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "brand"
    store = get_store(BRAND_STORE)
    n = 0
    for md in sorted(root.glob("*.md")):
        for i, ch in enumerate(chunk(md.read_text())):
            store.add(f"{md.stem}:{i}", ch, {"source": md.name})
            n += 1
    print(f"ingested {n} chunks from {root} -> store now has {store.count()} docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
