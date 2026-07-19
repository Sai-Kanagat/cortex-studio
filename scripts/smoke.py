"""CLI smoke run: execute one campaign end-to-end and pretty-print the package.

    python scripts/smoke.py            # mock LLM
    LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-... python scripts/smoke.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from app.graph import agents  # noqa: E402
from app.graph.build import build_graph  # noqa: E402

BRIEF = (
    "Launch a refill-signup campaign for Sai Energy LPG (Indane distributor, Coimbatore). "
    "Goal: more households booking cylinder refills via the app. Tone: trustworthy, local, safety-first."
)


def main() -> int:
    agents.reset_run("cli")
    graph = build_graph()
    print(f"\n== running brief ==\n{BRIEF}\n")
    final = graph.invoke({"run_id": "cli", "brief": BRIEF})
    for ev in final.get("events", []):
        print(f"  [{ev['agent']}] {ev['message']}")
    print("\n== package ==")
    print(json.dumps(final["package"], indent=2)[:1500])
    print("\n== cost ==")
    print(json.dumps(final["cost"]["total"], indent=2))
    assert final.get("package"), "no package produced"
    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
