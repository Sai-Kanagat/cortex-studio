"""Capture ONE real campaign run (events + package + cost + trace) into a JSON seed
that the keyless client-side demo replays. Run with a real provider once:

    set -a && . .env && set +a && python scripts/capture_seed.py
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
OUT = Path(__file__).resolve().parents[1] / "apps" / "web" / "seed.json"


def main() -> int:
    agents.reset_run("seed")
    state = build_graph().invoke({"run_id": "seed", "brief": BRIEF})
    seed = {
        "brief": BRIEF,
        "events": state.get("events", []),
        "package": state.get("package"),
        "cost": state.get("cost"),
        "trace": state.get("trace"),
    }
    OUT.write_text(json.dumps(seed, indent=2, ensure_ascii=False))
    print(f"seed written: {OUT} ({OUT.stat().st_size} bytes)")
    print("provider:", "REAL" if state['package']['copy'].get('headline','').find('[mock]') == -1 else "MOCK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
