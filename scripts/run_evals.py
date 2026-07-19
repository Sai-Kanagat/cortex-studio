"""Run the golden-set evaluation and fail (exit 1) if thresholds are missed.

    python scripts/run_evals.py

Wire this into CI so a prompt/agent regression breaks the build."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from app.eval.harness import run_all  # noqa: E402

PASS_RATE_MIN = 1.0        # every golden case must pass its structural checks
FAITHFULNESS_MIN = 0.5     # at least half of findings carry a citation


def main() -> int:
    report = run_all()
    print(json.dumps(report, indent=2))
    ok = report["pass_rate"] >= PASS_RATE_MIN and report["avg_faithfulness"] >= FAITHFULNESS_MIN
    print("\nEVAL", "PASS" if ok else "FAIL",
          f"(pass_rate={report['pass_rate']}, faithfulness={report['avg_faithfulness']}, judge={report['avg_judge']})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
