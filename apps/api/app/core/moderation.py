"""Output-side guardrails: moderate final copy before it can be published.

Complements the input guardrails in guardrails.py. Checks the generated campaign
copy against (a) a banned-content list and (b) brand Don'ts loaded from the brand
store, returning structured violations the critic/packager can act on."""
from __future__ import annotations

import re
from typing import Any

# Categories we never want in outbound marketing copy.
_BANNED = {
    "unsafe_claims": [r"guaranteed?\s+safe", r"no\s+risk", r"100%\s+safe"],
    "fearmongering": [r"deadly", r"explosion.*neighbou?r", r"will\s+kill"],
    "competitor_attack": [r"competitor.*dangerous", r"their\s+cylinders.*unsafe"],
}


def moderate_text(text: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    low = (text or "").lower()
    for category, pats in _BANNED.items():
        for pat in pats:
            if re.search(pat, low):
                violations.append({"category": category, "pattern": pat})
    return violations


def moderate_copy(copy: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a copy dict and moderate all human-facing strings."""
    blob = " ".join(
        str(v) for v in [
            copy.get("headline", ""),
            copy.get("email_subject", ""),
            copy.get("email_body", ""),
            *(copy.get("captions", []) or []),
        ]
    )
    return moderate_text(blob)
