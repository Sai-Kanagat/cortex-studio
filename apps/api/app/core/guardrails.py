"""Input guardrails: prompt-injection heuristics + PII redaction.

Deliberately lightweight and dependency-free for M1/M3 seed. Swap the regex PII
pass for Presidio and the injection heuristic for a classifier later; the call
sites (sanitize_input, redact_pii) stay stable."""
from __future__ import annotations

import re

# Phrases that commonly signal an injection attempt embedded in user content.
_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier|foregoing)?\s*instructions",
    r"disregard\s+(?:your|the|all|previous)\s+(?:system\s+)?(?:prompt|instructions)",
    r"you\s+are\s+now",
    r"reveal\s+(?:your|the)\s+(?:system\s+prompt|instructions|prompt)",
    r"leak\s+(?:your|the)\s+(?:system\s+)?prompt",
    r"</?(?:system|assistant)>",
]

_PII_PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE": r"(?:(?:\+|00)\d{1,3}[\s-]?)?(?:\d[\s-]?){9,12}\d",
    "CARD": r"\b(?:\d[ -]?){13,16}\b",
}


def detect_injection(text: str) -> list[str]:
    hits = []
    low = text.lower()
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, low):
            hits.append(pat)
    return hits


def redact_pii(text: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    out = text
    for label, pat in _PII_PATTERNS.items():
        if re.search(pat, out):
            flags.append(label)
            out = re.sub(pat, f"[REDACTED_{label}]", out)
    return out, flags


def sanitize_input(text: str) -> tuple[str, list[str]]:
    """Return (safe_text, flags). Neutralizes injection markers and redacts PII so
    nothing sensitive reaches the model or the traces."""
    flags = [f"INJECTION:{p}" for p in detect_injection(text)]
    # Neutralize by fencing rather than deleting, so the agent still sees intent.
    safe = text
    for pat in _INJECTION_PATTERNS:
        safe = re.sub(pat, "[neutralized-instruction]", safe, flags=re.IGNORECASE)
    safe, pii_flags = redact_pii(safe)
    flags.extend(f"PII:{f}" for f in pii_flags)
    return safe, flags
