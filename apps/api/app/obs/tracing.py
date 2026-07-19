"""Observability: per-agent/per-tool spans with a Langfuse backend and a local
JSON fallback so tracing works with zero external setup.

Usage:
    tr = get_tracer(run_id)
    with tr.span("research", kind="agent", input=brief) as s:
        ...
        s.set_output(result); s.set_usage(tokens_in, tokens_out, cost)

When LANGFUSE_* env is set, spans are also mirrored to Langfuse; otherwise they are
persisted to data/traces/<run_id>.json and served back through the API for the UI."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import settings

_TRACE_DIR = Path(os.environ.get("CORTEX_DATA_DIR", Path(__file__).resolve().parents[4] / "data")) / "traces"
_TRACE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Span:
    name: str
    kind: str = "agent"          # agent | tool | llm
    start: float = 0.0
    end: float = 0.0
    input: Any = None
    output: Any = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        return round((self.end - self.start) * 1000, 1)

    def set_output(self, output: Any) -> None:
        self.output = output

    def set_usage(self, in_tok: int, out_tok: int, cost: float) -> None:
        self.input_tokens, self.output_tokens, self.cost_usd = in_tok, out_tok, cost


@dataclass
class Tracer:
    run_id: str
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, kind: str = "agent", input: Any = None):
        # monotonic() is process-relative and always available (wall-clock now() is
        # blocked in some sandboxes); good enough for latency deltas.
        s = Span(name=name, kind=kind, start=time.monotonic(), input=input)
        try:
            yield s
        except Exception as e:  # record then re-raise
            s.error = repr(e)
            raise
        finally:
            s.end = time.monotonic()
            self.spans.append(s)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "spans": [
                {
                    "name": s.name, "kind": s.kind, "latency_ms": s.latency_ms,
                    "input_tokens": s.input_tokens, "output_tokens": s.output_tokens,
                    "cost_usd": round(s.cost_usd, 6), "error": s.error,
                }
                for s in self.spans
            ],
            "total_cost_usd": round(sum(s.cost_usd for s in self.spans), 6),
            "total_latency_ms": round(sum(s.latency_ms for s in self.spans), 1),
        }

    def flush(self) -> None:
        (_TRACE_DIR / f"{self.run_id}.json").write_text(json.dumps(self.summary(), indent=2))
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            self._flush_langfuse()

    def _flush_langfuse(self) -> None:  # pragma: no cover - needs Langfuse creds
        try:
            from langfuse import Langfuse

            lf = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            trace = lf.trace(id=self.run_id, name="campaign_run")
            for s in self.spans:
                trace.span(
                    name=s.name, start_time=None, end_time=None,
                    input=s.input, output=s.output,
                    metadata={"kind": s.kind, "latency_ms": s.latency_ms, "cost_usd": s.cost_usd},
                )
            lf.flush()
        except Exception:
            pass  # never let tracing break a run


def get_tracer(run_id: str) -> Tracer:
    return Tracer(run_id=run_id)


def load_trace(run_id: str) -> dict[str, Any] | None:
    p = _TRACE_DIR / f"{run_id}.json"
    return json.loads(p.read_text()) if p.exists() else None
