# Demo script & talking points

A 5-minute walkthrough that lands with both non-technical leaders and engineers.

## Setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r apps/api/requirements.txt
python scripts/ingest_brand.py
cd apps/api && uvicorn app.main:app --port 8000
# open http://localhost:8000
```

## The 5-minute run
1. **Frame it (leaders):** "Give it a marketing brief; a team of AI specialists produces a
   ready-to-review campaign, with a human sign-off before anything ships." Paste the Sai
   Energy brief. Click **Run (auto-approve)**.
2. **Watch the crew (engineers):** the Agent Activity log streams each specialist in order:
   ingest → planner → research → strategy → copy → creative → critic → packager. Point out
   this is a real LangGraph state machine, not one prompt.
3. **Grounding:** research pulls from the ingested brand doc (RAG) + web signals and cites
   sources. Not hallucinated.
4. **Guardrails:** the critic scores brand-safety; off-brand copy is blocked and cannot be
   published. Show a bad brief ("say our competitor's cylinders are deadly") getting blocked.
5. **Cost & trace:** the Cost & Trace panel shows per-agent latency + tokens + USD. The critic
   is priciest (heavy model); cheap steps run on Haiku; a repeat run costs ~$0 (prompt cache).
6. **Human-in-the-loop:** click **Run with approval**. The run pauses; you review the draft
   and Approve or Reject. Rejection loops back for a revision. This is a real graph interrupt.

## The one-liner
"Every pillar that separates a demo from production is here and demonstrable: guardrails,
three kinds of memory, security, tracing, evals, cost control, human-in-the-loop, scaling,
and grounded tool use, in one coherent system I can walk you through end to end."

## Toughest-question answers
- *Cost at scale?* Model-tier routing + prompt caching + per-agent metering; the critic is
  the cost driver and is bounded by `MAX_CRITIC_LOOPS`.
- *Hallucination?* RAG grounding + citation-coverage eval + the brand-safety critic gate.
- *Reliability?* Typed Pydantic contracts per agent, checkpointed resumable runs, CI eval gate.
- *Security?* Injection neutralization + PII redaction before the model, auth + rate limiting,
  env-only secrets. See docs/security.md.
