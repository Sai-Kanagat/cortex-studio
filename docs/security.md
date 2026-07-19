# Security (M3)

## Threat surface & controls

| Risk | Control | Where |
|------|---------|-------|
| Prompt injection in the brief | Heuristic detect + neutralize before any agent | `core/guardrails.py` |
| PII leaking into prompts/traces | Regex redaction on ingest | `core/guardrails.py` |
| Off-brand / unsafe outbound copy | Deterministic moderation gate + publish block | `core/moderation.py`, packager |
| Unauthenticated access | Opt-in `X-API-Key` header check | `api/routes.py::require_api_key` |
| Abuse / cost blowout | Per-client token-bucket rate limit | `core/ratelimit.py` |
| Runaway agent loops | `MAX_CRITIC_LOOPS` bound | `core/config.py`, critic |

## Secrets

- All secrets come from environment variables only (`.env`, never committed — see `.gitignore`).
- `ANTHROPIC_API_KEY`, `CORTEX_API_KEY`, `LANGFUSE_*`, DB/Redis URLs.
- Production: inject via the platform secret store (Railway/Fly/Vercel env), not `.env` files.
- No secret is ever logged or written into traces; PII is redacted before it reaches the model.

## Hardening backlog (post-portfolio)

- Replace regex injection heuristic with a classifier; swap PII regex for Presidio.
- Move rate-limit buckets to Redis so limits hold across scaled instances (M6).
- Add per-key quotas + audit logging on publish actions.
