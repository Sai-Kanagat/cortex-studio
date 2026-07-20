"""Export a campaign package to Markdown (and PDF if reportlab is available).

    python scripts/export.py                 # runs a mock campaign, writes out/campaign.md
    python scripts/export.py path/to/pkg.json # export an existing package JSON
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "api"))

from app.graph import agents  # noqa: E402
from app.graph.build import build_graph  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"
OUT.mkdir(exist_ok=True)


def to_markdown(pkg: dict) -> str:
    s = pkg.get("strategy", {}) or {}
    c = pkg.get("copy", {}) or {}
    cr = pkg.get("creative", {}) or {}
    loc = pkg.get("localization", {}) or {}
    q = pkg.get("critique", {}) or {}
    lines = [
        "# Campaign Package",
        f"\n**Status:** {pkg.get('status')}  ·  **Brand-safety score:** {q.get('score')}",
        f"\n## Brief\n{pkg.get('brief','')}",
        f"\n## Positioning\n{s.get('positioning','')}",
        f"\n## Channels\n" + ", ".join(s.get("channels", []) or []),
        f"\n## Headline\n{c.get('headline','')}",
        "\n## Captions\n" + "\n".join(f"- {x}" for x in c.get("captions", []) or []),
        f"\n## Email\n**{c.get('email_subject','')}**\n\n{c.get('email_body','')}",
        f"\n## Creative concept\n{cr.get('concept','')}",
        "\n## Image prompts\n" + "\n".join(f"- {x}" for x in cr.get("image_prompts", []) or []),
    ]
    if loc.get("headline_ta"):
        lines.append(f"\n## Tamil localization\n{loc.get('headline_ta','')}\n"
                     + "\n".join(f"- {x}" for x in loc.get("captions_ta", []) or []))
    return "\n".join(lines) + "\n"


def maybe_pdf(md: str, path: Path) -> bool:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        return False
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in md.splitlines():
        c.drawString(40, y, line[:100])
        y -= 14
        if y < 40:
            c.showPage(); y = 800
    c.save()
    return True


def main() -> int:
    if len(sys.argv) > 1:
        pkg = json.loads(Path(sys.argv[1]).read_text())
    else:
        agents.reset_run("export")
        pkg = build_graph().invoke({"run_id": "export", "brief": "Refill campaign for Sai Energy LPG, Coimbatore."})["package"]
    md = to_markdown(pkg)
    (OUT / "campaign.md").write_text(md)
    pdf = maybe_pdf(md, OUT / "campaign.pdf")
    print(f"wrote {OUT/'campaign.md'}" + (f" and {OUT/'campaign.pdf'}" if pdf else " (PDF skipped: pip install reportlab)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
