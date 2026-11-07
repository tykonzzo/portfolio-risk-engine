"""Build the self-contained dashboard (docs/index.html) from outputs/results.json.

docs/index.html is a plain static page: open it locally or serve it with GitHub Pages.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).with_name("dashboard_template.html")

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>html{color-scheme:light}</style>
</head>
<body>
"""
TAIL = "\n</body>\n</html>\n"


def fragment(results_json: str, plotly_src: str | None = None) -> str:
    """Page body + styles + scripts (no doctype). `plotly_src` overrides the CDN URL for offline tests."""
    html = TEMPLATE.read_text()
    html = html.replace("__DATA__", results_json.replace("</", "<\\/"))
    if plotly_src:
        html = html.replace("https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js", plotly_src)
    return html


def build(results_path: Path = ROOT / "outputs/results.json", out: Path = ROOT / "docs/index.html") -> Path:
    page = HEAD + fragment(Path(results_path).read_text()) + TAIL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out


if __name__ == "__main__":
    print(build())
