"""Render the group-meeting markdown to a local HTML page."""
from __future__ import annotations

import html as htmllib
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "group_meeting_2026-08-24.md"
OUT = ROOT / "reports" / "group_meeting_2026-08-24.html"

CSS = """
:root {
  --fg: #1a1a1a;
  --muted: #555;
  --border: #ddd;
  --bg: #f6f7f9;
  --card: #fff;
  --accent: #1f4e79;
}
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: "Segoe UI", system-ui, sans-serif;
  color: var(--fg);
  background: var(--bg);
  line-height: 1.55;
}
.wrap {
  max-width: 900px;
  margin: 0 auto;
  padding: 36px 28px 96px;
  background: var(--card);
  box-shadow: 0 0 0 1px var(--border);
}
h1, h2, h3 { color: var(--accent); line-height: 1.25; }
h1 { font-size: 1.85rem; margin-top: 0; }
h2 {
  margin-top: 2.2rem;
  padding-top: 0.5rem;
  border-top: 1px solid var(--border);
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0 1.4rem;
  font-size: 0.92rem;
}
th, td {
  border: 1px solid var(--border);
  padding: 6px 10px;
  text-align: left;
  vertical-align: top;
}
th { background: #eef3f8; }
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0.6rem auto 0.3rem;
  border: 1px solid var(--border);
  background: #fff;
}
em { color: var(--muted); }
code {
  font-family: ui-monospace, Consolas, monospace;
  font-size: 0.88em;
  background: #f3f3f3;
  padding: 0.1em 0.35em;
  border-radius: 4px;
}
pre {
  background: #f4f6f8;
  padding: 12px 14px;
  overflow-x: auto;
  border-radius: 6px;
  border: 1px solid var(--border);
}
pre code { background: none; padding: 0; }
blockquote {
  margin: 1rem 0;
  padding: 0.15rem 1rem;
  border-left: 4px solid var(--accent);
  background: #f4f7fb;
}
.mermaid {
  background: #fff;
  padding: 12px;
  margin: 1rem 0;
  text-align: center;
}
"""


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "smarty"],
    )
    body = re.sub(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        lambda m: '<div class="mermaid">' + htmllib.unescape(m.group(1)) + "</div>",
        body,
        flags=re.DOTALL,
    )
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Group meeting — LLM hypothesis loop</title>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
</script>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{body}
</div>
</body>
</html>
"""
    OUT.write_text(page, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
