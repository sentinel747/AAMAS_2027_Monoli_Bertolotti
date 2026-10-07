# -*- coding: utf-8 -*-
"""Rigenera docs/MARS_ABM_GUIDA_DIVULGATIVA.pdf dalla sorgente Markdown.

Uso (da radice repo):  python scripts/genera_pdf_guida.py

Pipeline: Markdown -> HTML (python-markdown, estensione tables) -> PDF via
Microsoft Edge headless (sempre presente su Windows 11). La sorgente di
verita' e' docs/MARS_ABM_GUIDA_DIVULGATIVA.md: dopo ogni modifica alle regole
degli agenti aggiornare il Markdown e rilanciare questo script.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "MARS_ABM_GUIDA_DIVULGATIVA.md"
PDF = ROOT / "docs" / "MARS_ABM_GUIDA_DIVULGATIVA.pdf"

EDGE_CANDIDATES = [
    pathlib.Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    pathlib.Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

TEMPLATE = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Mars Colony ABM — Guida divulgativa</title>
<style>
  @page { size: A4; margin: 17mm 16mm 19mm 16mm; }
  * { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  html { font-size: 10.6pt; }
  body { font-family: "Segoe UI", "Calibri", sans-serif; color: #26282e; line-height: 1.55; margin: 0; }
  h1 { font-size: 21pt; line-height: 1.2; color: #8c3319; margin: 0 0 2pt 0; }
  h1 + p strong { font-size: 12pt; color: #444a55; }
  h2 { font-size: 14pt; color: #8c3319; border-bottom: 1.4pt solid #d8a48f;
       padding-bottom: 3pt; margin: 20pt 0 8pt 0; page-break-after: avoid; }
  h3 { font-size: 11.5pt; color: #444a55; margin: 12pt 0 4pt 0; page-break-after: avoid; }
  p { margin: 5pt 0; text-align: justify; hyphens: auto; }
  ul, ol { margin: 5pt 0; padding-left: 18pt; }
  li { margin: 2.5pt 0; text-align: justify; hyphens: auto; }
  strong { color: #1d1f24; }
  em { color: #3d414a; }
  code { font-family: "Consolas", monospace; font-size: 9pt; background: #f1efec;
         padding: 0.5pt 3pt; border-radius: 2pt; }
  hr { border: 0; border-top: 0.7pt solid #cfd2d8; margin: 14pt 0; }
  table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9.2pt; line-height: 1.35; }
  thead { display: table-header-group; }
  th { background: #8c3319; color: #ffffff; text-align: left; padding: 4pt 6pt; font-weight: 600; }
  td { border-bottom: 0.6pt solid #d9dbe0; padding: 4pt 6pt; vertical-align: top; }
  tr:nth-child(even) td { background: #f6f3f0; }
  tr { page-break-inside: avoid; }
  blockquote { margin: 8pt 0; padding: 6pt 10pt; border-left: 3pt solid #d8a48f;
               background: #faf6f3; color: #3d414a; }
</style>
</head>
<body>
__BODY__
</body>
</html>
"""


def main() -> int:
    edge = next((p for p in EDGE_CANDIDATES if p.exists()), None)
    if edge is None:
        print("Edge non trovato: aggiorna EDGE_CANDIDATES in questo script.")
        return 1

    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables"], output_format="html5")
    with tempfile.TemporaryDirectory() as tmp:
        html_path = pathlib.Path(tmp) / "guida_divulgativa.html"
        html_path.write_text(TEMPLATE.replace("__BODY__", body), encoding="utf-8")
        cmd = [
            str(edge),
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={pathlib.Path(tmp) / 'edgeprofile'}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF}",
            html_path.as_uri(),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
    if not PDF.exists() or PDF.stat().st_size == 0:
        print("Conversione fallita:")
        print(result.stdout)
        print(result.stderr)
        return 1
    print(f"OK: {PDF} ({PDF.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
