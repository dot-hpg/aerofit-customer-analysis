"""Export the executed notebook to an A4 PDF with headless Chrome/Chromium
(the same engine as Chrome's Print -> Save as PDF).

Usage (from the repo root):
    python scripts/export_pdf.py [path/to/chrome]
"""
import shutil, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
nb = root / "notebooks" / "Aerofit_Case_Study.ipynb"
html = root / "reports" / "_print.html"
pdf = root / "reports" / "Aerofit_Case_Study.pdf"

chrome = (sys.argv[1] if len(sys.argv) > 1 else None) or next(
    (shutil.which(c) for c in ["google-chrome", "chromium", "chromium-browser", "chrome"] if shutil.which(c)), None)
if not chrome:
    sys.exit("Chrome/Chromium not found; pass its path as the first argument.")

subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", str(nb),
                "--output-dir", str(html.parent), "--output", html.stem], check=True)

PRINT_CSS = """<style>
@page { size: A4; margin: 11mm 9mm; }
body { zoom: 0.82; }
.jp-InputPrompt, .jp-OutputPrompt { display: none !important; }
.jp-OutputArea-output img { max-width: 100% !important; height: auto; break-inside: avoid; }
.jp-RenderedHTMLCommon table { font-size: 11px; }
.jp-RenderedHTMLCommon h2 { break-before: auto; break-after: avoid; }
.jp-Notebook { padding: 0 !important; }
</style>"""
html.write_text(html.read_text(encoding="utf-8").replace("</head>", PRINT_CSS + "</head>"), encoding="utf-8")

subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True)
html.unlink()
print("wrote", pdf)
