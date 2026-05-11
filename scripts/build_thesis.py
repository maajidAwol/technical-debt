"""
Convert ``docs/thesis.md`` to ``docs/thesis.docx`` using pandoc.

This is a thin wrapper around the pandoc command line so that the thesis
can be regenerated with a single, memorable invocation:

    python scripts/build_thesis.py

The script does not modify any artefact under ``results/`` and does not
rerun the experimental pipeline. It only converts the existing Markdown
source into a Word document suitable for AASTU submission per the
guideline section 4 (A4, Times New Roman 12pt, 1.5 line spacing, 1.5"
left margin, 1" elsewhere). If a reference Word template is found at
``docs/aastu_reference.docx`` the script passes it to pandoc via
``--reference-doc``; otherwise the student should adjust the formatting
once in Word and save that file as the reference template for future
runs.

Optional flags forwarded to pandoc:
    --pdf            Also produce ``docs/thesis.pdf`` if a LaTeX engine
                     is available on PATH (xelatex / lualatex / pdflatex).
    --no-toc         Skip the table-of-contents (default: include).
    --quiet          Suppress pandoc's normal stderr output.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MD_PATH = PROJECT_ROOT / "docs" / "thesis.md"
DOCX_PATH = PROJECT_ROOT / "docs" / "thesis.docx"
PDF_PATH = PROJECT_ROOT / "docs" / "thesis.pdf"
REFERENCE_DOCX = PROJECT_ROOT / "docs" / "aastu_reference.docx"


def _ensure_pandoc() -> str:
    """Return the absolute path to the pandoc executable or exit cleanly."""
    pandoc = shutil.which("pandoc")
    if pandoc is None and sys.platform == "win32":
        # Winget/MSI often installs here but does not update PATH for existing shells.
        candidates: list[Path] = [
            Path(r"C:\Program Files\Pandoc\pandoc.exe"),
        ]
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            candidates.insert(0, Path(local) / "Pandoc" / "pandoc.exe")
        for path in candidates:
            if path.is_file():
                pandoc = str(path)
                break
    if pandoc is None:
        sys.exit(
            "pandoc was not found on PATH. Install it from "
            "https://pandoc.org/installing.html (Windows: winget install pandoc; "
            "macOS: brew install pandoc; Linux: apt install pandoc) and re-run."
        )
    return pandoc


def _build_docx(pandoc: str, *, include_toc: bool, quiet: bool) -> None:
    if not MD_PATH.exists():
        sys.exit(f"Source not found: {MD_PATH}. Author docs/thesis.md first.")

    cmd: list[str] = [
        pandoc,
        str(MD_PATH),
        "-o",
        str(DOCX_PATH),
        f"--resource-path={PROJECT_ROOT}",
        "--number-sections",
    ]
    if include_toc:
        cmd += ["--toc", "--toc-depth=3"]
    if REFERENCE_DOCX.exists():
        cmd += [f"--reference-doc={REFERENCE_DOCX}"]

    if not quiet:
        print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"Wrote {DOCX_PATH.relative_to(PROJECT_ROOT)}", flush=True)


def _build_pdf(pandoc: str, *, quiet: bool) -> None:
    if not MD_PATH.exists():
        sys.exit(f"Source not found: {MD_PATH}.")
    cmd: list[str] = [
        pandoc,
        str(MD_PATH),
        "-o",
        str(PDF_PATH),
        f"--resource-path={PROJECT_ROOT}",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
    ]
    if not quiet:
        print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"Wrote {PDF_PATH.relative_to(PROJECT_ROOT)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true", help="Also produce a PDF.")
    parser.add_argument("--no-toc", action="store_true", help="Skip the table of contents.")
    parser.add_argument("--quiet", action="store_true", help="Suppress pandoc stderr.")
    args = parser.parse_args()

    pandoc = _ensure_pandoc()
    _build_docx(pandoc, include_toc=not args.no_toc, quiet=args.quiet)
    if args.pdf:
        _build_pdf(pandoc, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
