"""
Stage 0 - Workspace cleanup. Run before any rebuild.

Deletes all outputs of previous pipeline runs so the new
single-variant pipeline starts from a clean slate. Source
code, raw data, MSc proposal reference material, and the
thesis_v2 references are preserved.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DELETE_DIRS = [
    "results/tables",
    "results/figures",
    "results/appendix",
    "models",
    "data/processed",
    "notebooks",
    "tools",
    "__pycache__",
]

RECREATE_DIRS = [
    "results/tables",
    "results/figures",
    "data/processed",
    "models",
    "docs",
]

DELETE_ROOT_FILES = [
    "EXECUTION_PLAN.md",
    "RESEARCH_LOG.md",
    "THESIS_V2_CHANGES_SUMMARY.md",
    "UPDATE_TOC_INSTRUCTIONS.txt",
    "tempdiscussion1.md",
    "results/pip_install.log",
    "results/stage10_log.txt",
]

KEEP_IN_DOCS = {
    "thesis_v2.md",
    "thesis_v2_updated.docx",
    "thesis_references.bib",
}


def _rm_path(p: Path) -> None:
    if not p.exists():
        return
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    print(f"  deleted  {p.relative_to(PROJECT_ROOT)}")


def clean_docs() -> None:
    docs = PROJECT_ROOT / "docs"
    if not docs.exists():
        return
    print("[Stage 0] Pruning docs/ (preserving thesis_v2 references) ...")
    for entry in docs.iterdir():
        if entry.name in KEEP_IN_DOCS:
            print(f"  kept     docs/{entry.name}")
            continue
        _rm_path(entry)


def main() -> int:
    print(f"[Stage 0] Project root: {PROJECT_ROOT}")
    print("[Stage 0] Deleting previous outputs ...")
    for rel in DELETE_DIRS:
        _rm_path(PROJECT_ROOT / rel)
    for rel in DELETE_ROOT_FILES:
        _rm_path(PROJECT_ROOT / rel)

    clean_docs()

    print("[Stage 0] Recreating empty folders ...")
    for rel in RECREATE_DIRS:
        (PROJECT_ROOT / rel).mkdir(parents=True, exist_ok=True)
        print(f"  created  {rel}")

    print("\n[Stage 0] Preserved (untouched):")
    for label in (
        "data/raw/",
        "src/",
        "scripts/",
        "config.py",
        "requirements.txt",
        "run_pipeline.py",
        "proposal/",
        "results/run_logs/",
        "venv/",
        ".git/",
    ):
        path = PROJECT_ROOT / label.rstrip("/")
        exists = "exists" if path.exists() else "absent"
        print(f"  {label:<22} ({exists})")

    print("\n[Stage 0] Workspace clean. Ready for new run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
