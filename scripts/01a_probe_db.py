"""
Stage 1a - Minimal DB probe.

Smallest possible smoke test. Only:
  1. Opens the SQLite file.
  2. Lists table names.
  3. Runs COUNT(*) on each table.

No PRAGMA per-column reads, no sample rows, no pandas. Designed to finish
in well under a minute even on a 1.5 GB file. If this runs cleanly, we
know the DB is readable and we can proceed to the full inventory.

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/01a_probe_db.py
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import TD_DATASET_PATH  # noqa: E402


def main() -> None:
    t0 = time.time()
    size_mb = TD_DATASET_PATH.stat().st_size / (1024 ** 2)
    print(f"[probe] Path : {TD_DATASET_PATH}")
    print(f"[probe] Size : {size_mb:.1f} MB")
    print(f"[probe] Open ...", flush=True)

    conn = sqlite3.connect(str(TD_DATASET_PATH))
    print(f"[probe] Open OK ({time.time() - t0:.2f}s)")

    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [r[0] for r in cur.fetchall()]
    print(f"[probe] {len(tables)} tables found:")

    print(f"\n{'table':<40} {'rows':>15}   {'elapsed':>8}")
    print("-" * 68)
    for t in tables:
        t1 = time.time()
        try:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            n = cur.fetchone()[0]
        except Exception as exc:
            n = -1
            print(f"{t:<40} {'ERR':>15}   {time.time() - t1:>7.2f}s  {exc}")
            continue
        print(f"{t:<40} {n:>15,}   {time.time() - t1:>7.2f}s", flush=True)

    conn.close()
    print(f"\n[probe] Done in {time.time() - t0:.2f}s total.")


if __name__ == "__main__":
    main()
