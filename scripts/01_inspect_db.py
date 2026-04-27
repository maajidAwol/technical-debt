"""
Stage 1 - Database smoke test and schema inventory.

Opens ``data/raw/td_V2.db`` and produces a canonical inventory of tables,
columns, row counts, and sample rows. The purpose is to verify the exact
schema (names, types) before writing any downstream SQL or feature code.

Outputs
-------
- ``results/tables/db_schema.csv`` - one row per (table, column), with
  column types and example value extracted from the first sample row.
- ``results/tables/db_table_counts.csv`` - one row per table with row count.
- ``results/tables/db_samples/<table>.csv`` - 3 sample rows per table.

Run
---
.. code-block:: bash

    .\\venv\\Scripts\\python.exe scripts/01_inspect_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import TABLES_DIR, TD_DATASET_PATH  # noqa: E402
from src.data.load_data import (  # noqa: E402
    get_connection,
    get_row_count,
    get_sample_rows,
    get_table_names,
    get_table_schema,
)


SAMPLES_DIR = TABLES_DIR / "db_samples"


def main() -> None:
    print(f"[Stage 1] Opening: {TD_DATASET_PATH}")
    print(f"[Stage 1] Size (MB): {TD_DATASET_PATH.stat().st_size / (1024 ** 2):.1f}")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        tables = get_table_names(conn)
        print(f"[Stage 1] Found {len(tables)} tables:")
        for t in tables:
            print(f"   - {t}")

        schema_rows: list[dict] = []
        count_rows: list[dict] = []

        for table in tables:
            try:
                count = get_row_count(conn, table)
            except Exception as exc:
                print(f"   ! COUNT failed for {table}: {exc}")
                count = -1
            count_rows.append({"table": table, "row_count": count})

            try:
                schema = get_table_schema(conn, table)
                sample = get_sample_rows(conn, table, 3)
            except Exception as exc:
                print(f"   ! PRAGMA/sample failed for {table}: {exc}")
                continue

            sample.to_csv(SAMPLES_DIR / f"{table}.csv", index=False)

            first_row = sample.iloc[0] if len(sample) > 0 else pd.Series(dtype=object)
            for _, col in schema.iterrows():
                col_name = col["name"]
                example = first_row.get(col_name, None) if len(first_row) else None
                if isinstance(example, str) and len(example) > 120:
                    example = example[:117] + "..."
                schema_rows.append({
                    "table": table,
                    "column": col_name,
                    "type": col["type"],
                    "notnull": bool(col["notnull"]),
                    "pk": bool(col["pk"]),
                    "example_value": example,
                })

    schema_df = pd.DataFrame(schema_rows)
    counts_df = pd.DataFrame(count_rows).sort_values("row_count", ascending=False)

    schema_path = TABLES_DIR / "db_schema.csv"
    counts_path = TABLES_DIR / "db_table_counts.csv"
    schema_df.to_csv(schema_path, index=False)
    counts_df.to_csv(counts_path, index=False)

    print("\n[Stage 1] Row-count summary:")
    for _, r in counts_df.iterrows():
        print(f"   {r['table']:<40} {r['row_count']:>15,}")

    print(f"\n[Stage 1] Schema written to: {schema_path}")
    print(f"[Stage 1] Counts written to : {counts_path}")
    print(f"[Stage 1] Samples written to: {SAMPLES_DIR}")
    print("[Stage 1] Complete.")


if __name__ == "__main__":
    main()
