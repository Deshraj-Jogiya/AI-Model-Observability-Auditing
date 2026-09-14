"""Exports the real SQLite tables to CSV for Tableau Public to consume.

Tableau Public (the free desktop app, as opposed to paid Tableau Desktop)
only supports file-based data sources -- Excel, CSV, PDF, spatial files, web
data connectors -- not live database connections (SQLite/MySQL/Postgres
require a driver only the paid product ships). The README's "Connect -> To a
Server -> SQLite" instructions describe paid Tableau Desktop; this script is
the real path for Tableau Public.
"""
import os
import sqlite3

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "observability.db")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def export():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"{DB_PATH} not found. Run etl/log_ingestion.py and analytics/audit_engine.py first.")

    conn = sqlite3.connect(DB_PATH)

    logs = pd.read_sql_query("SELECT * FROM inference_logs", conn)
    audit = pd.read_sql_query("SELECT * FROM fairness_audit_history", conn)
    conn.close()

    logs_path = os.path.join(DATA_DIR, "inference_logs.csv")
    audit_path = os.path.join(DATA_DIR, "fairness_audit_history.csv")
    logs.to_csv(logs_path, index=False)
    audit.to_csv(audit_path, index=False)

    print(f"Exported {len(logs):,} rows to {logs_path}")
    print(f"Exported {len(audit):,} rows to {audit_path}")


if __name__ == "__main__":
    export()
