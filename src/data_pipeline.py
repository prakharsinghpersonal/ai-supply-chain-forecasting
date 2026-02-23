"""
SupplyGuard — Snowflake Data Ingestion & Processing Pipeline
Ingests raw API feeds into structured Snowflake datasets for
real-time logistical record tracking and analysis.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import snowflake.connector
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Snowflake connection
# ---------------------------------------------------------------------------

SNOWFLAKE_CONFIG = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "SUPPLY_WH"),
    "database": os.getenv("SNOWFLAKE_DATABASE", "SUPPLYGUARD"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "RAW"),
}


def get_connection():
    """Return an authenticated Snowflake connection."""
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)


# ---------------------------------------------------------------------------
# Data ingestion helpers
# ---------------------------------------------------------------------------

def ingest_shipment_feed(conn, feed_path: str) -> int:
    """Load a shipment feed CSV into the RAW.SHIPMENTS staging table."""
    df = pd.read_csv(feed_path, parse_dates=["ship_date", "expected_delivery"])
    df["ingested_at"] = datetime.utcnow()

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS RAW.SHIPMENTS (
            shipment_id     STRING,
            origin          STRING,
            destination     STRING,
            carrier         STRING,
            ship_date       TIMESTAMP,
            expected_delivery TIMESTAMP,
            weight_kg       FLOAT,
            status          STRING,
            risk_score      FLOAT,
            ingested_at     TIMESTAMP
        )
    """)

    rows = [tuple(r) for r in df.itertuples(index=False)]
    cursor.executemany(
        "INSERT INTO RAW.SHIPMENTS VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows
    )
    logger.info("Ingested %d shipment records from %s", len(rows), feed_path)
    return len(rows)


def ingest_supplier_feed(conn, feed_path: str) -> int:
    """Load a supplier performance feed into RAW.SUPPLIERS."""
    df = pd.read_csv(feed_path, parse_dates=["report_date"])
    df["ingested_at"] = datetime.utcnow()

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS RAW.SUPPLIERS (
            supplier_id         STRING,
            supplier_name       STRING,
            region              STRING,
            on_time_rate        FLOAT,
            defect_rate         FLOAT,
            avg_lead_time_days  FLOAT,
            report_date         TIMESTAMP,
            ingested_at         TIMESTAMP
        )
    """)

    rows = [tuple(r) for r in df.itertuples(index=False)]
    cursor.executemany(
        "INSERT INTO RAW.SUPPLIERS VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", rows
    )
    logger.info("Ingested %d supplier records from %s", len(rows), feed_path)
    return len(rows)


# ---------------------------------------------------------------------------
# Concurrent multi-source ingestion
# ---------------------------------------------------------------------------

def run_parallel_ingestion(feed_manifest: dict) -> dict:
    """
    Ingest multiple feeds concurrently.

    Parameters
    ----------
    feed_manifest : dict
        Mapping of feed type ("shipments" | "suppliers") to list of file paths.

    Returns
    -------
    dict  – summary with total rows per feed type.
    """
    conn = get_connection()
    results = {}

    ingest_fn = {
        "shipments": ingest_shipment_feed,
        "suppliers": ingest_supplier_feed,
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for feed_type, paths in feed_manifest.items():
            for path in paths:
                fut = executor.submit(ingest_fn[feed_type], conn, path)
                futures[fut] = (feed_type, path)

        for fut in as_completed(futures):
            feed_type, path = futures[fut]
            try:
                count = fut.result()
                results.setdefault(feed_type, 0)
                results[feed_type] += count
            except Exception:
                logger.exception("Failed to ingest %s from %s", feed_type, path)

    conn.close()
    logger.info("Ingestion complete — %s", results)
    return results


if __name__ == "__main__":
    manifest = {
        "shipments": ["data/raw/shipments_2024.csv"],
        "suppliers": ["data/raw/suppliers_2024.csv"],
    }
    run_parallel_ingestion(manifest)
