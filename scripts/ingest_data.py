#!/usr/bin/env python3
"""AI.AS MVP ingestion script for building dual FAISS indexes."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingest.pipeline import run_ingestion_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI.AS Data Ingestion Pipeline v2")
    parser.add_argument(
        "--images-dir", required=True, help="Path to product images directory"
    )
    parser.add_argument("--excel", required=True, help="Path to SKU Excel file")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Reserved for compatibility"
    )
    args = parser.parse_args()

    if args.batch_size != 16:
        logger.warning(
            "--batch-size is currently reserved and not used by this pipeline"
        )

    stats = run_ingestion_pipeline(images_dir=args.images_dir, excel_path=args.excel)
    logger.info(
        "Ingestion summary: total=%s, with_images=%s, multi_angle=%s, text_only=%s, elapsed=%ss",
        stats["total_products"],
        stats["products_with_images"],
        stats["products_multi_angle"],
        stats["products_text_only"],
        stats["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
