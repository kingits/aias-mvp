#!/usr/bin/env python3
"""
AI.AS MVP - Search Accuracy Evaluation

Tests search quality by using each product's own image as a query
and measuring whether the correct product appears in top-K results.

Usage:
    python scripts/evaluate_search.py --images-dir <path> --top-k 5
"""
import argparse
import json
import logging
import sys
import time
import random
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import INDEX_DIR, METADATA_PATH
from backend.models.clip_model import CLIPEmbedder
from backend.search.engine import SearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("evaluate")


def find_test_images(images_dir: Path, wise_code: str) -> list[Path]:
    """Find test images for a product (prefer with.background for realistic test)."""
    code_no_dash = wise_code.replace("-", "")
    results = []

    # Use with.background images as "user query" (more realistic — user photos have backgrounds)
    with_bg_dir = images_dir / "with.background"
    if with_bg_dir.exists():
        for f in with_bg_dir.iterdir():
            if f.stem.startswith(code_no_dash) and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                results.append(f)
    return results


def evaluate_visual_search(engine, clip, test_data, top_k=5):
    """Evaluate visual search: use product image as query, check if correct product in top-K."""
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks = []
    latencies = []

    for item in test_data:
        start = time.time()
        query_emb = clip.embed_image(item["image_path"])
        results = engine.visual_search(query_emb, top_k=top_k)
        elapsed = (time.time() - start) * 1000
        latencies.append(elapsed)

        result_codes = [r["wise_code"] for r in results]
        target = item["wise_code"]

        if target in result_codes:
            rank = result_codes.index(target) + 1
            reciprocal_ranks.append(1.0 / rank)
            if rank == 1:
                hits_at_1 += 1
            hits_at_k += 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(test_data)
    return {
        "recall_at_1": round(hits_at_1 / n * 100, 1) if n else 0,
        "recall_at_k": round(hits_at_k / n * 100, 1) if n else 0,
        "mrr": round(np.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0,
        "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
        "k": top_k,
        "n_tested": n,
    }


def evaluate_semantic_search(engine, clip, test_data, top_k=5):
    """Evaluate semantic search: use product description as query."""
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal_ranks = []

    for item in test_data:
        query_emb = clip.embed_text(item["description"])
        results = engine.semantic_search(query_emb, top_k=top_k)
        result_codes = [r["wise_code"] for r in results]
        target = item["wise_code"]

        if target in result_codes:
            rank = result_codes.index(target) + 1
            reciprocal_ranks.append(1.0 / rank)
            if rank == 1:
                hits_at_1 += 1
            hits_at_k += 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(test_data)
    return {
        "recall_at_1": round(hits_at_1 / n * 100, 1) if n else 0,
        "recall_at_k": round(hits_at_k / n * 100, 1) if n else 0,
        "mrr": round(np.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0,
        "k": top_k,
        "n_tested": n,
    }


def main():
    parser = argparse.ArgumentParser(description="AI.AS Search Evaluation")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=100, help="Max products to test (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)

    # Load model and index
    clip = CLIPEmbedder()
    clip.load()
    engine = SearchEngine()
    engine.load()

    if engine.total_products == 0:
        logger.error("No products indexed! Run ingest_data.py first.")
        sys.exit(1)

    # Build test set: products that have images
    test_data = []
    for product in engine.metadata:
        if not product.get("has_image"):
            continue
        images = find_test_images(images_dir, product["wise_code"])
        if images:
            test_data.append({
                "wise_code": product["wise_code"],
                "description": product["description"],
                "image_path": images[0],
            })

    if args.sample_size > 0 and len(test_data) > args.sample_size:
        random.seed(args.seed)
        test_data = random.sample(test_data, args.sample_size)

    logger.info(f"Testing with {len(test_data)} products (top-k={args.top_k})")

    # Run evaluations
    logger.info("\n=== Visual Search Evaluation ===")
    visual_results = evaluate_visual_search(engine, clip, test_data, args.top_k)
    logger.info(f"  Recall@1:  {visual_results['recall_at_1']}%")
    logger.info(f"  Recall@{args.top_k}:  {visual_results['recall_at_k']}%")
    logger.info(f"  MRR:       {visual_results['mrr']}")
    logger.info(f"  Avg latency: {visual_results['avg_latency_ms']}ms")

    logger.info("\n=== Semantic Search Evaluation ===")
    semantic_results = evaluate_semantic_search(engine, clip, test_data, args.top_k)
    logger.info(f"  Recall@1:  {semantic_results['recall_at_1']}%")
    logger.info(f"  Recall@{args.top_k}:  {semantic_results['recall_at_k']}%")
    logger.info(f"  MRR:       {semantic_results['mrr']}")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"top_k": args.top_k, "sample_size": len(test_data)},
        "visual_search": visual_results,
        "semantic_search": semantic_results,
    }
    output_path = INDEX_DIR / "eval_results.json"
    with open(str(output_path), "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
