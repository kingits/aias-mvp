#!/usr/bin/env python3
"""
AI.AS MVP - Data Ingestion Pipeline (v2 — Dual Index + Accuracy Improvements)

Generates SEPARATE image and text embeddings for hybrid search.
Uses multi-angle averaging and text augmentation for better accuracy.

Usage:
    python scripts/ingest_data.py --images-dir <path> --excel <path>
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import INDEX_DIR, EMBEDDING_DIM
from backend.models.clip_model import CLIPEmbedder
from backend.search.engine import SearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("ingest")


def load_sku_data(excel_path: str) -> pd.DataFrame:
    logger.info(f"Loading SKU data from: {excel_path}")
    df = pd.read_excel(excel_path)
    df.columns = ["no", "wise_code", "description", "unit"]
    df["wise_code"] = df["wise_code"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()
    logger.info(f"Loaded {len(df)} SKUs")
    return df


def find_product_images(images_dir: Path, wise_code: str) -> dict:
    """Find product images. Returns {without_bg: [...], with_bg: [...]}"""
    code_no_dash = wise_code.replace("-", "")

    with_bg = []
    without_bg = []

    with_bg_dir = images_dir / "with.background"
    if with_bg_dir.exists():
        for f in with_bg_dir.iterdir():
            if f.stem.startswith(code_no_dash) and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                with_bg.append(f)

    without_bg_dir = images_dir / "without.background"
    if without_bg_dir.exists():
        for f in without_bg_dir.iterdir():
            if f.stem.upper().startswith(wise_code.upper()) and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                without_bg.append(f)

    return {"with_bg": sorted(with_bg), "without_bg": sorted(without_bg)}


def augment_description(description: str, wise_code: str, unit: str) -> str:
    """Augment short product description for better CLIP text embedding.

    Industrial descriptions like 'Bearing, 15x35x11, 6202.2RS' are too terse
    for CLIP. Adding context helps CLIP understand the product domain.
    """
    parts = [f"Industrial spare part: {description}"]
    if unit and unit != "nan":
        parts.append(f"Sold per {unit}")
    parts.append(f"Product code {wise_code}")
    return ". ".join(parts)


def compute_multi_angle_embedding(clip: CLIPEmbedder, image_paths: list) -> np.ndarray:
    """Average embeddings from multiple image angles of the same product.

    This captures features from all angles rather than just one view,
    making the product more findable regardless of which angle the user photographs.
    """
    if len(image_paths) == 1:
        return clip.embed_image(image_paths[0])

    embeddings = [clip.embed_image(p) for p in image_paths]
    avg = np.mean(embeddings, axis=0)
    avg = avg / np.linalg.norm(avg)  # re-normalize after averaging
    return avg


def main():
    parser = argparse.ArgumentParser(description="AI.AS Data Ingestion Pipeline v2")
    parser.add_argument("--images-dir", required=True, help="Path to product images directory")
    parser.add_argument("--excel", required=True, help="Path to SKU Excel file")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)

    # 1. Load SKU data
    df = load_sku_data(args.excel)

    # 2. Initialize CLIP
    logger.info("Loading CLIP model...")
    clip = CLIPEmbedder()
    clip.load()

    # 3. Process each product — build SEPARATE image and text embeddings
    all_image_embeddings = []
    all_text_embeddings = []
    all_metadata = []

    start_time = time.time()
    stats = {"with_images": 0, "multi_angle": 0, "text_only": 0}

    for idx, row in df.iterrows():
        wise_code = row["wise_code"]
        description = row["description"]
        unit = row["unit"]

        images = find_product_images(images_dir, wise_code)

        # --- IMAGE EMBEDDING ---
        # Combine both with.bg and without.bg images for richer embedding
        # This reduces domain gap: index will match regardless of background presence
        all_images = images["without_bg"] + images["with_bg"]

        if all_images:
            # Multi-angle average: embed ALL angles from both folders, then average
            image_embedding = compute_multi_angle_embedding(clip, all_images)
            stats["with_images"] += 1
            if len(all_images) > 1:
                stats["multi_angle"] += 1
        else:
            # Fallback: use text embedding as image embedding too
            image_embedding = clip.embed_text(description)
            stats["text_only"] += 1

        # --- TEXT EMBEDDING ---
        augmented_desc = augment_description(description, wise_code, unit)
        text_embedding = clip.embed_text(augmented_desc)

        # --- METADATA ---
        with_bg_api = [f"/api/images/with.background/{img.name}" for img in images["with_bg"]]
        without_bg_api = [f"/api/images/without.background/{img.name}" for img in images["without_bg"]]

        metadata = {
            "id": int(row["no"]),
            "wise_code": wise_code,
            "description": description,
            "unit": unit,
            "images": {
                "with_bg": with_bg_api[0] if with_bg_api else None,
                "without_bg": without_bg_api[0] if without_bg_api else None,
                "with_bg_all": with_bg_api,
                "without_bg_all": without_bg_api,
            },
            "has_image": bool(all_images),
            "image_count": len(all_images) if all_images else 0,
        }

        all_image_embeddings.append(image_embedding)
        all_text_embeddings.append(text_embedding)
        all_metadata.append(metadata)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Processed {idx + 1}/{len(df)} ({(idx+1)/elapsed:.1f}/sec)")

    # 4. Build dual FAISS index
    image_emb_array = np.vstack(all_image_embeddings).astype(np.float32)
    text_emb_array = np.vstack(all_text_embeddings).astype(np.float32)

    logger.info(f"Image embeddings: {image_emb_array.shape}")
    logger.info(f"Text embeddings:  {text_emb_array.shape}")

    engine = SearchEngine()
    engine.build_index(image_emb_array, text_emb_array, all_metadata)

    elapsed = time.time() - start_time
    logger.info(f"\nIngestion complete in {elapsed:.1f}s!")
    logger.info(f"  Products with images: {stats['with_images']} ({stats['multi_angle']} multi-angle)")
    logger.info(f"  Products text-only:   {stats['text_only']}")
    logger.info(f"  Dual indexes saved to: {INDEX_DIR}")


if __name__ == "__main__":
    main()
