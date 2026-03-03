"""Reusable data ingestion pipeline for building dual FAISS indexes."""

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from backend.config import INDEX_DIR
from backend.models.clip_model import CLIPEmbedder
from backend.search.engine import SearchEngine

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


def load_sku_data(excel_path: str | Path) -> pd.DataFrame:
    """Load and normalize SKU data from Excel.

    Args:
        excel_path: Path to Excel file containing SKU rows.

    Returns:
        Normalized dataframe with columns: no, wise_code, description, unit.
    """
    excel_path = Path(excel_path)
    logger.info(f"Loading SKU data from: {excel_path}")
    df = pd.read_excel(excel_path)
    df.columns = ["no", "wise_code", "description", "unit"]
    df["wise_code"] = df["wise_code"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df["unit"] = df["unit"].astype(str).str.strip()
    logger.info(f"Loaded {len(df)} SKUs")
    return df


def find_product_images(images_dir: Path, wise_code: str) -> dict[str, list[Path]]:
    """Find product images in both naming conventions.

    Args:
        images_dir: Base directory containing with.background/without.background folders.
        wise_code: Product code from source data.

    Returns:
        Dict with sorted path lists: {"with_bg": [...], "without_bg": [...]}.
    """
    code_no_dash = wise_code.replace("-", "")

    with_bg: list[Path] = []
    without_bg: list[Path] = []

    with_bg_dir = images_dir / "with.background"
    if with_bg_dir.exists():
        for file_path in with_bg_dir.iterdir():
            if file_path.stem.startswith(code_no_dash) and file_path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                with_bg.append(file_path)

    without_bg_dir = images_dir / "without.background"
    if without_bg_dir.exists():
        for file_path in without_bg_dir.iterdir():
            if file_path.stem.upper().startswith(
                wise_code.upper()
            ) and file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                without_bg.append(file_path)

    return {"with_bg": sorted(with_bg), "without_bg": sorted(without_bg)}


def augment_description(description: str, wise_code: str, unit: str) -> str:
    """Add domain context to short descriptions for stronger text embeddings.

    Args:
        description: Product description text from source data.
        wise_code: Product code.
        unit: Sale/stock unit.

    Returns:
        Augmented text suitable for CLIP text embedding.
    """
    parts = [f"Industrial spare part: {description}"]
    if unit and unit != "nan":
        parts.append(f"Sold per {unit}")
    parts.append(f"Product code {wise_code}")
    return ". ".join(parts)


def compute_multi_angle_embedding(
    clip: CLIPEmbedder, image_paths: list[Path]
) -> np.ndarray:
    """Average image embeddings from multiple angles and re-normalize.

    Args:
        clip: Loaded CLIP embedder.
        image_paths: One or more image paths for the same product.

    Returns:
        Single normalized embedding vector.
    """
    if len(image_paths) == 1:
        return clip.embed_image(image_paths[0])

    embeddings = [clip.embed_image(path) for path in image_paths]
    avg = np.mean(embeddings, axis=0)
    return avg / np.linalg.norm(avg)


def _emit_progress(
    callback: ProgressCallback | None,
    current: int,
    total: int,
    stage: str,
) -> None:
    """Emit progress updates when callback is provided."""
    if callback is not None:
        callback(current, total, stage)


def run_ingestion_pipeline(
    images_dir: str | Path,
    excel_path: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Run full ingestion flow and persist dual indexes + metadata.

    Args:
        images_dir: Base directory containing product images.
        excel_path: Path to SKU Excel file.
        progress_callback: Optional callback called as (current, total, stage).

    Returns:
        Ingestion summary stats.
    """
    images_dir = Path(images_dir)
    excel_path = Path(excel_path)

    start_time = time.time()
    df = load_sku_data(excel_path)

    _emit_progress(progress_callback, 0, len(df), "loading_model")
    logger.info("Loading CLIP model...")
    clip = CLIPEmbedder()
    clip.load()

    all_image_embeddings: list[np.ndarray] = []
    all_text_embeddings: list[np.ndarray] = []
    all_metadata: list[dict] = []

    stats = {"with_images": 0, "multi_angle": 0, "text_only": 0}

    for idx, row in df.iterrows():
        wise_code = row["wise_code"]
        description = row["description"]
        unit = row["unit"]

        images = find_product_images(images_dir, wise_code)
        all_images = images["without_bg"] + images["with_bg"]

        if all_images:
            image_embedding = compute_multi_angle_embedding(clip, all_images)
            stats["with_images"] += 1
            if len(all_images) > 1:
                stats["multi_angle"] += 1
        else:
            image_embedding = clip.embed_text(description)
            stats["text_only"] += 1

        augmented_desc = augment_description(description, wise_code, unit)
        text_embedding = clip.embed_text(augmented_desc)

        with_bg_api = [
            f"/api/images/with.background/{img.name}" for img in images["with_bg"]
        ]
        without_bg_api = [
            f"/api/images/without.background/{img.name}" for img in images["without_bg"]
        ]

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

        current = idx + 1
        if current % 50 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Processed {current}/{len(df)} ({current / elapsed:.1f}/sec)")
        _emit_progress(progress_callback, current, len(df), "embedding")

    _emit_progress(progress_callback, len(df), len(df), "building_index")
    image_emb_array = np.vstack(all_image_embeddings).astype(np.float32)
    text_emb_array = np.vstack(all_text_embeddings).astype(np.float32)

    logger.info(f"Image embeddings: {image_emb_array.shape}")
    logger.info(f"Text embeddings:  {text_emb_array.shape}")

    engine = SearchEngine()
    engine.build_index(image_emb_array, text_emb_array, all_metadata)

    elapsed = time.time() - start_time
    logger.info(f"Ingestion complete in {elapsed:.1f}s")
    logger.info(
        f"  Products with images: {stats['with_images']} ({stats['multi_angle']} multi-angle)"
    )
    logger.info(f"  Products text-only:   {stats['text_only']}")
    logger.info(f"  Dual indexes saved to: {INDEX_DIR}")

    _emit_progress(progress_callback, len(df), len(df), "completed")
    return {
        "total_products": len(df),
        "products_with_images": stats["with_images"],
        "products_multi_angle": stats["multi_angle"],
        "products_text_only": stats["text_only"],
        "elapsed_seconds": round(elapsed, 2),
        "index_dir": str(INDEX_DIR),
    }
