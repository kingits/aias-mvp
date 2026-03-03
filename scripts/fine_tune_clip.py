#!/usr/bin/env python3
"""
AI.AS MVP - CLIP Fine-tuning on Wise Product Data

Fine-tunes CLIP using contrastive learning on product images + descriptions.
Uses LoRA for efficient fine-tuning (only trains ~1% of parameters).

Improvements over v1:
  - Reuses augment_description() from ingestion pipeline (text template alignment)
  - Uses images from BOTH folders (with.background + without.background)
  - Multi-template text augmentation (random per sample, matches query expansion)
  - LR warmup + cosine decay schedule
  - Per-SKU image cap to prevent data imbalance
  - Lower default LR (5e-6) and more epochs (12)

Usage:
    python scripts/fine_tune_clip.py \
        --images-dir "source_data" \
        --excel "source_data/SKUของตัวอย่างภาพสินค้า1K.xlsx" \
        --epochs 12 \
        --output-dir model/fine_tuned_clip_v2

Requirements (additional):
    pip install peft>=0.7.0  # for LoRA
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import CLIP_MODEL_NAME
from backend.ingest.pipeline import augment_description, find_product_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("fine_tune")

DEFAULT_BASE_MODEL_NAME = "openai/clip-vit-large-patch14"

# Text templates used during training. Each sample randomly picks one per epoch.
# These mirror: 1) the ingestion augment_description format, 2) query expansion
# variants from backend/main.py, and 3) additional diversity templates.
TEXT_TEMPLATES = [
    # Primary: matches ingestion pipeline augment_description() exactly
    lambda d, c, u: augment_description(d, c, u),
    # Query-expansion style (matches runtime _semantic_query_embedding variants)
    lambda d, c, u: d,
    lambda d, c, u: f"A photo of {d}",
    lambda d, c, u: f"Industrial {d} spare part",
    # Extra diversity
    lambda d, c, u: f"{d}, product code {c}",
    lambda d, c, u: f"Spare part {d}. Code: {c}",
]


class WiseProductDataset(Dataset):
    """Dataset of (image, augmented_description) pairs for contrastive learning.

    Each __getitem__ call randomly selects a text template, providing text
    diversity across epochs so the model learns robust text-image alignment.
    """

    def __init__(self, products: list[dict], processor: CLIPProcessor):
        self.products = products
        self.processor = processor

    def __len__(self):
        return len(self.products)

    def __getitem__(self, idx):
        item = self.products[idx]
        image = Image.open(item["image_path"]).convert("RGB")

        # Randomly pick a text template for diversity
        template = random.choice(TEXT_TEMPLATES)
        text = template(item["description"], item["wise_code"], item["unit"])

        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}


def contrastive_loss(image_features, text_features, temperature=0.07):
    """CLIP-style symmetric contrastive loss (InfoNCE)."""
    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Cosine similarity as logits
    logits = (image_features @ text_features.T) / temperature
    labels = torch.arange(len(logits), device=logits.device)

    loss_i2t = nn.functional.cross_entropy(logits, labels)
    loss_t2i = nn.functional.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


def resolve_model_name(model_name: str) -> str:
    """Resolve model name/path and fallback if a local path is missing."""
    model_path = Path(model_name).expanduser()
    looks_like_local_path = (
        model_path.is_absolute()
        or model_name.startswith(".")
        or "/" in model_name
        or "\\" in model_name
    )

    if looks_like_local_path:
        if model_path.exists():
            return str(model_path)

        logger.warning(
            "Model path not found: %s. Falling back to base model: %s",
            model_name,
            DEFAULT_BASE_MODEL_NAME,
        )
        return DEFAULT_BASE_MODEL_NAME

    return model_name


def build_training_data(
    df: pd.DataFrame, images_dir: Path, max_images_per_sku: int = 6
) -> list[dict]:
    """Build training pairs from Excel data + images from BOTH folders.

    Args:
        df: Dataframe with columns: no, wise_code, description, unit.
        images_dir: Base directory containing with.background/ and without.background/.
        max_images_per_sku: Maximum images per SKU to prevent imbalance.

    Returns:
        List of training dicts with keys: wise_code, description, unit, image_path.
    """
    products = []
    skus_with_images = 0
    skus_without_images = 0

    for _, row in df.iterrows():
        wise_code = str(row["wise_code"]).strip()
        description = str(row["description"]).strip()
        unit = str(row["unit"]).strip()

        # Use ingestion pipeline's find_product_images (returns both folders)
        images = find_product_images(images_dir, wise_code)
        all_images = images["without_bg"] + images["with_bg"]

        if not all_images:
            skus_without_images += 1
            continue

        skus_with_images += 1

        # Cap images per SKU to prevent imbalance
        if len(all_images) > max_images_per_sku:
            all_images = random.sample(all_images, max_images_per_sku)

        for img_path in all_images:
            products.append(
                {
                    "wise_code": wise_code,
                    "description": description,
                    "unit": unit,
                    "image_path": img_path,
                }
            )

    logger.info(
        f"Training data: {len(products)} pairs from {skus_with_images} SKUs "
        f"({skus_without_images} SKUs skipped — no images)"
    )
    return products


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on Wise product data")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--excel", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of total steps used for LR warmup (default: 0.1)",
    )
    parser.add_argument(
        "--max-images-per-sku",
        type=int,
        default=6,
        help="Cap images per SKU to prevent training imbalance (default: 6)",
    )
    parser.add_argument("--output-dir", default="model/fine_tuned_clip_v2")
    parser.add_argument("--model-name", default=CLIP_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    images_dir = Path(args.images_dir)

    # Load data
    df = pd.read_excel(args.excel)
    df.columns = ["no", "wise_code", "description", "unit"]

    # Build training data using BOTH image folders + per-SKU cap
    products = build_training_data(df, images_dir, args.max_images_per_sku)

    if not products:
        logger.error("No training data found! Check --images-dir path.")
        sys.exit(1)

    # Load model
    resolved_model_name = resolve_model_name(args.model_name)
    logger.info(f"Loading {resolved_model_name}...")
    model = CLIPModel.from_pretrained(resolved_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(resolved_model_name)

    # Try to use LoRA for efficient fine-tuning
    lora_enabled = False
    try:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],  # attention layers only
        )
        model = get_peft_model(model, lora_config)
        lora_enabled = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"LoRA enabled: training {trainable:,} / {total:,} params ({trainable / total * 100:.1f}%)"
        )
    except ImportError:
        logger.warning(
            "peft not installed — fine-tuning all parameters (slower). Install with: pip install peft"
        )

    # Dataset and DataLoader
    dataset = WiseProductDataset(products, processor)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    # Optimizer + LR schedule with warmup
    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()

    logger.info(
        f"\nStarting fine-tuning: {args.epochs} epochs, batch_size={args.batch_size}, "
        f"lr={args.lr}, warmup={warmup_steps}/{total_steps} steps"
    )
    logger.info(f"Text templates: {len(TEXT_TEMPLATES)} variants (random per sample)")
    logger.info(f"Max images/SKU: {args.max_images_per_sku}")

    best_loss = float("inf")
    for epoch in range(args.epochs):
        total_loss = 0
        n_batches = 0
        start = time.time()

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = contrastive_loss(
                outputs.image_embeds,
                outputs.text_embeds,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - start
        improved = " *" if avg_loss < best_loss else ""
        best_loss = min(best_loss, avg_loss)
        logger.info(
            f"  Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f} "
            f"lr={current_lr:.2e} ({elapsed:.1f}s){improved}"
        )

    # Save fine-tuned model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info(f"\nFine-tuned model saved to: {output_dir}")

    if lora_enabled and hasattr(model, "merge_and_unload"):
        merged_output_dir = output_dir / "merged"
        merged_output_dir.mkdir(parents=True, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_output_dir))
        processor.save_pretrained(str(merged_output_dir))
        logger.info(f"Merged full model saved to: {merged_output_dir}")
        logger.info(
            "To use merged model: set AIAS_CLIP_MODEL_NAME to this path, "
            "then re-run ingest_data.py and evaluate_search.py"
        )
    else:
        logger.info(
            "To use: set AIAS_CLIP_MODEL_NAME to this path, then re-run ingest_data.py"
        )


if __name__ == "__main__":
    main()
