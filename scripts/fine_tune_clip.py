#!/usr/bin/env python3
"""
AI.AS MVP - CLIP Fine-tuning on Wise Product Data

Fine-tunes CLIP using contrastive learning on product images + descriptions.
Uses LoRA for efficient fine-tuning (only trains ~1% of parameters).

Usage:
    python scripts/fine_tune_clip.py \
        --images-dir "../ตัวอย่างภาพสินค้า1K" \
        --excel "../ตัวอย่างภาพสินค้า1K/SKUของตัวอย่างภาพสินค้า1K.xlsx" \
        --epochs 5 \
        --output-dir data/fine_tuned_clip

Requirements (additional):
    pip install peft>=0.7.0  # for LoRA
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import CLIP_MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("fine_tune")


class WiseProductDataset(Dataset):
    """Dataset of (image, augmented_description) pairs for contrastive learning."""

    def __init__(self, products: list[dict], processor: CLIPProcessor):
        self.products = products
        self.processor = processor

    def __len__(self):
        return len(self.products)

    def __getitem__(self, idx):
        item = self.products[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        text = f"Industrial spare part: {item['description']}. Product code {item['wise_code']}"

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


def find_product_images(images_dir: Path, wise_code: str) -> list[Path]:
    """Find images for a product (prefer without.background)."""
    results = []
    without_bg_dir = images_dir / "without.background"
    if without_bg_dir.exists():
        for f in without_bg_dir.iterdir():
            if f.stem.upper().startswith(wise_code.upper()) and f.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                results.append(f)
    if not results:
        code_no_dash = wise_code.replace("-", "")
        with_bg_dir = images_dir / "with.background"
        if with_bg_dir.exists():
            for f in with_bg_dir.iterdir():
                if f.stem.startswith(code_no_dash) and f.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }:
                    results.append(f)
    return sorted(results)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on Wise product data")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--excel", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="data/fine_tuned_clip")
    parser.add_argument("--model-name", default=CLIP_MODEL_NAME)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    images_dir = Path(args.images_dir)

    # Load data
    df = pd.read_excel(args.excel)
    df.columns = ["no", "wise_code", "description", "unit"]

    # Build training data: only products with images
    products = []
    for _, row in df.iterrows():
        wise_code = str(row["wise_code"]).strip()
        images = find_product_images(images_dir, wise_code)
        if images:
            # Use each image angle as a separate training example
            for img_path in images:
                products.append(
                    {
                        "wise_code": wise_code,
                        "description": str(row["description"]).strip(),
                        "image_path": img_path,
                    }
                )

    logger.info(f"Training examples: {len(products)} (from {len(df)} SKUs)")

    # Load model
    logger.info(f"Loading {args.model_name}...")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)

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

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()

    logger.info(
        f"\nStarting fine-tuning: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}"
    )

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

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        elapsed = time.time() - start
        logger.info(
            f"  Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f} ({elapsed:.1f}s)"
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
            "To use merged model: set CLIP_MODEL_NAME to this path, then re-run ingest_data.py"
        )
    else:
        logger.info(
            "To use: set CLIP_MODEL_NAME in config.py to this path, then re-run ingest_data.py"
        )


if __name__ == "__main__":
    main()
