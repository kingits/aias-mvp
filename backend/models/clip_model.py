"""CLIP Model wrapper for image and text embedding generation."""

import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
from typing import Union
import io
import logging

from backend.config import CLIP_MODEL_NAME

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """Handles CLIP model loading and embedding generation for both images and text."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or CLIP_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        """Load CLIP model and processor."""
        if self._loaded:
            return

        logger.info(f"Loading CLIP model: {self.model_name} on {self.device}")

        model_path = Path(str(self.model_name))
        if (
            model_path.exists()
            and model_path.is_dir()
            and (model_path / "adapter_config.json").exists()
        ):
            self.model, self.processor = self._load_lora_adapter(model_path)
        else:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

        self.model.eval()
        self._loaded = True
        logger.info("CLIP model loaded successfully")

    def _load_lora_adapter(self, adapter_dir: Path) -> tuple[CLIPModel, CLIPProcessor]:
        """Load and merge a LoRA adapter directory into a base CLIP model.

        Args:
            adapter_dir: Directory containing adapter_config.json and adapter weights.

        Returns:
            Tuple of merged CLIP model and CLIP processor.
        """
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "Detected LoRA adapter directory, but peft is not installed. "
                "Install with: pip install peft>=0.7.0"
            ) from exc

        config_path = adapter_dir / "adapter_config.json"
        with open(config_path, "r", encoding="utf-8") as file:
            adapter_config = json.load(file)

        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError(
                f"Invalid adapter config: missing base_model_name_or_path in {config_path}"
            )

        logger.info(
            f"Detected LoRA adapter at {adapter_dir}. Base model: {base_model_name}"
        )
        base_model = CLIPModel.from_pretrained(base_model_name)
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged_model = peft_model.merge_and_unload().to(self.device)

        processor_source = (
            str(adapter_dir)
            if (adapter_dir / "preprocessor_config.json").exists()
            else base_model_name
        )
        processor = CLIPProcessor.from_pretrained(processor_source)
        logger.info("LoRA adapter merged successfully")
        return merged_model, processor

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    @torch.no_grad()
    def embed_image(self, image: Union[Image.Image, bytes, str, Path]) -> np.ndarray:
        """Generate embedding vector from an image.

        Args:
            image: PIL Image, bytes, file path, or Path object

        Returns:
            Normalized embedding vector (numpy array)
        """
        self._ensure_loaded()

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        embedding = features.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
        return embedding

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding vector from text.

        Args:
            text: Search query or product description

        Returns:
            Normalized embedding vector (numpy array)
        """
        self._ensure_loaded()

        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        features = self.model.get_text_features(**inputs)
        embedding = features.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    @torch.no_grad()
    def embed_images_batch(self, images: list, batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a batch of images.

        Args:
            images: List of PIL Images
            batch_size: Processing batch size

        Returns:
            Array of normalized embeddings
        """
        self._ensure_loaded()
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            pil_batch = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    pil_batch.append(Image.open(img).convert("RGB"))
                elif isinstance(img, Image.Image):
                    pil_batch.append(img.convert("RGB"))

            inputs = self.processor(
                images=pil_batch, return_tensors="pt", padding=True
            ).to(self.device)
            features = self.model.get_image_features(**inputs)
            embeddings = features.cpu().numpy()
            # Normalize each embedding
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)


# Global singleton
_clip_instance = None


def get_clip_model() -> CLIPEmbedder:
    global _clip_instance
    if _clip_instance is None:
        _clip_instance = CLIPEmbedder()
    return _clip_instance
