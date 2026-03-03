"""CLIP Model wrapper for image and text embedding generation."""
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
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()
        self._loaded = True
        logger.info("CLIP model loaded successfully")

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

        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
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
            batch = images[i:i + batch_size]
            pil_batch = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    pil_batch.append(Image.open(img).convert("RGB"))
                elif isinstance(img, Image.Image):
                    pil_batch.append(img.convert("RGB"))

            inputs = self.processor(images=pil_batch, return_tensors="pt", padding=True).to(self.device)
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
