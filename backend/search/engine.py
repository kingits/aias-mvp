"""FAISS-based search engine with Hybrid Scoring (image + text dual index)."""
import faiss
import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional

from backend.config import (
    INDEX_DIR, METADATA_PATH, TOP_K_RESULTS, CONFIDENCE_THRESHOLD, EMBEDDING_DIM,
    VISUAL_IMAGE_WEIGHT, VISUAL_TEXT_WEIGHT, SEMANTIC_IMAGE_WEIGHT, SEMANTIC_TEXT_WEIGHT,
)

logger = logging.getLogger(__name__)

IMAGE_INDEX_PATH = INDEX_DIR / "image_vectors.index"
TEXT_INDEX_PATH = INDEX_DIR / "text_vectors.index"
LEGACY_INDEX_PATH = INDEX_DIR / "product_vectors.index"


class SearchEngine:
    """Manages dual FAISS indexes (image + text) for hybrid product search."""

    def __init__(self):
        self.image_index: Optional[faiss.Index] = None
        self.text_index: Optional[faiss.Index] = None
        self.metadata: list[dict] = []
        self._loaded = False

    def load(self):
        """Load FAISS indexes and product metadata from disk."""
        meta_path = str(METADATA_PATH)

        # Try dual-index first, fall back to legacy single index
        has_dual = IMAGE_INDEX_PATH.exists() and TEXT_INDEX_PATH.exists()
        has_legacy = LEGACY_INDEX_PATH.exists()

        if not has_dual and not has_legacy:
            logger.warning("No FAISS index found. Run data ingestion first.")
            self.image_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.text_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.metadata = []
            self._loaded = True
            return

        if has_dual:
            logger.info("Loading dual indexes (image + text)...")
            self.image_index = faiss.read_index(str(IMAGE_INDEX_PATH))
            self.text_index = faiss.read_index(str(TEXT_INDEX_PATH))
        else:
            logger.info("Loading legacy single index (fallback)...")
            legacy = faiss.read_index(str(LEGACY_INDEX_PATH))
            self.image_index = legacy
            self.text_index = legacy

        if Path(meta_path).exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self._loaded = True
        logger.info(f"Search engine loaded: {self.image_index.ntotal} products")

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def _search_single(self, index: faiss.Index, query_vector: np.ndarray, top_k: int) -> dict[int, float]:
        """Search a single FAISS index, return {metadata_idx: score}."""
        if index.ntotal == 0:
            return {}
        query = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = index.search(query, min(top_k * 2, index.ntotal))
        return {int(idx): float(score) for score, idx in zip(scores[0], indices[0]) if idx >= 0}

    def hybrid_search(
        self,
        query_vector: np.ndarray,
        top_k: int = TOP_K_RESULTS,
        image_weight: float = 0.5,
        text_weight: float = 0.5,
    ) -> list[dict]:
        """Hybrid search combining image and text index scores.

        The query_vector is searched against BOTH indexes.
        Final score = (image_weight × image_similarity) + (text_weight × text_similarity)
        """
        self._ensure_loaded()

        image_scores = self._search_single(self.image_index, query_vector, top_k)
        text_scores = self._search_single(self.text_index, query_vector, top_k)

        # Merge scores
        all_ids = set(image_scores.keys()) | set(text_scores.keys())
        combined = {}
        for idx in all_ids:
            if idx >= len(self.metadata):
                continue
            img_s = image_scores.get(idx, 0.0)
            txt_s = text_scores.get(idx, 0.0)
            combined[idx] = (image_weight * img_s) + (text_weight * txt_s)

        # Sort and filter
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in sorted_results:
            if score < CONFIDENCE_THRESHOLD:
                continue
            product = self.metadata[idx].copy()
            product["score"] = round(score, 4)
            results.append(product)

        return results

    def visual_search(self, query_vector: np.ndarray, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """Visual search: emphasize image similarity, use text to disambiguate."""
        return self.hybrid_search(
            query_vector, top_k,
            image_weight=VISUAL_IMAGE_WEIGHT,
            text_weight=VISUAL_TEXT_WEIGHT,
        )

    def semantic_search(self, query_vector: np.ndarray, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """Semantic search: emphasize text similarity, use image to re-rank."""
        return self.hybrid_search(
            query_vector, top_k,
            image_weight=SEMANTIC_IMAGE_WEIGHT,
            text_weight=SEMANTIC_TEXT_WEIGHT,
        )

    def search(self, query_vector: np.ndarray, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """Legacy single-index search (backward compatible)."""
        return self.visual_search(query_vector, top_k)

    def build_index(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        metadata: list[dict],
    ):
        """Build dual FAISS indexes from image and text embeddings."""
        n = image_embeddings.shape[0]
        dim = image_embeddings.shape[1]
        logger.info(f"Building dual FAISS indexes: {n} products, {dim} dimensions")

        self.image_index = faiss.IndexFlatIP(dim)
        self.image_index.add(image_embeddings.astype(np.float32))

        self.text_index = faiss.IndexFlatIP(dim)
        self.text_index.add(text_embeddings.astype(np.float32))

        self.metadata = metadata

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.image_index, str(IMAGE_INDEX_PATH))
        faiss.write_index(self.text_index, str(TEXT_INDEX_PATH))

        with open(str(METADATA_PATH), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self._loaded = True
        logger.info(f"Dual indexes saved to: {INDEX_DIR}")

    @property
    def total_products(self) -> int:
        self._ensure_loaded()
        return self.image_index.ntotal if self.image_index else 0


# Global singleton
_engine_instance = None

def get_search_engine() -> SearchEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SearchEngine()
    return _engine_instance
