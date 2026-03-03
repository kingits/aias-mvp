"""AI.AS MVP - Configuration"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
# Images: ใช้ env var หรือ default ไปที่ sample data ข้างนอก project
_default_images = BASE_DIR.parent / "ตัวอย่างภาพสินค้า1K"
IMAGES_DIR = Path(os.getenv("AIAS_IMAGES_DIR", str(_default_images if _default_images.exists() else DATA_DIR / "images")))

# CLIP Model
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
EMBEDDING_DIM = 768

# FAISS
FAISS_INDEX_PATH = INDEX_DIR / "product_vectors.index"
METADATA_PATH = INDEX_DIR / "product_metadata.json"

# Search
TOP_K_RESULTS = 10
CONFIDENCE_THRESHOLD = 0.15  # minimum cosine similarity

# Hybrid Scoring Weights
# Visual Search: เน้นรูป แต่ใช้ text ช่วย disambiguate สินค้าหน้าตาคล้ายกัน
VISUAL_IMAGE_WEIGHT = float(os.getenv("AIAS_VISUAL_IMG_W", "0.7"))
VISUAL_TEXT_WEIGHT = float(os.getenv("AIAS_VISUAL_TXT_W", "0.3"))
# Semantic Search: เน้น text แต่ใช้ image ช่วย re-rank
SEMANTIC_IMAGE_WEIGHT = float(os.getenv("AIAS_SEMANTIC_IMG_W", "0.2"))
SEMANTIC_TEXT_WEIGHT = float(os.getenv("AIAS_SEMANTIC_TXT_W", "0.8"))

# Query Expansion: สร้างหลาย variant ของ query แล้ว average embedding
QUERY_EXPANSION_ENABLED = os.getenv("AIAS_QUERY_EXPANSION", "true").lower() == "true"

# Server
HOST = os.getenv("AIAS_HOST", "0.0.0.0")
PORT = int(os.getenv("AIAS_PORT", "8000"))
DEBUG = os.getenv("AIAS_DEBUG", "true").lower() == "true"

# Image
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
