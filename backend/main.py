"""AI.AS MVP - FastAPI Application"""
import logging
import time
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from backend.config import (
    HOST, PORT, DEBUG, TOP_K_RESULTS, MAX_IMAGE_SIZE,
    ALLOWED_EXTENSIONS, IMAGES_DIR, BASE_DIR, QUERY_EXPANSION_ENABLED,
)
from backend.models.clip_model import get_clip_model
from backend.search.engine import get_search_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI.AS - Smart Search API",
    description="AI Assistance System: Visual & Semantic Product Search for 100K+ SKUs",
    version="1.0.0-mvp",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Load models and search index on startup."""
    logger.info("Starting AI.AS MVP...")
    clip = get_clip_model()
    clip.load()
    engine = get_search_engine()
    engine.load()
    logger.info(f"Ready! {engine.total_products} products indexed.")


# --- API Endpoints ---

@app.get("/api/health")
async def health_check():
    engine = get_search_engine()
    return {
        "status": "healthy",
        "total_products": engine.total_products,
        "model": "CLIP ViT-B/32",
        "index": "FAISS IndexFlatIP",
    }


@app.post("/api/search/visual")
async def visual_search(
    image: UploadFile = File(...),
    top_k: int = Query(default=TOP_K_RESULTS, ge=1, le=50),
):
    """Search products by uploading an image."""
    # Validate file
    ext = Path(image.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    contents = await image.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(400, f"Image too large. Max size: {MAX_IMAGE_SIZE // (1024*1024)}MB")

    start = time.time()

    clip = get_clip_model()
    query_embedding = clip.embed_image(contents)

    engine = get_search_engine()
    results = engine.visual_search(query_embedding, top_k=top_k)

    elapsed = round((time.time() - start) * 1000, 1)

    return {
        "query_type": "visual",
        "results": results,
        "count": len(results),
        "elapsed_ms": elapsed,
    }


@app.get("/api/search/semantic")
async def semantic_search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query text"),
    top_k: int = Query(default=TOP_K_RESULTS, ge=1, le=50),
):
    """Search products by text description or keywords."""
    start = time.time()

    clip = get_clip_model()

    # Query Expansion: สร้างหลาย variant แล้ว average เพื่อเพิ่มความแม่นยำ
    if QUERY_EXPANSION_ENABLED:
        import numpy as np
        variants = [
            q,
            f"A photo of {q}",
            f"Industrial {q} spare part",
        ]
        embeddings = [clip.embed_text(v) for v in variants]
        query_embedding = np.mean(embeddings, axis=0)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
    else:
        query_embedding = clip.embed_text(q)

    engine = get_search_engine()
    results = engine.semantic_search(query_embedding, top_k=top_k)

    elapsed = round((time.time() - start) * 1000, 1)

    return {
        "query_type": "semantic",
        "query": q,
        "results": results,
        "count": len(results),
        "elapsed_ms": elapsed,
    }


@app.get("/api/products/{wise_code}")
async def get_product(wise_code: str):
    """Get product details by Wise Code."""
    engine = get_search_engine()
    engine._ensure_loaded()

    for product in engine.metadata:
        if product.get("wise_code") == wise_code:
            return product

    raise HTTPException(404, f"Product not found: {wise_code}")


@app.get("/api/products")
async def list_products(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
):
    """List all products with pagination."""
    engine = get_search_engine()
    engine._ensure_loaded()

    total = len(engine.metadata)
    start = (page - 1) * per_page
    end = start + per_page
    items = engine.metadata[start:end]

    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    }


@app.get("/api/images/{image_type}/{filename}")
async def serve_image(image_type: str, filename: str):
    """Serve product images (with.background or without.background)."""
    if image_type not in ("with.background", "without.background"):
        raise HTTPException(400, "image_type must be 'with.background' or 'without.background'")

    image_path = IMAGES_DIR / image_type / filename
    if not image_path.exists():
        raise HTTPException(404, f"Image not found: {filename}")

    return FileResponse(image_path)


# Serve frontend
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=DEBUG)
