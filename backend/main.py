"""AI.AS MVP - FastAPI Application"""

import json
import logging
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import (
    HOST,
    PORT,
    DEBUG,
    TOP_K_RESULTS,
    MAX_IMAGE_SIZE,
    MAX_UPLOAD_SIZE,
    ALLOWED_EXTENSIONS,
    IMAGES_DIR,
    BASE_DIR,
    QUERY_EXPANSION_ENABLED,
    UPLOAD_DIR,
    UPLOADED_IMAGES_DIR,
)
from backend.ingest.pipeline import run_ingestion_pipeline
from backend.models.clip_model import get_clip_model
from backend.search.engine import get_search_engine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
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


class IngestStartRequest(BaseModel):
    """Start request payload for ingestion pipeline."""

    upload_id: str


_ingest_lock = threading.Lock()
_ingest_thread: threading.Thread | None = None
_ingest_state = {
    "state": "idle",
    "upload_id": None,
    "progress": 0,
    "message": "No ingestion job started",
    "started_at": None,
    "completed_at": None,
    "stats": None,
    "error": None,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _history_path() -> Path:
    return UPLOAD_DIR / "history.json"


def _read_history() -> list[dict]:
    path = _history_path()
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _append_history(entry: dict) -> None:
    history = _read_history()
    history.insert(0, entry)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with open(_history_path(), "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)


def _set_ingest_state(**kwargs) -> None:
    with _ingest_lock:
        _ingest_state.update(kwargs)


def _get_ingest_state() -> dict:
    with _ingest_lock:
        return dict(_ingest_state)


def _progress_callback(current: int, total: int, stage: str) -> None:
    if stage == "loading_model":
        progress = 5
        message = "Loading CLIP model"
    elif stage == "embedding":
        ratio = current / total if total else 0.0
        progress = 5 + int(ratio * 85)
        message = f"Processing products ({current}/{total})"
    elif stage == "building_index":
        progress = 95
        message = "Building FAISS indexes"
    elif stage == "completed":
        progress = 100
        message = "Ingestion completed"
    else:
        progress = _get_ingest_state()["progress"]
        message = stage

    _set_ingest_state(progress=progress, message=message)


def _save_upload_file(upload: UploadFile, target_path: Path) -> int:
    if not upload.filename:
        raise HTTPException(400, "Uploaded file is missing filename")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    upload.file.seek(0, 2)
    size = upload.file.tell()
    upload.file.seek(0)
    if size > MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"File too large: {upload.filename}")

    with open(target_path, "wb") as out_file:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
    return size


def _extract_zip(zip_path: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    destination_resolved = destination.resolve()

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for member in zip_file.infolist():
            member_path = destination / member.filename
            resolved = member_path.resolve()
            if not str(resolved).startswith(str(destination_resolved)):
                raise HTTPException(400, "Zip contains invalid path")
            zip_file.extract(member, destination)
            if member.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                extracted += 1
    return extracted


def _normalize_uploaded_images(images_root: Path) -> None:
    with_bg_dir = images_root / "with.background"
    without_bg_dir = images_root / "without.background"
    with_bg_dir.mkdir(parents=True, exist_ok=True)
    without_bg_dir.mkdir(parents=True, exist_ok=True)

    for file_path in images_root.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
            file_path.replace(without_bg_dir / file_path.name)


def _run_ingestion_job(upload_id: str, excel_path: Path, images_root: Path) -> None:
    start_ts = _utc_now()
    _set_ingest_state(
        state="running",
        upload_id=upload_id,
        progress=1,
        message="Starting ingestion",
        started_at=start_ts,
        completed_at=None,
        stats=None,
        error=None,
    )

    try:
        stats = run_ingestion_pipeline(
            images_dir=images_root,
            excel_path=excel_path,
            progress_callback=_progress_callback,
        )
        engine = get_search_engine()
        engine.load()

        completed_ts = _utc_now()
        _set_ingest_state(
            state="completed",
            progress=100,
            message="Ingestion completed",
            completed_at=completed_ts,
            stats=stats,
            error=None,
        )
        _append_history(
            {
                "upload_id": upload_id,
                "status": "completed",
                "started_at": start_ts,
                "completed_at": completed_ts,
                "stats": stats,
            }
        )
    except Exception as exc:
        logger.exception("Ingestion job failed")
        completed_ts = _utc_now()
        _set_ingest_state(
            state="failed",
            message="Ingestion failed",
            completed_at=completed_ts,
            error=str(exc),
        )
        _append_history(
            {
                "upload_id": upload_id,
                "status": "failed",
                "started_at": start_ts,
                "completed_at": completed_ts,
                "error": str(exc),
            }
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
        "model": "CLIP ViT-L/14",
        "index": "FAISS IndexFlatIP",
    }


@app.post("/api/ingest/upload")
async def ingest_upload(
    excel: UploadFile = File(...),
    images: list[UploadFile] | None = File(default=None),
    with_bg_images: list[UploadFile] | None = File(default=None),
    without_bg_images: list[UploadFile] | None = File(default=None),
):
    """Upload ingestion files (Excel + images or zip archives)."""
    excel_ext = Path(excel.filename or "").suffix.lower()
    if excel_ext not in {".xlsx", ".xls"}:
        raise HTTPException(400, "Excel file must be .xlsx or .xls")

    upload_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    upload_dir = UPLOAD_DIR / upload_id
    images_root = UPLOADED_IMAGES_DIR / upload_id
    archives_dir = upload_dir / "archives"

    upload_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    excel_path = upload_dir / f"sku_data{excel_ext}"
    excel_size = _save_upload_file(excel, excel_path)

    uploaded_count = 0
    extracted_count = 0

    def handle_files(
        files: list[UploadFile] | None, target_dir: Path | None = None
    ) -> None:
        nonlocal uploaded_count, extracted_count
        if not files:
            return

        for file in files:
            file_ext = Path(file.filename or "").suffix.lower()
            if file_ext == ".zip":
                zip_path = archives_dir / (file.filename or "images.zip")
                _save_upload_file(file, zip_path)
                extracted_count += _extract_zip(zip_path, images_root)
            elif file_ext in ALLOWED_EXTENSIONS:
                destination = (target_dir or (images_root / "without.background")) / (
                    file.filename or "image"
                )
                _save_upload_file(file, destination)
                uploaded_count += 1
            else:
                raise HTTPException(400, f"Unsupported image file: {file.filename}")

    handle_files(images)
    handle_files(with_bg_images, images_root / "with.background")
    handle_files(without_bg_images, images_root / "without.background")

    _normalize_uploaded_images(images_root)

    return {
        "upload_id": upload_id,
        "excel": {
            "name": excel.filename,
            "size_bytes": excel_size,
            "stored_at": str(excel_path),
        },
        "images": {
            "uploaded_files": uploaded_count,
            "extracted_from_zip": extracted_count,
            "root_dir": str(images_root),
        },
    }


@app.post("/api/ingest/start")
async def ingest_start(request: IngestStartRequest):
    """Start ingestion job for an uploaded dataset."""
    global _ingest_thread

    with _ingest_lock:
        if _ingest_state["state"] == "running":
            raise HTTPException(409, "An ingestion job is already running")

    upload_id = request.upload_id
    upload_dir = UPLOAD_DIR / upload_id
    images_root = UPLOADED_IMAGES_DIR / upload_id

    if not upload_dir.exists():
        raise HTTPException(404, f"Upload session not found: {upload_id}")
    if not images_root.exists():
        raise HTTPException(404, f"Uploaded images not found: {upload_id}")

    excel_files = [
        path
        for path in upload_dir.iterdir()
        if path.suffix.lower() in {".xlsx", ".xls"}
    ]
    if not excel_files:
        raise HTTPException(400, "No Excel file found for this upload session")

    excel_path = excel_files[0]
    _set_ingest_state(
        state="starting",
        upload_id=upload_id,
        progress=0,
        message="Queued ingestion job",
        started_at=None,
        completed_at=None,
        stats=None,
        error=None,
    )
    _ingest_thread = threading.Thread(
        target=_run_ingestion_job,
        args=(upload_id, excel_path, images_root),
        daemon=True,
    )
    _ingest_thread.start()

    return {
        "status": "started",
        "upload_id": upload_id,
        "state": _get_ingest_state(),
    }


@app.get("/api/ingest/status")
async def ingest_status():
    """Return current ingestion job status."""
    return _get_ingest_state()


@app.get("/api/ingest/history")
async def ingest_history(limit: int = Query(default=20, ge=1, le=200)):
    """Return recent ingestion jobs."""
    history = _read_history()
    return {"items": history[:limit], "count": min(len(history), limit)}


def _semantic_query_embedding(query: str):
    """Build semantic query embedding with optional expansion."""
    clip = get_clip_model()
    if QUERY_EXPANSION_ENABLED:
        import numpy as np

        variants = [
            query,
            f"A photo of {query}",
            f"Industrial {query} spare part",
        ]
        embeddings = [clip.embed_text(variant) for variant in variants]
        query_embedding = np.mean(embeddings, axis=0)
        return query_embedding / np.linalg.norm(query_embedding)
    return clip.embed_text(query)


@app.post("/api/chat")
async def chat_search(
    message: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
):
    """Chat-style search endpoint (text, image, or both)."""
    if not message and not image:
        raise HTTPException(400, "Provide message text, image, or both")

    engine = get_search_engine()
    start = time.time()
    visual_results: list[dict] = []
    semantic_results: list[dict] = []

    if image is not None:
        ext = Path(image.filename or "").suffix.lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}"
            )

        contents = await image.read()
        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(
                400, f"Image too large. Max size: {MAX_IMAGE_SIZE // (1024 * 1024)}MB"
            )

        clip = get_clip_model()
        query_embedding = clip.embed_image(contents)
        visual_results = engine.visual_search(query_embedding, top_k=5)

    cleaned_message = (message or "").strip()
    if cleaned_message:
        query_embedding = _semantic_query_embedding(cleaned_message)
        semantic_results = engine.semantic_search(query_embedding, top_k=5)

    if visual_results and semantic_results:
        merged_by_code: dict[str, dict] = {}
        for result in visual_results + semantic_results:
            wise_code = result.get("wise_code")
            if not wise_code:
                continue
            existing = merged_by_code.get(wise_code)
            if existing is None or result.get("score", 0) > existing.get("score", 0):
                merged_by_code[wise_code] = result
        results = sorted(
            merged_by_code.values(), key=lambda item: item.get("score", 0), reverse=True
        )[:5]
        input_type = "multimodal"
    elif visual_results:
        results = visual_results[:5]
        input_type = "image"
    else:
        results = semantic_results[:5]
        input_type = "text"

    elapsed = round((time.time() - start) * 1000, 1)
    response_message = f"Found {len(results)} matching products"
    if cleaned_message and not image:
        response_message = (
            f"Found {len(results)} matching products for '{cleaned_message}'"
        )
    elif image and not cleaned_message:
        response_message = f"Found {len(results)} matching products for your image"

    return {
        "type": "search_results",
        "input_type": input_type,
        "message": response_message,
        "results": results,
        "count": len(results),
        "elapsed_ms": elapsed,
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
        raise HTTPException(
            400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}"
        )

    contents = await image.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(
            400, f"Image too large. Max size: {MAX_IMAGE_SIZE // (1024 * 1024)}MB"
        )

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

    query_embedding = _semantic_query_embedding(q)

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
        raise HTTPException(
            400, "image_type must be 'with.background' or 'without.background'"
        )

    image_path = IMAGES_DIR / image_type / filename
    if not image_path.exists():
        if UPLOADED_IMAGES_DIR.exists():
            session_dirs = [
                path for path in UPLOADED_IMAGES_DIR.iterdir() if path.is_dir()
            ]
            session_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            for session_dir in session_dirs:
                candidate = session_dir / image_type / filename
                if candidate.exists():
                    image_path = candidate
                    break

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
