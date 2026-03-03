# MEMORY.md — AI.AS MVP Change Log

> This file is maintained by agentic coding assistants.
> Every feature update, architecture change, or notable modification must be recorded here.
> Newest entries go at the top of each section.

---

## Project State Summary

- **CLIP Model**: `openai/clip-vit-large-patch14` (ViT-L/14, 768-dim) — upgraded from ViT-B/32
- **Search**: Dual FAISS index (image + text) with hybrid scoring
- **Frontend**: Single-file SPA (`frontend/index.html`), vanilla HTML/CSS/JS
- **Tests**: None configured (tests/ is empty)
- **Linting**: None configured (no ruff, black, flake8, mypy)
- **CI/CD**: None configured

---

## Change Log

### 2026-03-04 — Comprehensive fine-tuning accuracy improvements (Phase 1-3)

- **What**: Overhauled fine-tuning pipeline to fix accuracy regressions and improve overall search quality
- **Why**: Fine-tuned model showed strong visual search (Recall@1=85%) but weak semantic search (Recall@1=58%) due to mismatches between training and inference
- **Changes**:
  - **Phase 1 (Fix Mismatches)**:
    - Reused `augment_description()` from ingestion pipeline (text template now includes `unit` field)
    - Reused `find_product_images()` from ingestion (now uses BOTH `with.background/` and `without.background/` folders)
    - Added 6 random text templates per sample (matches query expansion variants + extra diversity)
  - **Phase 2 (Training Improvements)**:
    - Added LR warmup (10%) + cosine decay schedule via `get_cosine_schedule_with_warmup`
    - Added per-SKU image cap (`--max-images-per-sku`, default 6) to prevent training imbalance
    - Changed default args: `lr=5e-6` (was 1e-5), `epochs=12` (was 5), added `--seed=42`
    - Added reproducibility seeding
  - **Phase 3 (Evaluation Fix)**:
    - Added query expansion to semantic evaluation (mirrors runtime `_semantic_query_embedding`)
    - Eval now accurately reflects real-world search accuracy
- **Files modified**:
  - `scripts/fine_tune_clip.py`: All Phase 1+2 changes
  - `scripts/evaluate_search.py`: Added `build_semantic_embedding()` helper and query expansion in eval
- **New CLI defaults**:
  - `--epochs 12` (was 5)
  - `--lr 5e-6` (was 1e-5)
  - `--output-dir model/fine_tuned_clip_v2` (was data/fine_tuned_clip)
  - `--warmup-ratio 0.1` (new)
  - `--max-images-per-sku 6` (new)
  - `--seed 42` (new)
- **Expected impact**: Visual +2-5%, Semantic +10-20% (from ~58% to ~70-78% Recall@1)

### 2026-03-03 — Hardened fine-tune model loading fallback for missing local path

- **What**: Updated `scripts/fine_tune_clip.py` to resolve `--model-name` more safely and auto-fallback when a configured local path does not exist
- **Why**: Fine-tuning crashed at startup when `AIAS_CLIP_MODEL_NAME` pointed to a local merged-model path that was not created yet (e.g., `data/fine_tuned_clip/merged`)
- **Changes**:
  - Added `resolve_model_name()` to detect path-like model values
  - If local model path exists, load it normally
  - If local model path is missing, log a warning and fallback to base model `openai/clip-vit-large-patch14`
  - Updated model+processor loading to use resolved model source
- **Operational result**:
  - `scripts/fine_tune_clip.py` can start reliably even when runtime env is configured to a not-yet-generated fine-tuned path
  - Prevents `HFValidationError` / `Incorrect path_or_model_id` on fresh training runs

### 2026-03-03 — Fixed LoRA model loading path and added merged-model output

- **What**: Updated CLIP loading and fine-tuning workflow to handle LoRA adapter outputs correctly
- **Why**: Users hit "model not found/load failed" after fine-tuning because LoRA output can be adapter-only and not directly loadable as a full CLIP model
- **Changes**:
  - `backend/models/clip_model.py`: `CLIPEmbedder.load()` now detects adapter directories (`adapter_config.json`), auto-loads base model, applies adapter, and merges weights via `peft`
  - `scripts/fine_tune_clip.py`: after LoRA training, script now also saves a merged full model to `output_dir/merged` for direct use
  - `backend/config.py`: `CLIP_MODEL_NAME` now supports env override via `AIAS_CLIP_MODEL_NAME`
- **Operational result**:
  - Can point `CLIP_MODEL_NAME` (or `AIAS_CLIP_MODEL_NAME`) to either adapter dir or merged dir
  - Recommended production path: use merged model directory for simpler deployment

### 2026-03-03 — Started Sub-task 6: LoRA dependency + evaluation smoke run

- **What**: Added `peft>=0.7.0` to `requirements.txt` and installed `peft` in local virtualenv
- **Why**: Unblock LoRA fine-tuning flow in `scripts/fine_tune_clip.py` without requiring manual dependency install per session
- **Validation**:
  - `venv/bin/python scripts/fine_tune_clip.py --help` runs successfully
  - Ran a quick evaluation smoke test (`sample-size=10`) against current index:
    - Visual: Recall@1=80.0%, Recall@5=100.0%, MRR=0.8583, avg latency=512.0ms
    - Semantic: Recall@1=80.0%, Recall@5=100.0%, MRR=0.8833
  - Results file refreshed at `data/index/eval_results.json`
- **Execution note**:
  - Attempted to run one-epoch fine-tuning (`scripts/fine_tune_clip.py --epochs 1`) on CPU; process started successfully (dataset loaded + LoRA initialized) but exceeded session timeout during training loop
  - Full fine-tuning should be run in a longer session or on GPU hardware

### 2026-03-03 — Added uploaded-image fallback in image serving endpoint

- **What**: Updated `GET /api/images/{image_type}/{filename}` in `backend/main.py` to fall back to `data/images/uploaded/<upload_id>/...` when the file is not found under primary `IMAGES_DIR`
- **Why**: Ensure product cards can still display images after running ingestion from uploaded datasets, even when base `IMAGES_DIR` points to legacy/sample directories
- **Behavior**:
  - Tries primary path first: `IMAGES_DIR / image_type / filename`
  - If missing, scans upload session folders by most recently modified and serves first matching file
  - Returns 404 only when no match is found in either location

### 2026-03-03 — Added Chat tab UI integrated with `/api/chat`

- **What**: Implemented a new `Chat` tab in `frontend/index.html` with conversational UI and product-card responses
- **Why**: Deliver planned chat-style search experience while keeping implementation lightweight (search wrapper, no LLM dependency)
- **Chat features**:
  - New tab button + `#chat-panel` with message timeline, input box, image attachment, send button, and clear chat
  - Sends multipart requests to `POST /api/chat` with text, image, or both
  - Renders AI response message and top matching product mini-cards (image, wise code, description, score)
  - Supports attached-image preview and Enter-to-send flow
  - Added initial assistant greeting and reusable chat bubble rendering helpers

### 2026-03-03 — Added `/api/chat` multimodal search endpoint

- **What**: Added `POST /api/chat` in `backend/main.py` to support chat-style product retrieval for text, image, or both inputs
- **Why**: Prepare backend for Chat tab UI while reusing existing semantic/visual search logic without introducing LLM dependencies
- **Behavior**:
  - Accepts `message` (form text), `image` (file), or both
  - Text-only path uses semantic search with existing query expansion strategy
  - Image-only path uses visual search with existing file validation constraints
  - Image+text path merges and deduplicates by `wise_code`, then returns top 5 by best score
  - Returns chat-friendly payload: `type`, `input_type`, `message`, `results`, `count`, `elapsed_ms`
- **Refactor**:
  - Extracted semantic embedding construction into `_semantic_query_embedding()` and reused it from both `/api/search/semantic` and `/api/chat`

### 2026-03-03 — Added Data Management tab to frontend for ingestion workflow

- **What**: Extended `frontend/index.html` with a new "Data Management" tab and client-side ingestion workflow
- **Why**: Provide customer-facing upload UI for Excel + images and live index rebuild progress without CLI usage
- **Frontend updates**:
  - Refactored tab switching to use `data-tab` attributes (removed hardcoded index logic), enabling scalable multi-tab layout
  - Added new `#ingest-panel` with three steps: Excel upload, image/ZIP upload, and ingestion start
  - Added ingestion progress UI (progress bar, status text, error text) and polling integration for `GET /api/ingest/status`
  - Added ingestion history table wired to `GET /api/ingest/history`
  - Added drag-and-drop helpers reused across visual upload and ingestion upload zones
  - Added upload file summaries (selected Excel file, image file count/size, preview list)

### 2026-03-03 — Added Data Ingestion API endpoints with background processing

- **What**: Implemented ingestion upload/start/status/history APIs in `backend/main.py` and added upload path config in `backend/config.py`
- **Why**: Enable multi-session implementation of the Data Ingestion Web Module so customers can upload files and trigger full index rebuild from UI
- **Added endpoints**:
  - `POST /api/ingest/upload` — accepts Excel + image files (individual files and ZIP archives), stores uploads under `data/uploads/` and `data/images/uploaded/`
  - `POST /api/ingest/start` — starts ingestion in a background thread for a specific `upload_id`
  - `GET /api/ingest/status` — returns live job state (`idle|starting|running|completed|failed`) with progress/message/error/stats
  - `GET /api/ingest/history` — returns recent ingestion run history from `data/uploads/history.json`
- **Implementation details**:
  - Added ingestion runtime state management with lock-protected in-memory state + progress callback wiring
  - Added secure ZIP extraction guard against path traversal
  - Added automatic image-root normalization to ensure `with.background/` and `without.background/` folders exist
  - On successful ingestion, API now reloads `SearchEngine` so new indexes are available without server restart
  - Updated health endpoint model label from `CLIP ViT-B/32` to `CLIP ViT-L/14`
- **Config updates**:
  - Added `UPLOAD_DIR = data/uploads`
  - Added `UPLOADED_IMAGES_DIR = data/images/uploaded`
  - Added `MAX_UPLOAD_SIZE = 500MB`

### 2026-03-03 — Refactored ingestion logic into reusable backend module

- **What**: Added `backend/ingest/pipeline.py` and `backend/ingest/__init__.py` to centralize ingestion helpers and orchestration logic
- **Why**: Prepare for multi-session implementation of the Data Ingestion Web Module by making ingestion callable from API endpoints (not only CLI)
- **Details**:
  - Moved reusable functions out of `scripts/ingest_data.py`: `load_sku_data`, `find_product_images`, `augment_description`, `compute_multi_angle_embedding`
  - Added `run_ingestion_pipeline(images_dir, excel_path, progress_callback=None)` to run full ingestion and return summary stats
  - Added optional progress callback contract `(current, total, stage)` for future API status tracking
  - Updated `scripts/ingest_data.py` to delegate to `run_ingestion_pipeline` while keeping CLI arguments/behavior compatible
  - Kept `--batch-size` argument as reserved compatibility flag (warns when non-default is provided)

### 2026-03-03 — Enriched AGENTS.md with project-brief.md context

- **What**: Added "Data & Domain Context" section to `AGENTS.md` with content derived from `project-brief.md`
- **Why**: AGENTS.md was missing critical domain knowledge that agents need — image filename mapping, data flow, eval results, success criteria, and pending improvement strategies
- **Added sections**:
  - Image Filename Mapping (with.background vs without.background naming conventions)
  - Data Flow (ingestion pipeline + runtime search flow with hybrid scoring weights)
  - Current Evaluation Results (ViT-L/14: Visual Recall@5=98%, Semantic Recall@5=97%)
  - Success Criteria table (targets from project-brief.md)
  - Pending Improvements (Priority 1: both-folder indexing, Priority 3: LoRA fine-tune, Future: category-aware filtering)
  - Phase 2+ roadmap awareness (SQL Server, Chat, Pricing, scale)

### 2026-03-03 — Added mandatory MEMORY.md read rule to AGENTS.md

- **What**: Added "IMPORTANT: Read Memory First" section at the top of `AGENTS.md`
- **Why**: Ensure every agent reads `MEMORY.md` before starting work, and updates it after completing changes — prevents context loss between sessions

### 2026-03-03 — Created AGENTS.md

- **What**: Created `AGENTS.md` at project root (201 lines)
- **Why**: Provide agentic coding assistants with full context on build commands, architecture, code style, and project conventions
- **Sections**: Project Overview, Build/Run Commands, Testing & Linting gaps, Architecture map, Code Style Guidelines (imports, formatting, types, naming, docstrings, logging, paths, error handling, API response shape), Important Notes for Agents
- **Key details documented**:
  - No test/lint/format tools exist; `tests/` directory is empty
  - Singleton + lazy loading pattern for CLIPEmbedder and SearchEngine
  - Dual FAISS index with hybrid scoring (configurable weights)
  - All embeddings L2-normalized (Inner Product = Cosine Similarity)
  - Scripts use `sys.path.insert` hack for imports
  - Thai file/folder names in sample data paths
  - `.claude/settings.local.json` permissions

### 2026-03-03 — Created MEMORY.md

- **What**: Created this file (`MEMORY.md`) at project root
- **Why**: Persistent memory for tracking all project changes across agent sessions

---

## Architecture Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| Pre-existing | ViT-L/14 over ViT-B/32 | 768-dim embeddings capture finer detail for similar-looking industrial parts |
| Pre-existing | Dual FAISS index (image + text) | Hybrid scoring disambiguates visually similar products using text similarity |
| Pre-existing | No database (JSON + FAISS files) | MVP simplicity; metadata is small enough for in-memory JSON |
| Pre-existing | Vanilla JS frontend (no framework) | No build step needed, simplifies deployment |

---

## Known Issues / Tech Debt

- README.md still references ViT-B/32 but config.py uses ViT-L/14
- No tests exist anywhere in the project
- No linting or formatting tooling
- No CI/CD pipeline
- CORS is wide open (`allow_origins=["*"]`)
- API responses are plain dicts, not Pydantic response models
- `run_server.sh` checks for legacy `product_vectors.index` but dual index uses `image_vectors.index` + `text_vectors.index`
