# AGENTS.md — AI.AS MVP

> Instructions for agentic coding assistants operating in this repository.

## IMPORTANT: Read Memory First

**Before doing ANY work in this repository, you MUST read `MEMORY.md` at the project root.** It contains the running change log, current project state, architecture decisions, and known issues. This ensures you have full context and do not duplicate or contradict previous work.

After completing any feature, fix, or notable change, **you MUST update `MEMORY.md`** with a new Change Log entry (newest on top) documenting what changed and why.

---

## Project Overview

AI.AS (AI Assistance System) is a **Smart Product Search Engine** MVP for industrial spare parts (100K+ SKUs). It provides two search modes:

- **Visual Search** — upload a product image, find matching products via CLIP image embeddings
- **Semantic Search** — type a text query, find products via CLIP text embeddings

**Stack**: Python 3.12, FastAPI, OpenAI CLIP (ViT-L/14), FAISS (IndexFlatIP), vanilla HTML/CSS/JS frontend (no build step). No database — metadata is stored as JSON, vectors in FAISS index files.

---

## Build / Run Commands

All commands assume the virtualenv is active (`source venv/bin/activate`).

```bash
# Environment setup (first time only)
bash scripts/setup.sh
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Data ingestion (must run before search works)
python scripts/ingest_data.py \
  --images-dir "../ตัวอย่างภาพสินค้า1K" \
  --excel "../ตัวอย่างภาพสินค้า1K/SKUของตัวอย่างภาพสินค้า1K.xlsx"

# Start dev server (with hot reload)
bash scripts/run_server.sh
# — or equivalently:
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Evaluate search accuracy
python scripts/evaluate_search.py \
  --images-dir "../ตัวอย่างภาพสินค้า1K" \
  --top-k 5 --sample-size 100

# Fine-tune CLIP (requires peft>=0.7.0)
python scripts/fine_tune_clip.py \
  --images-dir "../ตัวอย่างภาพสินค้า1K" \
  --excel "../ตัวอย่างภาพสินค้า1K/SKUของตัวอย่างภาพสินค้า1K.xlsx" \
  --epochs 5 --output-dir data/fine_tuned_clip
```

### Testing & Linting

**No test framework, linter, or formatter is currently configured.** The `tests/` directory exists but is empty. There is no pytest, flake8, ruff, black, mypy, or pre-commit setup. If you add tests, place them in `tests/` and follow existing code conventions.

---

## Architecture

```
backend/
  main.py              # FastAPI app, routes, startup lifecycle
  config.py            # All configuration (env vars with defaults)
  models/
    clip_model.py      # CLIPEmbedder — image & text embedding generation
  search/
    engine.py          # SearchEngine — dual FAISS index, hybrid scoring
frontend/
  index.html           # Single-file SPA (HTML + CSS + JS, no build step)
scripts/
  setup.sh             # Virtualenv creation + dependency install
  run_server.sh        # Server launcher
  ingest_data.py       # Data ingestion pipeline (Excel + images -> FAISS index)
  evaluate_search.py   # Search accuracy evaluation (Recall@K, MRR)
  fine_tune_clip.py    # LoRA fine-tuning of CLIP on domain data
data/                  # Runtime data (gitignored)
  index/               # FAISS index files + product_metadata.json
  images/              # Product images (with.background / without.background)
```

### Key Patterns

- **Singleton + lazy loading**: `CLIPEmbedder` and `SearchEngine` are accessed via global functions `get_clip_model()` and `get_search_engine()`. Both use `_ensure_loaded()` guards.
- **Dual FAISS index**: Separate `image_vectors.index` and `text_vectors.index` for hybrid scoring. Falls back to legacy single `product_vectors.index` if dual indexes are absent.
- **Hybrid scoring**: `final_score = (image_weight * image_sim) + (text_weight * text_sim)`. Weights are configurable in `backend/config.py`.
- **All embeddings are L2-normalized** so Inner Product = Cosine Similarity.
- **Configuration**: Centralized in `backend/config.py` via `os.getenv()` with sensible defaults. See `.env.example` for available env vars.
- **Frontend served by FastAPI**: `app.mount("/", StaticFiles(...))` at the end of `main.py`.

---

## Code Style Guidelines

### Imports

Standard library first, then third-party, then local (`backend.*`). No import sorting tool is enforced — follow PEP 8 grouping manually.

```python
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from PIL import Image
import numpy as np

from backend.config import HOST, PORT, DEBUG
from backend.models.clip_model import get_clip_model
```

### Formatting

- **Indentation**: 4 spaces (no tabs)
- **Line length**: ~120 characters (no enforced limit, match existing code)
- **Strings**: Use f-strings for interpolation, double quotes preferred
- **No formatter configured** — match the style of surrounding code

### Type Hints

Use type hints on all function signatures and class attributes. Use Python 3.9+ built-in generics (`list[dict]`, `dict[int, float]`) rather than `typing.List`/`typing.Dict`. Use `Optional` and `Union` from `typing` where needed.

```python
def hybrid_search(self, query_vector: np.ndarray, top_k: int = 10) -> list[dict]:
```

### Naming

Follow PEP 8 strictly:
- `snake_case` for functions, methods, variables, module names
- `PascalCase` for classes (`CLIPEmbedder`, `SearchEngine`)
- `UPPER_CASE` for module-level constants (`EMBEDDING_DIM`, `TOP_K_RESULTS`)
- Private/internal methods prefixed with `_` (`_ensure_loaded`, `_search_single`)

### Docstrings

- Every `.py` file starts with a triple-quoted module-level docstring
- Classes get a one-line docstring describing their purpose
- Methods use Google-style docstrings with `Args:` and `Returns:` sections

```python
"""CLIP Model wrapper for image and text embedding generation."""

class CLIPEmbedder:
    """Handles CLIP model loading and embedding generation for both images and text."""

    def embed_image(self, image: Union[Image.Image, bytes, str, Path]) -> np.ndarray:
        """Generate embedding vector from an image.

        Args:
            image: PIL Image, bytes, file path, or Path object

        Returns:
            Normalized embedding vector (numpy array)
        """
```

### Logging

Each module creates its own logger. Use the standard library `logging` module:

```python
import logging
logger = logging.getLogger(__name__)
```

Format (set in `main.py`): `%(asctime)s [%(levelname)s] %(name)s: %(message)s`

### Path Handling

Always use `pathlib.Path`. Never concatenate strings for file paths.

```python
from pathlib import Path
image_path = IMAGES_DIR / image_type / filename  # correct
image_path = IMAGES_DIR + "/" + filename          # wrong
```

### Error Handling

- In FastAPI endpoints: raise `HTTPException(status_code, detail)` with appropriate codes (400, 404)
- In singletons: use `_ensure_loaded()` guard before accessing state
- No global error handling middleware exists — errors are handled per-endpoint

### API Response Shape

All search endpoints return consistent JSON:
```json
{ "query_type": "visual|semantic", "results": [...], "count": 10, "elapsed_ms": 45.2 }
```
Responses are plain Python dicts, not Pydantic response models.

---

## Data & Domain Context

### Image Filename Mapping

Product images live in two folders with **different naming conventions**:

```
Wise Code:          000-0300005-23
with.background/    000030000523.jpg       (dashes removed)
without.background/ 000-0300005-23.JPG     (dashes kept, may be uppercase ext)
```

A single product can have multiple angles: `_1.jpg`, `_2.jpg`, etc.

### Data Flow

1. **Ingestion** (one-time via `scripts/ingest_data.py`):
   - Read Excel -> get Wise Code + Description for 440 SKUs
   - Load product images -> CLIP embed -> 768-dim vector per image
   - Multi-angle images are averaged into a single embedding per product
   - Text descriptions are augmented with templates before embedding
   - Build dual FAISS indexes (`image_vectors.index` + `text_vectors.index`) + `product_metadata.json`

2. **Visual Search** (runtime): User uploads image -> CLIP embed -> hybrid FAISS search (image 0.7 + text 0.3)
3. **Semantic Search** (runtime): User types query -> query expansion (3 variants averaged) -> hybrid FAISS search (text 0.8 + image 0.2)

### Current Evaluation Results (ViT-L/14, dual index)

From `data/index/eval_results.json` (100 samples, top-k=5):

| Metric | Visual Search | Semantic Search |
|--------|--------------|-----------------|
| Recall@1 | 57.0% | 73.0% |
| Recall@5 | 98.0% | 97.0% |
| MRR | 0.7305 | 0.8270 |

### Success Criteria (from project-brief.md)

| Metric | Target |
|--------|--------|
| Visual Search | Top-5 accuracy > 80% |
| Semantic Search | Top-5 relevance > 70% |
| Response time | < 500ms per query |
| Ingestion | 1K SKUs in < 30 min |

### Pending Improvements (not yet implemented)

- **Priority 1**: Index images from BOTH folders (with + without background) to reduce domain gap
- **Priority 3**: LoRA fine-tune CLIP on Wise product data (`scripts/fine_tune_clip.py` is ready)
- **Future**: Category-aware filtering using Wise Code prefix as category signal

### Phase 2+ (out of MVP scope, for awareness only)

SQL Server integration, Chat Interface, Personalized Pricing, scale to 100K SKUs, 50 concurrent users.

---

## Important Notes for Agents

1. **CLIP model download**: First server start downloads ~2GB of model weights from Hugging Face. Expect a long initial startup.
2. **Ingestion required**: Search returns empty results until `scripts/ingest_data.py` has been run and FAISS indexes exist in `data/index/`.
3. **Gitignored directories**: `data/index/`, `data/images/`, `venv/`, `__pycache__/`, `.env` are all gitignored. Never commit these.
4. **Scripts path hack**: Scripts in `scripts/` use `sys.path.insert(0, project_root)` to import from `backend.*`. This is intentional — do not remove it.
5. **Current CLIP model**: `openai/clip-vit-large-patch14` (768-dim embeddings). The README mentions ViT-B/32 but config.py has been upgraded to ViT-L/14.
6. **Thai file/folder names**: Sample data paths contain Thai characters (e.g., `../ตัวอย่างภาพสินค้า1K`). These are external to the repo and must be passed as CLI arguments.
7. **No CI/CD pipeline**: No GitHub Actions, no automated checks. Manual verification only.
8. **`.claude/settings.local.json`**: Pre-configured permissions for `source`, `python scripts/ingest_data.py`, and `python scripts/evaluate_search.py`.
