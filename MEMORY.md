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
