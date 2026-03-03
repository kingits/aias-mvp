# AI.AS - Smart Product Search Engine (MVP)

AI Assistance System for product search across 100,000+ SKUs using Visual Search (image) and Semantic Search (text).

## Tech Stack

- **AI Engine**: OpenAI CLIP (ViT-B/32) for image & text embeddings
- **Vector Search**: FAISS (IndexFlatIP) for similarity search
- **Backend**: Python FastAPI with async support
- **Frontend**: Vanilla HTML/CSS/JS (no build step)

## Quick Start

```bash
# 1. Setup
bash scripts/setup.sh
source venv/bin/activate

# 2. Index product data (1K sample SKUs)
python scripts/ingest_data.py \
  --images-dir "../ตัวอย่างภาพสินค้า1K" \
  --excel "../ตัวอย่างภาพสินค้า1K/SKUของตัวอย่างภาพสินค้า1K.xlsx"

# 3. Start server
bash scripts/run_server.sh

# 4. Open http://localhost:8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/search/visual` | Search by image upload |
| GET | `/api/search/semantic?q=...` | Search by text query |
| GET | `/api/products/{wise_code}` | Get product details |
| GET | `/api/products` | List products (paginated) |
| GET | `/api/health` | System health check |

## Project Structure

```
aias-mvp/
├── backend/
│   ├── main.py           # FastAPI app & endpoints
│   ├── config.py          # Configuration
│   ├── models/
│   │   └── clip_model.py  # CLIP embedding model
│   └── search/
│       └── engine.py      # FAISS search engine
├── frontend/
│   └── index.html         # Web UI (single-file)
├── scripts/
│   ├── setup.sh           # Environment setup
│   ├── ingest_data.py     # Data ingestion pipeline
│   └── run_server.sh      # Server launcher
├── data/                  # Product images & FAISS index
├── requirements.txt
└── README.md
```
