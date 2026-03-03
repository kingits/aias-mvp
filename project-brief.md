# AI.AS — MVP Project Brief

> **AI Assistance System**: ระบบผู้ช่วยอัจฉริยะบริหารจัดการสินค้า
> Smart Product Search Engine สำหรับ 100,000+ SKUs

---

## 1. Project Overview

สร้าง MVP ของระบบ **Smart Search Engine** ที่รองรับการค้นหาสินค้า 2 แบบ:

- **Visual Search** — อัพโหลดรูปภาพสินค้า แล้วระบบค้นหาสินค้าที่ตรงกันจาก database
- **Semantic Search** — พิมพ์คำอธิบายหรือ keyword แล้วระบบค้นหาด้วยความหมาย (ไม่ใช่ keyword matching)

MVP นี้เป็นส่วนแรกของโปรเจกต์ใหญ่ที่มี scope เต็มรวม Chat Interface, Personalized Pricing, Human Takeover แต่ตอนนี้ **focus เฉพาะ Smart Search** ก่อน

---

## 2. Tech Stack

| Layer | Technology | เหตุผล |
|-------|-----------|--------|
| **AI Model** | OpenAI CLIP (ViT-B/32) | Embed ทั้งรูปภาพและข้อความลง vector space เดียวกัน ทำให้ Visual + Semantic Search ใช้ index เดียว |
| **Vector Search** | FAISS (IndexFlatIP) | Similarity search ระดับ millisecond รองรับ 100K+ vectors บน local hardware |
| **Backend** | Python FastAPI | Async, auto-docs (Swagger), รองรับ file upload |
| **Frontend** | HTML/CSS/JS (Vanilla) | ไม่ต้อง build step, responsive, ง่ายต่อการ deploy |
| **Data Storage** | SQLite + JSON | เก็บ metadata สินค้า, FAISS index files |
| **Image Processing** | Pillow + torchvision | Preprocess รูปภาพก่อนส่งเข้า CLIP |

---

## 3. ข้อมูลสินค้าตัวอย่าง (Sample Data)

### 3.1 Excel — `ตัวอย่างภาพสินค้า1K/SKUของตัวอย่างภาพสินค้า1K.xlsx`

มี 440 SKUs (subset จาก 100K) ประกอบด้วย 4 columns:

| Column | ตัวอย่าง |
|--------|---------|
| NO. | 1, 2, 3... |
| Wise Code | `000-0300005-23`, `010-0400074-82` |
| Description | `Alarm, Sound Back-up, 48V, 90dB, YEC` |
| Unit | `PC.`, `SET` |

### 3.2 รูปภาพสินค้า — 1,000 ภาพ แบ่ง 2 folder

```
ตัวอย่างภาพสินค้า1K/
├── with.background/      # รูปพร้อมพื้นหลัง (มี watermark WISE Enterprise)
│   ├── 000030000523.jpg        ← ชื่อไฟล์ = Wise Code ไม่มีขีด
│   ├── 000030000523_1.jpg      ← รูปมุมอื่นของสินค้าเดียวกัน (_1, _2, ...)
│   └── ...
└── without.background/   # รูปตัดพื้นหลังแล้ว
    ├── 000-0300005-23.JPG      ← ชื่อไฟล์ = Wise Code มีขีด
    ├── 000-0300005-23_1.JPG
    └── ...
```

**สำคัญ**: ชื่อไฟล์ใน `with.background/` ไม่มี dash (`000030000523`) แต่ `without.background/` มี dash (`000-0300005-23`) ต้อง map ให้ถูก

### 3.3 ประเภทสินค้า

อะไหล่และชิ้นส่วนอุตสาหกรรม เช่น Bearing, Cable, Connector, Alarm, Filter, Gasket, Seal, Bolt, Nut, Pump, Motor, Valve ฯลฯ — สินค้าหลายตัวหน้าตาคล้ายกัน ระบบต้องแยกแยะได้

---

## 4. Architecture

```
┌─────────────────────────────────────────────────┐
│                   Frontend (Web UI)              │
│         HTML/CSS/JS — Responsive Design          │
│    [Image Upload] [Text Search] [Results Grid]   │
└──────────────────────┬──────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────┐
│              Backend (FastAPI)                    │
│                                                  │
│  POST /api/search/visual    ← รับรูป, return     │
│  GET  /api/search/semantic  ← รับ text, return   │
│  GET  /api/products/{code}  ← product details    │
│  GET  /api/products         ← list + pagination  │
│  GET  /api/health           ← system status      │
│  GET  /api/images/{type}/{file} ← serve images   │
└──────┬───────────────────────────┬──────────────┘
       │                           │
┌──────▼──────┐          ┌────────▼─────────┐
│ CLIP Model  │          │   FAISS Index    │
│ (ViT-B/32)  │          │  (IndexFlatIP)   │
│             │          │                  │
│ embed_image │──512d──▶│ similarity search │
│ embed_text  │──512d──▶│ top-K results    │
└─────────────┘          └──────────────────┘
```

### Data Flow

1. **Ingestion** (ทำครั้งเดียว):
   - อ่าน Excel → ได้ Wise Code + Description
   - โหลดรูปสินค้า → CLIP embed → ได้ 512-dim vector ต่อรูป
   - สร้าง FAISS index จาก vectors ทั้งหมด
   - บันทึก index + metadata เป็นไฟล์

2. **Visual Search** (runtime):
   - User upload รูป → CLIP embed → FAISS search → top-K products

3. **Semantic Search** (runtime):
   - User พิมพ์ query → CLIP embed → FAISS search (index เดียวกัน!) → top-K products

---

## 5. Project Structure

```
aias-mvp/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routes, startup
│   ├── config.py            # Configuration constants
│   ├── models/
│   │   ├── __init__.py
│   │   └── clip_model.py    # CLIP model wrapper (embed_image, embed_text)
│   ├── search/
│   │   ├── __init__.py
│   │   └── engine.py        # FAISS index management & search
│   └── data/
│       └── __init__.py
├── frontend/
│   └── index.html           # Single-file web UI
├── scripts/
│   ├── setup.sh             # Environment setup
│   ├── ingest_data.py       # Data ingestion pipeline
│   └── run_server.sh        # Server launcher
├── data/                    # Runtime data (gitignored)
│   ├── index/               # FAISS index + metadata JSON
│   └── images/              # Symlink or copy of product images
├── tests/
├── docs/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 6. API Specification

### POST `/api/search/visual`

ค้นหาสินค้าด้วยรูปภาพ

```
Content-Type: multipart/form-data
Body: image (file), top_k (int, default 10)

Response:
{
  "query_type": "visual",
  "results": [
    {
      "id": 1,
      "wise_code": "000-0300005-23",
      "description": "Alarm, Sound Back-up, 48V, 90dB, YEC",
      "unit": "PC.",
      "score": 0.8532,
      "images": {
        "with_bg": "/api/images/with.background/000030000523.jpg",
        "without_bg": "/api/images/without.background/000-0300005-23.JPG"
      }
    },
    ...
  ],
  "count": 10,
  "elapsed_ms": 45.2
}
```

### GET `/api/search/semantic?q={query}&top_k=10`

ค้นหาสินค้าด้วยข้อความ (natural language)

```
Response: same format as visual search, with "query_type": "semantic" and "query": "..."
```

### GET `/api/products/{wise_code}`

ดึงข้อมูลสินค้าจาก Wise Code

### GET `/api/products?page=1&per_page=20`

แสดงรายการสินค้าทั้งหมด แบบ pagination

### GET `/api/health`

```json
{
  "status": "healthy",
  "total_products": 440,
  "model": "CLIP ViT-B/32",
  "index": "FAISS IndexFlatIP"
}
```

---

## 7. Frontend Requirements

- **Responsive**: ใช้ได้บนมือถือ, tablet, PC
- **2 โหมดค้นหา**: Tab สลับระหว่าง Semantic (text) กับ Visual (image upload)
- **Image Upload**: รองรับ click และ drag-and-drop พร้อม preview
- **Results Grid**: แสดง card สินค้า มี Wise Code, Description, Unit, Score, รูปภาพ
- **Image Toggle**: สลับดูรูป with/without background ได้
- **Score Badge**: แสดงคะแนนความคล้าย (cosine similarity) เป็น %
- **Status Bar**: แสดงสถานะระบบ (online/offline) และจำนวนสินค้าที่ indexed
- **Loading State**: แสดง spinner ระหว่าง AI กำลังค้นหา

---

## 8. Key Implementation Details

### CLIP Model (`backend/models/clip_model.py`)

- ใช้ `transformers` library โหลด `openai/clip-vit-base-patch32`
- Auto-detect GPU/CPU
- L2 normalize ทุก embedding vector (เพื่อให้ Inner Product = Cosine Similarity)
- รองรับ batch embedding สำหรับ ingestion
- Singleton pattern — โหลด model ครั้งเดียว

### FAISS Search Engine (`backend/search/engine.py`)

- ใช้ `IndexFlatIP` (Inner Product) — เหมาะกับ normalized vectors
- `search()` return top-K results พร้อม metadata
- `build_index()` สร้าง index จาก embeddings array แล้วบันทึกไฟล์
- Minimum score threshold: 0.15 (กรอง noise ออก)

### Data Ingestion (`scripts/ingest_data.py`)

- อ่าน Excel → map Wise Code กับ images
- สินค้าที่มีรูป → ใช้ image embedding เป็น primary
- สินค้าที่ไม่มีรูป → ใช้ text embedding จาก description
- สร้าง metadata JSON รวม image paths ทั้ง with/without bg
- Log progress ทุก 50 items

### Image Filename Mapping

```
Wise Code:       000-0300005-23
with.background: 000030000523.jpg       (remove dashes)
without.bg:      000-0300005-23.JPG     (keep dashes, may be uppercase ext)

สินค้าหนึ่งอาจมีหลายรูป: _1.jpg, _2.jpg = มุมต่างๆ
```

---

## 9. Success Criteria

| Metric | Target |
|--------|--------|
| Visual Search accuracy | Top-5 มีสินค้าถูกต้อง > 80% |
| Semantic Search relevance | Top-5 เกี่ยวข้อง > 70% |
| Response time | < 500ms ต่อ query (1K SKUs) |
| Data ingestion | Process 1K SKUs ใน < 30 นาที |
| UI usability | เซลล์สามารถค้นหาได้ใน < 10 วินาที |

---

## 10. Hardware Requirements (MVP)

- **GPU**: NVIDIA 8GB+ VRAM (RTX 3060 ขึ้นไป) — สำหรับ CLIP inference ที่เร็ว
- **RAM**: 32 GB ขั้นต่ำ
- **Storage**: 256 GB SSD
- **CPU**: 8+ cores
- **OS**: Linux (Ubuntu 22.04 recommended) หรือ Windows with WSL2

> หมายเหตุ: CLIP สามารถรันบน CPU ได้ แต่จะช้ากว่า GPU ประมาณ 5-10x

---

## 11. Dependencies

```
# requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
torch>=2.0.0
torchvision>=0.15.0
transformers==4.44.0
Pillow>=10.0.0
faiss-cpu==1.8.0
pandas>=2.0.0
openpyxl>=3.1.0
pydantic>=2.0.0
python-dotenv>=1.0.0
aiofiles>=23.0.0
```

---

## 12. Quick Start Commands

```bash
# 1. Setup environment
bash scripts/setup.sh
source venv/bin/activate

# 2. Ingest sample data (1K SKUs)
python scripts/ingest_data.py \
  --images-dir "../ตัวอย่างภาพสินค้า1K" \
  --excel "../ตัวอย่างภาพสินค้า1K/SKUของตัวอย่างภาพสินค้า1K.xlsx"

# 3. Start server
bash scripts/run_server.sh
# → http://localhost:8000 (UI)
# → http://localhost:8000/docs (Swagger API docs)
```

---

## 13. Phase 2+ Roadmap (Out of MVP Scope)

สิ่งที่จะทำต่อหลัง MVP:

1. **SQL Server Integration** — เชื่อมกับ Legacy Server (SQL Server 2012) สำหรับ live pricing, stock
2. **Chat Interface** — AI chatbot ตอบคำถามสินค้า + Human Takeover
3. **Personalized Pricing** — คำนวณราคาตามส่วนลดเฉพาะลูกค้า
4. **Scale to 100K SKUs** — ย้ายไป AI Workstation (RTX 4090, 128GB RAM)
5. **50 Concurrent Users** — Queue management, load balancing
6. **Transaction Logs** — เก็บประวัติการค้นหาและ conversation

---

## 14. Accuracy Improvement Strategies

### Baseline Evaluation Results (ViT-B/32, Dual Index, Quick Wins applied)

ผล eval จาก `scripts/evaluate_search.py` กับ sample 100 สินค้า (top-k=5):

| Metric | Visual Search | Semantic Search |
|--------|--------------|-----------------|
| **Recall@1** | 25.0% | 43.0% |
| **Recall@5** | 51.0% | **96.0%** |
| **MRR** | 0.3493 | 0.6502 |
| Avg Latency | 40.6ms | — |

**วิเคราะห์:**
- Semantic Search ดีมาก (96% Recall@5) — text augmentation + query expansion ช่วยได้เยอะ
- Visual Search ยังอ่อน (51% Recall@5) — สาเหตุหลัก:
  1. **Domain gap**: test query ใช้รูป with.background (มี watermark) แต่ index สร้างจาก without.background
  2. **ViT-B/32 resolution ต่ำ** (224px): สินค้าอุตสาหกรรมที่ต่างกันแค่ size/spec ต้องการ resolution สูงกว่า
  3. **CLIP ไม่ได้ถูก train กับสินค้าอุตสาหกรรม** โดยเฉพาะ

### สิ่งที่ implement ไปแล้ว (Quick Wins — Strategy A)

- [x] A1. Hybrid Scoring — dual FAISS index (image + text) พร้อม weighted average
- [x] A2. ใช้รูป without.background เป็น primary (ลด watermark noise)
- [x] A3. Multi-angle average embedding (embed ทุกมุม แล้ว average)
- [x] A4. Text augmentation (ขยาย description ด้วย template)
- [x] A5. Query expansion สำหรับ semantic search
- [x] C3. Evaluation pipeline (`scripts/evaluate_search.py`)

### สิ่งที่ต้องทำต่อเพื่อปรับปรุง Visual Search

#### Priority 1: ปรับ Image Index ให้ใช้รูปทั้ง 2 แบบ (with + without background)

**ปัญหา**: ตอนนี้ index ใช้แค่ without.background แต่ user อาจส่งรูปที่มีพื้นหลัง (ถ่ายจากหน้างาน) ทำให้เกิด domain gap
**วิธีแก้**: สร้าง image embedding จากทั้ง 2 แบบ แล้ว average รวมกัน

```python
# ใน ingest_data.py — ปรับ compute_multi_angle_embedding ให้รวมรูปจากทั้ง 2 folder
all_images = images["without_bg"] + images["with_bg"]  # รวมทุกรูปทุกแบบ
image_embedding = compute_multi_angle_embedding(clip, all_images)
```

ผลที่คาดหวัง: Visual Search Recall@5 ควรเพิ่มจาก ~51% เป็น ~65-75% เพราะ index จะจำได้ทั้งรูปมีพื้นหลังและไม่มี

#### Priority 2: Upgrade CLIP เป็น ViT-L/14

**เหตุผล**: ViT-L/14 ใช้ resolution สูงขึ้น (224→336px option) และมี 768-dim embedding ซึ่งจับ fine-grained detail ได้ดีกว่า ViT-B/32 มาก — สำคัญมากสำหรับสินค้าที่หน้าตาคล้ายกัน

```python
# backend/config.py
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"  # เปลี่ยนจาก vit-base-patch32
EMBEDDING_DIM = 768  # เปลี่ยนจาก 512
```

**ข้อควรระวัง:**
- ต้อง re-run `ingest_data.py` ใหม่ทั้งหมดหลังเปลี่ยน model (เพราะ dimension เปลี่ยน)
- ใช้ VRAM ~6GB (RTX 3060 ขึ้นไปพอ, production ใช้ RTX 4090 สบาย)
- Inference ช้าลง ~2x แต่ยังอยู่ในระดับ <100ms ต่อ query

ผลที่คาดหวัง: +10-15% accuracy ทั้ง visual และ semantic search

#### Priority 3: Fine-tune CLIP บนข้อมูลสินค้า Wise

**Script พร้อมแล้ว**: `scripts/fine_tune_clip.py`

```bash
# ติดตั้ง LoRA dependency
pip install peft>=0.7.0

# Fine-tune (ใช้เวลา ~1-2 ชม. บน GPU)
python scripts/fine_tune_clip.py \
  --images-dir "../ตัวอย่างภาพสินค้า1K" \
  --excel "../ตัวอย่างภาพสินค้า1K/SKUของตัวอย่างภาพสินค้า1K.xlsx" \
  --epochs 5 \
  --output-dir data/fine_tuned_clip

# ใช้ model ที่ fine-tune แล้ว: แก้ config.py
CLIP_MODEL_NAME = "data/fine_tuned_clip"

# Re-run ingestion
python scripts/ingest_data.py ...
```

**หลักการ**: LoRA fine-tuning train แค่ ~1% ของ parameters (attention layers) ทำให้เร็วและไม่ overfit
**ผลที่คาดหวัง**: +15-25% accuracy บน Visual Search เพราะ model จะเรียนรู้ว่า bearing กับ bearing ต่าง size ต่างกันยังไง

### Target Accuracy หลังปรับปรุงครบ

| Metric | Current | After All Improvements (estimate) |
|--------|---------|-----------------------------------|
| Visual Recall@1 | 25% | 55-70% |
| Visual Recall@5 | 51% | 80-90% |
| Semantic Recall@1 | 43% | 60-75% |
| Semantic Recall@5 | 96% | 98%+ |

### Evaluation Workflow

หลังทำแต่ละ improvement ให้รัน eval เปรียบเทียบ:

```bash
# รัน eval
python scripts/evaluate_search.py \
  --images-dir "../ตัวอย่างภาพสินค้า1K" \
  --top-k 5 \
  --sample-size 100

# ผลจะบันทึกที่ data/index/eval_results.json
```

---

### Reference: Strategy Details

#### A1. Hybrid Scoring — รวม image + text similarity

ปัญหา: ใช้ image embedding อย่างเดียวอาจสับสนระหว่างสินค้าที่หน้าตาคล้ายกัน
วิธีแก้: สร้าง **2 index แยกกัน** — image index + text index แล้วรวม score ด้วย weighted average

```
final_score = (α × image_similarity) + (β × text_similarity)
```

- Visual Search: α=0.7, β=0.3 (เน้นรูป แต่ใช้ text ช่วย disambiguate)
- Semantic Search: α=0.2, β=0.8 (เน้น text แต่ใช้ image ช่วย re-rank)

Implementation:
- `backend/search/engine.py` — เพิ่ม `hybrid_search()` method ที่ query ทั้ง 2 index แล้วรวม score
- `scripts/ingest_data.py` — สร้าง 2 FAISS index: `image_vectors.index` + `text_vectors.index`
- Config: เพิ่ม `VISUAL_WEIGHT`, `SEMANTIC_WEIGHT` ใน `config.py`

#### A2. ใช้รูป without.background เป็น primary

ปัญหา: รูป with.background มี watermark "WISE Enterprise" ซ้อน ทำให้ embedding มี noise
วิธีแก้: ใช้รูป `without.background/` เป็น primary embedding เพราะ background สะอาด ไม่มี watermark

Implementation:
- `scripts/ingest_data.py` — เปลี่ยน priority: `without_bg` ก่อน `with_bg`

#### A3. Multi-Angle Average Embedding

ปัญหา: สินค้า 1 ชิ้นมีหลายรูป (`_1.jpg`, `_2.jpg`) แต่ตอนนี้ใช้แค่รูปแรก
วิธีแก้: embed ทุกมุมของสินค้า แล้ว **average embeddings** เป็น vector เดียว — ทำให้จับ feature ได้ครบถ้วนกว่า

```python
all_angles = [clip.embed_image(img) for img in product_images]
avg_embedding = np.mean(all_angles, axis=0)
avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # re-normalize
```

#### A4. Text Augmentation for Descriptions

ปัญหา: description สั้นมาก เช่น `"Bearing, 15x35x11, 6202.2RS"` — CLIP อาจไม่เข้าใจ context
วิธีแก้: ขยาย description ด้วย structured template ก่อน embed:

```python
augmented = f"Industrial product: {description}. Category: {category}. Wise Code: {wise_code}"
```

หรือใช้ LLM (Cloud AI) สร้าง rich description เพิ่ม:
```python
augmented = f"This is a {description}, an industrial spare part used in manufacturing equipment."
```

#### A5. Query Expansion for Semantic Search

ปัญหา: user พิมพ์สั้นๆ เช่น `"bearing"` — ผลลัพธ์กว้างเกินไป
วิธีแก้: ส่ง query ผ่าน CLIP text encoder หลายแบบแล้ว average:

```python
queries = [
    original_query,
    f"A photo of {original_query}",
    f"Industrial {original_query} spare part",
]
embeddings = [clip.embed_text(q) for q in queries]
avg_query = np.mean(embeddings, axis=0)
avg_query = avg_query / np.linalg.norm(avg_query)
```

---

### Strategy B: Better Model (เปลี่ยน CLIP variant)

#### B1. Upgrade to Larger CLIP

| Model | Dim | Accuracy | VRAM | Speed |
|-------|-----|----------|------|-------|
| `ViT-B/32` (ปัจจุบัน) | 512 | Baseline | ~2GB | เร็ว |
| `ViT-B/16` | 512 | +5-8% | ~3GB | ปานกลาง |
| `ViT-L/14` | 768 | +10-15% | ~6GB | ช้าลง |
| `ViT-L/14@336px` | 768 | +12-18% | ~8GB | ช้าที่สุด |

Implementation:
- `backend/config.py` — เปลี่ยน `CLIP_MODEL_NAME` และ `EMBEDDING_DIM`
- ต้อง re-run ingestion หลังเปลี่ยน model
- RTX 4090 (24GB) รองรับทุก variant ได้สบาย

#### B2. พิจารณา SigLIP หรือ EVA-CLIP

- **SigLIP** (`google/siglip-large-patch16-384`): ใช้ Sigmoid loss แทน contrastive, มักแม่นกว่า CLIP บน fine-grained tasks
- **EVA-CLIP** (`BAAI/EVA-CLIP`): Pre-trained ด้วย data มากกว่า, ดีกับ industrial domain

Implementation:
- เปลี่ยน model loader ใน `clip_model.py` ให้รองรับ SigLIP processor
- API ยังเหมือนเดิม (embed_image, embed_text return normalized vector)

---

### Strategy C: Fine-tune on Domain Data (แม่นที่สุด แต่ต้องใช้เวลา)

#### C1. Contrastive Fine-tuning

Train CLIP เพิ่มเติมด้วยข้อมูลสินค้า Wise โดยเฉพาะ โดยสร้าง training pairs:

- **Positive pairs**: (รูปสินค้า, description ที่ถูกต้อง)
- **Hard negatives**: (รูปสินค้า, description ของสินค้าคล้ายๆ แต่ไม่ใช่)

```python
# Training data format
pairs = [
    {"image": "bearing_6202.jpg", "text": "Bearing, 15x35x11, 6202.2RS", "label": 1},
    {"image": "bearing_6202.jpg", "text": "Bearing, 30x62x16, 6206ZZ", "label": 0},  # hard negative
]
```

Implementation: `scripts/fine_tune_clip.py`
- ใช้ `transformers` Trainer + LoRA (lightweight fine-tuning)
- เทรนบน GPU ใช้เวลาประมาณ 1-2 ชั่วโมงสำหรับ 1K SKUs
- ได้ model ที่เข้าใจ domain อุตสาหกรรม + format Wise Code

#### C2. Category-Aware Embedding

เพิ่ม product category เป็น signal:
1. แยก category จาก Wise Code prefix (e.g., `020-` = Bearing, `030-` = Seal)
2. สร้าง category classifier เป็น first-stage filter
3. ค้นหา FAISS เฉพาะในหมวดหมู่ที่เกี่ยวข้อง → ลด false positives

#### C3. Evaluation Pipeline

สร้างระบบวัดผลอัตโนมัติเพื่อเปรียบเทียบแต่ละวิธี:

```python
# scripts/evaluate_search.py
# สุ่ม N สินค้า → ใช้รูปของมัน search กลับ → วัดว่า top-K มีตัวเองไหม
def evaluate(search_fn, test_data, k=5):
    hits = 0
    for product in test_data:
        results = search_fn(product["image"])
        if product["wise_code"] in [r["wise_code"] for r in results[:k]]:
            hits += 1
    return hits / len(test_data)  # Recall@K
```

Metrics:
- **Recall@1**: สินค้าถูกต้องอยู่อันดับ 1 กี่ %
- **Recall@5**: สินค้าถูกต้องอยู่ใน top-5 กี่ %
- **MRR** (Mean Reciprocal Rank): อันดับเฉลี่ยของคำตอบถูก

---

### Recommended Implementation Order

```
Phase 1 (ทำเลย — ไม่กี่ชั่วโมง):
  A2 → ใช้ without.bg เป็น primary
  A3 → Multi-angle average embedding
  A4 → Text augmentation

Phase 2 (1-2 วัน):
  A1 → Hybrid scoring (image + text dual index)
  A5 → Query expansion
  C3 → Evaluation pipeline (วัดผลก่อน-หลัง)

Phase 3 (3-5 วัน):
  B1 → Upgrade CLIP to ViT-L/14
  C1 → Fine-tune CLIP on Wise product data
  C2 → Category-aware filtering
```

---

## 15. Development Notes

- **CLIP model จะถูกโหลดตอน server startup** — ครั้งแรกจะ download model ~600MB
- **FAISS index ต้อง build ก่อน** ด้วย `ingest_data.py` ไม่งั้น search จะ return ผลว่าง
- **Frontend เป็น static HTML** serve โดย FastAPI เลย ไม่ต้อง build แยก
- **รูปสินค้า serve ผ่าน API endpoint** `/api/images/{type}/{filename}` — ไม่ได้ serve static files ตรงๆ เพื่อ control access ได้ในอนาคต
- **ทุก embedding ถูก L2 normalize** ดังนั้น Inner Product = Cosine Similarity พอดี
