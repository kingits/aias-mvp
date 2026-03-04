# AI.AS MVP — VM Deployment Manual

This guide covers deploying the AI.AS Smart Product Search Engine on a Virtual Machine (VM).

---

## 1. VM Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **OS** | Ubuntu 20.04 LTS or later | Ubuntu 22.04 LTS |
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 50 GB SSD | 100 GB SSD |
| **GPU** | Optional (CPU inference works) | NVIDIA GPU with CUDA for faster training |

**Note**: The app works on CPU. GPU is only needed for faster fine-tuning.

---

## 2. OS Setup

### 2.1 Update system

```bash
sudo apt update && sudo apt upgrade -y
```

### 2.2 Install system dependencies

```bash
sudo apt install -y python3.11 python3-pip python3.11-venv git wget curl
```

### 2.3 Create project directory

```bash
mkdir -p ~/aias && cd ~/aias
```

---

## 3. Project Setup

### 3.1 Clone or copy the project

If using Git:

```bash
git clone <your-repo-url> aias-mvp
cd aias-mvp
```

If copying files manually, copy the entire project folder to `~/aias/aias-mvp`.

### 3.2 Create virtual environment

```bash
cd ~/aias/aias-mvp
python3 -m venv venv
```

### 3.3 Activate environment

```bash
source venv/bin/activate
```

**Important**: Always activate the virtual environment before running any Python commands.

### 3.4 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.5 Install additional GPU support (optional)

If you have an NVIDIA GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 4. Data Preparation

### 4.1 Prepare your data

You need two things:

1. **Product images** — organized in two folders:
   ```
   data/
   └── images/
       ├── with.background/        # Product photos with background
       │   └── 000030000523.jpg
       └── without.background/     # Product photos with transparent/white background
           └── 000-0300005-23.jpg
   ```

2. **Excel file** — with columns: `no`, `wise_code`, `description`, `unit`
   ```
   data/
   └── SKUของตัวอย่างภาพสินค้า1K.xlsx
   ```

### 4.2 Set environment variables (optional)

Create a `.env` file in the project root:

```bash
# Example .env
AIAS_IMAGES_DIR=/path/to/your/images
AIAS_HOST=0.0.0.0
AIAS_PORT=8000
AIAS_DEBUG=false
```

---

## 5. Data Ingestion

Before the search API works, you must build the FAISS index from your product data.

### 5.1 Run ingestion

```bash
source venv/bin/activate

python scripts/ingest_data.py \
  --images-dir "/path/to/your/images" \
  --excel "/path/to/your/SKU.xlsx"
```

**Example** (if data is in a folder named `data` inside the project):

```bash
python scripts/ingest_data.py \
  --images-dir "data/images" \
  --excel "data/SKUของตัวอย่างภาพสินค้า1K.xlsx"
```

### 5.2 Ingestion output

```
Loading CLIP model: openai/clip-vit-large-patch14 on cpu
Loaded 440 SKUs
Processing products: 100%|████████████| 440/440
Image embeddings: (440, 768)
Text embeddings:  (440, 768)
Ingestion complete in 87.5s
  Products with images: 440 (0 multi-angle)
  Products text-only:   0
  Dual indexes saved to: data/index
```

---

## 6. Running the Server

### 6.1 Start the server

```bash
source venv/bin/activate

# Option 1: Use the run script
bash scripts/run_server.sh

# Option 2: Direct uvicorn
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 6.2 First startup

The first time you run the server, it will download the CLIP model (~2GB):

```
Loading CLIP model: openai/clip-vit-large-patch14 on cpu
Downloading model files: 100%|████████████| 2.0G/2.0G
CLIP model loaded successfully
```

### 6.3 Verify it's running

Open a browser and go to: `http://<VM-IP>:8000`

Or test the API:

```bash
curl http://localhost:8000/api/health
```

Expected response:

```json
{
  "status": "healthy",
  "model": "CLIP ViT-L/14",
  "indexed_products": 440
}
```

---

## 7. Using the Web App

### 7.1 Visual Search

1. Click the **Visual** tab
2. Drag & drop a product image (or click to browse)
3. Results appear with similarity scores

### 7.2 Semantic Search

1. Click the **Semantic** tab
2. Type a search query (e.g., "bearing", "bolt M10")
3. Results appear ranked by relevance

### 7.3 Chat Search

1. Click the **Chat** tab
2. Type a message or attach an image
3. Get AI response with product suggestions

---

## 8. API Reference

| Method | Endpoint | Description | Example |
|--------|----------|-------------|---------|
| GET | `/api/health` | Health check | `curl http://localhost:8000/api/health` |
| GET | `/api/search/semantic?q=bearing` | Text search | `curl "http://localhost:8000/api/search/semantic?q=bearing"` |
| POST | `/api/search/visual` | Image search | `curl -X POST -F "image=@photo.jpg" http://localhost:8000/api/search/visual` |
| GET | `/api/products/{wise_code}` | Get product | `curl http://localhost:8000/api/products/000-0300005-23` |
| GET | `/api/products` | List products | `curl "http://localhost:8000/api/products?limit=10"` |

---

## 9. (Optional) Fine-Tuning

To improve accuracy for your specific product domain:

```bash
source venv/bin/activate

python scripts/fine_tune_clip.py \
  --images-dir "data/images" \
  --excel "data/SKUของตัวอย่างภาพสินค้า1K.xlsx" \
  --epochs 12 \
  --batch-size 16 \
  --lr 5e-6 \
  --output-dir model/fine_tuned_clip_v2
```

After training:

```bash
# Re-index with the new model
export AIAS_CLIP_MODEL_NAME="model/fine_tuned_clip_v2/merged"
python scripts/ingest_data.py \
  --images-dir "data/images" \
  --excel "data/SKUของตัวอย่างภาพสินค้า1K.xlsx"
```

---

## 10. (Optional) Evaluation

To measure search accuracy:

```bash
source venv/bin/activate

python scripts/evaluate_search.py \
  --images-dir "data/images" \
  --top-k 5 \
  --sample-size 0
```

Results are saved to `data/index/eval_results.json`.

---

## 11. Troubleshooting

### Issue: "No products indexed"

**Cause**: Ingestion has not been run.

**Fix**: Run ingestion first (see Section 5).

---

### Issue: "Model download very slow"

**Cause**: Hugging Face connection from your region is slow.

**Fix**: Use a mirror or pre-download the model:

```bash
pip install huggingface_hub
python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-large-patch14'); CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')"
```

---

### Issue: "CUDA out of memory"

**Cause**: GPU memory insufficient.

**Fix**: Use CPU instead:

```bash
# No special command needed — the app automatically uses CPU if CUDA is unavailable
```

---

### Issue: "Port 8000 already in use"

**Cause**: Another process is using the port.

**Fix**: Kill the existing process or use a different port:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

---

### Issue: "No images found"

**Cause**: Image folder structure is incorrect.

**Fix**: Ensure images are in:
```
<images-dir>/
├── with.background/
│   └── 000030000523.jpg
└── without.background/
    └── 000-0300005-23.jpg
```

Filename mapping:
- `with.background/`: Use **no dashes** (e.g., `000030000523.jpg`)
- `without.background/`: Use **dashes** (e.g., `000-0300005-23.JPG`)

---

## 12. Production Deployment

### 12.1 Using systemd (recommended for production)

Create a service file:

```bash
sudo nano /etc/systemd/system/aias.service
```

Contents:

```ini
[Unit]
Description=AI.AS Product Search Engine
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/aias/aias-mvp
Environment="PATH=/home/ubuntu/aias/aias-mvp/venv/bin"
Environment="AIAS_HOST=0.0.0.0"
Environment="AIAS_PORT=8000"
Environment="AIAS_DEBUG=false"
ExecStart=/home/ubuntu/aias/aias-mvp/venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable aias
sudo systemctl start aias
sudo systemctl status aias
```

### 12.2 Using nginx as reverse proxy

```bash
sudo apt install nginx

sudo nano /etc/nginx/sites-available/aias
```

Contents:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/aias /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 13. Quick Reference

```bash
# Activate environment
source venv/bin/activate

# Run ingestion (first time or when data changes)
python scripts/ingest_data.py --images-dir "data/images" --excel "data/SKU.xlsx"

# Start server
bash scripts/run_server.sh

# Test health
curl http://localhost:8000/api/health

# Evaluate accuracy
python scripts/evaluate_search.py --images-dir "data/images" --sample-size 0
```

---

## 14. Support

- Check `data/index/eval_results.json` for accuracy metrics
- Check server logs for errors
- Ensure image filenames match the expected format (see Section 11)
