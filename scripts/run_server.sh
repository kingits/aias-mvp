#!/bin/bash
# AI.AS MVP - Start Server
# Usage: bash scripts/run_server.sh

set -e

echo "============================================"
echo "  AI.AS - Smart Product Search Engine"
echo "  MVP Server"
echo "============================================"

# Check if data is indexed
if [ ! -f "data/index/product_vectors.index" ]; then
    echo ""
    echo "[WARNING] No product index found!"
    echo "Run the data ingestion script first:"
    echo ""
    echo "  python scripts/ingest_data.py \\"
    echo "    --images-dir '../ตัวอย่างภาพสินค้า1K' \\"
    echo "    --excel '../ตัวอย่างภาพสินค้า1K/SKUของตัวอย่างภาพสินค้า1K.xlsx'"
    echo ""
fi

echo "Starting server at http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo ""

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
