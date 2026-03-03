#!/bin/bash
# AI.AS MVP - Setup Script
# Run this once to set up the project environment

set -e

echo "============================================"
echo "  AI.AS MVP - Environment Setup"
echo "============================================"

# 1. Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create data directories
echo "[3/4] Creating data directories..."
mkdir -p data/index
mkdir -p data/images/with.background
mkdir -p data/images/without.background

# 4. Symlink sample images (if available)
SAMPLE_DIR="../ตัวอย่างภาพสินค้า1K"
if [ -d "$SAMPLE_DIR/with.background" ]; then
    echo "[4/4] Linking sample product images..."
    # Create symlinks to avoid copying large image files
    ln -sf "$(cd "$SAMPLE_DIR/with.background" && pwd)" data/images/with.background.link 2>/dev/null || true
    # Or copy if symlinks don't work
    if [ ! -L "data/images/with.background.link" ]; then
        echo "  Copying images instead of symlinking..."
        cp -r "$SAMPLE_DIR/with.background/"* data/images/with.background/ 2>/dev/null || true
        cp -r "$SAMPLE_DIR/without.background/"* data/images/without.background/ 2>/dev/null || true
    fi
else
    echo "[4/4] No sample images found. Place product images in:"
    echo "  data/images/with.background/"
    echo "  data/images/without.background/"
fi

echo ""
echo "Setup complete! Next steps:"
echo ""
echo "1. Activate environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run data ingestion:"
echo "   python scripts/ingest_data.py \\"
echo "     --images-dir '../ตัวอย่างภาพสินค้า1K' \\"
echo "     --excel '../ตัวอย่างภาพสินค้า1K/SKUของตัวอย่างภาพสินค้า1K.xlsx'"
echo ""
echo "3. Start the server:"
echo "   bash scripts/run_server.sh"
echo ""
echo "4. Open browser:"
echo "   http://localhost:8000"
echo ""
