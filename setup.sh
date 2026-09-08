#!/bin/bash
# One-command setup for Reverse Turing Test.
#
# The app serves prebuilt artifacts, so this does NOT download any dataset.
# To retrain from scratch, see the "Retraining" section of the README.
set -e

echo "=== Reverse Turing Test — setup ==="

# --- Model weights (tracked via git-lfs) ---
if ! command -v git-lfs &> /dev/null; then
    echo "git-lfs is required to fetch model weights."
    if command -v brew &> /dev/null; then
        echo "  installing via brew..."
        brew install git-lfs
    else
        echo "  install it from https://git-lfs.com/ and re-run." >&2
        exit 1
    fi
fi
git lfs install

if [ ! -f models/bert_classifier/model.safetensors ]; then
    echo "Pulling model weights (~280 MB)..."
    git lfs pull
fi

# --- Dependencies ---
echo "Installing runtime dependencies..."
pip install -r requirements.txt

# --- Verify the app has everything it needs ---
MISSING=0
for f in artifacts/metadata.json artifacts/explorer.json artifacts/pca.pkl \
         models/meta_classifier.pkl models/stylometric_classifier.pkl \
         models/cnn_classifier.pt models/bert_classifier/model.safetensors; do
    [ -f "$f" ] || { echo "  MISSING: $f" >&2; MISSING=1; }
done

if [ "$MISSING" -ne 0 ]; then
    echo ""
    echo "Some artifacts are missing. Rebuild them with:" >&2
    echo "  pip install -r requirements-dev.txt" >&2
    echo "  python src/build_artifacts.py --dataset raid" >&2
    exit 1
fi

echo ""
echo "=== Setup complete ==="
echo "Run:  python app/app.py     ->  http://localhost:5001"
