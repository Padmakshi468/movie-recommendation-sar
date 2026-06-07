#!/usr/bin/env bash
# run.sh — One-command startup for the CineMatch SAR System
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "════════════════════════════════════════════"
echo "  CineMatch · SAR Movie Recommender"
echo "════════════════════════════════════════════"

echo ""
echo "[1/3] Installing dependencies..."
pip install pandas numpy scipy scikit-learn joblib fastapi uvicorn \
            pydantic requests streamlit --quiet --break-system-packages 2>/dev/null || \
pip install pandas numpy scipy scikit-learn joblib fastapi uvicorn \
            pydantic requests streamlit --quiet

if [ ! -f "model/artifacts/sar_model.joblib" ]; then
  echo ""
  echo "[2/3] Training SAR model (first run, ~30s)..."
  python scripts/train.py
else
  echo ""
  echo "[2/3] Trained model found — skipping training."
fi

echo ""
echo "[3/3] Starting services..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
sleep 2
streamlit run ui/app.py --server.port 8501 --server.headless true &

echo ""
echo "════════════════════════════════════════════"
echo "  🎬 CineMatch is running!"
echo "  API:   http://localhost:8000"
echo "  Docs:  http://localhost:8000/docs"
echo "  UI:    http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo "════════════════════════════════════════════"
wait
