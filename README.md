# 🎬 CineMatch — SAR Movie Recommendation System

A production-grade, end-to-end movie recommendation system built with
**Microsoft SAR (Smart Adaptive Recommendations)** on **MovieLens 100K**.

---

## Architecture

```
sar-recommender/
├── data/
│   ├── loader.py          ← Download, parse, positive-filter, split
│   └── raw/ml-100k/       ← Auto-downloaded dataset
│
├── model/
│   ├── sar.py             ← Full SAR: co-occurrence, Jaccard/Lift/Counts
│   │                        similarity, time-decay affinity, cold-start
│   ├── evaluate.py        ← Precision@K, Recall@K, NDCG@K, MAP@K, Coverage
│   ├── persistence.py     ← joblib save/load with compression
│   └── artifacts/         ← Generated: sar_model.joblib, metadata, metrics
│
├── api/
│   └── main.py            ← FastAPI: /recommend, /similar, /movies,
│                              /recommend/items, /users, /health
├── ui/
│   └── app.py             ← Streamlit: 3 modes, search-as-you-type,
│                              CSV export, health monitoring
├── scripts/
│   ├── train.py           ← Full pipeline + optional grid search
│   └── infer.py           ← CLI inference
│
├── requirements.txt
├── run.sh
└── README.md
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train (downloads MovieLens 100K automatically)
python scripts/train.py

# Start API (port 8000)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start UI (port 8501)
streamlit run ui/app.py

# OR — one command
bash run.sh
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System status + eval metrics |
| GET | `/recommend/{user_id}?k=` | Personalised top-K |
| GET | `/similar/{movie_id}?k=` | Similar movies |
| GET | `/movies?q=&limit=` | Title search / autocomplete |
| POST | `/recommend/items` | Cold-start from item list |
| GET | `/users?limit=` | List known users |

### Example calls

```bash
curl "http://localhost:8000/recommend/42?k=10"
curl "http://localhost:8000/similar/50?k=8"
curl "http://localhost:8000/movies?q=star&limit=10"
curl -X POST "http://localhost:8000/recommend/items" \
     -H "Content-Type: application/json" \
     -d '{"item_ids":[50,181,268],"k":10}'
```

---

## SAR Algorithm

1. **Positive interaction filtering** — keep ratings ≥ 4 (configurable)
2. **Chronological split** — per-user, most-recent interactions → test
3. **Item co-occurrence matrix** — C[i,j] = # users who rated both i and j
4. **Item similarity** (Jaccard by default):
   ```
   S[i,j] = C[i,j] / (freq(i) + freq(j) − C[i,j])
   ```
5. **Time-decay user affinity**:
   ```
   w(t) = rating × exp(−ln2 × Δt / half_life)
   ```
6. **Scoring**: `score[u,j] = affinity[u] · S[:,j]`
7. **Cold-start**: aggregate similarity scores over seed items

---

## Hyperparameter Tuning

```bash
python scripts/train.py --grid_search
# or manually:
python scripts/train.py --similarity lift --half_life 60 --threshold 2
```

Grid-search tests `{jaccard, lift} × {30d, 60d} × {1, 2}` and saves the best config.

---

## Three UI Modes

| Mode | Input | Output |
|------|-------|--------|
| User Recommendations | User ID dropdown | Personalised top-K with scores |
| Similar Movies | Movie title search | Most similar movies |
| Recommend From Movies | Multi-select movie list | Cold-start recommendations |

All movie selection uses **search-as-you-type** — no manual ID entry required.

---

## CLI Inference

```bash
python scripts/infer.py --user_id 42 --k 10
python scripts/infer.py --mode similar --movie_id 50 --k 8
python scripts/infer.py --mode cold_start --movie_ids 50,181,268 --k 10
```

---

## Tech Stack

`Python 3.9+` · `FastAPI` · `Streamlit` · `Pandas` · `NumPy` · `SciPy` · `Joblib` · `scikit-learn`
