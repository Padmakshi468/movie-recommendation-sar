"""
scripts/train.py
================
Full training pipeline:
  1. Load (or download) MovieLens 100K
  2. Positive interaction filtering (rating >= 4)
  3. Chronological train/test split
  4. Train SAR
  5. Evaluate Precision@K, Recall@K, NDCG@K, MAP@K, Coverage
  6. Save model + metadata
  7. Hyperparameter grid-search (optional flag)

Usage:
  python scripts/train.py
  python scripts/train.py --grid_search
"""
from __future__ import annotations
import sys, argparse, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import ensure_data, load_ratings, load_item_metadata, chronological_split
from model.sar import SAR
from model.evaluate import evaluate_all
from model.persistence import save

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

import pandas as pd


def run_training(
    similarity_type: str = "jaccard",
    time_decay_half_life: float = 30.0,
    threshold: int = 1,
    min_rating: float = 4.0,
    top_k: int = 10,
) -> dict:
    logger.info("=== STEP 1: DATA ===")
    data_dir   = ensure_data()
    ratings    = load_ratings(data_dir, min_rating=min_rating)
    item_meta  = load_item_metadata(data_dir)
    train, test = chronological_split(ratings, test_ratio=0.2, min_interactions=5)

    logger.info("=== STEP 2: TRAIN (%s, decay=%.0fd, thresh=%d) ===",
                similarity_type, time_decay_half_life, threshold)
    model = SAR(
        similarity_type=similarity_type,
        time_decay_half_life=time_decay_half_life,
        threshold=threshold,
    ).fit(train)

    logger.info("=== STEP 3: EVALUATE ===")
    test_users = pd.DataFrame({"user_id": test["user_id"].unique()})
    recs_list  = [model.recommend_k_items(uid, top_k=top_k)
                  for uid in test_users["user_id"]]
    recs = pd.concat([r for r in recs_list if len(r)], ignore_index=True)

    metrics = evaluate_all(recs, test, total_items=len(model.item2idx), k=top_k)
    metrics.update({"similarity_type": similarity_type,
                    "time_decay_half_life": time_decay_half_life,
                    "threshold": threshold,
                    "min_rating": min_rating,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_users": len(model.user2idx),
                    "n_items": len(model.item2idx)})

    logger.info("=== STEP 4: SAVE ===")
    save(model, item_meta, metrics)
    return metrics


def grid_search():
    """Mini grid search over key hyperparameters."""
    from itertools import product
    best, best_score = None, -1.0
    grid = {
        "similarity_type":      ["jaccard", "lift"],
        "time_decay_half_life": [30.0, 60.0],
        "threshold":            [1, 2],
    }
    for sim, hl, thr in product(*grid.values()):
        logger.info(">> Testing sim=%s hl=%.0f thr=%d", sim, hl, thr)
        m = run_training(similarity_type=sim, time_decay_half_life=hl, threshold=thr)
        ndcg = m.get("ndcg@10", 0.0)
        logger.info("   NDCG@10 = %.4f", ndcg)
        if ndcg > best_score:
            best_score = ndcg
            best = m
    logger.info("Best config: %s  NDCG@10=%.4f", best, best_score)
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument("--similarity", default="jaccard")
    parser.add_argument("--half_life",  type=float, default=30.0)
    parser.add_argument("--threshold",  type=int,   default=1)
    parser.add_argument("--min_rating", type=float, default=4.0)
    args = parser.parse_args()

    if args.grid_search:
        metrics = grid_search()
    else:
        metrics = run_training(
            similarity_type=args.similarity,
            time_decay_half_life=args.half_life,
            threshold=args.threshold,
            min_rating=args.min_rating,
        )

    print("\n✅  Training complete. Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")
