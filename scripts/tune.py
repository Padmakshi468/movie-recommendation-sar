"""
scripts/tune.py
Grid search over SAR hyperparameters, prints a comparison table.

Usage:
    python scripts/tune.py
"""

import sys
import logging
from itertools import product
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import get_dataset, filter_positive, chronological_split
from model.sar import SAR
from model.evaluate import evaluate_all
from config.settings import COL_USER

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def tune():
    print("Loading data …")
    ratings, items = get_dataset()
    pos_ratings = filter_positive(ratings)
    train, test = chronological_split(pos_ratings)
    all_items = set(ratings["item_id"].unique())

    import pandas as pd
    test_users = pd.DataFrame({"user_id": test["user_id"].unique()})

    param_grid = {
        "similarity_type":       ["jaccard", "lift"],
        "time_decay_coefficient": [0.0, 30.0, 60.0],
        "threshold":              [1, 3],
    }

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    results = []

    print(f"\nRunning {len(combos)} hyperparameter combinations …\n")
    for combo in combos:
        params = dict(zip(keys, combo))
        model = SAR(**params)
        model.fit(train)
        recs = model.recommend_k_items(list(test_users["user_id"]), top_k=10)
        m = evaluate_all(recs, test, all_items, k=10)
        row = {**params, **m}
        results.append(row)
        print(f"  {params}  →  P@10={m['precision@10']:.4f}  NDCG@10={m['ndcg@10']:.4f}")

    df = pd.DataFrame(results).sort_values("ndcg@10", ascending=False)
    print("\n── Top configurations by NDCG@10 ──")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    tune()
