"""
model/evaluate.py
=================
Offline evaluation metrics:
  Precision@K, Recall@K, NDCG@K, MAP@K, Coverage
"""
from __future__ import annotations
import logging
from typing import Dict, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COL_USER  = "user_id"
COL_ITEM  = "item_id"
COL_SCORE = "score"


def _test_sets(test_df: pd.DataFrame) -> Dict[int, set]:
    return test_df.groupby(COL_USER)[COL_ITEM].apply(set).to_dict()


def precision_at_k(recs: pd.DataFrame, test_df: pd.DataFrame, k: int) -> float:
    test_sets = _test_sets(test_df)
    scores = []
    for uid, group in recs.groupby(COL_USER):
        if uid not in test_sets:
            continue
        top = group.nlargest(k, COL_SCORE)[COL_ITEM].tolist()
        hits = sum(1 for it in top if it in test_sets[uid])
        scores.append(hits / k)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(recs: pd.DataFrame, test_df: pd.DataFrame, k: int) -> float:
    test_sets = _test_sets(test_df)
    scores = []
    for uid, group in recs.groupby(COL_USER):
        if uid not in test_sets:
            continue
        top  = group.nlargest(k, COL_SCORE)[COL_ITEM].tolist()
        n_rel = len(test_sets[uid])
        hits  = sum(1 for it in top if it in test_sets[uid])
        scores.append(hits / n_rel if n_rel > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(recs: pd.DataFrame, test_df: pd.DataFrame, k: int) -> float:
    test_sets = _test_sets(test_df)
    scores = []
    for uid, group in recs.groupby(COL_USER):
        if uid not in test_sets:
            continue
        top    = group.nlargest(k, COL_SCORE)[COL_ITEM].tolist()
        rel    = test_sets[uid]
        dcg    = sum(int(it in rel) / np.log2(i + 2) for i, it in enumerate(top))
        ideal  = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
        scores.append(dcg / ideal if ideal > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def map_at_k(recs: pd.DataFrame, test_df: pd.DataFrame, k: int) -> float:
    test_sets = _test_sets(test_df)
    scores = []
    for uid, group in recs.groupby(COL_USER):
        if uid not in test_sets:
            continue
        top = group.nlargest(k, COL_SCORE)[COL_ITEM].tolist()
        rel = test_sets[uid]
        hits, running_p = 0, 0.0
        for i, it in enumerate(top, 1):
            if it in rel:
                hits += 1
                running_p += hits / i
        n_rel = len(rel)
        scores.append(running_p / min(n_rel, k) if n_rel > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def catalog_coverage(recs: pd.DataFrame, total_items: int) -> float:
    """Fraction of the full catalog that appears in any recommendation."""
    recommended = recs[COL_ITEM].nunique()
    return recommended / total_items if total_items > 0 else 0.0


def evaluate_all(
    recs: pd.DataFrame,
    test_df: pd.DataFrame,
    total_items: int,
    k: int = 10,
) -> Dict[str, float]:
    metrics = {
        f"precision@{k}": precision_at_k(recs, test_df, k),
        f"recall@{k}":    recall_at_k(recs, test_df, k),
        f"ndcg@{k}":      ndcg_at_k(recs, test_df, k),
        f"map@{k}":       map_at_k(recs, test_df, k),
        "coverage":       catalog_coverage(recs, total_items),
    }
    for name, val in metrics.items():
        logger.info("  %s: %.4f", name, val)
    return metrics
