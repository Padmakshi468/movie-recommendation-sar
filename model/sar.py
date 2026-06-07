"""
model/sar.py
============
Microsoft SAR (Smart Adaptive Recommendations) — pure Python/NumPy/SciPy
implementation supporting:
  • Jaccard / Lift / Counts item-item similarity
  • Exponential time-decay user affinity
  • Top-K recommendations with seen-item removal
  • Similar items retrieval
  • Cold-start recommendations from a seed item list
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

COL_USER  = "user_id"
COL_ITEM  = "item_id"
COL_RATING = "rating"
COL_TIMESTAMP = "timestamp"
COL_SCORE = "score"

SIMILARITY_TYPES = ("jaccard", "lift", "counts")


class SAR:
    """
    Smart Adaptive Recommendations collaborative filter.

    Parameters
    ----------
    similarity_type      : 'jaccard' | 'lift' | 'counts'
    time_decay_half_life : Half-life in days (0 = no decay)
    threshold            : Minimum co-occurrence count to keep an edge
    normalize            : Normalise similarity rows to [0,1]
    """

    def __init__(
        self,
        similarity_type: str = "jaccard",
        time_decay_half_life: float = 30.0,
        threshold: int = 1,
        normalize: bool = True,
    ):
        if similarity_type not in SIMILARITY_TYPES:
            raise ValueError(f"similarity_type must be one of {SIMILARITY_TYPES}")
        self.similarity_type      = similarity_type
        self.time_decay_half_life = time_decay_half_life
        self.threshold            = threshold
        self.normalize            = normalize

        # populated after fit()
        self.item_similarity: Optional[np.ndarray] = None
        self.user_affinity:   Optional[sp.csr_matrix] = None
        self.item2idx:        Dict[int, int] = {}
        self.idx2item:        Dict[int, int] = {}
        self.user2idx:        Dict[int, int] = {}
        self.idx2user:        Dict[int, int] = {}
        self.seen_items:      Dict[int, set] = {}
        self.item_popularity: Dict[int, int] = {}
        self._time_now:       Optional[int] = None

    # ─── fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "SAR":
        """Train SAR on a ratings DataFrame (user_id, item_id, rating, timestamp)."""
        logger.info("Building index maps ...")
        users  = sorted(df[COL_USER].unique())
        items  = sorted(df[COL_ITEM].unique())
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.idx2user = dict(enumerate(users))
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2item = dict(enumerate(items))
        n_users, n_items = len(users), len(items)
        logger.info("  %d users, %d items", n_users, n_items)

        # seen items per user
        self.seen_items = df.groupby(COL_USER)[COL_ITEM].apply(set).to_dict()

        # item popularity (interaction count)
        self.item_popularity = df[COL_ITEM].value_counts().to_dict()

        # ── time-decay weights ─────────────────────────────────────────────
        self._time_now = int(df[COL_TIMESTAMP].max())
        if self.time_decay_half_life > 0:
            hl_sec = self.time_decay_half_life * 86_400.0
            decay  = np.exp(-np.log(2) * (self._time_now - df[COL_TIMESTAMP].values) / hl_sec)
            weights = df[COL_RATING].values.astype(np.float32) * decay.astype(np.float32)
        else:
            weights = df[COL_RATING].values.astype(np.float32)

        u_idx = df[COL_USER].map(self.user2idx).values
        i_idx = df[COL_ITEM].map(self.item2idx).values

        # ── user-affinity matrix ───────────────────────────────────────────
        logger.info("Building user-affinity matrix ...")
        self.user_affinity = sp.csr_matrix(
            (weights, (u_idx, i_idx)), shape=(n_users, n_items), dtype=np.float32
        )

        # ── item co-occurrence ─────────────────────────────────────────────
        logger.info("Computing item co-occurrence ...")
        presence = sp.csr_matrix(
            (np.ones(len(df), dtype=np.float32), (u_idx, i_idx)),
            shape=(n_users, n_items), dtype=np.float32,
        )
        co_occ = (presence.T @ presence).toarray()  # (n_items, n_items)
        np.fill_diagonal(co_occ, 0)

        # ── item similarity ────────────────────────────────────────────────
        logger.info("Computing %s similarity ...", self.similarity_type)
        item_freq = np.array(presence.sum(axis=0)).flatten()

        if self.similarity_type == "jaccard":
            denom = item_freq[:, None] + item_freq[None, :] - co_occ
            with np.errstate(divide="ignore", invalid="ignore"):
                sim = np.where(denom > 0, co_occ / denom, 0.0)

        elif self.similarity_type == "lift":
            n = float(n_users)
            with np.errstate(divide="ignore", invalid="ignore"):
                outer = item_freq[:, None] * item_freq[None, :]
                sim   = np.where(outer > 0, co_occ * n / outer, 0.0)
        else:
            sim = co_occ.astype(np.float64)

        # apply threshold
        sim[co_occ < self.threshold] = 0.0
        np.fill_diagonal(sim, 0)

        if self.normalize:
            row_max = sim.max(axis=1, keepdims=True)
            row_max[row_max == 0] = 1.0
            sim = sim / row_max

        self.item_similarity = sim.astype(np.float32)
        logger.info("SAR training complete.")
        return self

    # ─── recommend for a user ─────────────────────────────────────────────────

    def recommend_k_items(
        self,
        user_id: int,
        top_k: int = 10,
        remove_seen: bool = True,
    ) -> pd.DataFrame:
        """Return top-K recommendations for a single user."""
        self._assert_fitted()
        if user_id not in self.user2idx:
            return pd.DataFrame(columns=[COL_USER, COL_ITEM, COL_SCORE])

        u_idx     = self.user2idx[user_id]
        affinity  = np.array(self.user_affinity[u_idx].todense()).flatten()
        scores    = affinity @ self.item_similarity  # (n_items,)

        if remove_seen and user_id in self.seen_items:
            for it in self.seen_items[user_id]:
                if it in self.item2idx:
                    scores[self.item2idx[it]] = -np.inf

        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        top_idx = [i for i in top_idx if scores[i] != -np.inf]

        return pd.DataFrame({
            COL_USER:  user_id,
            COL_ITEM:  [self.idx2item[i] for i in top_idx],
            COL_SCORE: [float(scores[i]) for i in top_idx],
        })

    # ─── similar items ────────────────────────────────────────────────────────

    def similar_items(self, item_id: int, top_k: int = 10) -> pd.DataFrame:
        """Return the top-K most similar items to a given item."""
        self._assert_fitted()
        if item_id not in self.item2idx:
            return pd.DataFrame(columns=["item_id", "similar_item_id", COL_SCORE])

        idx  = self.item2idx[item_id]
        row  = self.item_similarity[idx].copy()
        row[idx] = -1  # exclude self

        k    = min(top_k, len(row))
        top  = np.argpartition(row, -k)[-k:]
        top  = top[np.argsort(row[top])[::-1]]

        return pd.DataFrame({
            "item_id":        item_id,
            "similar_item_id": [self.idx2item[i] for i in top],
            COL_SCORE:         [float(row[i])    for i in top],
        })

    # ─── cold-start / recommend from item list ────────────────────────────────

    def recommend_from_items(
        self,
        seed_item_ids: List[int],
        top_k: int = 10,
        exclude_seed: bool = True,
    ) -> pd.DataFrame:
        """
        Cold-start: aggregate similarity scores over a list of seed items
        and return top-K recommendations.
        """
        self._assert_fitted()
        valid = [it for it in seed_item_ids if it in self.item2idx]
        if not valid:
            return pd.DataFrame(columns=[COL_ITEM, COL_SCORE])

        scores = np.zeros(len(self.item2idx), dtype=np.float32)
        for it in valid:
            scores += self.item_similarity[self.item2idx[it]]

        if exclude_seed:
            for it in valid:
                scores[self.item2idx[it]] = -np.inf

        k    = min(top_k, (scores > -np.inf).sum())
        if k == 0:
            return pd.DataFrame(columns=[COL_ITEM, COL_SCORE])
        top  = np.argpartition(scores, -k)[-k:]
        top  = top[np.argsort(scores[top])[::-1]]
        top  = [i for i in top if scores[i] > -np.inf]

        return pd.DataFrame({
            COL_ITEM:  [self.idx2item[i] for i in top],
            COL_SCORE: [float(scores[i]) for i in top],
        })

    # ─── popularity fallback ─────────────────────────────────────────────────

    def popular_items(self, top_k: int = 10, exclude: Optional[set] = None) -> List[int]:
        """Return the most popular item IDs (fallback for new/unknown users)."""
        ranked = sorted(self.item_popularity.items(), key=lambda x: -x[1])
        result = []
        for item_id, _ in ranked:
            if exclude and item_id in exclude:
                continue
            result.append(item_id)
            if len(result) >= top_k:
                break
        return result

    # ─── utils ───────────────────────────────────────────────────────────────

    def _assert_fitted(self):
        if self.item_similarity is None:
            raise RuntimeError("Model not trained. Call fit() first.")
