"""
data/loader.py
MovieLens 100K data ingestion, preprocessing, and chronological train/test split.
"""
from __future__ import annotations
import logging, urllib.request, zipfile
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
COL_USER      = "user_id"
COL_ITEM      = "item_id"
COL_RATING    = "rating"
COL_TIMESTAMP = "timestamp"
COL_TITLE     = "title"
COL_GENRES    = "genres"

GENRE_COLUMNS = [
    "Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western","unknown",
]

_DEFAULT_DATA_DIR = Path(__file__).parent / "raw" / "ml-100k"


def ensure_data(data_dir: Path = _DEFAULT_DATA_DIR) -> Path:
    """Download and extract MovieLens 100K if not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "u.data").exists():
        logger.info("MovieLens 100K already present at %s", data_dir)
        return data_dir
    zip_path = data_dir.parent / "ml-100k.zip"
    logger.info("Downloading MovieLens 100K ...")
    try:
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
    except Exception as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir.parent)
    logger.info("Extracted to %s", data_dir)
    return data_dir


def load_ratings(data_dir: Path = _DEFAULT_DATA_DIR, min_rating: float = 1.0) -> pd.DataFrame:
    """Load u.data ratings."""
    path = data_dir / "u.data"
    df = pd.read_csv(path, sep="\t",
                     names=[COL_USER, COL_ITEM, COL_RATING, COL_TIMESTAMP],
                     dtype={COL_USER: np.int32, COL_ITEM: np.int32,
                            COL_RATING: np.float32, COL_TIMESTAMP: np.int64})
    if min_rating > 1.0:
        before = len(df)
        df = df[df[COL_RATING] >= min_rating].reset_index(drop=True)
        logger.info("Kept %d/%d ratings (>= %.1f stars)", len(df), before, min_rating)
    logger.info("Ratings: %d | Users: %d | Items: %d",
                len(df), df[COL_USER].nunique(), df[COL_ITEM].nunique())
    return df


def load_item_metadata(data_dir: Path = _DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load u.item and return item_id, title, genres."""
    path = data_dir / "u.item"
    col_names = [COL_ITEM, COL_TITLE, "release_date", "video_date", "imdb_url"] + GENRE_COLUMNS
    df = pd.read_csv(path, sep="|", encoding="latin-1",
                     header=None, names=col_names,
                     dtype={COL_ITEM: np.int32, COL_TITLE: str})
    def _genres(row):
        return "|".join(g for g in GENRE_COLUMNS if row.get(g, 0) == 1) or "Unknown"
    df[COL_GENRES] = df[GENRE_COLUMNS].apply(_genres, axis=1)
    result = df[[COL_ITEM, COL_TITLE, COL_GENRES]].copy()
    result[COL_TITLE] = result[COL_TITLE].str.strip()
    logger.info("Loaded metadata for %d movies", len(result))
    return result


def chronological_split(df: pd.DataFrame, test_ratio: float = 0.2,
                        min_interactions: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user chronological train/test split."""
    df = df.sort_values([COL_USER, COL_TIMESTAMP])
    train_list, test_list = [], []
    for _, group in df.groupby(COL_USER, sort=False):
        n = len(group)
        if n < min_interactions:
            train_list.append(group)
            continue
        n_test = max(1, int(np.ceil(n * test_ratio)))
        train_list.append(group.iloc[:-n_test])
        test_list.append(group.iloc[-n_test:])
    train = pd.concat(train_list, ignore_index=True)
    test  = pd.concat(test_list,  ignore_index=True) if test_list else pd.DataFrame(columns=df.columns)
    logger.info("Train: %d rows | Test: %d rows", len(train), len(test))
    return train, test
