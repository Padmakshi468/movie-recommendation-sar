"""
config/settings.py
Central configuration for the SAR Movie Recommendation System.
"""
from pathlib import Path

# ── Project paths ─────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw" / "ml-100k"
ARTIFACTS_DIR = ROOT_DIR / "model" / "artifacts"

# ── Dataset columns ───────────────────────────────────────────────────
COL_USER      = "user_id"
COL_ITEM      = "item_id"
COL_RATING    = "rating"
COL_TIMESTAMP = "timestamp"
COL_TITLE     = "title"
COL_PRED      = "prediction"
COL_GENRES    = "genres"

# ── SAR defaults ──────────────────────────────────────────────────────
DEFAULT_SIMILARITY    = "jaccard"     # jaccard | lift | counts
DEFAULT_TIME_DECAY    = 30.0          # half-life in days (0 = off)
DEFAULT_THRESHOLD     = 1             # min co-occurrence
DEFAULT_TOP_K         = 10
MIN_RATING_THRESHOLD  = 4.0           # positive interaction filter

# ── API ───────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Logging ───────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
