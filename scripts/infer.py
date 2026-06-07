"""
scripts/infer.py
================
Command-line inference utility.

Usage:
  python scripts/infer.py --user_id 42 --k 10
  python scripts/infer.py --movie_id 50 --mode similar --k 8
  python scripts/infer.py --movie_ids 50,181,268 --mode cold_start
"""
from __future__ import annotations
import sys, argparse, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.persistence import load, exists

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="SAR Inference CLI")
    p.add_argument("--user_id",   type=int,  help="User ID for personalised recs")
    p.add_argument("--movie_id",  type=int,  help="Movie ID for similar-items mode")
    p.add_argument("--movie_ids", type=str,  help="Comma-separated movie IDs for cold-start")
    p.add_argument("--mode",      choices=["user","similar","cold_start"], default="user")
    p.add_argument("--k",         type=int,  default=10)
    p.add_argument("--no_seen",   action="store_true", help="Include seen items")
    args = p.parse_args()

    if not exists():
        logger.error("No trained model. Run: python scripts/train.py")
        sys.exit(1)

    model, item_meta = load()
    title_map  = dict(zip(item_meta["item_id"], item_meta["title"]))
    genres_map = dict(zip(item_meta["item_id"], item_meta.get("genres", {})))

    def show(rank, item_id, score=None):
        t = title_map.get(item_id, f"Movie #{item_id}")
        g = genres_map.get(item_id, "")
        s = f"  score={score:.4f}" if score is not None else ""
        print(f"  {rank:>2}. [{item_id:>4}] {t}  |  {g}{s}")

    # ── user mode ─────────────────────────────────────────────────────────────
    if args.mode == "user":
        uid = args.user_id
        if uid is None:
            logger.error("Provide --user_id for user mode")
            sys.exit(1)
        if uid not in model.user2idx:
            logger.warning("User %d unknown — showing popular fallback", uid)
            ids = model.popular_items(top_k=args.k)
            print(f"\n🎬 Popular fallback (top {args.k}):")
            for i, it in enumerate(ids, 1): show(i, it)
        else:
            df = model.recommend_k_items(uid, top_k=args.k,
                                         remove_seen=not args.no_seen)
            print(f"\n🎬 Top-{args.k} for User {uid}:")
            for i, (_, r) in enumerate(df.iterrows(), 1):
                show(i, int(r["item_id"]), r["score"])

    # ── similar items ─────────────────────────────────────────────────────────
    elif args.mode == "similar":
        mid = args.movie_id
        if mid is None:
            logger.error("Provide --movie_id for similar mode")
            sys.exit(1)
        df = model.similar_items(mid, top_k=args.k)
        print(f"\n🔗 Movies similar to [{mid}] {title_map.get(mid,'?')}:")
        for i, (_, r) in enumerate(df.iterrows(), 1):
            show(i, int(r["similar_item_id"]), r["score"])

    # ── cold start ────────────────────────────────────────────────────────────
    elif args.mode == "cold_start":
        if not args.movie_ids:
            logger.error("Provide --movie_ids for cold_start mode")
            sys.exit(1)
        seed = list(map(int, args.movie_ids.split(",")))
        df   = model.recommend_from_items(seed, top_k=args.k)
        seed_titles = ", ".join(title_map.get(s, str(s)) for s in seed)
        print(f"\n🎲 Cold-start recs based on: {seed_titles}")
        for i, (_, r) in enumerate(df.iterrows(), 1):
            show(i, int(r["item_id"]), r["score"])


if __name__ == "__main__":
    main()
