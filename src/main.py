"""
End-to-end pipeline.

Run this to:
  1. Load and preprocess MovieLens.
  2. Train content-based, collaborative (item-CF + user-CF), matrix-factorization
     (SVD), and hybrid recommenders.
  3. Evaluate and compare them on a held-out test set.
  4. Print example recommendations for a sample user.

Usage:
    python -m src.main
    python -m src.main --user 42 --topn 10
"""
from __future__ import annotations

import argparse
import time

from . import config
from .als import ALSRecommender
from .collaborative_filtering import ItemBasedCF, UserBasedCF
from .content_based import ContentBasedRecommender
from .data_loader import load_dataset
from .evaluation import evaluate_all
from .hybrid import HybridRecommender
from .matrix_factorization import SVDRecommender


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main(user_id: int | None = None, top_n: int = config.TOP_N) -> None:
    banner("1. Loading MovieLens dataset")
    t0 = time.time()
    ds = load_dataset()
    print(f"Users: {ds.n_users} | Movies: {ds.n_movies} | "
          f"Train: {len(ds.train):,} | Test: {len(ds.test):,}")
    print(f"Loaded in {time.time() - t0:.1f}s")

    banner("2. Training models")
    models = {}

    t = time.time()
    models["ContentBased"] = ContentBasedRecommender(ds).fit()
    print(f"  ContentBased       trained in {time.time() - t:.1f}s")

    t = time.time()
    models["ItemCF"] = ItemBasedCF(ds).fit()
    print(f"  ItemBasedCF        trained in {time.time() - t:.1f}s")

    t = time.time()
    models["UserCF"] = UserBasedCF(ds).fit()
    print(f"  UserBasedCF        trained in {time.time() - t:.1f}s")

    t = time.time()
    models["SVD"] = SVDRecommender(ds).fit()
    print(f"  SVD (MatrixFact.)  trained in {time.time() - t:.1f}s")

    t = time.time()
    models["Hybrid"] = HybridRecommender(ds).fit()
    print(f"  Hybrid             trained in {time.time() - t:.1f}s")

    t = time.time()
    models["ALS"] = ALSRecommender(ds).fit()
    print(f"  ALS (implicit)     trained in {time.time() - t:.1f}s")

    banner("3. Evaluation & comparison (held-out test set)")
    results = evaluate_all(models, ds, k=top_n)
    print(results.round(5).to_string())
    print("\nHigher Precision/Recall/Coverage is better; lower RMSE/MAE is better.")
    results.round(5).to_csv(config.ARTIFACTS_DIR / "model_comparison.csv")
    print(f"\nSaved table -> {config.ARTIFACTS_DIR / 'model_comparison.csv'}")

    banner("4. Example recommendations")
    if user_id is None:
        user_id = ds.test.userId.iloc[0]
    print(f"Sample user: {user_id}\n")

    # what this user already liked (context)
    liked = (
        ds.train[(ds.train.userId == user_id) & (ds.train.rating >= 4.0)]
        .sort_values("rating", ascending=False)
        .head(5)
    )
    print("A few things this user rated highly:")
    for row in liked.itertuples(index=False):
        print(f"   {row.rating}  {ds.title(row.movieId)}")

    for name in ["ContentBased", "ItemCF", "SVD", "Hybrid", "ALS"]:
        print(f"\n--- {name} recommends ---")
        for mid, score in models[name].recommend(user_id, top_n=5):
            print(f"   {score:6.3f}  {ds.title(mid)}")

    banner("Done")
    print("Open notebooks/recommendation_walkthrough.ipynb for an interactive tour,")
    print("or extend any model in src/ — each follows the same .fit()/.recommend() interface.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommendation engine pipeline")
    parser.add_argument("--user", type=int, default=None, help="User ID to demo")
    parser.add_argument("--topn", type=int, default=config.TOP_N, help="List length")
    args = parser.parse_args()
    main(user_id=args.user, top_n=args.topn)
