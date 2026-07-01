"""
Evaluation metrics.

Two complementary views:

* Rating-prediction accuracy (RMSE, MAE) -- how close are predicted ratings to
  the truth on held-out data? Reported for models that predict a rating value
  (SVD, item-based CF).

* Top-N ranking quality (Precision@K, Recall@K, Coverage) -- of the K items we
  recommend, how many does the user actually like in the test set? This is what
  matters for "what should we show on the homepage". Reported for every model.

A test rating counts as "relevant" when it is >= RELEVANCE_THRESHOLD.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import config
from .data_loader import Dataset


def rating_metrics(predict_fn, test_df) -> dict:
    """RMSE and MAE for a model exposing predict(user_id, movie_id)."""
    errors = []
    for row in test_df.itertuples(index=False):
        pred = predict_fn(row.userId, row.movieId)
        errors.append(pred - row.rating)
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAE": float(np.mean(np.abs(errors))),
    }


def ranking_metrics(recommender, dataset: Dataset, k: int = config.TOP_N) -> dict:
    """Precision@K, Recall@K and catalog coverage on the test split."""
    # ground-truth relevant items per user (from the held-out test set)
    relevant = defaultdict(set)
    for row in dataset.test.itertuples(index=False):
        if row.rating >= config.RELEVANCE_THRESHOLD:
            relevant[row.userId].add(row.movieId)

    precisions, recalls = [], []
    recommended_items = set()

    for user_id in dataset.user_index:
        truth = relevant.get(user_id)
        if not truth:
            continue
        recs = [mid for mid, _ in recommender.recommend(user_id, top_n=k)]
        recommended_items.update(recs)
        hits = sum(1 for mid in recs if mid in truth)
        precisions.append(hits / k)
        recalls.append(hits / len(truth))

    coverage = len(recommended_items) / dataset.n_movies
    return {
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "Coverage": float(coverage),
    }


def evaluate_all(models: dict, dataset: Dataset, k: int = config.TOP_N) -> "pd.DataFrame":
    """models: name -> object with .recommend(); optionally .predict()."""
    import pandas as pd

    rows = []
    for name, model in models.items():
        result = {"Model": name}
        result.update(ranking_metrics(model, dataset, k=k))
        if hasattr(model, "predict"):
            result.update(rating_metrics(model.predict, dataset.test))
        rows.append(result)
    df = pd.DataFrame(rows).set_index("Model")
    return df
