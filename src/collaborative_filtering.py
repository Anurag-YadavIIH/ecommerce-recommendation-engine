"""
Neighborhood collaborative filtering (memory-based).

Two flavors:

* Item-based CF: "users who liked X also liked Y". We precompute item-item
  cosine similarity from the rating matrix and predict a user's rating for an
  unseen item as a similarity-weighted average of their ratings on similar items.
  This is the workhorse used by many real e-commerce sites because item-item
  similarities are stable and can be cached.

* User-based CF: find users with similar taste and borrow their ratings.

Both rely ONLY on the interaction matrix -- no item content needed.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .data_loader import Dataset


class ItemBasedCF:
    def __init__(self, dataset: Dataset, top_k: int = config.CF_TOP_K_NEIGHBORS):
        self.ds = dataset
        self.top_k = top_k
        self.sim: np.ndarray | None = None      # (n_movies x n_movies)
        self.user_means: np.ndarray | None = None

    def fit(self) -> "ItemBasedCF":
        ui = self.ds.user_item                  # users x movies
        # mean-center each user's ratings to remove "harsh vs generous rater" bias
        self.user_means = np.array(
            [row.data.mean() if row.nnz else 0.0 for row in ui]
        )
        # item-item similarity on the (movies x users) matrix
        self.sim = cosine_similarity(ui.T, dense_output=True).astype(np.float32)
        np.fill_diagonal(self.sim, 0.0)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        u = self.ds.user_index[user_id]
        if movie_id not in self.ds.movie_index:
            return float(self.user_means[u])
        i = self.ds.movie_index[movie_id]

        urow = self.ds.user_item.getrow(u)
        rated_cols = urow.indices
        if len(rated_cols) == 0:
            return float(self.user_means[u])

        sims = self.sim[i, rated_cols]
        # keep only the top-k most similar rated items
        if len(sims) > self.top_k:
            keep = np.argpartition(sims, -self.top_k)[-self.top_k:]
            sims = sims[keep]
            ratings = urow.data[keep]
        else:
            ratings = urow.data

        denom = np.abs(sims).sum()
        if denom == 0:
            return float(self.user_means[u])
        return float(np.dot(sims, ratings) / denom)

    def score_all(self, user_id: int) -> np.ndarray:
        """Predicted rating for every movie for this user."""
        u = self.ds.user_index[user_id]
        urow = self.ds.user_item.getrow(u)
        rated = urow.indices
        if len(rated) == 0:
            return np.full(self.ds.n_movies, self.user_means[u], dtype=np.float32)
        # (n_movies x n_rated) @ (n_rated,) -> weighted sums
        sub = self.sim[:, rated]                      # n_movies x n_rated
        num = sub @ urow.data
        den = np.abs(sub).sum(axis=1)
        den[den == 0] = 1e-8
        return (num / den).astype(np.float32)

    def recommend(self, user_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        scores = self.score_all(user_id)
        seen = set(self.ds.user_item.getrow(self.ds.user_index[user_id]).indices)
        ranked = np.argsort(scores)[::-1]
        out = []
        for i in ranked:
            if i in seen:
                continue
            out.append((self.ds.index_movie[i], float(scores[i])))
            if len(out) == top_n:
                break
        return out


class UserBasedCF:
    def __init__(self, dataset: Dataset, top_k: int = config.CF_TOP_K_NEIGHBORS):
        self.ds = dataset
        self.top_k = top_k
        self.sim: np.ndarray | None = None      # (n_users x n_users)

    def fit(self) -> "UserBasedCF":
        self.sim = cosine_similarity(self.ds.user_item, dense_output=True).astype(np.float32)
        np.fill_diagonal(self.sim, 0.0)
        return self

    def score_all(self, user_id: int) -> np.ndarray:
        u = self.ds.user_index[user_id]
        sims = self.sim[u].copy()
        if self.top_k < len(sims):
            cut = np.argpartition(sims, -self.top_k)[-self.top_k:]
            mask = np.zeros_like(sims)
            mask[cut] = sims[cut]
            sims = mask
        num = sims @ self.ds.user_item            # (n_movies,)
        den = np.abs(sims).sum()
        den = den if den else 1e-8
        return np.asarray(num / den).ravel().astype(np.float32)

    def recommend(self, user_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        scores = self.score_all(user_id)
        seen = set(self.ds.user_item.getrow(self.ds.user_index[user_id]).indices)
        ranked = np.argsort(scores)[::-1]
        out = []
        for i in ranked:
            if i in seen:
                continue
            out.append((self.ds.index_movie[i], float(scores[i])))
            if len(out) == top_n:
                break
        return out


if __name__ == "__main__":
    from .data_loader import load_dataset

    ds = load_dataset()
    item_cf = ItemBasedCF(ds).fit()
    print("Item-based CF picks for user 1:")
    for mid, score in item_cf.recommend(1, top_n=5):
        print(f"  {score:.3f}  {ds.title(mid)}")
