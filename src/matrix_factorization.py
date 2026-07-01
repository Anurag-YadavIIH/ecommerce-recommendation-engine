"""
Matrix factorization with truncated SVD.

Idea
----
The user-item matrix is huge and mostly empty. We approximate it as the product
of two skinny matrices:

    R  ~=  U  @  V^T

where U holds a short latent vector per user and V holds one per movie. Those
latent factors capture hidden taste dimensions ("gritty crime drama",
"feel-good animation", ...) discovered automatically from the ratings.

We use scikit-learn's TruncatedSVD, which runs directly on the sparse matrix and
scales to large, sparse data -- exactly the situation in real e-commerce.
Predictions add back each user's mean rating so the numbers land on the 0.5-5
scale.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD

from . import config
from .data_loader import Dataset


class SVDRecommender:
    def __init__(self, dataset: Dataset, n_factors: int = config.SVD_N_FACTORS):
        self.ds = dataset
        self.n_factors = n_factors
        self.user_factors: np.ndarray | None = None   # n_users x k
        self.item_factors: np.ndarray | None = None   # n_movies x k
        self.user_means: np.ndarray | None = None
        self.predicted: np.ndarray | None = None       # dense reconstruction

    def fit(self) -> "SVDRecommender":
        # subtract each user's mean from their observed entries (bias removal)
        self.user_means = np.zeros(self.ds.n_users, dtype=np.float32)
        for u in range(self.ds.n_users):
            row = self.ds.user_item.getrow(u)
            if row.nnz:
                self.user_means[u] = row.data.mean()

        centered = self.ds.user_item.tocoo().copy().astype(np.float32)
        centered.data = centered.data - self.user_means[centered.row]
        centered = centered.tocsr()

        k = min(self.n_factors, min(centered.shape) - 1)
        svd = TruncatedSVD(n_components=k, random_state=config.RANDOM_SEED)
        self.user_factors = svd.fit_transform(centered)           # n_users x k
        self.item_factors = svd.components_.T                     # n_movies x k
        return self

    def score_all(self, user_id: int) -> np.ndarray:
        u = self.ds.user_index[user_id]
        preds = self.user_factors[u] @ self.item_factors.T
        return (preds + self.user_means[u]).astype(np.float32)

    def predict(self, user_id: int, movie_id: int) -> float:
        if movie_id not in self.ds.movie_index:
            return float(self.user_means[self.ds.user_index[user_id]])
        u = self.ds.user_index[user_id]
        i = self.ds.movie_index[movie_id]
        pred = self.user_factors[u] @ self.item_factors[i] + self.user_means[u]
        return float(np.clip(pred, 0.5, 5.0))

    def recommend(self, user_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        scores = self.score_all(user_id)
        seen = set(self.ds.user_item.getrow(self.ds.user_index[user_id]).indices)
        ranked = np.argsort(scores)[::-1]
        out = []
        for i in ranked:
            if i in seen:
                continue
            out.append((self.ds.index_movie[i], float(np.clip(scores[i], 0.5, 5.0))))
            if len(out) == top_n:
                break
        return out


if __name__ == "__main__":
    from .data_loader import load_dataset

    ds = load_dataset()
    svd = SVDRecommender(ds).fit()
    print(f"Latent factors per user/movie: {svd.user_factors.shape[1]}")
    print("\nSVD picks for user 1:")
    for mid, score in svd.recommend(1, top_n=5):
        print(f"  {score:.3f}  {ds.title(mid)}")
