"""
Hybrid recommendation.

We blend a collaborative signal (SVD matrix factorization, which captures
behavioral taste patterns) with a content signal (TF-IDF similarity over genres
and tags, which captures the *meaning* of items). The blend gives the best of
both worlds:

* SVD is accurate for users/items with lots of interactions.
* Content rescues cold-start items and keeps recommendations topically coherent.

Scores from each model live on different scales, so we min-max normalize them to
[0, 1] per user before taking a weighted sum:

    final = w_cf * norm(collaborative) + w_cb * norm(content)
"""
from __future__ import annotations

import numpy as np

from . import config
from .content_based import ContentBasedRecommender
from .data_loader import Dataset
from .matrix_factorization import SVDRecommender


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridRecommender:
    def __init__(
        self,
        dataset: Dataset,
        w_cf: float = config.HYBRID_WEIGHT_CF,
        w_cb: float = config.HYBRID_WEIGHT_CB,
    ):
        self.ds = dataset
        self.w_cf = w_cf
        self.w_cb = w_cb
        self.cf = SVDRecommender(dataset)
        self.cb = ContentBasedRecommender(dataset)

    def fit(self) -> "HybridRecommender":
        self.cf.fit()
        self.cb.fit()
        return self

    def score_all(self, user_id: int) -> np.ndarray:
        cf_scores = _minmax(self.cf.score_all(user_id))
        cb_scores = _minmax(self.cb.score_all(user_id))
        return self.w_cf * cf_scores + self.w_cb * cb_scores

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

    def explain(self, user_id: int, movie_id: int) -> dict:
        """Return the component scores behind a recommendation (transparency)."""
        i = self.ds.movie_index[movie_id]
        cf = _minmax(self.cf.score_all(user_id))[i]
        cb = _minmax(self.cb.score_all(user_id))[i]
        return {
            "movie": self.ds.title(movie_id),
            "collaborative": round(float(cf), 3),
            "content": round(float(cb), 3),
            "final": round(float(self.w_cf * cf + self.w_cb * cb), 3),
        }


if __name__ == "__main__":
    from .data_loader import load_dataset

    ds = load_dataset()
    hybrid = HybridRecommender(ds).fit()
    print("Hybrid picks for user 1:")
    recs = hybrid.recommend(1, top_n=5)
    for mid, score in recs:
        print(f"  {score:.3f}  {ds.title(mid)}")
    print("\nWhy was the top pick recommended?")
    print(hybrid.explain(1, recs[0][0]))
