"""
Content-based filtering.

Idea
----
Describe every movie by its text content (genres + tags), turn that into a
TF-IDF vector, and recommend movies whose vectors are close (cosine similarity)
to the things a user already liked.

This needs NO other users' data, so it works for brand-new items ("cold start"
on the item side) where collaborative filtering has nothing to go on.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import Dataset


class ContentBasedRecommender:
    def __init__(self, dataset: Dataset):
        self.ds = dataset
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf = None          # sparse (n_movies x n_terms)

    def fit(self) -> "ContentBasedRecommender":
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"[A-Za-z][A-Za-z\-]+",
            min_df=2,
            max_features=5000,
        )
        self.tfidf = self.vectorizer.fit_transform(self.ds.movies["content_soup"])
        return self

    # ---- item -> item -----------------------------------------------------
    def similar_movies(self, movie_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        col = self.ds.movie_index[movie_id]
        sims = cosine_similarity(self.tfidf[col], self.tfidf).ravel()
        sims[col] = -1.0  # exclude itself
        best = np.argsort(sims)[::-1][:top_n]
        return [(self.ds.index_movie[i], float(sims[i])) for i in best]

    # ---- user profile -----------------------------------------------------
    def _user_profile(self, user_id: int):
        """Average the TF-IDF vectors of movies the user liked, weighted by rating."""
        urow = self.ds.user_index[user_id]
        user_ratings = self.ds.user_item.getrow(urow)
        if user_ratings.nnz == 0:
            return None
        cols = user_ratings.indices
        weights = user_ratings.data.astype(np.float32)
        # center weights so disliked movies pull the profile away
        weights = weights - weights.mean() if len(weights) > 1 else weights
        profile = self.tfidf[cols].multiply(weights[:, None]).sum(axis=0)
        return np.asarray(profile)

    def score_all(self, user_id: int) -> np.ndarray:
        """Return a content-based score for every movie (length n_movies)."""
        profile = self._user_profile(user_id)
        if profile is None:
            return np.zeros(self.ds.n_movies, dtype=np.float32)
        scores = cosine_similarity(profile, self.tfidf).ravel()
        return scores.astype(np.float32)

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
    cb = ContentBasedRecommender(ds).fit()
    print("Movies similar to Toy Story (1995):")
    for mid, score in cb.similar_movies(1, top_n=5):
        print(f"  {score:.3f}  {ds.title(mid)}")
    print("\nContent-based picks for user 1:")
    for mid, score in cb.recommend(1, top_n=5):
        print(f"  {score:.3f}  {ds.title(mid)}")
