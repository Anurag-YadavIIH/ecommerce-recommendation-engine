"""
Lightweight smoke tests. Run with:  pytest -q

They confirm the data loads and every model produces sensible, correctly-shaped
recommendations without crashing. Keep them fast so they run on every change.
"""
import numpy as np
import pytest

from src.collaborative_filtering import ItemBasedCF, UserBasedCF
from src.content_based import ContentBasedRecommender
from src.data_loader import load_dataset
from src.hybrid import HybridRecommender
from src.matrix_factorization import SVDRecommender


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_dataset_shapes(ds):
    assert ds.n_users > 100
    assert ds.n_movies > 1000
    assert len(ds.train) > len(ds.test)
    assert ds.user_item.shape == (ds.n_users, ds.n_movies)


@pytest.mark.parametrize("Model", [
    ContentBasedRecommender, ItemBasedCF, UserBasedCF, SVDRecommender, HybridRecommender
])
def test_recommend_contract(ds, Model):
    model = Model(ds).fit()
    user_id = next(iter(ds.user_index))
    recs = model.recommend(user_id, top_n=10)
    assert len(recs) == 10
    # no duplicates
    assert len({mid for mid, _ in recs}) == 10
    # nothing the user already rated in train
    seen = set(ds.user_item.getrow(ds.user_index[user_id]).indices)
    for mid, _ in recs:
        assert ds.movie_index[mid] not in seen


def test_svd_predict_in_range(ds):
    svd = SVDRecommender(ds).fit()
    row = ds.test.iloc[0]
    pred = svd.predict(row.userId, row.movieId)
    assert 0.5 <= pred <= 5.0


def test_content_similarity_symmetry(ds):
    cb = ContentBasedRecommender(ds).fit()
    sims = cb.similar_movies(1, top_n=5)
    scores = [s for _, s in sims]
    assert scores == sorted(scores, reverse=True)   # ranked descending
    assert all(0.0 <= s <= 1.0 for s in scores)
