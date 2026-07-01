"""
Data loading and preprocessing.

Responsibilities
----------------
* Read the raw MovieLens CSV files.
* Filter out very sparse users/movies (the long tail hurts CF quality).
* Build clean, contiguous integer index mappings (user -> row, movie -> col).
* Produce a sparse user-item rating matrix.
* Produce a per-user train/test split for honest evaluation.
* Assemble a "content soup" (genres + tags) used by the content-based model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from . import config


@dataclass
class Dataset:
    """Everything the models need, bundled in one object."""
    ratings: pd.DataFrame          # full filtered ratings (userId, movieId, rating)
    train: pd.DataFrame            # training ratings
    test: pd.DataFrame             # held-out test ratings
    movies: pd.DataFrame           # movieId, title, genres, content_soup
    user_item: csr_matrix          # sparse (n_users x n_movies) train matrix
    user_index: dict[int, int]     # userId -> row position
    movie_index: dict[int, int]    # movieId -> col position
    index_user: dict[int, int]     # row position -> userId
    index_movie: dict[int, int]    # col position -> movieId

    @property
    def n_users(self) -> int:
        return len(self.user_index)

    @property
    def n_movies(self) -> int:
        return len(self.movie_index)

    def title(self, movie_id: int) -> str:
        row = self.movies.loc[self.movies.movieId == movie_id, "title"]
        return row.iloc[0] if len(row) else f"<movie {movie_id}>"


def _filter_sparse(ratings: pd.DataFrame) -> pd.DataFrame:
    """Iteratively drop users/movies with too few ratings."""
    while True:
        before = len(ratings)
        user_counts = ratings.userId.value_counts()
        keep_users = user_counts[user_counts >= config.MIN_RATINGS_PER_USER].index
        ratings = ratings[ratings.userId.isin(keep_users)]

        movie_counts = ratings.movieId.value_counts()
        keep_movies = movie_counts[movie_counts >= config.MIN_RATINGS_PER_MOVIE].index
        ratings = ratings[ratings.movieId.isin(keep_movies)]

        if len(ratings) == before:        # converged
            break
    return ratings.reset_index(drop=True)


def _train_test_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user split so every user appears in both train and test."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    train_parts, test_parts = [], []

    for _, group in ratings.groupby("userId"):
        group = group.sample(frac=1.0, random_state=config.RANDOM_SEED)
        n_test = max(1, int(len(group) * config.TEST_SIZE))
        test_parts.append(group.iloc[:n_test])
        train_parts.append(group.iloc[n_test:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, test


def _build_content_soup(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Merge genres and user-supplied tags into one text field per movie."""
    movies = movies.copy()
    # genres are pipe-separated, e.g. "Adventure|Comedy"
    movies["genre_text"] = movies.genres.str.replace("|", " ", regex=False)
    movies["genre_text"] = movies["genre_text"].replace("(no genres listed)", "")

    tag_text = (
        tags.groupby("movieId")["tag"]
        .apply(lambda s: " ".join(str(t) for t in s))
        .rename("tag_text")
    )
    movies = movies.merge(tag_text, on="movieId", how="left")
    movies["tag_text"] = movies["tag_text"].fillna("")

    # genres weighted twice (repeated) so they matter more than incidental tags
    movies["content_soup"] = (
        movies["genre_text"] + " " + movies["genre_text"] + " " + movies["tag_text"]
    ).str.strip()
    return movies


def load_dataset() -> Dataset:
    ratings = pd.read_csv(config.RATINGS_CSV)[["userId", "movieId", "rating"]]
    movies = pd.read_csv(config.MOVIES_CSV)
    tags = pd.read_csv(config.TAGS_CSV)

    ratings = _filter_sparse(ratings)

    # keep only movies that survived filtering, but retain metadata for all of them
    movies = _build_content_soup(movies, tags)
    movies = movies[movies.movieId.isin(ratings.movieId.unique())].reset_index(drop=True)

    # contiguous index maps
    users = sorted(ratings.userId.unique())
    movie_ids = sorted(ratings.movieId.unique())
    user_index = {u: i for i, u in enumerate(users)}
    movie_index = {m: i for i, m in enumerate(movie_ids)}
    index_user = {i: u for u, i in user_index.items()}
    index_movie = {i: m for m, i in movie_index.items()}

    train, test = _train_test_split(ratings)

    rows = train.userId.map(user_index).to_numpy()
    cols = train.movieId.map(movie_index).to_numpy()
    vals = train.rating.to_numpy(dtype=np.float32)
    user_item = csr_matrix(
        (vals, (rows, cols)), shape=(len(user_index), len(movie_index))
    )

    # ensure movies frame is aligned to movie_index ordering
    movies = movies.set_index("movieId").loc[movie_ids].reset_index()

    return Dataset(
        ratings=ratings,
        train=train,
        test=test,
        movies=movies,
        user_item=user_item,
        user_index=user_index,
        movie_index=movie_index,
        index_user=index_user,
        index_movie=index_movie,
    )


def load_amazon_dataset(csv_path: str) -> Dataset:
    """Load an Amazon product-reviews CSV and return the same Dataset the models expect.

    Expected CSV columns
    --------------------
    user_id     : str or int  — reviewer identifier
    product_id  : str or int  — product / ASIN identifier
    rating      : float       — star rating (e.g. 1.0–5.0)
    review_text : str         — free-text review body (used for content soup)
    category    : str         — product category (e.g. "Electronics", "Books")

    The function applies the same sparse-filtering, train/test split, and index-
    building logic as load_dataset(), so every model works without modification.
    The content soup is built as:
        category + " " + category + " " + review_text_per_product
    mirroring the genre-doubling trick used for MovieLens.

    Example
    -------
    >>> ds = load_amazon_dataset("data/amazon/reviews.csv")
    >>> from src.hybrid import HybridRecommender
    >>> HybridRecommender(ds).fit().recommend(ds.test.userId.iloc[0], top_n=5)
    """
    raw = pd.read_csv(csv_path)

    # Normalise column names to the internal convention used by every model.
    raw = raw.rename(columns={"user_id": "userId", "product_id": "movieId"})
    ratings = raw[["userId", "movieId", "rating"]].copy()

    ratings = _filter_sparse(ratings)

    # Build a products frame analogous to movies (movieId, title, content_soup).
    # Aggregate all review texts per product after filtering to keep only the
    # products that survived the sparsity filter.
    surviving_products = ratings.movieId.unique()
    product_rows = raw[raw.movieId.isin(surviving_products)].copy()

    # One row per product: aggregate review texts and take the first category.
    agg = product_rows.groupby("movieId").agg(
        title=("movieId", "first"),          # product_id doubles as title
        category=("category", "first"),
        review_text=("review_text", lambda s: " ".join(str(t) for t in s)),
    ).reset_index()

    # Mirror the MovieLens genre-doubling: repeat category twice so it outweighs
    # the noisy free-text bag-of-words.
    agg["content_soup"] = (
        agg["category"].fillna("") + " "
        + agg["category"].fillna("") + " "
        + agg["review_text"].fillna("")
    ).str.strip()

    # Use product_id as the display title if no separate title column exists.
    agg["title"] = agg["title"].astype(str)

    # Contiguous index maps (identical to load_dataset logic).
    users = sorted(ratings.userId.unique())
    movie_ids = sorted(ratings.movieId.unique())
    user_index = {u: i for i, u in enumerate(users)}
    movie_index = {m: i for i, m in enumerate(movie_ids)}
    index_user = {i: u for u, i in user_index.items()}
    index_movie = {i: m for m, i in movie_index.items()}

    train, test = _train_test_split(ratings)

    rows_idx = train.userId.map(user_index).to_numpy()
    cols_idx = train.movieId.map(movie_index).to_numpy()
    vals = train.rating.to_numpy(dtype=np.float32)
    user_item = csr_matrix(
        (vals, (rows_idx, cols_idx)), shape=(len(user_index), len(movie_index))
    )

    # Align products frame to the contiguous movie_index order.
    movies = agg.set_index("movieId").loc[movie_ids].reset_index()

    return Dataset(
        ratings=ratings,
        train=train,
        test=test,
        movies=movies,
        user_item=user_item,
        user_index=user_index,
        movie_index=movie_index,
        index_user=index_user,
        index_movie=index_movie,
    )


if __name__ == "__main__":
    ds = load_dataset()
    print(f"Users: {ds.n_users}  Movies: {ds.n_movies}")
    print(f"Train ratings: {len(ds.train):,}  Test ratings: {len(ds.test):,}")
    density = ds.user_item.nnz / (ds.n_users * ds.n_movies)
    print(f"Matrix density: {density:.3%}")
    print(ds.movies[["title", "content_soup"]].head(3).to_string())
