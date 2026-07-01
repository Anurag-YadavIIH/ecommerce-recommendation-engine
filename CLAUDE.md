# Project context for Claude Code

This is a **movie/product recommendation engine** on the MovieLens dataset. Use
this file to orient yourself before making changes.

## Run / test
- Full pipeline: `python -m src.main` (optionally `--user N --topn K`)
- Tests: `pytest -q`
- Single module demo: `python -m src.content_based`, `python -m src.matrix_factorization`, etc.

## Architecture
- `src/config.py` — all tunable parameters and file paths. Change knobs here.
- `src/data_loader.py` — reads CSVs from `data/movielens/`, filters sparse
  users/movies, builds a sparse `user_item` matrix, makes a per-user train/test
  split, and returns a single `Dataset` dataclass that every model consumes.
- Each recommender exposes the SAME interface:
  - `.fit()` -> self
  - `.recommend(user_id, top_n)` -> list of `(movie_id, score)`
  - `.score_all(user_id)` -> np.ndarray of length `n_movies`
  - some also expose `.predict(user_id, movie_id)` -> float rating
- `src/evaluation.py` — `evaluate_all(models, dataset)` returns a comparison
  DataFrame (Precision@K, Recall@K, Coverage, and RMSE/MAE where available).
- `src/hybrid.py` — blends SVD + content; weights in `config.py`.
- `src/llm_explainer.py` — optional; needs `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

## Conventions
- Keep the common interface above when adding a new model, so it drops straight
  into `main.py` and `evaluation.py`.
- `userId` / `movieId` are the original dataset IDs; `user_index` / `movie_index`
  map them to contiguous matrix positions. Use `index_movie` / `index_user` to
  go back.
- Prefer sparse matrices (`scipy.sparse`) for anything user×movie sized.
- Don't commit secrets; keys live in `.env` (gitignored).

## Good first tasks if asked to extend
1. Add an ALS / BPR implicit-feedback model in a new `src/als.py` following the
   shared interface, then register it in `src/main.py`.
2. Add a FastAPI app in `app/` exposing `GET /recommend/{user_id}`.
3. Add novelty/diversity metrics to `src/evaluation.py`.
