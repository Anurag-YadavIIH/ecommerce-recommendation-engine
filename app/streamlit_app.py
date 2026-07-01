"""
Streamlit front-end for the Hybrid Recommender.

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# Ensure project root is on the path when launched as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.data_loader import load_dataset
from src.hybrid import HybridRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading data and training model…")
def _load():
    import pickle
    from src import config

    pkl = config.ARTIFACTS_DIR / "model.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            saved = pickle.load(f)
        return saved["ds"], saved["model"]

    ds = load_dataset()
    model = HybridRecommender(ds).fit()
    return ds, model


ds, model = _load()

# ── Sidebar: user picker ────────────────────────────────────────────────────
st.sidebar.title("🎬 Movie Recommender")
user_ids = sorted(ds.user_index.keys())
user_id = st.sidebar.selectbox("Pick a user", user_ids, index=0)

top_n = st.sidebar.slider("Number of recommendations", min_value=5, max_value=20, value=10)

# ── Main panel ───────────────────────────────────────────────────────────────
st.title("Hybrid Movie Recommendations")
st.caption(
    "Blends SVD collaborative filtering (60 %) with TF-IDF content similarity (40 %). "
    "Toggle the sidebar to explore different users."
)

col_left, col_right = st.columns([1, 2])

# Left: what the user already liked
with col_left:
    st.subheader(f"User {user_id} — highly rated")
    liked = (
        ds.train[(ds.train.userId == user_id) & (ds.train.rating >= 4.0)]
        .sort_values("rating", ascending=False)
        .head(10)
    )
    if liked.empty:
        st.info("No ratings ≥ 4.0 found in training data.")
    else:
        for row in liked.itertuples(index=False):
            stars = "★" * int(row.rating) + ("½" if row.rating % 1 else "")
            st.write(f"**{stars}** {ds.title(row.movieId)}")

# Right: recommendations with score breakdown
with col_right:
    st.subheader(f"Top {top_n} Recommendations")

    recs = model.recommend(user_id, top_n=top_n)
    rows = []
    for movie_id, _ in recs:
        ex = model.explain(user_id, movie_id)
        rows.append(
            {
                "Movie": ex["movie"],
                "Collaborative (SVD)": ex["collaborative"],
                "Content (TF-IDF)": ex["content"],
                "Hybrid Score": ex["final"],
            }
        )

    df = pd.DataFrame(rows)

    st.dataframe(
        df.style.background_gradient(
            subset=["Collaborative (SVD)", "Content (TF-IDF)", "Hybrid Score"],
            cmap="YlGn",
        ).format(
            {
                "Collaborative (SVD)": "{:.3f}",
                "Content (TF-IDF)": "{:.3f}",
                "Hybrid Score": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "All scores are min-max normalised to [0, 1] before blending. "
        "Higher = stronger signal from that component."
    )
