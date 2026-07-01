"""
Pre-train and pickle the Hybrid model so the Streamlit app starts instantly.

Run once locally before deploying (or after changing model weights):

    python scripts/build_models.py

Saves artifacts/model.pkl (~5 MB).

The pickle is gitignored — it is a local dev convenience and is NOT required
on Streamlit Community Cloud (the app trains at startup, which takes ~1 s on
a warm process, and @st.cache_resource holds it in memory for the lifetime of
the server).
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

# allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data_loader import load_dataset
from src.hybrid import HybridRecommender


def main() -> None:
    t0 = time.time()

    print("Loading dataset...")
    ds = load_dataset()
    print(f"  {ds.n_users} users, {ds.n_movies} movies ({time.time() - t0:.1f}s)")

    print("Training HybridRecommender...")
    t1 = time.time()
    model = HybridRecommender(ds).fit()
    print(f"  done ({time.time() - t1:.1f}s)")

    out = config.ARTIFACTS_DIR / "model.pkl"
    with open(out, "wb") as f:
        pickle.dump({"ds": ds, "model": model}, f)

    size_mb = out.stat().st_size / 1e6
    print(f"Saved {out}  ({size_mb:.1f} MB)  total={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
