# E-Commerce Product Recommendation Engine

A complete, runnable recommendation system built on the **MovieLens** dataset
(the same techniques power product recommendations on Amazon, Netflix, and most
e-commerce sites). It implements **content-based filtering**, **collaborative
filtering**, **matrix factorization (SVD)**, and a **hybrid** model, then
**evaluates and compares** all of them on held-out data.

> The dataset is already bundled in `data/movielens/` — nothing to download.
> The core engine runs **fully offline** and needs **no API key**.

---

## What's inside

| Model | File | Idea | Needs |
|-------|------|------|-------|
| Content-based | `src/content_based.py` | TF-IDF over genres + tags, cosine similarity | item text |
| Item-based CF | `src/collaborative_filtering.py` | "users who liked X also liked Y" | ratings only |
| User-based CF | `src/collaborative_filtering.py` | find similar users, borrow their taste | ratings only |
| Matrix Factorization | `src/matrix_factorization.py` | truncated SVD into latent taste factors | ratings only |
| **Hybrid** | `src/hybrid.py` | weighted blend of SVD + content | both |
| Evaluation | `src/evaluation.py` | RMSE, MAE, Precision@K, Recall@K, Coverage | — |
| LLM explanations *(bonus)* | `src/llm_explainer.py` | "Because you watched…" via an LLM API | API key |

---

## Quick start (Windows + VS Code)

### 1. Open the project
1. Unzip this folder somewhere easy, e.g. `C:\Users\You\ecommerce-recommendation-engine`.
2. Open **VS Code** → **File → Open Folder** → select that folder.
3. Open the integrated terminal (**Terminal → New Terminal**) — you'll run all
   commands from here.

### 2. Set up Python (one time)
Open a terminal in VS Code (**Terminal → New Terminal**) and run:

```powershell
# create an isolated environment
python -m venv .venv

# activate it (PowerShell)
.\.venv\Scripts\Activate.ps1
# if PowerShell blocks the script, run this once then retry:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# install dependencies
pip install -r requirements.txt
```

> Tip: bottom-right in VS Code, click the Python version and pick the one inside
> `.venv` so the editor uses the same interpreter as your terminal.

### 3. Run the whole thing

```powershell
python -m src.main
```

You'll see the dataset stats, training times, a **model comparison table**, and
**example recommendations** for a sample user. A CSV of the comparison is saved
to `artifacts/model_comparison.csv`.

Try a specific user or list length:

```powershell
python -m src.main --user 42 --topn 10
```

### 4. (Optional) Streamlit web app

Pick a user from a dropdown, see their highly-rated movies, and explore the top
recommendations with a per-component score breakdown (collaborative vs. content):

```powershell
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501`. The model is cached after the first
load, so switching users is instant.

### 5. (Optional) Explore the notebook

```powershell
jupyter notebook notebooks/recommendation_walkthrough.ipynb
```

### 6. (Optional) Turn on natural-language explanations
Copy `.env.example` to `.env`, paste in **one** key, then:

```powershell
# PowerShell, current session only:
$env:OPENAI_API_KEY="sk-..."          # or $env:ANTHROPIC_API_KEY="sk-ant-..."
python -m src.llm_explainer
```

Without a key it still runs and prints a templated explanation instead.

### Windows setup notes

**Jupyter long-path issue:** `pip install -r requirements.txt` can fail on
Windows with `OSError: [Errno 2] No such file or directory` deep inside
`.venv\share\jupyter\labextensions\...`. Jupyter's labextension assets create
file paths that exceed Windows's 260-character limit when the project sits deep
in a user directory.

**Workarounds — pick one:**

- Install core packages individually (this is what the steps above do; Jupyter
  is listed as optional):

  ```powershell
  pip install "pandas>=2.0" "numpy>=1.24" "scikit-learn>=1.3" "scipy>=1.10" "pytest>=7.0" "matplotlib>=3.7" streamlit
  ```

- Keep the project close to the drive root (e.g. `C:\rec\`) so paths stay
  under 260 characters, then `pip install -r requirements.txt` works in full.

**Permanent fix (if you want the notebook walkthrough):** enable Windows
long-path support once, then reinstall. Run in an elevated PowerShell:

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name LongPathsEnabled -Value 1
```

Or via Group Policy: *Computer Configuration → Administrative Templates →
System → Filesystem → Enable Win32 long paths*.

> The core engine (`python -m src.main`) and Streamlit app do **not** need
> Jupyter — this only matters for the notebook walkthrough in step 5.

---

## How it works (the 60-second version)

1. **Load & clean** (`data_loader.py`) — drop users/movies with very few
   ratings, build a sparse user×movie matrix, and split each user's ratings into
   train/test so evaluation is honest.
2. **Content-based** — turn each movie's genres + tags into a TF-IDF vector;
   recommend movies similar to what a user already liked. Works for brand-new
   items (no ratings needed).
3. **Collaborative** — learn from behavior only. Item-CF and user-CF use cosine
   similarity; SVD compresses the matrix into latent "taste factors".
4. **Hybrid** — normalize each model's scores and blend them (weights in
   `src/config.py`). Captures behavioral patterns *and* keeps recommendations
   topically coherent + improves catalog coverage.
5. **Evaluate** — `RMSE`/`MAE` for rating accuracy, `Precision@K`/`Recall@K` for
   ranking quality, and `Coverage` for how much of the catalog gets surfaced.

### Reading the comparison
Different models win on different metrics — that's the point, not a bug:
- **Item-CF** tends to have the best **RMSE** (it's tuned for rating accuracy)
  but ranks niche items highly, hurting Precision@K. A classic recsys tradeoff.
- **User-CF / SVD** usually lead on **ranking** (Precision/Recall@K).
- **Hybrid** sacrifices a little precision for much higher **Coverage** and more
  coherent, explainable picks — often what you actually want in production.

Tune the behavior in `src/config.py` (latent factors, neighbor count, hybrid
weights, relevance threshold) and re-run to see the table change.

---

## Results & model comparison

Numbers below come directly from `python -m src.main` on the bundled MovieLens
*ml-latest-small* dataset (610 users, 3 650 movies, 72 434 train / 17 840 test
ratings, relevance threshold ≥ 4.0 stars).

| Model        | Precision@10 | Recall@10 | Coverage | RMSE    | MAE     |
|--------------|-------------|-----------|----------|---------|---------|
| ContentBased | 0.00956     | 0.00767   | 0.53890  | —       | —       |
| ItemCF       | 0.00117     | 0.00024   | 0.14877  | 0.86536 | 0.65967 |
| UserCF       | 0.19413     | 0.19064   | 0.06603  | —       | —       |
| SVD          | 0.11309     | 0.10917   | 0.16411  | 0.91825 | 0.70993 |
| Hybrid       | 0.06930     | 0.07263   | 0.40301  | —       | —       |
| ALS          | 0.09513     | 0.12901   | 0.26082  | —       | —       |

Higher Precision / Recall / Coverage is better; lower RMSE / MAE is better.
`—` means the model does not output a rating prediction so RMSE/MAE are undefined.

### Why each model lands where it does

**UserCF** leads on ranking (Precision@10 = 0.194, Recall@10 = 0.191) because
it borrows top-rated picks directly from the nearest-neighbour user cluster. On
a dataset dense enough that most users share several co-raters, this signal is
extremely clean.

**ItemCF** posts the best RMSE (0.865) but near-zero ranking quality
(Precision@10 = 0.00117). It is the only non-hybrid model that predicts an
actual star rating, which is why RMSE is meaningful for it. The disconnect
illustrates the classic accuracy-vs-ranking tradeoff: RMSE rewards getting close
to the observed rating on items that *have already been rated*; Precision@K
rewards surfacing items that turn up in a user's *held-out test set*. ItemCF's
top-scored items tend to be niche films whose rating pattern matches the user's
history in item-space — but those films are too obscure to appear in most users'
test sets, so almost no hits register. A model can ace one objective and bomb
the other.

**ContentBased** achieves the widest catalog coverage (~54 %) because it can
score any item that has genre or tag text, with no interaction data required.
That breadth comes at the cost of ranking quality: a TF-IDF user profile captures
topical similarity, not whether the user will actually want those specific films.

**SVD** sits in the middle on both ranking (Precision@10 = 0.113) and rating
accuracy (RMSE = 0.918). Truncated SVD discovers latent taste dimensions from
the interaction matrix and generalises across users, avoiding the extremes of the
pure neighbourhood methods.

**Hybrid** (SVD 60 % + ContentBased 40 %) trades some precision against SVD for
substantially higher coverage (~40 %). Normalising and blending both score
vectors keeps recommendations topically coherent while still tracking behavioural
patterns — and makes each pick explainable via `hybrid.explain()`.

**ALS** (Alternating Least Squares, Hu-Koren-Volinsky 2008) treats each rating
as a *confidence signal* rather than a target score: confidence = 1 + 40 × rating.
It does not learn to predict star values; instead it finds user and item factors
that explain which items users interacted with at all, weighted by interaction
strength. This makes it the only purely implicit-feedback model in the suite. It
lands between SVD and Hybrid on recall (0.129) with solid coverage (26 %) and
trains in under 10 seconds.

### Metric definitions

**Precision@K** = (relevant items in the top-K list) ÷ K, averaged across users.

**Recall@K** = (relevant items in the top-K list) ÷ (total relevant items *for
that user* in the test set), averaged across users. This is a per-user ratio, not
aggregate hits divided by catalog size. A user with 50 relevant test items who
gets 1 hit contributes Recall = 0.02; a user with 1 relevant item who gets that
1 hit contributes Recall = 1.0. Because ItemCF barely hits any relevant items and
the median user has 8 relevant test items, its per-user recall averages to
~0.00024 — tiny but non-zero.

**Coverage** = fraction of the full catalog that appears in at least one
recommendation list across all users.

### Which would you ship?

The right choice depends on the business objective — no single model wins on all
fronts.

- **Dense user base, click-through optimisation:** start with **UserCF** or
  **ALS**. UserCF's precision is the highest here; ALS is a better long-term bet
  because it can scale to implicit signals (page views, add-to-cart) without
  needing explicit star ratings.
- **Broad catalog exposure, explainability required:** use the **Hybrid**. Its
  ~40 % coverage surfaces far more of your inventory; the `hybrid.explain()`
  breakdown of collaborative vs. content score is auditable on product pages or
  in customer-service tooling.
- **Cold-start items (no interaction history yet):** only **ContentBased** and
  **Hybrid** can handle a brand-new item. The purely behavioural models (ItemCF,
  UserCF, SVD, ALS) have nothing to go on until at least a few ratings arrive.
- **When you need a rating estimate** (e.g. "would this user give this 4 stars?"):
  only **ItemCF** and **SVD** output a calibrated rating value. RMSE is the
  tiebreaker there, favouring ItemCF.

---

## Project structure

```
ecommerce-recommendation-engine/
├── README.md
├── requirements.txt
├── .env.example               # template for optional API keys
├── data/
│   ├── movielens/             # bundled dataset (movies, ratings, tags, links)
│   └── download_data.py       # re-download a fresh / larger copy
├── src/
│   ├── config.py              # all tunable knobs live here
│   ├── data_loader.py
│   ├── content_based.py
│   ├── collaborative_filtering.py
│   ├── matrix_factorization.py
│   ├── hybrid.py
│   ├── evaluation.py
│   ├── llm_explainer.py       # optional bonus
│   └── main.py                # end-to-end pipeline
├── notebooks/
│   └── recommendation_walkthrough.ipynb
└── tests/
    └── test_recommenders.py
```

---

## Ideas to extend (great for an interview story)
- Add **BPR** or **ALS** implicit-feedback factorization.
- Swap MovieLens for **Amazon product reviews** — only `data_loader.py` changes.
- Build a small **Streamlit** or **FastAPI** front end over `HybridRecommender`.
- Add **diversity / novelty** metrics and re-rank for them.
- Use the LLM explainer to generate **marketing copy** per recommendation.

---

## Dataset credit
MovieLens *ml-latest-small* — F. Maxwell Harper and Joseph A. Konstan. 2015.
*The MovieLens Datasets: History and Context.* ACM TiiS.
GroupLens Research, University of Minnesota. https://grouplens.org/datasets/movielens/
For learning/development use.
