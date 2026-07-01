"""
Central configuration for the recommendation engine.

Edit values here (or override via environment variables) to tune the models
without touching the algorithm code.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "movielens"

MOVIES_CSV = DATA_DIR / "movies.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"
TAGS_CSV = DATA_DIR / "tags.csv"
LINKS_CSV = DATA_DIR / "links.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"   # saved models / outputs go here
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2          # fraction of each user's ratings held out for testing
RANDOM_SEED = 42

# A user/movie must have at least this many ratings to be kept. Filtering the
# long tail makes collaborative filtering far more reliable.
MIN_RATINGS_PER_USER = 5
MIN_RATINGS_PER_MOVIE = 5

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------
# Matrix factorization (truncated SVD)
SVD_N_FACTORS = 50       # latent dimensions

# Collaborative filtering
CF_TOP_K_NEIGHBORS = 40  # how many similar items/users to consider

# Hybrid weighting: final = w_cf * collaborative + w_cb * content_based
HYBRID_WEIGHT_CF = 0.6
HYBRID_WEIGHT_CB = 0.4

# Evaluation
TOP_N = 10               # length of the recommendation list for precision/recall
RELEVANCE_THRESHOLD = 4.0  # a rating >= this counts as a "relevant" / liked item
