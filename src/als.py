"""
ALS (Alternating Least Squares) implicit-feedback recommender.

Follows the Hu, Koren & Volinsky (2008) confidence-weighted matrix
factorization. Every observed rating becomes a positive interaction whose
confidence grows with the rating value:

    confidence(u, i) = 1 + alpha * rating(u, i)

The preference matrix P is binary (1 if the user rated the item). We find
low-rank factors U (n_users x k) and V (n_items x k) that minimize:

    sum_{u,i} c_{ui} (p_{ui} - u_u · v_i)^2  +  lambda * (||U||^2 + ||V||^2)

The Hu et al. trick allows each factor to be solved in closed form:

    u_u = (V^T C_u V + λI)^{-1} V^T C_u p_u
        = (V^T V + V_u^T diag(c_u - 1) V_u + λI)^{-1} V_u^T c_u

where V_u contains only the rows of V for items user u has interacted with.
This keeps inner-loop complexity to O(n_u * k^2 + k^3) per user rather than
O(n_items * k^2), making the algorithm scale to large catalogs.
"""
from __future__ import annotations

import numpy as np

from . import config
from .data_loader import Dataset


class ALSRecommender:
    def __init__(
        self,
        dataset: Dataset,
        n_factors: int = config.SVD_N_FACTORS,
        n_iters: int = 10,
        reg: float = 0.1,
        alpha: float = 40.0,
    ):
        self.ds = dataset
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.reg = reg
        self.alpha = alpha
        self.user_factors: np.ndarray | None = None   # n_users x k
        self.item_factors: np.ndarray | None = None   # n_items x k

    def fit(self) -> "ALSRecommender":
        rng = np.random.default_rng(config.RANDOM_SEED)
        n_users, n_items = self.ds.n_users, self.ds.n_movies
        k = self.n_factors

        self.user_factors = (rng.standard_normal((n_users, k)) * 0.01).astype(np.float32)
        self.item_factors = (rng.standard_normal((n_items, k)) * 0.01).astype(np.float32)

        R = self.ds.user_item          # CSR  (n_users x n_items)
        Rt = R.T.tocsr()               # CSR  (n_items x n_users)
        reg_eye = self.reg * np.eye(k, dtype=np.float32)

        for _ in range(self.n_iters):
            # ---- update user factors ----------------------------------------
            # Precompute V^T V once per iteration; only the confidence residual
            # V_u^T diag(c_u-1) V_u is user-specific.
            VtV = self.item_factors.T @ self.item_factors
            for u in range(n_users):
                row = R.getrow(u)
                if row.nnz == 0:
                    continue
                c = (1.0 + self.alpha * row.data).astype(np.float32)
                V_u = self.item_factors[row.indices]         # n_rated x k
                A = VtV + (V_u * (c - 1)[:, None]).T @ V_u + reg_eye
                b = (V_u * c[:, None]).sum(axis=0)
                self.user_factors[u] = np.linalg.solve(A, b)

            # ---- update item factors ----------------------------------------
            UtU = self.user_factors.T @ self.user_factors
            for i in range(n_items):
                col = Rt.getrow(i)
                if col.nnz == 0:
                    continue
                c = (1.0 + self.alpha * col.data).astype(np.float32)
                U_i = self.user_factors[col.indices]         # n_raters x k
                A = UtU + (U_i * (c - 1)[:, None]).T @ U_i + reg_eye
                b = (U_i * c[:, None]).sum(axis=0)
                self.item_factors[i] = np.linalg.solve(A, b)

        return self

    def score_all(self, user_id: int) -> np.ndarray:
        u = self.ds.user_index[user_id]
        return (self.user_factors[u] @ self.item_factors.T).astype(np.float32)

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
    als = ALSRecommender(ds).fit()
    print("ALS picks for user 1:")
    for mid, score in als.recommend(1, top_n=5):
        print(f"  {score:.3f}  {ds.title(mid)}")
