"""
Model ladder: deterministic baseline, Logistic Regression, LightGBM.
All models share the same interface: fit(X, y) / predict_proba(X).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from src.utils import get_config, Timer
from src.features import FEATURE_NAMES


def _cfg() -> dict:
    return get_config()["model"]


# ---------------------------------------------------------------------------
# L0 – Deterministic baseline
# ---------------------------------------------------------------------------

class DeterministicBaseline:
    """
    Score = weighted average of key similarity features.
    No training required.
    """
    # Feature indices by name
    _FI = {n: i for i, n in enumerate(FEATURE_NAMES)}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeterministicBaseline":
        return self  # no-op

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        fi = self._FI
        name_score = (
            0.4 * X[:, fi["name_exact"]]
            + 0.2 * X[:, fi["name_edit_sim"]]
            + 0.2 * X[:, fi["name_token_set_sim"]]
            + 0.2 * X[:, fi["name_token_jaccard"]]
        )
        addr_score = (
            0.3 * X[:, fi["addr_exact"]]
            + 0.2 * X[:, fi["addr_edit_sim"]]
            + 0.2 * X[:, fi["addr_token_set_sim"]]
            + 0.2 * X[:, fi["digit_overlap"]]
            + 0.1 * X[:, fi["digit_exact"]]
        )
        # Penalize contradictions
        penalty = (
            0.3 * X[:, fi["contr_same_name_diff_addr"]]
            + 0.3 * X[:, fi["contr_same_addr_diff_name"]]
            + 0.2 * X[:, fi["contr_digit_conflict"]]
            + 0.2 * X[:, fi["contr_country_conflict"]]
        )
        score = 0.5 * name_score + 0.5 * addr_score - 0.3 * penalty
        score = np.clip(score, 0.0, 1.0)
        return np.column_stack([1 - score, score])

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "DeterministicBaseline":
        return pickle.loads(path.read_bytes())


# ---------------------------------------------------------------------------
# L1 – Logistic Regression
# ---------------------------------------------------------------------------

class LRModel:
    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 20260925) -> None:
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self._model: Pipeline | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LRModel":
        cfg = _cfg()
        with Timer("LR fit"):
            self._model = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    C=cfg["lr_C"],
                    max_iter=cfg["lr_max_iter"],
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=cfg["random_seed"],
                    n_jobs=-1,
                )),
            ])
            self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "LRModel":
        return pickle.loads(path.read_bytes())


# ---------------------------------------------------------------------------
# L2 – LightGBM
# ---------------------------------------------------------------------------

class LGBMModel:
    def __init__(self) -> None:
        self._model: lgb.LGBMClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMModel":
        cfg = _cfg()
        pos = int(y.sum())
        neg = int((y == 0).sum())
        scale = neg / max(pos, 1)

        with Timer("LightGBM fit"):
            self._model = lgb.LGBMClassifier(
                n_estimators=cfg["lgbm_n_estimators"],
                learning_rate=cfg["lgbm_learning_rate"],
                num_leaves=cfg["lgbm_num_leaves"],
                min_child_samples=cfg["lgbm_min_child_samples"],
                subsample=cfg["lgbm_subsample"],
                colsample_bytree=cfg["lgbm_colsample_bytree"],
                reg_alpha=cfg["lgbm_reg_alpha"],
                reg_lambda=cfg["lgbm_reg_lambda"],
                scale_pos_weight=scale,
                random_state=cfg["random_seed"],
                n_jobs=-1,
                verbose=-1,
            )
            self._model.fit(
                X, y,
                feature_name=FEATURE_NAMES,
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        assert self._model is not None
        return pd.DataFrame({
            "feature": FEATURE_NAMES,
            "importance": self._model.feature_importances_,
        }).sort_values("importance", ascending=False)

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "LGBMModel":
        return pickle.loads(path.read_bytes())
