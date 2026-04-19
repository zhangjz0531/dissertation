from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

try:
    from model_layer.data.loader import (
        compute_class_weight_from_train_bundle, load_numpy_split,
    )
except Exception:
    from data.loader import compute_class_weight_from_train_bundle, load_numpy_split

warnings.filterwarnings("ignore")


InputView = Literal["flatten", "last_step"]


@dataclass
class LogisticConfig:
    input_view: InputView = "last_step"
    penalty: str = "l2"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 2000
    fit_intercept: bool = True
    positive_class_weight: Optional[float] = None
    n_jobs: Optional[int] = None
    random_state: int = 42


class LogisticBaseline:
    """
    Logistic baseline. 默认 input_view='last_step' 避免 flatten 视图下维度爆炸。
    """

    def __init__(
        self,
        input_view: InputView = "last_step",
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 2000,
        fit_intercept: bool = True,
        positive_class_weight: Optional[float] = None,
        n_jobs: Optional[int] = None,
        random_state: int = 42,
    ):
        self.config = LogisticConfig(
            input_view=input_view, penalty=penalty, C=float(C),
            solver=solver, max_iter=int(max_iter),
            fit_intercept=bool(fit_intercept),
            positive_class_weight=None if positive_class_weight is None else float(positive_class_weight),
            n_jobs=n_jobs, random_state=int(random_state),
        )

        class_weight = None
        if positive_class_weight is not None:
            class_weight = {0: 1.0, 1: float(positive_class_weight)}

        # L1 需要 liblinear/saga
        if penalty == "l1" and solver == "lbfgs":
            solver = "liblinear"

        self.model = LogisticRegression(
            penalty=penalty, C=float(C), solver=solver,
            max_iter=int(max_iter), fit_intercept=bool(fit_intercept),
            class_weight=class_weight, n_jobs=n_jobs,
            random_state=int(random_state),
        )

        self.is_fitted_: bool = False
        self.feature_names_: Optional[Sequence[str]] = None
        self.n_features_in_: Optional[int] = None

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 3:
            if self.config.input_view == "flatten":
                n, l, f = arr.shape
                arr = arr.reshape(n, l * f)
            elif self.config.input_view == "last_step":
                arr = arr[:, -1, :]
            else:
                raise ValueError(self.config.input_view)
        elif arr.ndim != 2:
            raise ValueError(f"X 必须是 2D/3D，收到 shape={arr.shape}")
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _prepare_y(y: np.ndarray) -> np.ndarray:
        arr = np.asarray(y)
        if arr.ndim == 2:
            arr = arr[:, 0]
        arr = arr.reshape(-1).astype(np.int64)
        uniq = np.unique(arr)
        if not np.all(np.isin(uniq, [0, 1])):
            raise ValueError(f"y 必须是 0/1，unique={uniq.tolist()}")
        return arr

    def fit(self, X, y, feature_names=None):
        Xp = self._prepare_X(X)
        yp = self._prepare_y(y)
        self.model.fit(Xp, yp)
        self.is_fitted_ = True
        self.feature_names_ = None if feature_names is None else list(feature_names)
        self.n_features_in_ = int(Xp.shape[1])
        return self

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("未 fit")
        return self.model.predict_proba(self._prepare_X(X))[:, 1].astype(np.float32)

    def predict_logit(self, X):
        if not self.is_fitted_:
            raise RuntimeError("未 fit")
        return np.asarray(self.model.decision_function(self._prepare_X(X)), dtype=np.float32)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= float(threshold)).astype(np.int64)

    def evaluate(self, X, y, threshold: float = 0.5) -> Dict[str, float]:
        yp = self._prepare_y(y)
        pred = self.predict(X, threshold=threshold)
        proba = self.predict_proba(X)
        out = {
            "acc": float(accuracy_score(yp, pred)),
            "f1": float(f1_score(yp, pred, zero_division=0)),
            "precision": float(precision_score(yp, pred, zero_division=0)),
            "recall": float(recall_score(yp, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(yp, pred)),
        }
        try:
            out["auc"] = float(roc_auc_score(yp, proba))
        except Exception:
            out["auc"] = float("nan")
        out["positive_rate_pred"] = float(pred.mean()) if len(pred) else float("nan")
        out["positive_rate_true"] = float(yp.mean()) if len(yp) else float("nan")
        return out

    def save(self, path):
        if not self.is_fitted_:
            raise RuntimeError("未 fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "models": self.model,
            "feature_names": None if self.feature_names_ is None else list(self.feature_names_),
            "n_features_in": self.n_features_in_,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        cfg = payload["config"]
        obj = cls(
            input_view=cfg["input_view"], penalty=cfg["penalty"], C=cfg["C"],
            solver=cfg["solver"], max_iter=cfg["max_iter"],
            fit_intercept=cfg["fit_intercept"],
            positive_class_weight=cfg["positive_class_weight"],
            n_jobs=cfg["n_jobs"], random_state=cfg["random_state"],
        )
        obj.model = payload["models"]
        obj.feature_names_ = payload.get("feature_names")
        obj.n_features_in_ = payload.get("n_features_in")
        obj.is_fitted_ = True
        return obj

    def summary(self) -> Dict:
        out = {"is_fitted": bool(self.is_fitted_), "config": asdict(self.config),
               "n_features_in": self.n_features_in_}
        if self.is_fitted_:
            out["coef_shape"] = tuple(self.model.coef_.shape)
            out["intercept_shape"] = tuple(self.model.intercept_.shape)
        return out


def build_logistic_from_tensor_root(
    tensor_root,
    input_view: InputView = "last_step",
    penalty: str = "l2",
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 2000,
    fit_intercept: bool = True,
    use_class_weight: bool = False,   # 默认关闭，pos_rate=0.567 不需要 reweight
    n_jobs: Optional[int] = None,
    random_state: int = 42,
) -> LogisticBaseline:
    pos_weight = None
    if use_class_weight:
        cw = compute_class_weight_from_train_bundle(tensor_root=tensor_root)
        pos_weight = float(cw.pos_weight)
    return LogisticBaseline(
        input_view=input_view, penalty=penalty, C=C, solver=solver,
        max_iter=max_iter, fit_intercept=fit_intercept,
        positive_class_weight=pos_weight, n_jobs=n_jobs, random_state=random_state,
    )
