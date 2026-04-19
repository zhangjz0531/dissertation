from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except Exception as e:
    XGBClassifier = None
    _xgb_import_error = e
else:
    _xgb_import_error = None

try:
    from model_layer.data.loader import (
        compute_class_weight_from_train_bundle, load_numpy_split,
    )
except Exception:
    from data.loader import compute_class_weight_from_train_bundle, load_numpy_split

warnings.filterwarnings("ignore")


InputView = Literal["flatten", "last_step"]


@dataclass
class XGBConfig:
    input_view: InputView = "last_step"
    n_estimators: int = 600
    learning_rate: float = 0.03
    max_depth: int = 3
    min_child_weight: float = 3.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    gamma: float = 0.0
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    tree_method: str = "hist"
    max_bin: int = 256
    scale_pos_weight: Optional[float] = None
    random_state: int = 42
    n_jobs: int = 8
    early_stopping_rounds: int = 50


class XGBBaseline:
    """
    XGBoost baseline. 默认 input_view='last_step' 避免维度爆炸。
    """

    def __init__(self, **kwargs):
        if XGBClassifier is None:
            raise ImportError(f"未安装 xgboost: {_xgb_import_error}")

        cfg_kwargs = {k: v for k, v in kwargs.items() if k in XGBConfig.__dataclass_fields__}
        self.config = XGBConfig(**cfg_kwargs)

        self.model = XGBClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            gamma=self.config.gamma,
            objective=self.config.objective,
            eval_metric=self.config.eval_metric,
            tree_method=self.config.tree_method,
            max_bin=self.config.max_bin,
            scale_pos_weight=self.config.scale_pos_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

        self.is_fitted_: bool = False
        self.feature_names_: Optional[Sequence[str]] = None
        self.n_features_in_: Optional[int] = None
        self.best_iteration_: Optional[int] = None

    def _prepare_X(self, X):
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
    def _prepare_y(y):
        arr = np.asarray(y)
        if arr.ndim == 2:
            arr = arr[:, 0]
        arr = arr.reshape(-1).astype(np.int64)
        uniq = np.unique(arr)
        if not np.all(np.isin(uniq, [0, 1])):
            raise ValueError(f"y 必须是 0/1，unique={uniq.tolist()}")
        return arr

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, verbose=False):
        Xtr = self._prepare_X(X_train)
        ytr = self._prepare_y(y_train)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            Xva = self._prepare_X(X_val)
            yva = self._prepare_y(y_val)
            fit_kwargs["eval_set"] = [(Xva, yva)]
            fit_kwargs["verbose"] = bool(verbose)
            try:
                self.model.set_params(early_stopping_rounds=int(self.config.early_stopping_rounds))
            except Exception:
                fit_kwargs["early_stopping_rounds"] = int(self.config.early_stopping_rounds)

        self.model.fit(Xtr, ytr, **fit_kwargs)
        self.is_fitted_ = True
        self.feature_names_ = None if feature_names is None else list(feature_names)
        self.n_features_in_ = int(Xtr.shape[1])

        if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None:
            self.best_iteration_ = int(self.model.best_iteration)
        return self

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("未 fit")
        return self.model.predict_proba(self._prepare_X(X))[:, 1].astype(np.float32)

    def predict_logit(self, X):
        """返回 log(p/(1-p))，用于 calibration 的 platt 分支。"""
        p = np.clip(self.predict_proba(X).astype(np.float64), 1e-8, 1 - 1e-8)
        return np.log(p / (1.0 - p)).astype(np.float32)

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

    def get_feature_importance(self, topk: int = 30) -> Dict[str, float]:
        if not self.is_fitted_:
            raise RuntimeError("未 fit")
        scores = np.asarray(self.model.feature_importances_, dtype=np.float32)
        names = self.feature_names_ or [f"f{i}" for i in range(len(scores))]
        pairs = sorted(list(zip(names, scores.tolist())), key=lambda kv: -kv[1])[:topk]
        return {k: float(v) for k, v in pairs}

    def save(self, path):
        if not self.is_fitted_:
            raise RuntimeError("未 fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config), "model": self.model,
            "feature_names": None if self.feature_names_ is None else list(self.feature_names_),
            "n_features_in": self.n_features_in_, "best_iteration": self.best_iteration_,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path):
        with open(Path(path), "rb") as f:
            payload = pickle.load(f)
        cfg = payload["config"]
        obj = cls(**cfg)
        obj.model = payload["model"]
        obj.feature_names_ = payload.get("feature_names")
        obj.n_features_in_ = payload.get("n_features_in")
        obj.best_iteration_ = payload.get("best_iteration")
        obj.is_fitted_ = True
        return obj

    def summary(self) -> Dict:
        out = {"is_fitted": bool(self.is_fitted_), "config": asdict(self.config),
               "n_features_in": self.n_features_in_, "best_iteration": self.best_iteration_}
        if self.is_fitted_:
            try:
                out["feature_importance_top10"] = self.get_feature_importance(topk=10)
            except Exception:
                out["feature_importance_top10"] = {}
        return out


def build_xgb_from_tensor_root(
    tensor_root,
    input_view: InputView = "last_step",
    n_estimators: int = 600,
    learning_rate: float = 0.03,
    max_depth: int = 3,
    min_child_weight: float = 3.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    tree_method: str = "hist",
    max_bin: int = 256,
    use_class_weight: bool = False,   # 默认关闭
    random_state: int = 42,
    n_jobs: int = 8,
    early_stopping_rounds: int = 50,
) -> XGBBaseline:
    scale_pos_weight = None
    if use_class_weight:
        cw = compute_class_weight_from_train_bundle(tensor_root=tensor_root)
        scale_pos_weight = float(cw.pos_weight)
    return XGBBaseline(
        input_view=input_view, n_estimators=n_estimators, learning_rate=learning_rate,
        max_depth=max_depth, min_child_weight=min_child_weight, subsample=subsample,
        colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
        gamma=gamma, objective="binary:logistic", eval_metric="logloss",
        tree_method=tree_method, max_bin=max_bin, scale_pos_weight=scale_pos_weight,
        random_state=random_state, n_jobs=n_jobs, early_stopping_rounds=early_stopping_rounds,
    )
