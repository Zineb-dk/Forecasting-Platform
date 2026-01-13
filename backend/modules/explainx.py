# explainx.py

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List

import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error

ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]


class ExplainX:
    """
    Generic explainer for models.

    Supports:
      - model_family = 'gbm'   -> TreeExplainer (RF, XGB, LGBM, CatBoost, etc.)
      - model_family = 'linear'-> LinearExplainer (Ridge, Lasso, ElasticNet)
      - model_family = 'deep'  -> GradientExplainer on sequence models (LSTM, TCN, TFT...)
    """

    def __init__(
        self,
        model: Any,
        model_family: str,
        X_train: ArrayLike,
        X_val: ArrayLike,
        y_val: Optional[ArrayLike] = None,
        feature_names: Optional[List[str]] = None,
        task_name: str = "target",
        background_size: int = 200,
    ):
        self.model = model
        self.model_family = model_family.lower()
        self.task_name = task_name

        self.X_train = X_train
        self.X_val = X_val
        self.y_val = None if y_val is None else np.asarray(y_val).reshape(-1)

        self.feature_names = feature_names
        self.background_size = background_size

        # internal
        self.shap_values: Optional[np.ndarray] = None
        self.expected_value: Optional[Union[float, np.ndarray]] = None
        self.explainer: Optional[shap.Explainer] = None

        # sanity checks
        if self.model_family not in {"gbm", "linear", "deep"}:
            raise ValueError("model_family must be one of: 'gbm', 'linear', 'deep'")

        if self.model_family == "deep":
            if not isinstance(X_train, np.ndarray) or X_train.ndim != 3:
                raise ValueError("For 'deep', X_train must be ndarray of shape (n, L, F).")
            if not isinstance(X_val, np.ndarray) or X_val.ndim != 3:
                raise ValueError("For 'deep', X_val must be ndarray of shape (n, L, F).")

    # ------------------------------------------------------------------
    # 1) SHAP GLOBAL
    # ------------------------------------------------------------------
    def compute_shap_global(self, max_samples: int = 1000) -> Dict[str, Any]:
        """
        Fit a SHAP explainer and compute global shap values on X_val.

        Returns
        -------
        dict with keys: 'shap_values', 'expected_value'
        """
        # ===== TABULAR FAMILIES (GBM / LINEAR) =====
        if self.model_family in {"gbm", "linear"}:
            X_train_arr = (
                self.X_train.values if isinstance(self.X_train, pd.DataFrame) else np.asarray(self.X_train)
            )
            X_val_arr = (
                self.X_val.values if isinstance(self.X_val, pd.DataFrame) else np.asarray(self.X_val)
            )

            # subsample val for speed if needed
            if len(X_val_arr) > max_samples:
                X_val_arr = X_val_arr[:max_samples]

            if self.model_family == "gbm":
                self.explainer = shap.TreeExplainer(self.model)
            else:
                self.explainer = shap.LinearExplainer(self.model, X_train_arr)

            shap_vals = self.explainer.shap_values(X_val_arr)

            # For regression, SHAP can return:
            #  - array (n_samples, n_features)
            #  - or [array (n_samples, n_features)]
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]

            self.shap_values = np.asarray(shap_vals)
            self.expected_value = self.explainer.expected_value

            return {"shap_values": self.shap_values, "expected_value": self.expected_value}

        # ===== DEEP FAMILY (SEQUENCE MODELS) =====
        X_train_seq: np.ndarray = np.asarray(self.X_train)
        X_val_seq: np.ndarray = np.asarray(self.X_val)

        n_train, L, F = X_train_seq.shape
        n_val = X_val_seq.shape[0]

        # background sample for GradientExplainer
        bg_size = min(self.background_size, n_train)
        background = X_train_seq[:bg_size] 

        self.explainer = shap.GradientExplainer(self.model, background)
        n_use = min(max_samples, n_val)
        X_val_sub = X_val_seq[:n_use]    

        shap_vals = self.explainer.shap_values(X_val_sub)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        sv = np.asarray(shap_vals)

        if sv.ndim == 4:
            if sv.shape[-1] == 1:
                sv = sv[..., 0]
            else:
                sv = np.mean(sv, axis=-1) 

        if sv.ndim != 3:
            raise ValueError(f"Unexpected SHAP shape for deep model: {sv.shape}")

        self.shap_values = sv 

        self.expected_value = float(
            np.mean(self.model.predict(background, verbose=0))
        )

        return {"shap_values": self.shap_values, "expected_value": self.expected_value}

    # ------------------------------------------------------------------
    # 2) PLOTS – GLOBAL
    # ------------------------------------------------------------------
    def _get_feature_names_for_global(self) -> List[str]:
        """Returns sensible feature names for global plots."""
        if self.model_family in {"gbm", "linear"}:
            if self.feature_names is not None:
                return self.feature_names
            if isinstance(self.X_val, pd.DataFrame):
                return list(self.X_val.columns)
            return [f"f{i}" for i in range(self.X_val.shape[1])]
        _, L, F = self.X_val.shape
        if self.feature_names is not None and len(self.feature_names) == F:
            return self.feature_names
        return [f"feat_{i}" for i in range(F)]

    def _prepare_global_shap_for_deep(self):
        """
        For deep models we aggregate shap values over time dimension so we
        can use standard SHAP summary plots per feature.
        """
        assert self.model_family == "deep", "Only for deep models."
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_global() first for deep model.")

        sv = np.asarray(self.shap_values)
        if sv.ndim != 3:
            raise ValueError(f"Expected deep SHAP values with 3 dims, got {sv.shape}")

        n_val, L, F = sv.shape
        shap_feat = np.mean(np.abs(sv), axis=1) 


        X_seq = np.asarray(self.X_val)[:n_val] 
        X_feat = np.mean(X_seq, axis=1)         

        return shap_feat, X_feat

    def plot_shap_summary(self, max_display: int = 15):
        """Global SHAP beeswarm summary plot."""
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_global() first.")

        feature_names = self._get_feature_names_for_global()

        if self.model_family in {"gbm", "linear"}:
            X_val_arr = (
                self.X_val.values if isinstance(self.X_val, pd.DataFrame) else np.asarray(self.X_val)
            )

            n = self.shap_values.shape[0]
            shap.summary_plot(
                self.shap_values,
                X_val_arr[:n],
                feature_names=feature_names,
                max_display=max_display,
            )
        else:  
            shap_feat, X_feat = self._prepare_global_shap_for_deep()
            shap.summary_plot(
                shap_feat,
                X_feat,
                feature_names=feature_names,
                max_display=max_display,
            )

    def plot_shap_bar(self, max_display: int = 15):
        """Global SHAP bar plot."""
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_global() first.")

        feature_names = self._get_feature_names_for_global()

        if self.model_family in {"gbm", "linear"}:
            X_val_arr = (
                self.X_val.values if isinstance(self.X_val, pd.DataFrame) else np.asarray(self.X_val)
            )
            n = self.shap_values.shape[0]
            shap.summary_plot(
                self.shap_values,
                X_val_arr[:n],
                feature_names=feature_names,
                plot_type="bar",
                max_display=max_display,
            )
        else:
            shap_feat, X_feat = self._prepare_global_shap_for_deep()
            shap.summary_plot(
                shap_feat,
                X_feat,
                feature_names=feature_names,
                plot_type="bar",
                max_display=max_display,
            )

    # ------------------------------------------------------------------
    # 3) LOCAL EXPLANATION
    # ------------------------------------------------------------------
    def plot_local_instance(self, idx: int = -1):
        if self.shap_values is None or self.expected_value is None:
            raise RuntimeError("Call compute_shap_global() first.")

        if idx < 0:
            idx = self.shap_values.shape[0] + idx

        feature_names = self._get_feature_names_for_global()

        if self.model_family in {"gbm", "linear"}:
            X_val_arr = (
                self.X_val.values if isinstance(self.X_val, pd.DataFrame) else np.asarray(self.X_val)
            )
            shap.force_plot(
                self.expected_value,
                self.shap_values[idx, :],
                X_val_arr[idx, :],
                feature_names=feature_names,
                matplotlib=True,
            )
        else:
            sv = np.asarray(self.shap_values)
            if sv.ndim != 3:
                raise ValueError(f"Expected deep SHAP values with 3 dims, got {sv.shape}")

            seq_shap = sv[idx]                  
            sv_feat = np.mean(seq_shap, axis=0)

            x_seq = np.asarray(self.X_val)[idx]  
            x_feat = np.mean(x_seq, axis=0)    

            shap.force_plot(
                self.expected_value,
                sv_feat,
                x_feat,
                feature_names=feature_names,
                matplotlib=True,
            )

    def plot_deep_heatmap(self, idx: int = -1):

        if self.model_family != "deep":
            raise RuntimeError("plot_deep_heatmap is only for deep models.")
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_global() first.")

        sv = np.asarray(self.shap_values)
        if sv.ndim != 3:
            raise ValueError(f"Expected deep SHAP values with 3 dims, got {sv.shape}")

        if idx < 0:
            idx = sv.shape[0] + idx

        seq_shap = sv[idx]          # (L, F)
        L, F = seq_shap.shape

        feat_names = self._get_feature_names_for_global()
        if len(feat_names) != F:
            feat_names = [f"feat_{i}" for i in range(F)]

        plt.figure(figsize=(8, 4))
        plt.imshow(seq_shap.T, aspect="auto", cmap="RdBu_r")
        plt.colorbar(label="SHAP value")
        plt.yticks(range(F), feat_names)
        plt.xlabel("Lookback time step")
        plt.title(f"{self.task_name} – SHAP heatmap (sample {idx})")
        plt.tight_layout()
        plt.show()
    # ------------------------------------------------------------------
    # 4) PERMUTATION IMPORTANCE (TABULAR + DEEP)
    # ------------------------------------------------------------------
    def compute_permutation_importance(
        self, n_repeats: int = 10, random_state: int = 42
    ) -> pd.DataFrame:
        rng = np.random.RandomState(random_state)

        # ------------------------------
        # TABULAR CASE (unchanged)
        # ------------------------------
        if self.model_family in {"gbm", "linear"}:
            X_val_df = (
                self.X_val
                if isinstance(self.X_val, pd.DataFrame)
                else pd.DataFrame(self.X_val)
            )
            if self.y_val is None:
                raise RuntimeError("y_val is required for permutation importance.")

            scoring = lambda est, X, y: -mean_squared_error(y, est.predict(X))

            r = permutation_importance(
                self.model,
                X_val_df,
                self.y_val,
                n_repeats=n_repeats,
                random_state=random_state,
                scoring=scoring,
            )

            df_imp = pd.DataFrame(
                {
                    "feature": X_val_df.columns,
                    "importance_mean": r.importances_mean,
                    "importance_std": r.importances_std,
                }
            ).sort_values("importance_mean", ascending=False)

            return df_imp.reset_index(drop=True)

        # ------------------------------
        # DEEP CASE (sequence models)
        # ------------------------------
        if self.model_family != "deep":
            raise RuntimeError(
                "compute_permutation_importance supports only 'gbm', 'linear', or 'deep'."
            )

        if self.y_val is None:
            raise RuntimeError("y_val is required for permutation importance (deep).")

        X_val_seq = np.asarray(self.X_val)
        y_val = np.asarray(self.y_val).reshape(-1)

        if X_val_seq.ndim != 3:
            raise ValueError(
                f"For deep models, X_val must have shape (n, L, F). Got {X_val_seq.shape}"
            )

        n_samples, L, F = X_val_seq.shape
        if len(y_val) != n_samples:
            raise ValueError(
                f"X_val and y_val must have same number of samples. "
                f"Got X_val={n_samples}, y_val={len(y_val)}."
            )

        # baseline performance
        y_pred_base = self.model.predict(X_val_seq, verbose=0).reshape(-1)
        base_mse = mean_squared_error(y_val, y_pred_base)

        # feature names 
        if self.feature_names is not None and len(self.feature_names) == F:
            feat_names = self.feature_names
        else:
            feat_names = [f"feat_{i}" for i in range(F)]

        importances = np.zeros((F, n_repeats), dtype=float)

        for j in range(F):
            for r_idx in range(n_repeats):
                X_perm = X_val_seq.copy()
                perm = rng.permutation(n_samples)
                X_perm[:, :, j] = X_perm[perm, :, j]

                y_pred_perm = self.model.predict(X_perm, verbose=0).reshape(-1)
                mse_perm = mean_squared_error(y_val, y_pred_perm)

                importances[j, r_idx] = mse_perm - base_mse

        imp_mean = importances.mean(axis=1)
        imp_std = importances.std(axis=1)

        df_imp = pd.DataFrame(
            {
                "feature": feat_names,
                "importance_mean": imp_mean,
                "importance_std": imp_std,
            }
        ).sort_values("importance_mean", ascending=False)

        return df_imp.reset_index(drop=True)


