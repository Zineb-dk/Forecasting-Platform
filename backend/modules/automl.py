from dataclasses import dataclass
from typing import Dict, Callable, Tuple, List, Optional, Any


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use("Agg")  

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models

from modules.data_forge import DataForge, DataConfig

@dataclass
class AutoMLConfig:
    models_to_train: List[str]
    primary_metric: str = "RMSE"    
    test_size: float = 0.2       
    clip_threshold: Optional[float] = None  
    top_k: int = 2                
    n_splits: int = 5             
    lookback: int = 30             
    epochs: int = 50
    batch_size: int = 32
    do_plots: bool = True

    def __post_init__(self):
        if not self.models_to_train:
            raise ValueError(
                "AutoMLConfig.models_to_train must contain at least one model. "
                "You must explicitly choose which models to train."
            )

        valid_metrics = {"RMSE", "MAE", "R2", "MAPE", "sMAPE"}
        if self.primary_metric not in valid_metrics:
            raise ValueError(
                f"primary_metric must be one of {valid_metrics}, "
                f"got '{self.primary_metric}'."
            )

        if not (0 < self.test_size < 1):
            raise ValueError("test_size must be between 0 and 1.")


class AutoMLPipeline:
    # registry of ALL models this pipeline knows how to build   
    TABULAR_MODEL_BUILDERS = {
        "Random Forest": lambda self: self.build_random_forest(),
        "XGBoost":       lambda self: self.build_xgboost(),
        # later: "LightGBM", "Linear",...
    }

    # All the sequence (deep) models
    SEQ_MODEL_BUILDERS = {
        "LSTM": lambda self, input_shape: self.build_lstm(input_shape),
        "TCN":  lambda self, input_shape: self.build_tcn(input_shape),
        "TFT":  lambda self, input_shape: self.build_tft(input_shape),
    }

    def __init__(
        self,
        data_cfg: DataConfig,
        automl_cfg: AutoMLConfig,
    ):
        self.cfg = data_cfg
        self.automl_cfg = automl_cfg
        self.clip_threshold = automl_cfg.clip_threshold
        self.test_size = automl_cfg.test_size
        self.do_plots= automl_cfg.do_plots

        self.models_to_train = automl_cfg.models_to_train
        self.primary_metric = automl_cfg.primary_metric
        self.feature_scaler_ = None  # will be fitted during sequence building
        self.target_scaler_ = None   # will be fitted during sequence building

        # validate that the requested models exist
        unknown = [
            name for name in self.models_to_train
            if name not in self.TABULAR_MODEL_BUILDERS
            and name not in self.SEQ_MODEL_BUILDERS
        ]
        if unknown:
            raise ValueError(
                f"Unknown model(s) in AutoMLConfig.models_to_train: {unknown}. "
                f"Available tabular: {list(self.TABULAR_MODEL_BUILDERS.keys())}, "
                f"sequence: {list(self.SEQ_MODEL_BUILDERS.keys())}"
            )

        self.tabular_models = [
            name for name in self.models_to_train
            if name in self.TABULAR_MODEL_BUILDERS
        ]
        self.seq_models = [
            name for name in self.models_to_train
            if name in self.SEQ_MODEL_BUILDERS
        ]

        if not self.tabular_models and not self.seq_models:
            raise ValueError("No valid models selected in models_to_train.")


        # will be filled after run_benchmark()
        self.results_: Optional[Dict[str, Dict[str, Any]]] = None
        self.best_model_name_: Optional[str] = None
        self.best_model_ = None
        self.meta_: Optional[Dict[str, Any]] = None

        self.plot_payload_: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # SMALL PLOTTING HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _to_1d_array(arr) -> np.ndarray:
        """Ensure a 1D numpy array."""
        return np.asarray(arr).reshape(-1)

    @staticmethod
    def _downsample_indices(n: int, max_points: Optional[int]):
        """
        Return indices for downsampling:
        - if max_points is None or n <= max_points: slice(None)
        - else: int index array of length max_points
        """
        if max_points is None or max_points >= n:
            return slice(None)
        return np.linspace(0, n - 1, max_points).astype(int)

    # ------------------------------------------------------------------
    # DATA PREPARATION
    # ------------------------------------------------------------------
    def prepare_X_y(
        self,
        df_clean: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Build X, y from cleaned dataframe + config.
        Optionally clip the target at self.clip_threshold.
        """
        cfg = self.cfg
        df = df_clean.copy()

        if self.clip_threshold is not None:
            df["target_clipped"] = df[cfg.target_col].clip(upper=self.clip_threshold)
            y = df["target_clipped"]
            helper_col = "target_clipped"
        else:
            y = df[cfg.target_col]
            helper_col = None

        cols_to_remove = [cfg.time_col, cfg.target_col]
        if cfg.is_multi_entity and cfg.entity_col is not None:
            cols_to_remove.append(cfg.entity_col)
        if helper_col is not None:
            cols_to_remove.append(helper_col)

        if helper_col is not None:
            cols_to_remove.append(helper_col)
            
        train_features = df.columns.difference(cols_to_remove)
        X = df[train_features]

        meta = {
            "clip_threshold": self.clip_threshold,
            "helper_col": helper_col,
            "train_features": list(train_features),
            "n_samples": len(df),
            "n_features": len(train_features),
        }
        self.meta_ = meta
        return X, y, meta

    def time_train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Simple chronological split (no shuffle).
        Assumes df is already sorted by time.
        """
        test_size = self.test_size
        n = len(X)
        if len(y) != n:
            raise ValueError("X and y must have the same length.")
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1.")

        n_test = int(np.floor(n * test_size))
        n_train = n - n_test

        X_train = X.iloc[:n_train]
        y_train = y.iloc[:n_train]
        X_test = X.iloc[n_train:]
        y_test = y.iloc[n_train:]

        return X_train, X_test, y_train, y_test

    def build_sequences_from_df(
        self,
        df_clean: pd.DataFrame,
        scale_features: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Build (X_seq, y_seq, y_seq_original) from df_clean using a sliding window.
        
        Returns
        -------
        X_seq : np.ndarray, shape (n_samples, lookback, n_features)
            Scaled features if scale_features=True.
        y_seq : np.ndarray, shape (n_samples,)
            Scaled targets if scale_features=True
        y_seq_original : np.ndarray or None, shape (n_samples,)
            Original unscaled targets 
            None if scale_features=False.
        """
        lookback = self.automl_cfg.lookback
        cfg = self.cfg

        if cfg.time_col not in df_clean.columns:
            raise ValueError(f"time_col '{cfg.time_col}' not found in df_clean.")
        
        time_series = df_clean[cfg.time_col]
        if np.issubdtype(time_series.dtype, np.number):
            time_kind = "numeric"
        elif np.issubdtype(time_series.dtype, np.datetime64):
            time_kind = "datetime"
        else:
            raise ValueError(
                f"time_col '{cfg.time_col}' must be numeric or datetime, "
                f"got dtype: {time_series.dtype}"
            )
        
        if cfg.is_multi_entity and cfg.entity_col is not None:
            if cfg.entity_col not in df_clean.columns:
                raise ValueError(f"entity_col '{cfg.entity_col}' not found in df_clean.")
            df_sorted = df_clean.sort_values([cfg.entity_col, cfg.time_col]).reset_index(drop=True)
        else:
            df_sorted = df_clean.sort_values(cfg.time_col).reset_index(drop=True)
        
        X_tab, y, meta = self.prepare_X_y(df_sorted)
        
        y_original = np.asarray(y).reshape(-1).astype("float32")
        
        if scale_features:
            print(f"[Sequence Building] Scaling features and target for neural networks...")
            
            self.feature_scaler_ = StandardScaler()
            X_scaled = self.feature_scaler_.fit_transform(X_tab.values)
            
            self.target_scaler_ = StandardScaler()
            y_scaled = self.target_scaler_.fit_transform(y.values.reshape(-1, 1)).reshape(-1)
            
            X_vals = X_scaled.astype("float32")
            y_vals = y_scaled.astype("float32")
            
            print(f"  Feature scale: mean={X_vals.mean():.4f}, std={X_vals.std():.4f}")
            print(f"  Target scale:  mean={y_vals.mean():.4f}, std={y_vals.std():.4f}")
        else:
            X_vals = X_tab.values.astype("float32")
            y_vals = y_original.copy()

        # Build sequences
        if cfg.is_multi_entity and cfg.entity_col is not None:
            entity_ids = df_sorted[cfg.entity_col].values
            unique_entities = np.unique(entity_ids)
            
            seq_X_list = []
            seq_y_list = []
            seq_y_original_list = [] 
            
            for entity in unique_entities:
                entity_mask = (entity_ids == entity)
                entity_indices = np.where(entity_mask)[0]
                
                entity_X = X_vals[entity_mask]
                entity_y = y_vals[entity_mask]
                entity_y_orig = y_original[entity_mask] 
                
                if len(entity_X) <= lookback:
                    print(f"Warning: Entity '{entity}' has only {len(entity_X)} samples, "
                        f"need > {lookback}. Skipping this entity.")
                    continue
                
                entity_times = df_sorted.loc[entity_indices, cfg.time_col].values
                if not self._is_monotonic_increasing(entity_times):
                    raise ValueError(
                        f"Entity '{entity}' time values are not monotonic increasing "
                        f"even after sorting. Check for duplicate timestamps or data corruption."
                    )
                
                for i in range(lookback, len(entity_X)):
                    seq_X_list.append(entity_X[i - lookback : i, :])
                    seq_y_list.append(entity_y[i])
                    seq_y_original_list.append(entity_y_orig[i])
            
            if len(seq_X_list) == 0:
                raise ValueError(
                    f"No valid sequences created. All entities have <= {lookback} samples. "
                    f"Reduce lookback or use longer entity sequences."
                )
            
            X_seq = np.stack(seq_X_list, axis=0)
            y_seq = np.asarray(seq_y_list, dtype="float32")
            y_seq_original = np.asarray(seq_y_original_list, dtype="float32")   
        
        else:
            time_vals = df_sorted[cfg.time_col].values
            if not self._is_monotonic_increasing(time_vals):
                raise ValueError(
                    f"Time column '{cfg.time_col}' is not monotonic increasing "
                    f"even after sorting. Check for duplicate timestamps or data corruption."
                )
            
            if len(X_vals) <= lookback:
                raise ValueError(
                    f"Dataset has only {len(X_vals)} samples, need > {lookback} for lookback window. "
                    f"Reduce lookback value."
                )
            
            seq_X = []
            seq_y = []
            seq_y_original = [] 

            for i in range(lookback, len(X_vals)):
                seq_X.append(X_vals[i - lookback : i, :])
                seq_y.append(y_vals[i])
                seq_y_original.append(y_original[i])

            X_seq = np.stack(seq_X, axis=0)
            y_seq = np.asarray(seq_y, dtype="float32")
            y_seq_original = np.asarray(seq_y_original, dtype="float32")  

        return X_seq, y_seq, (y_seq_original if scale_features else None)


    @staticmethod
    def _is_monotonic_increasing(arr: np.ndarray) -> bool:
        """
        Check if array is monotonically increasing (allows equal consecutive values).
        Works for both numeric and datetime64 arrays.
        """
        if len(arr) <= 1:
            return True
        return np.all(arr[1:] >= arr[:-1])
    
    def seq_train_test_split(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        if len(X_seq) != len(y_seq):
            raise ValueError("X_seq and y_seq must have the same length.")

        n = len(X_seq)
        n_test = int(np.floor(n * self.test_size))
        n_train = n - n_test

        X_train = X_seq[:n_train]
        y_train = y_seq[:n_train]
        X_test  = X_seq[n_train:]
        y_test  = y_seq[n_train:]

        return X_train, X_test, y_train, y_test


    # TABULAR MODEL BUILDERS
    @staticmethod
    def build_random_forest() -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

    @staticmethod
    def build_xgboost() -> XGBRegressor:
        return XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror",
        )

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------
    @staticmethod
    def regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        denom = np.clip(np.abs(y_true), 1e-8, None)
        mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

        smape = np.mean(
            2.0 * np.abs(y_true - y_pred)
            / np.clip(np.abs(y_true) + np.abs(y_pred), 1e-8, None)
        ) * 100.0

        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
            "MAPE": float(mape),
            "sMAPE": float(smape),
        }

    def _metric_for_sort(self, metrics: Dict[str, float]) -> float:
        """
        Returns a scalar key for sorting according to self.primary_metric.
        """
        val = metrics[self.primary_metric]
        if self.primary_metric in {"RMSE", "MAE", "MAPE", "sMAPE"}:
            return val
        elif self.primary_metric == "R2":
            return -val
        else:
            return val

    # MODEL BENCHMARKING
    def benchmark_tabular_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Dict[str, Any]]:

        results: Dict[str, Dict[str, Any]] = {}

        print("\n--- Tabular Model Benchmarking ---")
        for name in self.tabular_models:
            builder = self.TABULAR_MODEL_BUILDERS[name]
            model = builder(self) 

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            m = self.regression_metrics(y_test.values, preds)

            print(f"\nModel: {name}")
            print(f"  RMSE:  {m['RMSE']:.2f}")
            print(f"  MAE:   {m['MAE']:.2f}")
            print(f"  R2:    {m['R2']:.4f}")
            print(f"  MAPE:  {m['MAPE']:.2f}%")
            print(f"  sMAPE: {m['sMAPE']:.2f}%")

            results[name] = {
                "model": model,
                "metrics": m,
                "y_pred": preds,
            }

        return results

    def benchmark_seq_models(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate deep sequence models (LSTM, TCN, TFT)
        using a simple chronological train/val split.
        """
        if not self.seq_models:
            return {}

        X_train, X_val, y_train, y_val = self.seq_train_test_split(X_seq, y_seq)

        input_shape = X_train.shape[1:]
        results: Dict[str, Dict[str, Any]] = {}

        print("\n--- Sequence Model Benchmarking ---")
        for name in self.seq_models:
            builder = self.SEQ_MODEL_BUILDERS[name]
            model = builder(self, input_shape)

            res = self.train_deep_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=self.automl_cfg.epochs,
                batch_size=self.automl_cfg.batch_size,
                target_scaler=self.target_scaler_,
            )

            m = res["metrics"]
            print(f"\nModel: {name}")
            print(f"  RMSE:  {m['RMSE']:.2f}")
            print(f"  MAE:   {m['MAE']:.2f}")
            print(f"  R2:    {m['R2']:.4f}")
            print(f"  MAPE:  {m['MAPE']:.2f}%")
            print(f"  sMAPE: {m['sMAPE']:.2f}%")

            results[name] = res

        return results

    # WALK-FORWARD VALIDATION (TABULAR ONLY)
    def walk_forward_validation(
        self,
        df_clean: pd.DataFrame,
        model_name: str,
        n_splits: int = 5,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Walk-forward validation using TimeSeriesSplit for tabular models only.
        """
        if model_name is None:
            raise ValueError(
                "model_name must be provided explicitly for walk_forward_validation. "
                "This function is only for tabular models (e.g. 'Random Forest'). "
                "For deep models use walk_forward_validation_seq."
            )

        if model_name not in self.tabular_models:
            raise ValueError(
                f"walk_forward_validation is for tabular models only.\n"
                f"Got '{model_name}'. Tabular models selected: {self.tabular_models}"
            )

        if model_name not in self.TABULAR_MODEL_BUILDERS:
            raise ValueError(
                f"Unknown tabular model_name: {model_name}. "
                f"Available tabular: {list(self.TABULAR_MODEL_BUILDERS.keys())}"
            )

        build_model = self.TABULAR_MODEL_BUILDERS[model_name]

        X, y, _ = self.prepare_X_y(df_clean)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        oof_pred = pd.Series(index=y.index, dtype=float)
        fold_results: List[Dict[str, Any]] = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

            model = build_model(self) 
            model.fit(X_tr, y_tr)
            y_hat = model.predict(X_te)
            oof_pred.iloc[test_idx] = y_hat

            m = self.regression_metrics(y_te.values, y_hat)
            fold_results.append(
                {
                    "fold": fold,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    **m,
                }
            )

            print(
                f"[WF Tabular {model_name}] Fold {fold}: "
                f"RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, R2={m['R2']:.3f}"
            )

        metrics_df = pd.DataFrame(fold_results)
        return oof_pred, metrics_df


    # WALK-FORWARD VALIDATION (SEQUENCE / DEEP)
    def walk_forward_validation_seq(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        model_name: str,
        n_splits: int = 5,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Walk-forward validation for deep sequence models (LSTM, TCN, TFT).

        Parameters
        ----------
        X_seq : np.ndarray
            Sequence features, shape (n_samples, lookback, n_features).
        y_seq : np.ndarray
            Targets, shape (n_samples,).
        model_name : str
            One of self.seq_models (e.g. 'LSTM').
        n_splits : int
            Number of TimeSeriesSplit folds.

        Returns
        -------
        oof_pred : np.ndarray
            Out-of-fold predictions, shape (n_samples,).
        metrics_df : pd.DataFrame
            Per-fold metrics.
        """
        if model_name not in self.seq_models:
            raise ValueError(
                f"walk_forward_validation_seq is for sequence models only.\n"
                f"Got '{model_name}'. Sequence models selected: {self.seq_models}"
            )

        if model_name not in self.SEQ_MODEL_BUILDERS:
            raise ValueError(
                f"Unknown sequence model_name: {model_name}. "
                f"Available sequence: {list(self.SEQ_MODEL_BUILDERS.keys())}"
            )

        n_samples = len(X_seq)
        if n_samples != len(y_seq):
            raise ValueError("X_seq and y_seq must have same number of samples.")

        builder = self.SEQ_MODEL_BUILDERS[model_name]
        input_shape = X_seq.shape[1:]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_pred = np.zeros(n_samples, dtype="float32")
        fold_results: List[Dict[str, Any]] = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(np.arange(n_samples)), start=1):
            X_tr, y_tr = X_seq[train_idx], y_seq[train_idx]
            X_te, y_te = X_seq[test_idx], y_seq[test_idx]

            model = builder(self, input_shape)
            res = self.train_deep_model(
                model=model,
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_te,
                y_val=y_te,
                epochs=self.automl_cfg.epochs,
                batch_size=self.automl_cfg.batch_size,
            )

            y_hat = res["y_pred"] 
            oof_pred[test_idx] = y_hat

            m = res["metrics"]
            fold_results.append(
                {
                    "fold": fold,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    **m,
                }
            )

            print(
                f"[WF Seq {model_name}] Fold {fold}: "
                f"RMSE={m['RMSE']:.2f}, MAE={m['MAE']:.2f}, R2={m['R2']:.3f}"
            )

        metrics_df = pd.DataFrame(fold_results)
        return oof_pred, metrics_df


    # VISUALIZATIONS 

    @staticmethod
    def plot_metrics_bar(results, metric: str = "RMSE", title: str = None):
        model_names = list(results.keys())
        values = [results[m]["metrics"][metric] for m in model_names]

        plt.figure(figsize=(8, 4))
        plt.bar(model_names, values, edgecolor="k", alpha=0.8)
        plt.ylabel(metric)
        plt.title(title or f"Model comparison – {metric}")
        plt.xticks(rotation=25)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        # plt.show()

    @staticmethod
    def plot_true_vs_pred_line(
        y_test,
        pred_dict,
        max_points: int = None,
        title_prefix: str = "True vs Predicted"
    ):
        y_test = AutoMLPipeline._to_1d_array(y_test)
        n_models = len(pred_dict)

        idx = AutoMLPipeline._downsample_indices(len(y_test), max_points)
        y_plot = y_test[idx]

        fig, axes = plt.subplots(n_models, 1, figsize=(10, 4 * n_models), sharex=True)
        if n_models == 1:
            axes = np.array([axes])

        for ax, (model_name, preds) in zip(axes, pred_dict.items()):
            preds = AutoMLPipeline._to_1d_array(preds)
            preds_plot = preds[idx]

            ax.plot(y_plot, label="True", linewidth=2)
            ax.plot(preds_plot, label="Predicted", alpha=0.75)
            ax.set_ylabel("Target")
            ax.set_title(f"{title_prefix} – {model_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Sample index")
        plt.tight_layout()

    @staticmethod
    def plot_true_vs_pred_scatter(
        y_test,
        pred_dict,
        max_points: int = 500,
        title_prefix: str = "Actual vs Predicted"
    ):
        y_test = AutoMLPipeline._to_1d_array(y_test)
        n_models = len(pred_dict)

        idx = AutoMLPipeline._downsample_indices(len(y_test), max_points)
        y_small = y_test[idx]

        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = np.array([axes])

        for ax, (model_name, preds) in zip(axes, pred_dict.items()):
            preds = AutoMLPipeline._to_1d_array(preds)
            preds_small = preds[idx]

            ax.scatter(y_small, preds_small, alpha=0.5, edgecolor="k")
            lim_min = min(y_small.min(), preds_small.min())
            lim_max = max(y_small.max(), preds_small.max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=2)

            ax.set_title(f"{title_prefix} – {model_name}")
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

    @staticmethod
    def plot_residuals(
        y_test,
        pred_dict,
        max_points: int = 500,
        title_prefix: str = "Residuals"
    ):
        y_test = AutoMLPipeline._to_1d_array(y_test)
        n_models = len(pred_dict)

        idx = AutoMLPipeline._downsample_indices(len(y_test), max_points)
        y_small = y_test[idx]

        fig, axes = plt.subplots(n_models, 1, figsize=(10, 4 * n_models), sharex=True)
        if n_models == 1:
            axes = np.array([axes])

        for ax, (model_name, preds) in zip(axes, pred_dict.items()):
            preds = AutoMLPipeline._to_1d_array(preds)
            preds_small = preds[idx]
            residuals = y_small - preds_small

            ax.scatter(range(len(residuals)), residuals, alpha=0.5, edgecolor="k")
            ax.axhline(0, color="red", linestyle="--")
            ax.set_title(f"{title_prefix} – {model_name}")
            ax.set_ylabel("Error")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Sample index (downsampled)")
        plt.tight_layout()

    @staticmethod
    def plot_full_diagnostics(
        y_test,
        pred_dict,
        max_points: int = 500,
    ):
        """
        High-level wrapper that calls:
        - plot_true_vs_pred_line
        - plot_true_vs_pred_scatter
        - plot_residuals
        on the same data.
        """
        AutoMLPipeline.plot_true_vs_pred_line(
            y_test=y_test,
            pred_dict=pred_dict,
            max_points=max_points,
            title_prefix="True vs Predicted"
        )

        AutoMLPipeline.plot_true_vs_pred_scatter(
            y_test=y_test,
            pred_dict=pred_dict,
            max_points=max_points,
            title_prefix="Actual vs Predicted"
        )

        AutoMLPipeline.plot_residuals(
            y_test=y_test,
            pred_dict=pred_dict,
            max_points=max_points,
            title_prefix="Residuals"
        )


    def full_diagnostics_tabular(
        self,
        y_true: pd.Series,
        results: Dict[str, Dict[str, Any]],
        max_points: int = 500,
        metric: str = "RMSE",
    ):
        """
        High-level diagnostics for tabular models:
        - bar plot of metric 
        - full diagnostics (line + scatter + residuals)

        Parameters
        ----------
        y_true : array-like
            Ground truth on the evaluation set (e.g. y_test or y for OOF).
        results : dict
            Output of benchmark_models or combined results; each entry:
            { "model": ..., "metrics": {...}, "y_pred": ... }
        max_points : int
            Downsampling for scatter/residuals.
        metric : str
            Metric key to plot on bar chart (e.g. "RMSE").
        """
        if not results:
            print("No tabular results to plot.")
            return

        self.plot_metrics_bar(results, metric=metric)
        pred_dict = {name: res["y_pred"] for name, res in results.items()}
        self.plot_full_diagnostics(y_true, pred_dict, max_points=max_points)


    def full_diagnostics_seq(
        self,
        y_val: np.ndarray,
        seq_results: Dict[str, Dict[str, Any]],
        max_points: int = 500,
        metric: str = "RMSE",
        show_history: bool = True,
    ):
        """
        High-level diagnostics for sequence (deep) models:
        - bar plot of metric across deep models
        - per-model training history
        - multi-model diagnostics (line + scatter + residuals)

        Parameters
        ----------
        y_val : array-like
            Ground truth on the validation set used in seq benchmark.
        seq_results : dict
            Output of benchmark_seq_models:
            { name: { "model": ..., "metrics": {...}, "y_pred": ..., "history": ... } }
        max_points : int
            Downsampling for scatter/residuals.
        metric : str
            Metric key to rank/plot (e.g. "RMSE").
        show_history : bool
            Whether to plot training curves for each model.
        """
        if not seq_results:
            print("No sequence model results to plot.")
            return

        metrics_like = {
            name: {"metrics": res["metrics"]} for name, res in seq_results.items()
        }
        self.plot_metrics_bar(metrics_like, metric=metric, title="Seq models – metrics")

        if show_history:
            for name, res in seq_results.items():
                hist = res.get("history", None)
                if hist is not None:
                    self.plot_training_history(
                        hist, title=f"Training history – {name}"
                    )

        pred_dict = {name: res["y_pred"] for name, res in seq_results.items()}
        self.plot_full_diagnostics(y_val, pred_dict, max_points=max_points)


    @staticmethod
    def plot_walk_forward_metrics(
        metrics_df: pd.DataFrame,
        metric: str = "RMSE",
        title: str = "Walk-forward metrics by fold",
    ):
        """
        Plot a simple line/bar chart of a given metric over folds.
        """
        if metric not in metrics_df.columns:
            raise ValueError(f"Metric '{metric}' not in metrics_df columns: {metrics_df.columns}")

        folds = metrics_df["fold"] if "fold" in metrics_df.columns else np.arange(len(metrics_df)) + 1
        values = metrics_df[metric].values

        plt.figure(figsize=(6, 4))
        plt.plot(folds, values, marker="o")
        plt.title(title)
        plt.xlabel("Fold")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.xticks(folds)
        plt.tight_layout()

    def plot_walk_forward_diagnostics(
        self,
        y_full: np.ndarray,
        oof_pred: pd.Series,
        model_name: str,
        max_points: int = 500,
    ):
        """
        Diagnostics for walk-forward predictions:
        - True vs Pred line (over all samples)
        - Scatter + residuals (via plot_full_diagnostics with one model)
        """
        if isinstance(oof_pred, pd.Series):
            preds = oof_pred.values
        else:
            preds = np.asarray(oof_pred)
        pred_dict = {model_name: preds}
        self.plot_full_diagnostics(y_full, pred_dict, max_points=max_points)

    # DEEP MODELS HELPERS 

    @staticmethod
    def build_lstm(input_shape) -> tf.keras.Model:
        model = models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.LSTM(128, return_sequences=True),
                layers.Dropout(0.3),
                layers.LSTM(64),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        return model

    @staticmethod
    def build_tcn(input_shape) -> tf.keras.Model:
        inputs = layers.Input(shape=input_shape)
        x = inputs

        for d in [1, 2, 4, 8]:
            y = layers.Conv1D(
                filters=64,
                kernel_size=5,
                padding="causal",
                dilation_rate=d,
                activation="relu",
            )(x)
            y = layers.Dropout(0.2)(y)
            if x.shape[-1] != y.shape[-1]:
                x = layers.Conv1D(64, kernel_size=1, padding="same")(x)
            x = layers.Add()([x, y])

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1)(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        return model

    @staticmethod
    def build_tft(input_shape) -> tf.keras.Model:
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(64, activation="relu")(inputs)

        attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization()(x)

        ff = layers.Dense(128, activation="relu")(x)
        ff = layers.Dropout(0.3)(ff)
        ff = layers.Dense(64)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1)(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        return model

    @staticmethod
    def train_deep_model(
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        target_scaler = None, 
    ) -> Dict[str, Any]:

        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1,
        )

        curves = {
            "loss": history.history.get("loss", []),
            "val_loss": history.history.get("val_loss", []),
            "mae": history.history.get("mae", []),
            "val_mae": history.history.get("val_mae", []),
            }

        y_pred_scaled = model.predict(X_val, verbose=0).reshape(-1)
        
        if target_scaler is not None:
            y_val_original = target_scaler.inverse_transform(
                y_val.reshape(-1, 1)
            ).reshape(-1)
            y_pred_original = target_scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).reshape(-1)
        else:
            y_val_original = y_val.reshape(-1)
            y_pred_original = y_pred_scaled
        
        m = AutoMLPipeline.regression_metrics(y_val_original, y_pred_original)

        return {
            "model": model,
            "metrics": m,
            "y_pred": y_pred_original,
            "history": history,
            "curves": curves,
        }

    @staticmethod
    def plot_training_history(
        history: tf.keras.callbacks.History,
        title: str = "Training History",
    ):
        hist = history.history
        plt.figure(figsize=(8, 4))
        plt.plot(hist["loss"], label="Train loss")
        if "val_loss" in hist:
            plt.plot(hist["val_loss"], label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)


    # ------------------------------------------------------------------
    # RUN PIPELINE
    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        df_clean: pd.DataFrame,
        top_k: Optional[int] = None,
        n_splits: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end AutoML run:

        1) Prepare data (tabular + sequences).
        2) Benchmark selected tabular models (train/test split).
        3) Benchmark selected seq models (train/val split on sequences).
        4) Rank all models by self.primary_metric.
        5) Take top-K and run walk-forward (tabular and seq).
        6) Select best by average primary_metric across folds.
        7) Retrain best model on full data.
        8) Optionally plot diagnostics.

        Returns
        -------
        summary : dict with keys:
            - 'tabular_results'
            - 'seq_results'
            - 'all_results'
            - 'wf_results'
            - 'best_model_name'
            - 'best_model_type' ('tabular' or 'seq')
            - 'best_model'
            - 'best_avg_metric'
            - 'primary_metric'
            - 'meta'
        """
        if top_k is None:
            top_k = self.automl_cfg.top_k
        if n_splits is None:
            n_splits = self.automl_cfg.n_splits

        # Tabular data
        X, y, meta = self.prepare_X_y(df_clean)
        self.meta_ = meta

        X_train, X_test, y_train, y_test = self.time_train_test_split(X, y)

        tabular_results: Dict[str, Dict[str, Any]] = {}
        if self.tabular_models:
            tabular_results = self.benchmark_tabular_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

        if tabular_results:
            self.plot_payload_["tabular"] = {
                "y_test": y_test.tolist(),
                "models": {
                    name: {
                        "y_pred": res["y_pred"].tolist(),
                        "metrics": res["metrics"],
                    }
                    for name, res in tabular_results.items()
                },
            }


        # Sequence data
        seq_results: Dict[str, Dict[str, Any]] = {}
        X_seq = y_seq = y_seq_original = None  
        y_val_original_for_plots = None

        if self.seq_models:
            X_seq, y_seq, y_seq_original = self.build_sequences_from_df(
                df_clean, 
                scale_features=True
            )
            
            X_tr_seq, X_val_seq, y_tr_seq, y_val_seq = self.seq_train_test_split(X_seq, y_seq)
            if y_seq_original is not None:
                _, _, _, y_val_original_for_plots = self.seq_train_test_split(
                    X_seq, y_seq_original
                )
            seq_results = self.benchmark_seq_models(X_seq, y_seq)
            self.plot_payload_["training_curves"] = {
                name: (res.get("curves") or {}) for name, res in seq_results.items()
            }

        if seq_results and y_val_original_for_plots is not None:
            self.plot_payload_["seq"] = {
                "y_val": y_val_original_for_plots.tolist(),
                "models": {
                    name: {
                        "y_pred": res["y_pred"].tolist(),
                        "metrics": res["metrics"],
                    }
                    for name, res in seq_results.items()
                },
            }
        if self.do_plots:
            if tabular_results:
                self.full_diagnostics_tabular(
                    y_true=y_test,
                    results=tabular_results,
                    max_points=500,
                    metric=self.primary_metric,
                )
            if seq_results and y_val_original_for_plots is not None:
                self.full_diagnostics_seq(
                    y_val=y_val_original_for_plots,  
                    seq_results=seq_results,
                    max_points=500,
                    metric=self.primary_metric,
                    show_history=True,
                )

        # Rank all models by primary_metric
        all_results: Dict[str, Dict[str, Any]] = {}

        for name, res in tabular_results.items():
            all_results[name] = {
                "type": "tabular",
                "metrics": res["metrics"],
            }
        for name, res in seq_results.items():
            all_results[name] = {
                "type": "seq",
                "metrics": res["metrics"],
            }

        if not all_results:
            raise RuntimeError("No models were benchmarked. Check models_to_train.")

        # sort by primary_metric 
        sorted_models = sorted(
            all_results.items(),
            key=lambda kv: self._metric_for_sort(kv[1]["metrics"]),
        )

        # clip K
        top_k = min(top_k, len(sorted_models))
        candidate_names = [name for name, _ in sorted_models[:top_k]]
        print(f"\nTop-{top_k} models overall by {self.primary_metric}: {candidate_names}")

        # Walk-forward validation

        wf_results: Dict[str, Dict[str, Any]] = {}

        # tabular WF uses df_clean
        for name in candidate_names:
            m_type = all_results[name]["type"]
            print(f"\n>>> Walk-forward validation for model: {name} ({m_type})")

            if m_type == "tabular":
                oof_pred, metrics_df = self.walk_forward_validation(
                    df_clean=df_clean,
                    model_name=name,
                    n_splits=n_splits,
                )
            else:  # seq
                if X_seq is None or y_seq is None:
                    raise RuntimeError("Sequences not built but seq model selected.")
                oof_pred, metrics_df = self.walk_forward_validation_seq(
                    X_seq=X_seq,
                    y_seq=y_seq,
                    model_name=name,
                    n_splits=n_splits,
                )

            avg_val = metrics_df[self.primary_metric].mean()
            wf_results[name] = {
                "type": m_type,
                "oof_pred": oof_pred,
                "metrics_df": metrics_df,
                "avg_metric": float(avg_val),
            }

            if self.do_plots:
                self.plot_walk_forward_metrics(
                    metrics_df,
                    metric=self.primary_metric,
                    title=f"WF {self.primary_metric} per fold – {name}",
                )

                y_full = y.values if m_type == "tabular" else y_seq
                self.plot_walk_forward_diagnostics(
                    y_full=y_full,
                    oof_pred=oof_pred,
                    model_name=name,
                    max_points=500,
                )
        if wf_results:
            self.plot_payload_["walk_forward"] = {
                name: {
                    "type": info["type"],
                    "avg_metric": info["avg_metric"],
                    "folds": info["metrics_df"].to_dict(orient="records"),
                }
                for name, info in wf_results.items()
            }

        # Pick best by WF avg primary_metric
        def wf_sort_key(item):
            name, info = item
            avg = info["avg_metric"]
            if self.primary_metric in {"RMSE", "MAE", "MAPE", "sMAPE"}:
                return avg
            elif self.primary_metric == "R2":
                return -avg
            else:
                return avg

        best_name, best_info = min(wf_results.items(), key=wf_sort_key)
        best_type = best_info["type"]
        best_avg_metric = best_info["avg_metric"]

        print(
            f"\nBest model by WF avg {self.primary_metric}: "
            f"{best_name} ({best_type}) = {best_avg_metric:.3f}"
        )

        # Retrain best model on full data
        if best_type == "tabular":
            builder = self.TABULAR_MODEL_BUILDERS[best_name]
            best_model = builder(self)
            best_model.fit(X, y)
        else:
            if X_seq is None or y_seq is None:
                raise RuntimeError("Sequences not built but best model is seq.")
            builder = self.SEQ_MODEL_BUILDERS[best_name]
            input_shape = X_seq.shape[1:]
            best_model = builder(self, input_shape)

            val_split = 0.1
            split_idx = int(len(X_seq) * (1 - val_split))

            X_train_final = X_seq[:split_idx]
            y_train_final = y_seq[:split_idx]
            X_val_final = X_seq[split_idx:]
            y_val_final = y_seq[split_idx:]
            final_res = self.train_deep_model(
                model=best_model,
                X_train=X_train_final,
                y_train=y_train_final,
                X_val=X_val_final,
                y_val=y_val_final,
                epochs=self.automl_cfg.epochs,
                batch_size=self.automl_cfg.batch_size,
                target_scaler=self.target_scaler_, 
            )

            self.plot_payload_["training_curves_best"] = final_res.get("curves") or {}

        self.best_model_name_ = best_name
        self.best_model_ = best_model
        
        self.results_ = all_results

        summary = {
            "tabular_results": tabular_results,
            "seq_results": seq_results,
            "all_results": all_results,
            "wf_results": wf_results,
            "y_val_original": y_val_original_for_plots, 
            "best_model_name": best_name,
            "best_model_type": best_type,
            "best_model": best_model,
            "best_avg_metric": best_avg_metric,
            "primary_metric": self.primary_metric,
            "meta": meta,
            "plot_payload": self.plot_payload_,
        }

        if best_type == "seq" and self.feature_scaler_ is not None and self.target_scaler_ is not None:
            summary["feature_scaler"] = self.feature_scaler_
            summary["target_scaler"] = self.target_scaler_

        return summary
