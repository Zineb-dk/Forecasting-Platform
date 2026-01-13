from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


@dataclass
class PredictionConfig:
    time_col: str
    target_col: str
    is_multi_entity: bool = False
    entity_col: Optional[str] = None
    horizon: int = 1

    mode: str = "multi_step"     # "one_step" | "multi_step"
    steps: int = 1               # used if multi_step

    entity_scope: str = "one"    # "one" | "all"
    entity_value: Optional[Union[str, int, float]] = None

    # feature control
    feature_cols: Optional[List[str]] = None
    max_entities: int = 200      # safety cap
    
    # NEW: for sequence models
    lookback: int = 30
    model_type: str = "tabular"  # "tabular" or "seq"


class PredictX:
    """
    PredictX handles predictions for both tabular and sequence models.
    
    For sequence models:
    - Builds lookback windows from the most recent data
    - Applies feature/target scaling if scalers are provided
    - Returns predictions in original scale
    """

    def __init__(
        self, 
        *, 
        model: Any, 
        cfg: PredictionConfig, 
        training_report: Optional[Dict[str, Any]] = None,
        feature_scaler: Optional[StandardScaler] = None,
        target_scaler: Optional[StandardScaler] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.training_report = training_report or {}
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        
        if self.cfg.model_type == "tabular":
            if hasattr(model, 'predict') and 'keras' in str(type(model)).lower():
                self.cfg.model_type = "seq"

    def validate(self, df: pd.DataFrame) -> None:
        c = self.cfg
        if c.time_col not in df.columns:
            raise ValueError(f"time_col '{c.time_col}' not found")
        if c.target_col not in df.columns:
            raise ValueError(f"target_col '{c.target_col}' not found")

        if c.is_multi_entity:
            if not c.entity_col:
                raise ValueError("entity_col is required for multi-entity dataset")
            if c.entity_col not in df.columns:
                raise ValueError(f"entity_col '{c.entity_col}' not found")

        c.horizon = int(c.horizon) if c.horizon else 1
        if c.mode == "one_step":
            c.steps = 1
        else:
            c.steps = max(1, min(int(c.steps), c.horizon))

        if c.is_multi_entity and c.entity_scope == "one":
            if c.entity_value is None or str(c.entity_value) == "":
                raise ValueError("entity_value required when entity_scope='one'")
            
            entity_col = c.entity_col
            if entity_col:
                available = df[entity_col].astype(str).unique().tolist()
                if str(c.entity_value) not in available:
                    raise ValueError(
                        f"Entity '{c.entity_value}' not found. "
                        f"Available entities: {', '.join(available[:10])}"
                        f"{' (showing first 10)' if len(available) > 10 else ''}"
                    )

    def _infer_feature_cols(self, df: pd.DataFrame) -> List[str]:
        fc = self.training_report.get("feature_columns") or self.training_report.get("train_features")
        if isinstance(fc, list) and fc:
            fc2 = [str(x) for x in fc if str(x) in df.columns]
            if fc2:
                return fc2
        exclude = {self.cfg.target_col, self.cfg.time_col}
        if self.cfg.is_multi_entity and self.cfg.entity_col:
            exclude.add(self.cfg.entity_col)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in num_cols if c not in exclude]

    def _sort_df(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg
        if c.is_multi_entity and c.entity_col:
            return df.sort_values([c.entity_col, c.time_col]).reset_index(drop=True)
        return df.sort_values(c.time_col).reset_index(drop=True)

    def _predict_one_row_tabular(self, Xrow: np.ndarray) -> float:
        """Predict a single row for tabular models."""
        # XGBoost Booster
        if self.model.__class__.__name__.lower() == "booster":
            import xgboost as xgb
            d = xgb.DMatrix(Xrow)
            y = self.model.predict(d)
            return float(np.ravel(y)[0])

        # Sklearn-like
        if hasattr(self.model, "predict"):
            y = self.model.predict(Xrow)
            return float(np.ravel(y)[0])

        raise ValueError(f"Unsupported model type: {type(self.model)}")

    def _build_sequence_window(
        self, 
        df_sorted: pd.DataFrame, 
        feature_cols: List[str]
    ) -> np.ndarray:
        """
        Build a sequence window from the most recent data.
        Returns: (1, lookback, n_features) array.
        """
        lookback = self.cfg.lookback
        
        # Get last lookback rows
        df_window = df_sorted.tail(lookback)
        
        if len(df_window) < lookback:
            raise ValueError(
                f"Need at least {lookback} rows for sequence prediction. "
                f"Got {len(df_window)} rows."
            )
        
        # Extract features
        X = df_window[feature_cols].apply(pd.to_numeric, errors='coerce').values
        
        # Apply feature scaling if available
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)
        
        # Shape: (1, lookback, n_features)
        return X.astype(np.float32).reshape(1, lookback, -1)

    def _predict_sequence(self, X_seq: np.ndarray) -> float:
        """
        Predict from a sequence window.
        Returns prediction in original scale.
        """
        if hasattr(self.model, 'predict') and 'keras' in str(type(self.model)).lower():
            y_scaled = self.model.predict(X_seq, verbose=0)
            y_value = float(np.ravel(y_scaled)[0])

            if self.target_scaler is not None:
                y_value = float(
                    self.target_scaler.inverse_transform([[y_value]])[0, 0]
                )
            
            return y_value
        
        raise ValueError(f"Unsupported sequence model type: {type(self.model)}")

    def predict_for_entity_df(
        self, 
        df_ent: pd.DataFrame, 
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Predict for a single entity's data."""
        
        if self.cfg.model_type == "seq":
            try:
                X_seq = self._build_sequence_window(df_ent, feature_cols)
            except ValueError as e:
                return {
                    "error": str(e),
                    "y_pred": None,
                }
            
            if self.cfg.mode == "one_step":
                y1 = self._predict_sequence(X_seq)
                return {"y_pred": y1}
            
            # Multi-step: predict iteratively
            # This is a simplified version. For production, we'll
            # implement later a proper autoregressive prediction with feature updates
            preds: List[float] = []
            for _ in range(self.cfg.steps):
                y = self._predict_sequence(X_seq)
                preds.append(y)
                # Then we can update X_seq with new prediction for next step
                # For now, we just repeat with same window 
                # should be done in v2
            
            return {"y_pred": preds}
        
        else:
            last = df_ent.tail(1).copy()
            X = last[feature_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)

            if self.cfg.mode == "one_step":
                y1 = self._predict_one_row_tabular(X)
                return {"y_pred": y1}

            preds: List[float] = []
            for _ in range(self.cfg.steps):
                preds.append(self._predict_one_row_tabular(X))
            return {"y_pred": preds}

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.validate(df)
        df = self._sort_df(df)

        feature_cols = self.cfg.feature_cols or self._infer_feature_cols(df)
        if not feature_cols:
            raise ValueError("No feature columns found")

        c = self.cfg

        # Single-entity dataset
        if not c.is_multi_entity:
            result = self.predict_for_entity_df(df, feature_cols)
            return {
                "mode": c.mode,
                "steps": c.steps,
                "model_type": c.model_type,
                "feature_columns": feature_cols,
                "lookback": c.lookback if c.model_type == "seq" else None,
                "result": result,
            }

        # Multi-entity dataset
        ent_col = c.entity_col
        assert ent_col is not None

        if c.entity_scope == "one":
            df_ent = df[df[ent_col].astype(str) == str(c.entity_value)]
            if df_ent.empty:
                raise ValueError(f"Entity '{c.entity_value}' not found")
            
            result = self.predict_for_entity_df(df_ent, feature_cols)
            return {
                "mode": c.mode,
                "steps": c.steps,
                "model_type": c.model_type,
                "entity_column": ent_col,
                "entity_value": c.entity_value,
                "feature_columns": feature_cols,
                "lookback": c.lookback if c.model_type == "seq" else None,
                "result": result,
            }

        # All entities (cap)
        uniq = df[ent_col].dropna().astype(str).unique().tolist()
        uniq = uniq[: c.max_entities]

        results: Dict[str, Any] = {}
        for ev in uniq:
            df_ent = df[df[ent_col].astype(str) == ev]
            results[ev] = self.predict_for_entity_df(df_ent, feature_cols)

        return {
            "mode": c.mode,
            "steps": c.steps,
            "model_type": c.model_type,
            "entity_column": ent_col,
            "feature_columns": feature_cols,
            "lookback": c.lookback if c.model_type == "seq" else None,
            "results": results,
            "capped_entities": len(uniq),
        }