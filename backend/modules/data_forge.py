#data_forge.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from dateutil import parser
from sklearn.preprocessing import StandardScaler  


# --- Configuration ---
@dataclass
class DataConfig:
    """
    Generic configuration for a time-series dataset.

    - time_col: name of the time axis (datetime, cycles, or integer index)
    - target_col: name of the column to predict 
    - is_multi_entity: True if you have multiple machines / engines / units
    - sensor_cols: list of sensor feature columns
    - entity_col: required if is_multi_entity=True 
    - horizon_steps: how many steps ahead the model will predict
    """
    # mandatory
    time_col: str
    target_col: str
    is_multi_entity: bool
    sensor_cols: List[str] 

    # optional
    entity_col: Optional[str] = None
    horizon_steps: int = 1
    freq_tolerance: float = 0.95


class DataForge:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    # DATA OVERVIEW DICTIONARY
    def get_data_overview(self,df: pd.DataFrame) -> Dict[str, Any]:
        """
        Return a dictionary with the main information we need
        about a dataset, given its DatasetConfig.
        """
        cfg = self.cfg
        info: Dict[str, Any] = {}
        n_rows, n_cols = df.shape

        # Basic structure
        info["n_rows"] = n_rows
        info["n_cols"] = n_cols
        info["columns"] = list(df.columns)
        info["dtypes"] = df.dtypes.astype(str).to_dict()

        # Missing values
        missing_counts = df.isna().sum()
        info["missing_counts"] = missing_counts.to_dict()
        info["missing_ratio"] = (
            (missing_counts / n_rows).to_dict() if n_rows > 0 else {}
        )

        # Constant columns
        nunique = df.nunique(dropna=False)
        const_cols = nunique[nunique <= 1].index.tolist()
        info["constant_columns"] = const_cols

        # Column types
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        info["numeric_columns"] = numeric_cols
        info["datetime_columns"] = datetime_cols
        info["categorical_columns"] = cat_cols

        # Target stats
        if cfg.target_col in df.columns:
            target_series = df[cfg.target_col]
            info["target_dtype"] = str(target_series.dtype)
            info["target_missing"] = int(target_series.isna().sum())
            if pd.api.types.is_numeric_dtype(target_series):
                info["target_describe"] = target_series.describe().to_dict()
            else:
                info["target_describe"] = None
        else:
            info["target_dtype"] = None
            info["target_missing"] = None
            info["target_describe"] = None

        # Sensor stats
        existing_sensors = [c for c in cfg.sensor_cols if c in df.columns]
        info["sensor_cols_effective"] = existing_sensors

        if existing_sensors:
            info["sensors_describe"] = df[existing_sensors].describe().to_dict()
        else:
            info["sensors_describe"] = {}

        # Entity-level info (if multi-entity)
        if cfg.is_multi_entity:
            if cfg.entity_col is None:
                raise ValueError("DatasetConfig.is_multi_entity=True but entity_col is None.")

            if cfg.entity_col not in df.columns:
                raise ValueError(f"entity_col '{cfg.entity_col}' not found in DataFrame.")

            entity_counts = df[cfg.entity_col].value_counts().sort_index()
            info["n_entities"] = int(entity_counts.shape[0])
            info["entity_counts"] = entity_counts.to_dict()
            info["entity_min_length"] = int(entity_counts.min())
            info["entity_max_length"] = int(entity_counts.max())
        else:
            info["n_entities"] = 1
            info["entity_counts"] = None
            info["entity_min_length"] = n_rows
            info["entity_max_length"] = n_rows

        # Time axis info
        if cfg.time_col in df.columns:
            time_series = df[cfg.time_col]

            info["time_raw_dtype"] = str(time_series.dtype)

            if np.issubdtype(time_series.dtype, np.number):
                time_kind = "numeric"
            elif np.issubdtype(time_series.dtype, np.datetime64):
                time_kind = "datetime" 
            else:
                time_kind = "string"     

            info["time_kind"] = time_kind

            try:
                info["time_min"] = time_series.min()
                info["time_max"] = time_series.max()
            except Exception:
                info["time_min"] = None
                info["time_max"] = None

            if time_kind == "datetime" and not cfg.is_multi_entity:
                try:
                    info["time_inferred_freq"] = pd.infer_freq(
                        pd.DatetimeIndex(time_series)
                    )
                except Exception:
                    info["time_inferred_freq"] = None
            else:
                info["time_inferred_freq"] = None

            info["time_needs_parsing"] = (time_kind == "string")

        else:
            info["time_raw_dtype"] = None
            info["time_kind"] = None
            info["time_min"] = None
            info["time_max"] = None
            info["time_inferred_freq"] = None
            info["time_needs_parsing"] = None

        return info

    def _date_parser(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
        Parses dates without ANY user input regarding format.
        It statistically infers Day-First vs. Month-First logic.
        """
        df = df.copy()
        raw_vals = df[col_name].astype(str).str.strip()

        day_first_evidence = raw_vals.str.contains(
            r'^(?:1[3-9]|2[0-9]|3[01])[\/\-\.]', regex=True
        ).any()

        df[col_name] = pd.to_datetime(
            raw_vals,
            dayfirst=day_first_evidence,
            errors='coerce',
            format='mixed',
        )
        mask_failed = df[col_name].isna() & raw_vals.notna() & (~raw_vals.isin(['nan', '']))
        if mask_failed.any():
            unique_failed_strs = raw_vals[mask_failed].unique()
            fallback_map = {}

            for date_str in unique_failed_strs:
                try:
                    fallback_map[date_str] = parser.parse(
                        date_str,
                        dayfirst=day_first_evidence,
                        fuzzy=True
                    )
                except (ValueError, TypeError):
                    fallback_map[date_str] = pd.NaT

            df.loc[mask_failed, col_name] = raw_vals[mask_failed].map(fallback_map)

        return df

    def _parse_and_sort_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses the time column and sorts the DataFrame by it."""
        cfg = self.cfg
        df = df.copy()

        if cfg.time_col not in df.columns:
            raise ValueError(f"time_col '{cfg.time_col}' not found in DataFrame.")

        df = self._date_parser(df, cfg.time_col)
        return df.sort_values(cfg.time_col).reset_index(drop=True)

    # DROP CONSTANT COLUMNS & FULL DUPLICATES
    @staticmethod
    def _drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Drops columns with <= 1 unique value."""
        nunique = df.nunique(dropna=False)
        const_cols = nunique[nunique <= 1].index.tolist()

        df = df.drop(columns=const_cols, errors='ignore')
        return df, const_cols

    def drop_empty_rows(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows where all values are empty, blank, whitespace, or NaN.
        Uses df.replace instead of applymap (no deprecation warnings).
        """
        df = df.copy()

        # Patterns that represent NULL-like values
        NULL_PATTERNS = [
            r"^\s*$",           # empty or whitespace-only
            r"(?i)^null$",      # "null" (case-insensitive)
            r"(?i)^none$",      # "none" (case-insensitive)
            r"(?i)^nan$",       # "nan" written as string (case-insensitive)
        ]

        # Replace all placeholder strings with NaN
        df = df.replace(NULL_PATTERNS, np.nan, regex=True)

        # Now drop rows where all values are NaN
        df = df.dropna(how="all").reset_index(drop=True)

        return df


    @staticmethod
    def _drop_full_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Drops full duplicates and returns the count dropped."""
        n_dup = int(df.duplicated().sum())
        if n_dup > 0:
            df = df.drop_duplicates().reset_index(drop=True)
        return df, n_dup

    # TIME REGULARITY ANALYSIS

    def analyze_datetime_regularity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Checks if time intervals are roughly regular."""
        cfg = self.cfg
        time_col = cfg.time_col
        if time_col not in df.columns:
            raise ValueError(f"time_col '{time_col}' not found in DataFrame.")

        df_sorted = df.sort_values(time_col)
        times = df_sorted[time_col].values

        if len(times) < 3:
            return {
                "is_regular": False,
                "inferred_freq": None,
                "mode_gap_ns": None,
                "mode_ratio": 0.0,
            }

        deltas = np.diff(times).astype("timedelta64[ns]").astype(np.int64)
        unique, counts = np.unique(deltas, return_counts=True)

        if len(unique) == 0:
            return {
                "is_regular": False,
                "inferred_freq": None,
                "mode_gap_ns": None,
                "mode_ratio": 0.0,
            }

        mode_idx = np.argmax(counts)
        mode_gap = int(unique[mode_idx])
        mode_ratio = float(counts[mode_idx] / counts.sum())

        try:
            inferred_freq = pd.infer_freq(pd.DatetimeIndex(times))
        except Exception:
            inferred_freq = None

        is_regular = bool(mode_ratio >= cfg.freq_tolerance and inferred_freq is not None)

        return {
            "is_regular": is_regular,
            "inferred_freq": inferred_freq,
            "mode_gap_ns": mode_gap,
            "mode_ratio": mode_ratio,
        }

    # REINDEX + INTERPOLATE
    def reindex_regular_and_interpolate(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, int]:
        """
        If frequency can be inferred, reindex to a full regular grid on time_col
        and interpolate the target over time. Otherwise, returns df unchanged.
        """
        cfg = self.cfg
        df = df.sort_values(cfg.time_col).copy()

        try:
            freq = pd.infer_freq(df[cfg.time_col])
        except Exception:
            freq = None

        if freq is None:
            return df, 0

        full_range = pd.date_range(
            start=df[cfg.time_col].iloc[0],
            end=df[cfg.time_col].iloc[-1],
            freq=freq,
        )

        n_before = len(df)
        df_reindexed = df.set_index(cfg.time_col).reindex(full_range)
        n_after = len(df_reindexed)
        n_added = n_after - n_before

        # Interpolate target over time, then ffill/bfill
        if cfg.target_col in df_reindexed.columns:
            df_reindexed[cfg.target_col] = (
                df_reindexed[cfg.target_col]
                .interpolate(method="time")
                .ffill()
                .bfill()
            )

        # For other columns, simple forward/backward fill
        for col in df_reindexed.columns:
            if col != cfg.target_col:
                df_reindexed[col] = df_reindexed[col].ffill().bfill()

        df_reindexed.index.name = cfg.time_col
        df_reindexed = df_reindexed.reset_index().rename(columns={"index": cfg.time_col})

        return df_reindexed, int(n_added)

    # FEATURE ENGINEERING 

    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic calendar features based on the time column."""
        cfg = self.cfg
        df = df.copy()
        dt = pd.to_datetime(df[cfg.time_col])
        df["dow"] = dt.dt.dayofweek   
        df["month"] = dt.dt.month        
        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        max_lag: int = 12,
        lag_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Add lag features for a list of numeric columns."""
        df = df.copy()
        if not lag_cols:
            return df
        for col in lag_cols:
            if col in df.columns:
                for lag in range(1, max_lag + 1):
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [6, 12, 24],
        min_periods: int = 2,
        roll_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Add rolling mean/std over the numeric columns."""
        df = df.copy()
        if not roll_cols:
            return df

        for col in roll_cols:
            if col in df.columns:
                for w in windows:
                    roll = df[col].shift(1).rolling(window=w, min_periods=min_periods)
                    df[f"{col}_roll_mean_{w}"] = roll.mean()
                    df[f"{col}_roll_std_{w}"] = roll.std()

        return df

    # PreProcessing Pipeline
    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
          1) gets overview_before
          2) parses datetime if time_col is object
          3) sorts by time_col
          4) drops duplicates, constant columns, fully empty rows
          5) if time_col is datetime: checks regularity + reindex + interpolate

        Returns:
          df_final, report dict
        """
        cfg = self.cfg

        # overview BEFORE
        overview_before = self.get_data_overview(df)

        df_proc = df.copy()

        # handle time_col
        if cfg.time_col in df_proc.columns:
            ts = df_proc[cfg.time_col]

            if pd.api.types.is_object_dtype(ts):
                df_proc = self._parse_and_sort_datetime(df_proc)

            elif pd.api.types.is_datetime64_any_dtype(ts):
                df_proc = df_proc.sort_values(cfg.time_col).reset_index(drop=True)

        # drop duplicates, constant cols, empty rows
        df_proc, n_dup = self._drop_full_duplicates(df_proc)
        df_proc, const_cols = self._drop_constant_columns(df_proc)
        df_proc = self.drop_empty_rows(df_proc)

        # datetime regularity + reindex if applicable
        time_regularity = None
        n_added = 0
        if (
            cfg.time_col in df_proc.columns
            and pd.api.types.is_datetime64_any_dtype(df_proc[cfg.time_col])
        ):
            time_regularity = self.analyze_datetime_regularity(df_proc)
            # reindex and interpolate
            df_proc, n_added = self.reindex_regular_and_interpolate(df_proc)

        # overview AFTER
        overview_after = self.get_data_overview(df_proc)

        report = {
            "overview_before": overview_before,
            "overview_after": overview_after,
            "n_full_duplicates_dropped": n_dup,
            "dropped_constant_columns": const_cols,
            "time_regularity": time_regularity,
            "n_rows_added_by_reindex": n_added,
        }

        return df_proc, report   