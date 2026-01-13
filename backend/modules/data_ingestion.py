# modules/data_ingestion.py
from __future__ import annotations

import re
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import pandas as pd


SUPPORTED_EXT = {"csv", "json", "xlsx", "xls", "parquet"}


class DataIngestionError(Exception):
    """Custom error for ingestion failures."""


class DataIngestion:
    """
    Robust ingestion + profiling for time-series / tabular datasets.

    - Read file (CSV / Excel / Parquet / JSON)
    - Validate basic structure (non-empty, header row)
    - Standardise column names
    - Detect & parse time columns
    - Coerce numeric / boolean columns
    - Compute profiling report for `ingestion_info` JSONB
    """

    # ---------- PUBLIC API ----------

    def read_to_dataframe(self, file_bytes: bytes, filename: str) -> pd.DataFrame:
        ext = self._get_ext(filename)
        if ext not in SUPPORTED_EXT:
            raise DataIngestionError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXT))}"
            )

        try:
            bio = BytesIO(file_bytes)
            if ext == "csv":
                df = pd.read_csv(bio)
            elif ext == "json":
                df = pd.read_json(bio)
            elif ext in {"xlsx", "xls"}:
                df = pd.read_excel(bio)
            elif ext == "parquet":
                df = pd.read_parquet(bio)
            else:
                raise DataIngestionError(f"Extension '{ext}' not implemented.")
        except Exception as e:
            raise DataIngestionError(f"Could not read {ext} file: {e}") from e

        self._validate_dataframe(df)
        return df

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        - Normalize column names (snake_case, no spaces)
        - Try to parse datetime-like columns
        - Try to coerce numeric / boolean columns
        """
        df = df.copy()

        # Normalize column names
        df.columns = [self._norm_col(c) for c in df.columns]

        # 1) datetime detection on object-like columns
        for col in df.columns:
            s = df[col]
            if s.dtype == "O":
                sample = s.dropna().astype(str).head(50)
                if sample.empty:
                    continue
                try:
                    parsed = pd.to_datetime(sample, errors="raise")
                except Exception:
                    continue

                # If most of sample parses fine → treat as datetime
                if parsed.notna().mean() >= 0.7:
                    df[col] = pd.to_datetime(s, errors="coerce")

        # 2) numeric / boolean coercion on remaining object cols
        for col in df.columns:
            s = df[col]
            if s.dtype == "O":
                # Try numeric
                num = pd.to_numeric(s, errors="coerce")
                if num.notna().sum() >= max(5, int(0.7 * s.notna().sum())):
                    df[col] = num
                    continue

                # Try boolean-like
                lower_vals = s.dropna().astype(str).str.lower().unique().tolist()
                if lower_vals and set(lower_vals) <= {"true", "false", "0", "1", "yes", "no"}:
                    mapping = {
                        "true": True,
                        "1": True,
                        "yes": True,
                        "false": False,
                        "0": False,
                        "no": False,
                    }
                    df[col] = s.astype(str).str.lower().map(mapping)

        return df

    def profile(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Compute the JSON report stored in `ingestion_info`.
        Includes:
        - shape
        - columns
        - dtypes
        - time_columns
        - target_column
        - missing_values per column
        - duplicate_rows count
        - numeric summary (describe)
        - categorical summary (cardinality + top values)
        - small sample (with datetimes formatted)
        """
        if target_column not in df.columns:
            raise DataIngestionError(
                f"Target column '{target_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        time_cols = self.detect_time_columns(df)
        missing = {c: int(df[c].isna().sum()) for c in df.columns}
        duplicate_rows = int(df.duplicated().sum())

        numeric_df = df.select_dtypes(include=[np.number])
        numeric_summary: Dict[str, Any] = {}
        if not numeric_df.empty:
            numeric_summary = numeric_df.describe().to_dict()

        cat_cols = [
            c for c in df.columns
            if c not in numeric_df.columns and not np.issubdtype(df[c].dtype, np.datetime64)
        ]
        categorical_summary: Dict[str, Any] = {}
        for c in cat_cols:
            s = df[c].astype("object")
            vc = s.value_counts(dropna=True)
            top = [
                {"value": str(idx), "count": int(cnt)}
                for idx, cnt in vc.head(5).items()
            ]
            categorical_summary[c] = {
                "distinct": int(s.nunique(dropna=True)),
                "top_values": top,
            }

        sample_records = self._sample(df, n=5)

        return {
            "shape": (int(df.shape[0]), int(df.shape[1])),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "time_columns": time_cols,
            "target_column": target_column,
            "missing_values": missing,
            "duplicate_rows": duplicate_rows,
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
            "sample": sample_records,
        }

    # ---------- HELPERS ----------

    @staticmethod
    def _get_ext(filename: str) -> str:
        if "." not in filename:
            return ""
        return filename.rsplit(".", 1)[-1].lower()

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        if df is None or df.empty:
            raise DataIngestionError("The uploaded file is empty.")

        if df.columns.size == 0:
            raise DataIngestionError("No columns found in file.")

        # Check header row: detect "unnamed" columns
        cols = [str(c) for c in df.columns]
        unnamed_like = [c for c in cols if re.match(r"^unnamed[:\s]*\d*", c.lower())]
        if len(unnamed_like) >= max(1, int(0.5 * len(cols))):
            raise DataIngestionError(
                "The file seems to have no proper header row. "
                "Please include a first row with column names."
            )

    @staticmethod
    def detect_time_columns(df: pd.DataFrame) -> List[str]:
        """Return a list of columns that look like time / datetime."""
        time_cols: List[str] = []

        # 1) Already datetime dtypes
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                time_cols.append(c)

        # 2) Object columns that can be parsed as datetime
        for c in df.columns:
            if c in time_cols:
                continue
            s = df[c]
            if s.dtype != "O":
                continue
            sample = s.dropna().astype(str).head(50)
            if sample.empty:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="raise")
            except Exception:
                continue
            if parsed.notna().mean() >= 0.7:
                time_cols.append(c)

        return time_cols

    @staticmethod
    def _norm_col(name: str) -> str:
        """
        Normalize column names:
        - strip
        - replace / and - with _
        - collapse spaces
        - lowercase
        """
        name = str(name)
        name = name.replace("/", "_").replace("-", "_")
        parts = name.strip().split()
        return "_".join(parts).lower()

    @staticmethod
    def _sample(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
        head = df.head(n).copy()
        for c in head.columns:
            if np.issubdtype(head[c].dtype, np.datetime64):
                head[c] = head[c].dt.strftime("%Y-%m-%d %H:%M:%S").astype("object")
        return head.to_dict(orient="records")
