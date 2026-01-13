from __future__ import annotations

import os
import pickle
import hashlib
import uuid
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union
import numpy as np
from datetime import datetime
import tempfile

import pandas as pd
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    Body,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from supabase import create_client, Client
from dotenv import load_dotenv

from modules.data_ingestion import DataIngestion, DataIngestionError
from modules.data_forge import DataForge, DataConfig
from modules.automl import AutoMLPipeline, AutoMLConfig
from modules.predict_x import PredictX, PredictionConfig
from modules.explainx import ExplainX
from modules.automl_v2 import AutoMLPipelineV2, AutoMLConfigV2
from modules.predictx_v2 import PredictXV2, PredictionConfigV2

import joblib
import tensorflow as tf


# ---------------------------------------------------------
# Environment & Supabase client
# ---------------------------------------------------------

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY missing in environment")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

ingestor = DataIngestion()

# ---------------------------------------------------------
# FastAPI app & CORS
# ---------------------------------------------------------

app = FastAPI(title="Predictive Platform API – Dashboard + Ingestion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ---------------------------------------------------------
# Auth helper
# ---------------------------------------------------------

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Any:
    token = credentials.credentials
    try:
        user_resp = supabase.auth.get_user(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth error: {e}")

    if not user_resp or user_resp.user is None:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    return user_resp.user


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    return str(obj)


# ---------------------------------------------------------
# Helper: upload / download storage
# ---------------------------------------------------------

def _upload_to_storage(
    bucket: str,
    path: str,
    data: bytes,
    content_type: str = "application/octet-stream",
):
    try:
        return supabase.storage.from_(bucket).upload(
            path,
            data,
            file_options={
                "cache-control": "3600",
                "content-type": content_type,
                "upsert": "true",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload to storage ({bucket}:{path}): {e}",
        )


def _download_bytes_from_storage(bucket: str, path: str) -> bytes:
    if not path:
        raise HTTPException(status_code=400, detail="No path provided.")

    try:
        res = supabase.storage.from_(bucket).download(path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Storage download error ({bucket}:{path}): {e}",
        )

    raw = res if isinstance(res, (bytes, bytearray)) else getattr(res, "data", None) or getattr(res, "content", None) or res
    if raw is None:
        raise HTTPException(status_code=500, detail=f"Empty response when downloading {bucket}:{path}")

    return raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)


def _download_parquet_from_storage(bucket: str, path: str) -> pd.DataFrame:
    raw = _download_bytes_from_storage(bucket, path)
    try:
        bio = BytesIO(raw)
        return pd.read_parquet(bio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read parquet: {e}")


def _remove_from_storage(bucket: str, path: str | None):
    if not path:
        return
    try:
        supabase.storage.from_(bucket).remove([path])
    except Exception as e:
        print(f"[WARN] Failed to remove {bucket}:{path} from storage: {e}")


def _download_model_artifact(bucket: str, path: str) -> Any:
    """
    Load model artifact 
    """
    if not path:
        raise HTTPException(status_code=400, detail="No artifacts_path set")

    raw = _download_bytes_from_storage(bucket, path)

    # joblib
    if path.endswith(".joblib"):
        try:
            return joblib.load(BytesIO(raw))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load joblib model: {e}")

    # xgboost json
    if path.endswith(".json"):
        try:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(bytearray(raw))
            return booster
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load xgboost model: {e}")

    # keras
    if path.endswith(".keras"):
        try:
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "model.keras")
                with open(p, "wb") as f:
                    f.write(raw)
                return tf.keras.models.load_model(p)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load keras model: {e}")

    # fallback: pickle
    try:
        return pickle.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model artifact: {e}")


def save_explainx_to_npz_bytes(payload: Dict[str, Any]) -> bytes:
    bio = BytesIO()
    np.savez_compressed(bio, **payload)
    return bio.getvalue()


def _upload_explain_artifacts_bytes(user_id: str, model_id: str, data: bytes) -> str:
    bucket = "models"
    path = f"{user_id}/{model_id}_explain.npz"
    _upload_to_storage(bucket, path, data, content_type="application/octet-stream")
    return path

def save_model_to_bytes(model_obj: Any) -> Tuple[bytes, str, str]:
    """
    Returns: (data_bytes, content_type, extension)

    Priority:
      1) XGBoost native JSON
      2) Keras native .keras
      3) joblib for sklearn/others
    """
    # ---- XGBoost ----
    try:
        import xgboost as xgb
        if hasattr(model_obj, "get_booster"):
            booster = model_obj.get_booster()
            raw = booster.save_raw("json")
            return raw, "application/json", "json"
        if isinstance(model_obj, xgb.Booster):
            raw = model_obj.save_raw("json")
            return raw, "application/json", "json"
    except Exception:
        pass

    # ---- Keras / TF ----
    try:
        if isinstance(model_obj, tf.keras.Model):
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "model.keras")
                model_obj.save(p)
                with open(p, "rb") as f:
                    return f.read(), "application/octet-stream", "keras"
    except Exception:
        pass

    # ---- joblib ----
    try:
        bio = BytesIO()
        joblib.dump(model_obj, bio, compress=3)
        return bio.getvalue(), "application/octet-stream", "joblib"
    except Exception as e:
        msg = f"SERIALIZATION_FAILED: {e}"
        return msg.encode("utf-8"), "text/plain", "txt"


def _upload_model_artifact_bytes(
    user_id: str,
    model_id: str,
    data: bytes,
    ext: str,
    content_type: str,
) -> str:
    bucket = "models"
    path = f"{user_id}/{model_id}.{ext}"
    _upload_to_storage(bucket, path, data, content_type=content_type)
    return path


# ---------------------------------------------------------
# Build evaluation_report 
# ---------------------------------------------------------

def build_evaluation_report(summary: Dict[str, Any]) -> Dict[str, Any]:
    pm = summary.get("primary_metric", "RMSE")
    best_name = summary.get("best_model_name")
    best_type = summary.get("best_model_type")
    wf_results = summary.get("wf_results") or {}
    all_results = summary.get("all_results") or {}
    meta = summary.get("meta") or {}

    wf_payload: Dict[str, Any] = {}
    for mname, info in wf_results.items():
        metrics_df = info.get("metrics_df")
        if isinstance(metrics_df, pd.DataFrame):
            folds = metrics_df.to_dict(orient="records")
        else:
            folds = make_json_safe(metrics_df) if metrics_df is not None else None

        wf_payload[mname] = {
            "type": info.get("type"),
            "avg_metric": info.get("avg_metric"),
            "primary_metric": pm,
            "folds": folds,
        }

    best_metrics = (all_results.get(best_name) or {}).get("metrics") if best_name else None
    best_wf_avg = (wf_results.get(best_name) or {}).get("avg_metric") if best_name else None

    return make_json_safe({
        "primary_metric": pm,
        "best_model_name": best_name,
        "best_model_type": best_type,
        "best_benchmark_metrics": best_metrics,
        "best_wf_avg_metric": best_wf_avg,
        "walk_forward": wf_payload,
        "meta": meta,
    })


# ---------------------------------------------------------
# ExplainX
# ---------------------------------------------------------
def build_and_store_best_explainx(
    *,
    automl: Any,
    best_model: Any,
    best_model_type: str,  
    df_clean: pd.DataFrame,
    user_id: str,
    model_id: str,
) -> tuple[str | None, dict, str]:

    if best_model is None:
        return None, {"kind": best_model_type, "error": "best_model is None"}, "none"

    bt = (best_model_type or "").lower().strip()
    if bt not in {"tabular", "seq"}:
        return None, {"kind": bt, "error": f"Unknown best_model_type={best_model_type}"}, "none"

    try:
        # -------------------------
        # TABULAR
        # -------------------------
        if bt == "tabular":
            if hasattr(automl, "train_test_split_from_df"):
                X_train, X_test, y_train, y_test = automl.train_test_split_from_df(df_clean)
            else:
                target_col = automl.cfg.target_col
                time_col = getattr(automl.cfg, "time_col", None)
                entity_col = getattr(automl.cfg, "entity_col", None)
                is_multi_entity = bool(getattr(automl.cfg, "is_multi_entity", False))

                df = df_clean.copy()
                if time_col and time_col in df.columns:
                    if is_multi_entity and entity_col and entity_col in df.columns:
                        df = df.sort_values([entity_col, time_col]).reset_index(drop=True)
                    else:
                        df = df.sort_values(time_col).reset_index(drop=True)

                exclude = [target_col]
                if time_col and time_col in df.columns:
                    exclude.append(time_col)
                if is_multi_entity and entity_col and entity_col in df.columns:
                    exclude.append(entity_col)

                feature_cols = df.columns.difference(exclude).tolist()
                X_all = df[feature_cols]
                y_all = df[target_col]

                test_size = float(getattr(automl.automl_cfg, "test_size", 0.2) if hasattr(automl, "automl_cfg") else 0.2)
                test_size = max(0.05, min(test_size, 0.5))
                n = len(df)
                n_test = max(1, int(n * test_size))
                n_train = n - n_test

                X_train = X_all.iloc[:n_train, :]
                X_test  = X_all.iloc[n_train:, :]
                y_train = y_all.iloc[:n_train]
                y_test  = y_all.iloc[n_train:]

            feature_names = automl.meta_.get("train_features") if hasattr(automl, "meta_") else list(X_train.columns)

            expl = ExplainX(
                model=best_model,
                model_family="gbm",
                X_train=X_train,
                X_val=X_test,
                y_val=y_test,
                feature_names=feature_names,
                task_name=automl.cfg.target_col,
            )

            g = expl.compute_shap_global(max_samples=1000)
            shap_values = np.asarray(g["shap_values"]) 
            x_values = np.asarray(X_test.values)[: shap_values.shape[0], :]  

            mean_abs = np.mean(np.abs(shap_values), axis=0)
            global_importance = sorted(
                [{"feature": str(f), "mean_abs_shap": float(v)} for f, v in zip(feature_names, mean_abs)],
                key=lambda r: r["mean_abs_shap"],
                reverse=True,
            )

            try:
                imp_df = expl.compute_permutation_importance(n_repeats=5, random_state=42)
                perm_importance = imp_df.to_dict(orient="records")
            except Exception as e:
                perm_importance = []
                print(f"[WARN] permutation_importance(tabular) failed: {e}")


            arrays_payload = {
                "shap_values": shap_values.astype(np.float32),
                "x_values": x_values.astype(np.float32),
                "feature_names": np.asarray([str(x) for x in feature_names], dtype=str),
            }

            npz_bytes = save_explainx_to_npz_bytes(arrays_payload)
            explain_path = _upload_explain_artifacts_bytes(user_id, model_id, npz_bytes)

            explain_report = {
                "kind": "tabular",
                "task_name": automl.cfg.target_col,
                "n_val": int(shap_values.shape[0]),
                "n_features": int(shap_values.shape[1]),
                "global_importance": global_importance,
                "permutation_importance": perm_importance,
            }
            return explain_path, explain_report, "ready"

        X_seq, y_seq, y_seq_original = automl.build_sequences_from_df(
            df_clean,
            scale_features=True,
        )

        X_tr_seq, X_val_seq, y_tr_seq, y_val_seq = automl.seq_train_test_split(X_seq, y_seq)

        feature_names = automl.meta_.get("train_features") if hasattr(automl, "meta_") else [f"feat_{i}" for i in range(X_tr_seq.shape[-1])]

        expl = ExplainX(
            model=best_model,
            model_family="deep",
            X_train=X_tr_seq,
            X_val=X_val_seq,
            y_val=y_val_seq,
            feature_names=feature_names,
            task_name=automl.cfg.target_col,
        )

        g = expl.compute_shap_global(max_samples=500)  
        shap_seq = np.asarray(g["shap_values"])        
        x_seq = np.asarray(X_val_seq)[: shap_seq.shape[0], :, :] 

        heatmap_last = shap_seq[-1]  
        shap_feat = np.mean(np.abs(shap_seq), axis=1)
        x_feat = np.mean(x_seq, axis=1)

        mean_abs = np.mean(shap_feat, axis=0)
        global_importance = sorted(
            [{"feature": str(f), "mean_abs_shap": float(v)} for f, v in zip(feature_names, mean_abs)],
            key=lambda r: r["mean_abs_shap"],
            reverse=True,
        )

        try:
            imp_df = expl.compute_permutation_importance(n_repeats=5, random_state=42)
            perm_importance = imp_df.to_dict(orient="records")
        except Exception as e:
            perm_importance = []
            print(f"[WARN] permutation_importance(seq) failed: {e}")

        arrays_payload = {
            "shap_seq": shap_seq.astype(np.float32),   
            "x_seq": x_seq.astype(np.float32),               
            "shap_feat": shap_feat.astype(np.float32),    
            "x_feat": x_feat.astype(np.float32),            
            "heatmap_last": heatmap_last.astype(np.float32), 
            "feature_names": np.asarray([str(x) for x in feature_names], dtype=str),
        }

        npz_bytes = save_explainx_to_npz_bytes(arrays_payload)
        explain_path = _upload_explain_artifacts_bytes(user_id, model_id, npz_bytes)

        explain_report = {
            "kind": "seq",
            "task_name": automl.cfg.target_col,
            "n_val": int(shap_seq.shape[0]),
            "lookback": int(shap_seq.shape[1]),
            "n_features": int(shap_seq.shape[2]),
            "global_importance": global_importance,
            "permutation_importance": perm_importance,
        }
        return explain_path, explain_report, "ready"

    except Exception as e:
        print(f"[ERROR] build_and_store_best_explainx failed: {e}")
        return None, {"kind": bt, "error": str(e)}, "failed"



# ---------------------------------------------------------
# Basic health
# ---------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------
# Datasets endpoints
# ---------------------------------------------------------

@app.get("/api/v1/datasets")
async def list_datasets(current_user: Any = Depends(get_current_user)):
    resp = (
        supabase.table("datasets")
        .select("*")
        .eq("user_id", current_user.id)
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data or []


@app.get("/api/v1/datasets/{dataset_id}")
async def get_dataset(dataset_id: str, current_user: Any = Depends(get_current_user)):
    resp = (
        supabase.table("datasets")
        .select("*")
        .eq("id", dataset_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return resp.data


# ---------------------------------------------------------
# Models endpoints
# ---------------------------------------------------------

@app.get("/api/v1/models")
async def list_models(current_user: Any = Depends(get_current_user)):
    resp = (
        supabase.table("models")
        .select(
            "id, user_id, dataset_id, model_name, algorithm, status, "
            "primary_metric, test_metrics, created_at, explain_status"
        )
        .eq("user_id", current_user.id)
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data or []


@app.get("/api/v1/models/{model_id}")
async def get_model(model_id: str, current_user: Any = Depends(get_current_user)):
    resp = (
        supabase.table("models")
        .select("*")
        .eq("id", model_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Model not found")
    return resp.data


@app.get("/api/v1/models/{model_id}/explainx")
async def get_model_explainx(model_id: str, current_user: Any = Depends(get_current_user)):
    m_resp = (
        supabase.table("models")
        .select("id, user_id, explain_status, explain_report, explain_artifacts_path")
        .eq("id", model_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not m_resp.data:
        raise HTTPException(status_code=404, detail="Model not found")

    row = m_resp.data
    path = row.get("explain_artifacts_path")

    arrays = None
    if path:
        raw = _download_bytes_from_storage("models", path)
        bio = BytesIO(raw)
        npz = np.load(bio, allow_pickle=False)
        arrays = {k: npz[k].tolist() for k in npz.files}

    report = row.get("explain_report") or {}
    kind = (report.get("kind") or "").lower() 

    return make_json_safe({
        "model_id": model_id,
        "explain_status": row.get("explain_status", "none"),
        "explain_report": report,
        "kind": kind,
        "arrays": arrays,  
        "keys": list(arrays.keys()) if arrays else [],
    })



# ---------------------------------------------------------
# Upload dataset
# ---------------------------------------------------------

@app.post("/api/v1/data/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_feature: str = Form(...),
    time_column: str = Form(...),
    forecast_horizon: int = Form(...),
    is_multi_entity: bool = Form(False),
    entity_column: str | None = Form(None),
    current_user: Any = Depends(get_current_user),
):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    filename = file.filename or "dataset"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    size_bytes = len(content)
    file_hash = hashlib.sha256(content).hexdigest()

    try:
        df = ingestor.read_to_dataframe(content, filename)
        df_std = ingestor.standardize(df)
    except DataIngestionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    if target_feature not in df_std.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_feature}' not found. Available: {list(df_std.columns)}",
        )

    if time_column not in df_std.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Time column '{time_column}' not found. Available: {list(df_std.columns)}",
        )

    if is_multi_entity:
        if not entity_column:
            raise HTTPException(status_code=400, detail="is_multi_entity=True but no entity_column provided.")
        if entity_column not in df_std.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Entity column '{entity_column}' not found. Available: {list(df_std.columns)}",
            )
        if entity_column in {target_feature, time_column}:
            raise HTTPException(
                status_code=400,
                detail="Entity column must be different from time_column and target_feature.",
            )
    else:
        entity_column = None

    exclude_cols = [time_column, target_feature]
    if entity_column:
        exclude_cols.append(entity_column)
    sensors = df_std.columns.difference(exclude_cols).tolist()

    try:
        ingestion_info = ingestor.profile(df_std, target_column=target_feature)
    except DataIngestionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {e}")

    time_cols = ingestion_info.get("time_columns") or []
    if time_column not in time_cols:
        time_cols = [time_column] + time_cols
    else:
        time_cols = [time_column] + [c for c in time_cols if c != time_column]
    ingestion_info["time_columns"] = time_cols

    ingestion_info["time_column"] = time_column
    ingestion_info["is_multi_entity"] = bool(is_multi_entity)
    ingestion_info["entity_column"] = entity_column
    ingestion_info["sensor_columns"] = sensors

    rows_count, columns_count = df_std.shape

    user_id = current_user.id
    base_id = str(uuid.uuid4())
    original_path = f"{user_id}/{base_id}.{ext or 'dat'}"
    parquet_path = f"{user_id}/{base_id}.parquet"

    _upload_to_storage("datasets", original_path, content)

    buf = BytesIO()
    df_std.to_parquet(buf, index=False)
    buf.seek(0)
    _upload_to_storage("datasets", parquet_path, buf.read())

    payload: Dict[str, Any] = {
        "user_id": user_id,
        "original_filename": filename,
        "file_extension": ext,
        "file_size_bytes": size_bytes,
        "storage_original_path": original_path,
        "storage_parquet_path": parquet_path,
        "processed_parquet_path": None,
        "rows_count": int(rows_count),
        "columns_count": int(columns_count),
        "file_hash": file_hash,
        "time_column": time_column,
        "target_column": target_feature,
        "entity_column": entity_column,
        "is_multi_entity": bool(is_multi_entity),
        "forecast_horizon": int(forecast_horizon),
        "status": "ready",
        "ingestion_info": ingestion_info,
        "processing_report": None,
        "raw_plots": None,
        "processed_plots": None,
    }

    try:
        resp = supabase.table("datasets").insert(payload).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB insert error: {e}")

    if not resp.data:
        raise HTTPException(status_code=500, detail="Failed to insert dataset row")

    row = resp.data[0]
    return {
        "message": "Dataset uploaded and registered successfully",
        "dataset_id": row["id"],
        "dataset": row,
        "original_filename": row["original_filename"],
    }


# ---------------------------------------------------------
# EDA endpoint
# ---------------------------------------------------------

@app.get("/api/v1/datasets/{dataset_id}/eda")
async def get_dataset_eda(dataset_id: str, current_user: Any = Depends(get_current_user)):
    resp = (
        supabase.table("datasets")
        .select("*")
        .eq("id", dataset_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    row: Dict[str, Any] = resp.data

    parquet_path = row.get("processed_parquet_path") or row.get("storage_parquet_path")
    if not parquet_path:
        raise HTTPException(status_code=400, detail="No parquet stored for this dataset.")

    df = _download_parquet_from_storage("datasets", parquet_path)

    preview_records = df.head(50).to_dict(orient="records")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_histograms: Dict[str, Any] = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        counts, bin_edges = np.histogram(s.values, bins=20)
        numeric_histograms[col] = {"bin_edges": bin_edges.tolist(), "counts": counts.tolist()}

    target_col = row.get("target_column")
    time_col = row.get("time_column") or None

    target_series: Dict[str, Any] | None = None
    if target_col and target_col in df.columns:
        if time_col and time_col in df.columns:
            df_sorted = df.sort_values(time_col)
            ts = df_sorted[[time_col, target_col]].dropna()
            target_series = {
                "kind": "time",
                "time_col": time_col,
                "target_col": target_col,
                "time": ts[time_col].astype(str).tolist(),
                "values": ts[target_col].tolist(),
            }
        else:
            ts = df[[target_col]].dropna().reset_index(drop=True)
            target_series = {
                "kind": "index",
                "target_col": target_col,
                "index": ts.index.tolist(),
                "values": ts[target_col].tolist(),
            }

    return {
        "dataset": row,
        "ingestion_info": row.get("ingestion_info") or {},
        "preview": preview_records,
        "numeric_histograms": numeric_histograms,
        "target_series": target_series,
    }


# ---------------------------------------------------------
# Preprocess endpoint 
# ---------------------------------------------------------

@app.post("/api/v1/datasets/{dataset_id}/preprocess")
async def preprocess_dataset(dataset_id: str, current_user: Any = Depends(get_current_user)):
    resp = (
        supabase.table("datasets")
        .select("*")
        .eq("id", dataset_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    row = resp.data

    target_col = row.get("target_column")
    time_col = row.get("time_column")
    is_multi_entity = bool(row.get("is_multi_entity"))
    entity_col = row.get("entity_column")

    if not target_col:
        raise HTTPException(status_code=400, detail="Dataset has no 'target_column' set.")
    if not time_col:
        raise HTTPException(status_code=400, detail="Dataset has no 'time_column' set.")
    if is_multi_entity and not entity_col:
        raise HTTPException(status_code=400, detail="Dataset is_multi_entity=True but 'entity_column' not set.")

    ingestion_info = row.get("ingestion_info") or {}
    sensor_cols = ingestion_info.get("sensor_columns") or []

    if not sensor_cols:
        parquet_path_tmp = row.get("storage_parquet_path")
        if not parquet_path_tmp:
            raise HTTPException(status_code=400, detail="No 'storage_parquet_path' to recompute sensor columns.")
        df_tmp = _download_parquet_from_storage("datasets", parquet_path_tmp)
        exclude_cols = [time_col, target_col]
        if entity_col:
            exclude_cols.append(entity_col)
        sensor_cols = df_tmp.columns.difference(exclude_cols).tolist()

    parquet_path = row.get("storage_parquet_path")
    if not parquet_path:
        raise HTTPException(status_code=400, detail="Dataset has no 'storage_parquet_path'.")

    df_raw = _download_parquet_from_storage("datasets", parquet_path)

    cfg = DataConfig(
        time_col=time_col,
        target_col=target_col,
        is_multi_entity=is_multi_entity,
        sensor_cols=sensor_cols,
        entity_col=entity_col,
    )
    forge = DataForge(cfg)

    df_clean, clean_report = forge.preprocess(df_raw)
    clean_report_safe = make_json_safe(clean_report)

    if is_multi_entity and entity_col in df_clean.columns:
        df_clean = df_clean.sort_values([entity_col, time_col]).reset_index(drop=True)
    else:
        df_clean = df_clean.sort_values(time_col).reset_index(drop=True)

    user_id = current_user.id
    clean_path = row.get("processed_parquet_path") or f"{user_id}/{dataset_id}_clean.parquet"

    buf = BytesIO()
    df_clean.to_parquet(buf, index=False)
    buf.seek(0)
    _upload_to_storage("datasets", clean_path, buf.read())

    update_payload = {
        "processed_parquet_path": clean_path,
        "processing_report": clean_report_safe,
        "status": "processed",
    }

    try:
        supabase.table("datasets").update(update_payload).eq("id", dataset_id).eq("user_id", current_user.id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update dataset preprocessing info: {e}")

    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "time_col": time_col,
        "target_col": target_col,
        "entity_col": entity_col,
        "is_multi_entity": is_multi_entity,
        "sensor_cols": sensor_cols,
        "processed_parquet_path": clean_path,
        "processing_report": clean_report_safe,
        "n_rows_clean": int(df_clean.shape[0]),
        "n_cols_clean": int(df_clean.shape[1]),
    }


@app.get("/api/v1/datasets/{dataset_id}/report")
def get_preprocessing_report(dataset_id: str, current_user=Depends(get_current_user)):
    resp = (
        supabase.table("datasets")
        .select("*")
        .eq("id", dataset_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    row = resp.data
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "dataset_id": dataset_id,
        "report": row.get("processing_report"),
        "processed_parquet_path": row.get("processed_parquet_path"),
        "status": row.get("status"),
    }


# ---------------------------------------------------------
# AutoML Run Endpoint
# ---------------------------------------------------------

def _upload_scaler_bytes(user_id: str, model_id: str, kind: str, obj: Any) -> str:
    """
    kind: 'feature' or 'target'
    """
    bucket = "models"
    path = f"{user_id}/{model_id}_{kind}_scaler.pkl"
    raw = pickle.dumps(obj)
    _upload_to_storage(bucket, path, raw, content_type="application/octet-stream")
    return path


def _download_scaler_obj(path: str) -> Any:
    raw = _download_bytes_from_storage("models", path)
    return pickle.loads(raw)

@app.post("/api/v1/datasets/{dataset_id}/automl/run")
async def run_automl_for_dataset(
    dataset_id: str,
    body: Dict[str, Any] = Body(...),
    current_user: Any = Depends(get_current_user),
):
    resp = (
        supabase.table("datasets")
        .select("*")
        .eq("id", dataset_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    row = resp.data
    processed_path = row.get("processed_parquet_path")
    if not processed_path:
        raise HTTPException(status_code=400, detail="Dataset is not preprocessed yet. Run preprocessing first.")

    df_clean = _download_parquet_from_storage("datasets", processed_path)

    time_col = row.get("time_column")
    target_col = row.get("target_column")
    is_multi_entity = bool(row.get("is_multi_entity"))
    entity_col = row.get("entity_column")

    if not time_col or not target_col:
        raise HTTPException(status_code=400, detail="Dataset missing time_column or target_column.")

    cols_to_remove: List[str] = [time_col, target_col]
    if is_multi_entity and entity_col is not None:
        cols_to_remove.append(entity_col)
    sensor_cols = df_clean.columns.difference(cols_to_remove).tolist()

    data_cfg = DataConfig(
        time_col=time_col,
        target_col=target_col,
        is_multi_entity=is_multi_entity,
        sensor_cols=sensor_cols,
        entity_col=entity_col,
    )

    use_clip = bool(body.get("use_clip", False))
    clip_threshold_in = body.get("clip_threshold", None)
    clip_threshold = (125.0 if clip_threshold_in is None else float(clip_threshold_in)) if use_clip else None

    try:
        cfg_automl = AutoMLConfig(
            models_to_train=body.get("models_to_train", []),
            primary_metric=body.get("primary_metric", "RMSE"),
            test_size=body.get("test_size", 0.2),
            clip_threshold=clip_threshold,
            top_k=body.get("top_k", 2),
            n_splits=body.get("n_splits", 3),
            lookback=body.get("lookback", 50),
            epochs=body.get("epochs", 10),
            batch_size=body.get("batch_size", 64),
            do_plots=body.get("do_plots", False),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid AutoML config: {e}")

    summary: Dict[str, Any] = {}
    best_model_id: str | None = None

    try:
        automl = AutoMLPipeline(data_cfg=data_cfg, automl_cfg=cfg_automl)
        summary = automl.run_pipeline(df_clean)

        best_name = summary.get("best_model_name")
        best_type = summary.get("best_model_type")
        wf_results = summary.get("wf_results") or {}
        all_results = summary.get("all_results") or {}
        pm = summary.get("primary_metric", "RMSE")

        if best_name:
            best_res = all_results.get(best_name) or {}
            best_metrics = best_res.get("metrics") or {}

            pm_val = None
            wf_info = wf_results.get(best_name)
            if wf_info:
                pm_val = wf_info.get("avg_metric")
            if pm_val is None:
                pm_val = best_metrics.get(pm)
            pm_val = float(pm_val) if pm_val is not None else float("nan")

            evaluation_report = build_evaluation_report(summary)

            best_obj = summary.get("best_model") or getattr(automl, "best_model_", None)
            artifact_path = None

            candidate_id = str(uuid.uuid4())

            if best_obj is not None:
                data_bytes, ctype, ext = save_model_to_bytes(best_obj)
                artifact_path = _upload_model_artifact_bytes(
                    user_id=current_user.id,
                    model_id=candidate_id,
                    data=data_bytes,
                    ext=ext,
                    content_type=ctype,
                )

            fs_path = None
            ts_path = None

            if summary.get("feature_scaler") is not None and summary.get("target_scaler") is not None:
                try:
                    fs_path = _upload_scaler_bytes(current_user.id, candidate_id, "feature", summary["feature_scaler"])
                    ts_path = _upload_scaler_bytes(current_user.id, candidate_id, "target", summary["target_scaler"])
                    print(f"[INFO] Saved scalers for model candidate {candidate_id}")
                except Exception as e:
                    print(f"[WARN] Failed to save scalers: {e}")
                    fs_path, ts_path = None, None


            explain_path, explain_report, explain_status = build_and_store_best_explainx(
                automl=automl,
                best_model=best_obj,
                best_model_type=best_type,
                df_clean=df_clean,
                user_id=current_user.id,
                model_id=candidate_id,
            )

            model_payload = {
                "id": candidate_id,
                "user_id": current_user.id,
                "dataset_id": dataset_id,
                "model_name": best_name,
                "algorithm": best_name,
                "status": "ready",
                "primary_metric": f"{pm}: {pm_val:.4f}",
                "train_metrics": make_json_safe(best_metrics),
                "test_metrics": make_json_safe({pm: pm_val}),
                "training_report": make_json_safe(
                    {
                        "primary_metric": pm,
                        "test_size": cfg_automl.test_size,
                        "top_k": cfg_automl.top_k,
                        "n_splits": cfg_automl.n_splits,
                        "clip_threshold": cfg_automl.clip_threshold,
                        "models_to_train": cfg_automl.models_to_train,
                        "best_model_type": best_type,
                        "data_meta": summary.get("meta"),
                        "plot_payload": summary.get("plot_payload", {}),
                    }
                ),
                "evaluation_report": evaluation_report,
                "model_plots": make_json_safe(summary.get("plot_payload", {})),
                "artifacts_path": artifact_path,
                "target_column": row.get("target_column"),
                "forecast_horizon": row.get("forecast_horizon"),
                "feature_scaler_path": fs_path,
                "target_scaler_path": ts_path,

                "explain_status": explain_status,
                "explain_report": make_json_safe(explain_report),
                "explain_artifacts_path": explain_path,
            }

            try:
                ins = supabase.table("models").insert(model_payload).execute()
                if not ins.data:
                    raise HTTPException(status_code=500, detail="Failed to insert best model row.")
                best_model_id = ins.data[0]["id"]

            except Exception as insert_err:
                model_payload.pop("id", None)

                model_payload["artifacts_path"] = None
                model_payload["explain_artifacts_path"] = None

                ins2 = supabase.table("models").insert(model_payload).execute()
                if not ins2.data:
                    raise HTTPException(status_code=500, detail=f"Failed to insert best model row (fallback): {insert_err}")
                best_model_id = ins2.data[0]["id"]

                if best_obj is not None:
                    data_bytes, ctype, ext = save_model_to_bytes(best_obj)
                    artifact_path = _upload_model_artifact_bytes(
                        user_id=current_user.id,
                        model_id=best_model_id,
                        data=data_bytes,
                        ext=ext,
                        content_type=ctype,
                    )

                    exp_path2, exp_report2, exp_status2 = build_and_store_best_explainx(
                        automl=automl,
                        best_model=best_obj,
                        best_model_type=best_type,
                        df_clean=df_clean,
                        user_id=current_user.id,
                        model_id=best_model_id,
                    )

                    supabase.table("models").update(
                        {
                            "artifacts_path": artifact_path,
                            "explain_status": exp_status2,
                            "explain_report": make_json_safe(exp_report2),
                            "explain_artifacts_path": exp_path2,
                        }
                    ).eq("id", best_model_id).eq("user_id", current_user.id).execute()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AutoML run failed: {e}")

    tabular_results = summary.get("tabular_results") or {}
    seq_results = summary.get("seq_results") or {}
    wf_results = summary.get("wf_results") or {}
    best_name = summary.get("best_model_name")
    best_type = summary.get("best_model_type")

    best_avg_metric_raw = summary.get("best_avg_metric", None)
    best_avg_metric = None
    try:
        if best_avg_metric_raw is not None:
            v = float(best_avg_metric_raw)
            best_avg_metric = v if np.isfinite(v) else None
    except Exception:
        best_avg_metric = None

    steps = [
        {"key": "prepare_data", "label": "Prepare data (X, y, sequences)", "done": True},
        {"key": "benchmark_tabular", "label": "Benchmark tabular models", "done": bool(tabular_results)},
        {"key": "benchmark_seq", "label": "Benchmark sequence models", "done": bool(seq_results)},
        {"key": "rank_models", "label": "Rank models by primary metric", "done": True},
        {"key": "walk_forward", "label": "Walk-forward validation (top-K)", "done": bool(wf_results)},
        {"key": "retrain_best", "label": "Retrain best model on full data", "done": bool(best_name)},
    ]

    primary_metric_name: str = summary.get("primary_metric", "RMSE")
    all_results = summary.get("all_results") or {}
    tabular_results = summary.get("tabular_results") or {}
    seq_results = summary.get("seq_results") or {}

    models_summary: List[Dict[str, Any]] = []
    for name, base_res in all_results.items():
        base_metrics = base_res.get("metrics") or {}
        pm_val = base_metrics.get(primary_metric_name, np.nan)
        try:
            primary_metric_value = float(pm_val)
        except Exception:
            primary_metric_value = float("nan")

        if name in tabular_results:
            m_type = "tabular"
        elif name in seq_results:
            m_type = "seq"
        else:
            t = base_res.get("type")
            m_type = "seq" if t == "seq" else "tabular"

        wf_info = wf_results.get(name, {})
        avg_metric = wf_info.get("avg_metric")

        models_summary.append(
            {
                "name": name,
                "type": m_type,
                "primary_metric_value": primary_metric_value,
                "metrics": make_json_safe(base_metrics),
                "wf_avg_metric": float(avg_metric) if avg_metric is not None else None,
            }
        )

    return make_json_safe(
        {
            "dataset_id": dataset_id,
            "primary_metric": primary_metric_name,
            "steps": steps,
            "models": models_summary,
            "best_model_id": best_model_id,
            "best_model_name": best_name,
            "best_model_type": best_type,
            "best_avg_metric": best_avg_metric,
            "plots": summary.get("plot_payload", {}),
        }
    )


@app.post("/api/v1/datasets/{dataset_id}/models/save-best")
async def save_best_model_for_dataset(
    dataset_id: str,
    body: Dict[str, Any] = Body(...),
    current_user: Any = Depends(get_current_user),
):
    best_model = body.get("best_model")
    artifact_path = body.get("artifact_path") 
    if not best_model:
        raise HTTPException(status_code=400, detail="Missing 'best_model' in payload.")

    ds_resp = (
        supabase.table("datasets")
        .select("*")
        .eq("id", dataset_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not ds_resp.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ds = ds_resp.data

    bm_name = best_model.get("name")
    bm_metrics = best_model.get("metrics") or {}
    primary_metric = best_model.get("primary_metric") or "RMSE"
    primary_metric_value = float(best_model.get("primary_metric_value", np.nan))

    if not bm_name:
        raise HTTPException(status_code=400, detail="Best model 'name' is required.")

    if artifact_path:
        try:
            res = supabase.storage.from_("models").download(artifact_path)
            if res is None:
                raise Exception("Empty artifact.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cannot read model artifact at '{artifact_path}': {e}")

    payload = {
        "user_id": current_user.id,
        "dataset_id": dataset_id,
        "model_name": bm_name,
        "algorithm": bm_name,
        "status": "ready",
        "primary_metric": f"{primary_metric}: {primary_metric_value:.4f}",
        "train_metrics": make_json_safe(bm_metrics),
        "test_metrics": make_json_safe({primary_metric: primary_metric_value}),
        "training_report": make_json_safe(
            {
                "primary_metric": primary_metric,
                "test_size": body.get("test_size"),
                "top_k": body.get("top_k"),
                "n_splits": body.get("n_splits"),
                "clip": body.get("clip"),
            }
        ),
        "evaluation_report": make_json_safe(best_model.get("evaluation_report") or {}),
        "model_plots": make_json_safe(best_model.get("model_plots") or {}),
        "artifacts_path": artifact_path,
        "target_column": ds.get("target_column"),
        "forecast_horizon": ds.get("forecast_horizon"),
    }

    existing = (
        supabase.table("models")
        .select("id")
        .eq("user_id", current_user.id)
        .eq("dataset_id", dataset_id)
        .limit(1)
        .execute()
    )

    if existing.data:
        model_id = existing.data[0]["id"]
        up_resp = supabase.table("models").update(payload).eq("id", model_id).eq("user_id", current_user.id).execute()
        saved = up_resp.data[0]
    else:
        ins_resp = supabase.table("models").insert(payload).execute()
        saved = ins_resp.data[0]

    return {"model_id": saved["id"]}


@app.delete("/api/v1/datasets/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str, current_user: Any = Depends(get_current_user)):
    resp = (
        supabase.table("datasets")
        .select("id, user_id, storage_original_path, storage_parquet_path, processed_parquet_path")
        .eq("id", dataset_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ds = resp.data

    mdl_resp = (
        supabase.table("models")
        .select("id, artifacts_path, explain_artifacts_path, feature_scaler_path, target_scaler_path")
        .eq("user_id", current_user.id)
        .eq("dataset_id", dataset_id)
        .execute()
    )
    models = mdl_resp.data or []
    for m in models:
        _remove_from_storage("models", m.get("artifacts_path"))
        _remove_from_storage("models", m.get("explain_artifacts_path"))
        _remove_from_storage("models", m.get("feature_scaler_path"))
        _remove_from_storage("models", m.get("target_scaler_path"))

    supabase.table("models").delete().eq("user_id", current_user.id).eq("dataset_id", dataset_id).execute()

    _remove_from_storage("datasets", ds.get("storage_original_path"))
    _remove_from_storage("datasets", ds.get("storage_parquet_path"))
    _remove_from_storage("datasets", ds.get("processed_parquet_path"))

    supabase.table("datasets").delete().eq("id", dataset_id).eq("user_id", current_user.id).execute()
    return


@app.delete("/api/v1/models/{model_id}", status_code=204)
async def delete_model(model_id: str, current_user: Any = Depends(get_current_user)):
    resp = (
        supabase.table("models")
        .select("id, user_id, artifacts_path, explain_artifacts_path, feature_scaler_path, target_scaler_path")
        .eq("id", model_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Model not found")
    
    _remove_from_storage("models", resp.data.get("artifacts_path"))
    _remove_from_storage("models", resp.data.get("explain_artifacts_path"))
    _remove_from_storage("models", resp.data.get("feature_scaler_path"))
    _remove_from_storage("models", resp.data.get("target_scaler_path"))

    supabase.table("models").delete().eq("id", model_id).eq("user_id", current_user.id).execute()
    return

# ---------------------------------------------------------
# Explain endpoint 
# ---------------------------------------------------------

def _infer_model_family(algorithm: str) -> str:
    alg = (algorithm or "").lower()
    if any(x in alg for x in ["random forest", "xgboost", "lightgbm", "catboost", "gradient boosting"]):
        return "gbm"
    if any(x in alg for x in ["linear", "ridge", "lasso", "elastic"]):
        return "linear"
    if any(x in alg for x in ["lstm", "tcn", "tft", "gru"]):
        return "deep"
    return "gbm"


@app.get("/api/v1/models/{model_id}/explain")
async def explain_model(model_id: str, current_user: Any = Depends(get_current_user)):
    m_resp = (
        supabase.table("models")
        .select("*")
        .eq("id", model_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not m_resp.data:
        raise HTTPException(status_code=404, detail="Model not found")

    mrow: Dict[str, Any] = m_resp.data

    ex_status = (mrow.get("explain_status") or "none").lower()
    ex_path = mrow.get("explain_artifacts_path")
    ex_report = mrow.get("explain_report") or {}

    if ex_status == "ready" and ex_path:
        raw = _download_bytes_from_storage("models", ex_path)
        bio = BytesIO(raw)
        npz = np.load(bio, allow_pickle=False)
        arrays = {k: npz[k].tolist() for k in npz.files}

        return make_json_safe({
            "mode": "precomputed",
            "model_id": model_id,
            "dataset_id": mrow.get("dataset_id"),
            "algorithm": mrow.get("algorithm") or mrow.get("model_name"),
            "explain_status": ex_status,
            "explain_report": ex_report,
            "arrays": arrays,
        })

    dataset_id = mrow.get("dataset_id")
    algorithm = mrow.get("algorithm") or mrow.get("model_name")
    artifacts_path = mrow.get("artifacts_path")
    target_col = mrow.get("target_column")

    if not dataset_id:
        raise HTTPException(status_code=400, detail="Model row has no dataset_id.")
    if not target_col:
        raise HTTPException(status_code=400, detail="Model row has no target_column.")
    if not artifacts_path:
        raise HTTPException(status_code=400, detail="Model has no artifacts_path (no saved artifact).")

    d_resp = (
        supabase.table("datasets")
        .select("*")
        .eq("id", dataset_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not d_resp.data:
        raise HTTPException(status_code=404, detail="Dataset for this model not found")

    drow: Dict[str, Any] = d_resp.data
    time_col = drow.get("time_column")
    is_multi_entity = bool(drow.get("is_multi_entity"))
    entity_col = drow.get("entity_column")

    processed_path = drow.get("processed_parquet_path") or drow.get("storage_parquet_path")
    if not processed_path:
        raise HTTPException(status_code=400, detail="Dataset has no parquet to explain.")

    df = _download_parquet_from_storage("datasets", processed_path)

    training_report = mrow.get("training_report") or {}
    test_size = training_report.get("test_size", 0.2)
    try:
        test_size = float(test_size)
    except Exception:
        test_size = 0.2
    test_size = max(0.05, min(test_size, 0.5))

    if is_multi_entity and entity_col in df.columns and time_col in df.columns:
        df = df.sort_values([entity_col, time_col]).reset_index(drop=True)
    else:
        if time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)

    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found for explain.")

    n = len(df)
    n_test = max(1, int(n * test_size))
    n_train = n - n_test
    if n_train <= 0:
        raise HTTPException(status_code=400, detail="Not enough rows for train/test split in explain.")

    exclude_cols = [target_col]
    if time_col:
        exclude_cols.append(time_col)
    if is_multi_entity and entity_col:
        exclude_cols.append(entity_col)

    feature_cols = df.columns.difference(exclude_cols).tolist()

    X_all = df[feature_cols]
    y_all = df[target_col]

    X_train = X_all.iloc[:n_train, :]
    X_val = X_all.iloc[n_train:, :]
    y_val = y_all.iloc[n_train:]

    model = _download_model_artifact("models", artifacts_path)

    model_family = _infer_model_family(str(algorithm))

    expl = ExplainX(
        model=model,
        model_family=model_family,
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=list(feature_cols),
        task_name=target_col,
    )

    try:
        g = expl.compute_shap_global(max_samples=1000)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute global SHAP: {e}")

    shap_values = np.asarray(g["shap_values"])
    if shap_values.ndim != 2:
        raise HTTPException(status_code=500, detail=f"Unexpected SHAP shape: {shap_values.shape}")

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0).tolist()

    try:
        imp_df = expl.compute_permutation_importance(n_repeats=10, random_state=42)
        perm_importance = imp_df.to_dict(orient="records")
    except Exception as e:
        print(f"[WARN] Permutation importance failed: {e}")
        perm_importance = []

    global_importance = sorted(
        [{"feature": f, "mean_abs_shap": float(v)} for f, v in zip(list(feature_cols), mean_abs_shap)],
        key=lambda x: x["mean_abs_shap"],
        reverse=True,
    )

    return make_json_safe(
        {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "algorithm": algorithm,
            "model_family": model_family,
            "target_column": target_col,
            "feature_names": list(feature_cols),
            "global_importance": global_importance,
            "permutation_importance": perm_importance,
            "meta": {
                "n_train": int(n_train),
                "n_val": int(n_test),
                "test_size": test_size,
                "artifacts_path": artifacts_path,
            },
        }
    )


@app.get("/api/v1/monitoring/summary")
async def monitoring_summary(current_user: Any = Depends(get_current_user)):
    """
    Small summary for monitoring page:
      - counts datasets/models
      - last dataset/model timestamps
      - status breakdown (basic)
    """
    # datasets
    ds_resp = (
        supabase.table("datasets")
        .select("id, status, created_at")
        .eq("user_id", current_user.id)
        .order("created_at", desc=True)
        .execute()
    )
    ds = ds_resp.data or []

    # models
    m_resp = (
        supabase.table("models")
        .select("id, status, created_at")
        .eq("user_id", current_user.id)
        .order("created_at", desc=True)
        .execute()
    )
    ms = m_resp.data or []

    def _count_by_status(rows):
        out = {}
        for r in rows:
            s = (r.get("status") or "unknown")
            out[s] = out.get(s, 0) + 1
        return out

    return {
        "datasets_count": len(ds),
        "models_count": len(ms),
        "datasets_by_status": _count_by_status(ds),
        "models_by_status": _count_by_status(ms),
        "last_dataset_created_at": ds[0]["created_at"] if ds else None,
        "last_model_created_at": ms[0]["created_at"] if ms else None,
    }

from pydantic import BaseModel, Field
from typing import Literal, Optional, Union

class PredictRequestBody(BaseModel):
    dataset_id: str
    model_id: str

    # entity controls
    entity_scope: Literal["one", "all"] = "one"
    entity_value: Optional[Union[str, int, float]] = None

    # forecast controls
    mode: Literal["one_step", "multi_step"] = "multi_step"
    steps: int = Field(1, ge=1)

    horizon: Optional[int] = None


@app.get("/api/v1/datasets/{dataset_id}/entities")
async def list_dataset_entities(dataset_id: str, current_user: Any = Depends(get_current_user)):
    """
    Returns distinct entity values for multi-entity datasets.
    FE expects exactly:
      { entity_column: string | null, values: string[] }
    """
    resp = (
        supabase.table("datasets")
        .select("id,user_id,is_multi_entity,entity_column,time_column,processed_parquet_path,storage_parquet_path")
        .eq("id", dataset_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    row = resp.data
    is_multi_entity = bool(row.get("is_multi_entity"))
    entity_col = row.get("entity_column")
    time_col = row.get("time_column")

    if not is_multi_entity or not entity_col:
        return {"entity_column": None, "values": []}

    parquet_path = row.get("processed_parquet_path") or row.get("storage_parquet_path")
    if not parquet_path:
        raise HTTPException(status_code=400, detail="No parquet available for this dataset.")

    df = _download_parquet_from_storage("datasets", parquet_path)

    if entity_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Entity column '{entity_col}' not found in parquet.")

    if time_col and time_col in df.columns:
        df = df.sort_values([entity_col, time_col])

    values = df[entity_col].dropna().astype(str).unique().tolist()
    values = values[:5000]

    return {"entity_column": entity_col, "values": values}

@app.post("/api/v1/predict")
async def predict_api(body: PredictRequestBody, current_user: Any = Depends(get_current_user)):
    # 1) dataset
    d_resp = (
        supabase.table("datasets")
        .select("*")
        .eq("id", body.dataset_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not d_resp.data:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ds: Dict[str, Any] = d_resp.data

    # 2) model
    m_resp = (
        supabase.table("models")
        .select("*")
        .eq("id", body.model_id)
        .eq("user_id", current_user.id)
        .single()
        .execute()
    )
    if not m_resp.data:
        raise HTTPException(status_code=404, detail="Model not found")
    mrow: Dict[str, Any] = m_resp.data

    if str(mrow.get("dataset_id") or "") != str(ds.get("id") or ""):
        raise HTTPException(status_code=400, detail="Model is not trained on this dataset")

    # 3) parquet
    parquet_path = ds.get("processed_parquet_path") or ds.get("storage_parquet_path")
    if not parquet_path:
        raise HTTPException(status_code=400, detail="No parquet available for dataset")
    df = _download_parquet_from_storage("datasets", parquet_path)

    # 4) load artifact
    artifacts_path = mrow.get("artifacts_path")
    if not artifacts_path:
        raise HTTPException(status_code=400, detail="Model has no artifacts_path")
    model_obj = _download_model_artifact("models", artifacts_path)

    # 5) Infer model type
    training_report = mrow.get("training_report") or {}
    best_model_type = training_report.get("best_model_type") or mrow.get("algorithm") or ""
    
    is_seq = (
        "keras" in str(type(model_obj)).lower() or
        any(name in str(best_model_type).lower() for name in ["lstm", "tcn", "tft", "gru"])
    )
    model_type = "seq" if is_seq else "tabular"
    
    # 6) Load scalers if sequence model 
    feature_scaler = None
    target_scaler = None

    if is_seq:
        fs_path = mrow.get("feature_scaler_path")
        ts_path = mrow.get("target_scaler_path")

        if fs_path and ts_path:
            try:
                feature_scaler = _download_scaler_obj(fs_path)
            except Exception as e:
                print(f"[WARN] Failed to load feature_scaler from {fs_path}: {e}")

            try:
                target_scaler = _download_scaler_obj(ts_path)
            except Exception as e:
                print(f"[WARN] Failed to load target_scaler from {ts_path}: {e}")
        else:
            print(f"[WARN] Seq model but scaler paths missing in DB for model {body.model_id}")


    # 7) config 
    ds_horizon = int(ds.get("forecast_horizon") or 1)
    horizon = int(body.horizon or ds_horizon)
    if horizon < 1:
        horizon = 1

    steps = 1 if body.mode == "one_step" else int(body.steps or horizon)
    steps = max(1, min(steps, max(1, horizon)))

    lookback = int(training_report.get("lookback", 30))

    cfg = PredictionConfig(
        time_col=ds.get("time_column"),
        target_col=ds.get("target_column"),
        is_multi_entity=bool(ds.get("is_multi_entity")),
        entity_col=ds.get("entity_column"),
        horizon=horizon,
        mode=body.mode,
        steps=steps,
        entity_scope=body.entity_scope,
        entity_value=body.entity_value,
        max_entities=200,
        lookback=lookback,
        model_type=model_type,
    )

    # 8) run
    try:
        px = PredictX(
            model=model_obj, 
            cfg=cfg, 
            training_report=training_report,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
        )
        out = px.run(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict failed: {e}")

    return make_json_safe({
        "ok": True,
        "dataset_id": body.dataset_id,
        "model_id": body.model_id,
        "mode": body.mode,
        "steps": steps,
        "horizon": horizon,
        "model_type": model_type,
        "entity_scope": body.entity_scope,
        "entity_value": body.entity_value,
        "artifacts_path": artifacts_path,
        "scalers_loaded": {
            "feature_scaler": feature_scaler is not None,
            "target_scaler": target_scaler is not None,
        },
        "payload": out,
    })
