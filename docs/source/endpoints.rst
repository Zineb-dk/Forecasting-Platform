Backend API Endpoints (Developer Reference)
===========================================

This section documents the FastAPI backend endpoints implemented in the project.
  
Base URL
--------

The backend is a FastAPI application. All routes below are defined in the main API module.

CORS is configured to allow:

- ``http://localhost:3000``
- ``http://127.0.0.1:3000``

Authentication
--------------

Most endpoints require authentication using an HTTP Bearer token.

**Auth mechanism (from code):**
- The API uses ``HTTPBearer`` and validates tokens via ``supabase.auth.get_user(token)``.
- If the token is missing/invalid, the API returns ``401``.

**Header:**

.. code-block:: http

   Authorization: Bearer <SUPABASE_ACCESS_TOKEN>

Endpoints without auth
^^^^^^^^^^^^^^^^^^^^^

- ``GET /api/health`` does **not** require authentication.

Response conventions
--------------------

- Success responses are JSON.
- Errors are raised using ``HTTPException`` and use the JSON shape:

.. code-block:: json

   { "detail": "..." }

The server also converts complex objects into JSON-safe formats via ``make_json_safe()`` (e.g., numpy types, pandas timestamps).

Health
------

GET /api/health
^^^^^^^^^^^^^^^

**Description**
  Basic service health check.

**Auth**
  No

**Response**
- ``200 OK``

.. code-block:: json

   { "status": "ok" }

Datasets
--------

GET /api/v1/datasets
^^^^^^^^^^^^^^^^^^^^

**Description**
  List datasets belonging to the authenticated user (ordered by ``created_at`` descending).

**Auth**
  Yes

**Response**
- ``200 OK``: returns a JSON array (rows from Supabase ``datasets`` table).

GET /api/v1/datasets/{dataset_id}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Fetch a single dataset row by ``dataset_id`` for the authenticated user.

**Auth**
  Yes

**Path params**
- ``dataset_id``: string

**Responses**
- ``200 OK``: dataset row
- ``404``: ``Dataset not found``

POST /api/v1/data/upload
^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Upload a dataset file, ingest it into a pandas DataFrame, standardize it, profile it, store:
  - original file bytes in Supabase Storage bucket ``datasets``
  - standardized parquet in Supabase Storage bucket ``datasets``
  - dataset metadata row in Supabase table ``datasets``

**Auth**
  Yes

**Request**
Multipart form-data with:

- ``file`` (UploadFile, required)
- ``target_feature`` (string, required)
- ``time_column`` (string, required)
- ``forecast_horizon`` (int, required)
- ``is_multi_entity`` (bool, optional, default false)
- ``entity_column`` (string, optional)

**Validation rules enforced**
- File must be non-empty.
- ``target_feature`` must exist in standardized columns.
- ``time_column`` must exist in standardized columns.
- If ``is_multi_entity=True``:
  - ``entity_column`` is required
  - must exist in columns
  - must be different from target and time columns

**Server-side computed fields (high level)**
- ``file_hash`` = sha256 over uploaded bytes
- ``sensor_columns`` = all columns except time/target/(entity)

**Responses**
- ``200 OK``: includes dataset row + dataset_id

.. code-block:: json

   {
     "message": "Dataset uploaded and registered successfully",
     "dataset_id": "...",
     "dataset": { "...": "..." },
     "original_filename": "..."
   }

- ``400``: empty file / missing columns / ingestion or profiling errors
- ``500``: ingestion failure / storage failure / DB insert error

GET /api/v1/datasets/{dataset_id}/eda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Returns:
  - dataset row
  - stored ``ingestion_info``
  - preview of first 50 rows
  - numeric histograms for numeric columns (20 bins)
  - a ``target_series`` either against time (if time column exists) or index

The parquet used is:
- ``processed_parquet_path`` if present, else
- ``storage_parquet_path``

**Auth**
  Yes

**Path params**
- ``dataset_id``: string

**Responses**
- ``200 OK``

Response keys:

- ``dataset``: dataset row
- ``ingestion_info``: JSON object (or empty)
- ``preview``: list of up to 50 records
- ``numeric_histograms``: dict of per-column histogram data
- ``target_series``: optional dict (depends on target/time availability)

- ``404``: dataset not found
- ``400``: no parquet stored
- ``500``: parquet download/read failures

POST /api/v1/datasets/{dataset_id}/preprocess
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Loads the dataset parquet, runs preprocessing via:

- ``DataForge(cfg).preprocess(df_raw)``

Then stores:
- cleaned parquet to ``datasets`` bucket (path stored in ``processed_parquet_path``)
- preprocessing report to ``processing_report``
- updates dataset ``status`` to ``processed``

Sorting behavior after preprocessing:
- if multi-entity and entity column exists: sort by ``[entity_col, time_col]``
- else: sort by ``time_col``

**Auth**
  Yes

**Path params**
- ``dataset_id``: string

**Responses**
- ``200 OK`` returns:

.. code-block:: json

   {
     "status": "ok",
     "dataset_id": "...",
     "time_col": "...",
     "target_col": "...",
     "entity_col": "... or null",
     "is_multi_entity": true/false,
     "sensor_cols": ["..."],
     "processed_parquet_path": "...",
     "processing_report": { "...": "..." },
     "n_rows_clean": 123,
     "n_cols_clean": 45
   }

- ``404``: dataset not found
- ``400``: missing required dataset metadata (time/target/entity) or missing parquet
- ``500``: storage update failures / DB update failures

GET /api/v1/datasets/{dataset_id}/report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Returns the dataset preprocessing report stored in the dataset row.

**Auth**
  Yes

**Path params**
- ``dataset_id``: string

**Responses**
- ``200 OK``

.. code-block:: json

   {
     "dataset_id": "...",
     "report": { "...": "..." },
     "processed_parquet_path": "...",
     "status": "..."
   }

- ``404``: dataset not found

DELETE /api/v1/datasets/{dataset_id}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Deletes:
  - dataset row in ``datasets``
  - all model rows in ``models`` linked to this dataset
  - storage objects for:
    - dataset original file
    - dataset parquet
    - processed parquet
    - each model artifact
    - each ExplainX artifact
    - each feature/target scaler object

**Auth**
  Yes

**Responses**
- ``204 No Content``
- ``404``: dataset not found

Models
------

GET /api/v1/models
^^^^^^^^^^^^^^^^^^

**Description**
  Lists models belonging to the authenticated user, selecting only:

- ``id, user_id, dataset_id, model_name, algorithm, status, primary_metric, test_metrics, created_at, explain_status``

Ordered by ``created_at`` descending.

**Auth**
  Yes

**Response**
- ``200 OK``: array of model rows (selected fields only)

GET /api/v1/models/{model_id}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Fetch a single model row (all columns) for the authenticated user.

**Auth**
  Yes

**Path params**
- ``model_id``: string

**Responses**
- ``200 OK`` model row
- ``404`` model not found

DELETE /api/v1/models/{model_id}
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Deletes:
  - model row in ``models``
  - storage objects for:
    - model artifact (``artifacts_path``)
    - ExplainX npz artifact (``explain_artifacts_path``)
    - feature scaler (``feature_scaler_path``)
    - target scaler (``target_scaler_path``)

**Auth**
  Yes

**Responses**
- ``204 No Content``
- ``404`` model not found

AutoML
------

POST /api/v1/datasets/{dataset_id}/automl/run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Runs AutoML on the dataset's **processed parquet**:

1) Load processed parquet (requires ``processed_parquet_path``).
2) Build ``DataConfig`` (time/target/entity/sensors).
3) Build ``AutoMLConfig`` from request JSON.
4) Run ``AutoMLPipeline(data_cfg, cfg_automl).run_pipeline(df_clean)``.
5) Serialize and upload the best model artifact to Supabase Storage bucket ``models``.
6) Optionally upload feature/target scalers (if present in summary).
7) Build and store ExplainX artifacts + JSON report using ``build_and_store_best_explainx``.
8) Insert a row into Supabase ``models`` table (best model).

**Auth**
  Yes

**Path params**
- ``dataset_id``: string

**Request body**
JSON dict (all keys are optional unless noted):

- ``models_to_train``: list (required logically; AutoMLConfig enforces non-empty)
- ``primary_metric``: string default ``"RMSE"``
- ``test_size``: float default ``0.2``
- ``use_clip``: bool default false
- ``clip_threshold``: float optional (used if ``use_clip`` true; defaults to 125.0 if missing)
- ``top_k``: int default 2
- ``n_splits``: int default 3
- ``lookback``: int default 50
- ``epochs``: int default 10
- ``batch_size``: int default 64
- ``do_plots``: bool default false

**Responses**
- ``200 OK`` returns a summary payload:

Top-level keys:

- ``dataset_id``
- ``primary_metric``
- ``steps``: list of step status objects
- ``models``: list of per-model summary objects
- ``best_model_id``
- ``best_model_name``
- ``best_model_type``  (as returned by the pipeline)
- ``best_avg_metric``  (if available and finite)
- ``plots`` (from AutoML summary plot payload)

- ``400`` dataset not preprocessed / config invalid
- ``404`` dataset not found
- ``500`` AutoML run failed / storage/db failures

POST /api/v1/datasets/{dataset_id}/models/save-best
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Backward-compatibility endpoint.

It expects a payload containing ``best_model`` and stores a model record in the DB.
If a model row already exists for (user_id, dataset_id), it updates it; otherwise inserts a new one.

**Auth**
  Yes

**Request body**
JSON dict:

- ``best_model``: required object containing at least:
  - ``name`` (required)
  - ``metrics`` (optional dict)
  - ``primary_metric`` (optional string)
  - ``primary_metric_value`` (optional number)
  - ``evaluation_report`` (optional)
  - ``model_plots`` (optional)
- ``artifact_path``: optional string (if provided, server attempts to download it from ``models`` bucket)
- plus optional values used in training_report:
  - ``test_size``, ``top_k``, ``n_splits``, ``clip``

**Responses**
- ``200 OK``

.. code-block:: json

   { "model_id": "..." }

- ``400`` missing best_model / missing best_model.name
- ``404`` dataset not found
- ``500`` artifact not readable if artifact_path provided

Explainability
--------------

GET /api/v1/models/{model_id}/explainx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Returns **stored ExplainX artifacts** for the model (if present).
It loads ``explain_artifacts_path`` from storage bucket ``models`` and returns all arrays from the NPZ as JSON lists.

**Auth**
  Yes

**Path params**
- ``model_id``: string

**Response**
- ``200 OK``

.. code-block:: json

   {
     "model_id": "...",
     "explain_status": "ready|failed|none|...",
     "explain_report": { ... },
     "kind": "tabular|deep|...",
     "arrays": { "key": [...], "...": ... },
     "keys": ["..."]
   }

Notes:
- ``arrays`` is ``null`` if no artifact path is stored.
- ``kind`` is derived from ``explain_report.kind``.

GET /api/v1/models/{model_id}/explain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Explain endpoint with two modes:

1) If model row has:
   - ``explain_status == "ready"`` AND
   - ``explain_artifacts_path`` present
   then it returns the saved arrays/report (no recomputation).

2) Otherwise, it recomputes SHAP + permutation importance:
   - Loads dataset parquet (processed if available)
   - Performs time-ordered train/test split using ``training_report.test_size`` (clamped to [0.05, 0.5])
   - Loads model artifact from storage (supports joblib/json/keras/pickle per loader helper)
   - Infers model family using ``_infer_model_family(algorithm)``
   - Uses ``ExplainX.compute_shap_global(max_samples=1000)``
   - Computes permutation importance (best-effort; errors return empty list)

**Auth**
  Yes

**Path params**
- ``model_id``: string

**Responses**
- ``200 OK``:
  - precomputed mode includes arrays
  - recomputed mode returns importance summaries

- ``400`` missing required model metadata (dataset_id, target_column, artifacts_path, etc.)
- ``404`` model not found / dataset not found
- ``500`` failures in SHAP computation, model loading, parquet loading

Monitoring
----------

GET /api/v1/monitoring/summary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Returns a small monitoring summary:

- dataset and model counts
- breakdown by ``status``
- last created timestamps (dataset + model)

**Auth**
  Yes

**Response**
- ``200 OK``

.. code-block:: json

   {
     "datasets_count": 0,
     "models_count": 0,
     "datasets_by_status": { "ready": 1, "processed": 2, "...": 0 },
     "models_by_status": { "ready": 1, "...": 0 },
     "last_dataset_created_at": "... or null",
     "last_model_created_at": "... or null"
   }

Entities
--------

GET /api/v1/datasets/{dataset_id}/entities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Description**
  Returns distinct entity values for multi-entity datasets.
If dataset is not multi-entity or has no entity column, returns empty list.

Parquet used:
- ``processed_parquet_path`` if present, else ``storage_parquet_path``.

**Auth**
  Yes

**Path params**
- ``dataset_id``: string

**Response**
- ``200 OK``

.. code-block:: json

   { "entity_column": "machine_id", "values": ["1","2","3"] }

Notes:
- Values are converted to strings.
- Returned list is capped to 5000 values.

Predictions
-----------

POST /api/v1/predict
^^^^^^^^^^^^^^^^^^^^

**Description**
  Runs prediction using a stored model artifact and dataset parquet:

Flow (as implemented):
1) Load dataset row by ``dataset_id``.
2) Load model row by ``model_id``.
3) Ensure the model belongs to the dataset.
4) Load parquet (processed if available).
5) Download model artifact from ``models`` bucket via ``artifacts_path``.
6) Infer model type (tabular vs sequence) using:
   - model object type contains "keras", or
   - algorithm/name contains one of: lstm/tcn/tft/gru (string heuristic)
7) If sequence model, attempt to load scalers from:
   - ``feature_scaler_path``
   - ``target_scaler_path``
8) Build ``PredictionConfig`` (server enforces horizon/steps bounds).
9) Run ``PredictX(...).run(df)`` and return result.

**Auth**
  Yes

**Request body schema**
The request body is a Pydantic model:

- ``dataset_id`` (string, required)
- ``model_id`` (string, required)
- ``entity_scope``: ``"one"`` or ``"all"`` (default ``"one"``)
- ``entity_value``: optional (string/int/float)
- ``mode``: ``"one_step"`` or ``"multi_step"`` (default ``"multi_step"``)
- ``steps``: int >= 1 (default 1)
- ``horizon``: optional int (if absent, server uses dataset ``forecast_horizon``)

**Important server-side rules**
- Horizon used = ``body.horizon`` if provided, else dataset ``forecast_horizon``.
- Steps:
  - if mode == one_step -> steps forced to 1
  - else steps = min(requested steps, horizon), with lower bound 1
- ``lookback`` is read from model ``training_report.lookback`` (default 30).

**Response**
- ``200 OK``

.. code-block:: json

   {
     "ok": true,
     "dataset_id": "...",
     "model_id": "...",
     "mode": "multi_step",
     "steps": 3,
     "horizon": 10,
     "model_type": "tabular|seq",
     "entity_scope": "one|all",
     "entity_value": "...",
     "artifacts_path": "...",
     "scalers_loaded": { "feature_scaler": true/false, "target_scaler": true/false },
     "payload": { "...": "PredictX output (json-safe)" }
   }

**Errors**
- ``400``: validation errors (e.g., entity missing, no feature columns found, etc.)
- ``404``: dataset/model not found
- ``500``: storage/model loading errors, unexpected prediction failures
