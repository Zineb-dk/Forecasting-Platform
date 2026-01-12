AutoML Module
=============

Location
--------

``backend/modules/automl.py``

Overview
--------

This module defines an AutoML pipeline that can:

- prepare feature/target matrices from a cleaned DataFrame
- train and benchmark multiple tabular regression models
- train and benchmark deep sequence models using sliding-window sequences
- compute regression metrics (MAE, RMSE, R2, MAPE, sMAPE)
- perform walk-forward validation using ``TimeSeriesSplit`` for both tabular and sequence models
- retrain the best model on the full dataset
- optionally produce matplotlib diagnostic plots (non-interactive backend)

The module uses scikit-learn for tabular models and metrics, XGBoost for gradient boosting,
and TensorFlow/Keras for deep sequence models.

Configuration
------------

AutoMLConfig
^^^^^^^^^^^^

.. class:: AutoMLConfig

A dataclass specifying which models to train and how the pipeline should evaluate them.

**Fields**

- ``models_to_train`` (List[str]): Required. List of model names to train.
- ``primary_metric`` (str): Metric used to rank models. Default: ``"RMSE"``.
- ``test_size`` (float): Hold-out fraction used for chronological splits. Default: ``0.2``.
- ``clip_threshold`` (Optional[float]): Optional upper bound applied to the target. Default: ``None``.
- ``top_k`` (int): Number of top models (by benchmark metric) selected for walk-forward. Default: ``2``.
- ``n_splits`` (int): Number of ``TimeSeriesSplit`` folds for walk-forward. Default: ``5``.
- ``lookback`` (int): Sliding window length for sequence models. Default: ``30``.
- ``epochs`` (int): Training epochs for deep models. Default: ``50``.
- ``batch_size`` (int): Batch size for deep models. Default: ``32``.
- ``do_plots`` (bool): Whether to run plotting routines. Default: ``True``.

Validation (``__post_init__``)

- Raises ``ValueError`` if ``models_to_train`` is empty.
- Validates ``primary_metric`` ∈ ``{"RMSE", "MAE", "R2", "MAPE", "sMAPE"}``.
- Validates ``0 < test_size < 1``.

AutoMLPipeline
--------------

.. class:: AutoMLPipeline

Main class implementing training, benchmarking, walk-forward validation, and selection.

Construction
^^^^^^^^^^^^

.. method:: AutoMLPipeline.__init__(data_cfg: DataConfig, automl_cfg: AutoMLConfig)

Stores configuration values and validates requested model names.

Known model registries (as defined in the code)

Tabular model builders (``TABULAR_MODEL_BUILDERS``):

- ``"Random Forest"`` → :meth:`build_random_forest`
- ``"XGBoost"`` → :meth:`build_xgboost`

Sequence model builders (``SEQ_MODEL_BUILDERS``):

- ``"LSTM"`` → :meth:`build_lstm`
- ``"TCN"`` → :meth:`build_tcn`
- ``"TFT"`` → :meth:`build_tft`

Validation performed at initialization:

- Builds a list of any requested models not present in either registry.
- Raises ``ValueError`` for unknown model names, listing the available names.
- Populates:

  - ``self.tabular_models``: requested models that are in ``TABULAR_MODEL_BUILDERS``
  - ``self.seq_models``: requested models that are in ``SEQ_MODEL_BUILDERS``

State attributes
^^^^^^^^^^^^^^^^

The pipeline stores results and selections after running:

- ``results_``: dict of per-model summary metrics for ranking (set in :meth:`run_pipeline`)
- ``best_model_name_``: name of best model selected by walk-forward average metric
- ``best_model_``: fitted model instance
- ``meta_``: metadata produced by :meth:`prepare_X_y`
- ``plot_payload_``: dictionary containing plot-ready arrays and summaries

It also stores scalers used for sequence models:

- ``feature_scaler_``: fitted ``StandardScaler`` for features (sequence models only)
- ``target_scaler_``: fitted ``StandardScaler`` for target (sequence models only)

Small Helpers (Plotting Support)
--------------------------------

_to_1d_array
^^^^^^^^^^^^

.. method:: AutoMLPipeline._to_1d_array(arr) -> numpy.ndarray
   :staticmethod:

Converts input into a flattened 1D numpy array using ``np.asarray(arr).reshape(-1)``.

_downsample_indices
^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline._downsample_indices(n: int, max_points: Optional[int])
   :staticmethod:

Returns indices for downsampling:

- If ``max_points`` is None or ``max_points >= n``, returns ``slice(None)``.
- Otherwise returns an integer index array created by ``np.linspace(0, n-1, max_points).astype(int)``.

Data Preparation
----------------

prepare_X_y
^^^^^^^^^^^

.. method:: AutoMLPipeline.prepare_X_y(df_clean: pandas.DataFrame) -> (pandas.DataFrame, pandas.Series, dict)

Builds the feature matrix ``X`` and target vector ``y`` from a cleaned DataFrame using ``DataConfig``.

**Target handling**

- If ``self.clip_threshold`` is not None:
  - creates column ``"target_clipped"`` as ``df[cfg.target_col].clip(upper=clip_threshold)``
  - uses it as ``y``
  - sets ``helper_col = "target_clipped"``
- Else uses ``df[cfg.target_col]`` directly and ``helper_col = None``

**Features selection**

Builds ``cols_to_remove`` from:

- ``cfg.time_col``
- ``cfg.target_col``
- ``cfg.entity_col`` (only if ``cfg.is_multi_entity`` and ``cfg.entity_col is not None``)
- ``helper_col`` if used

Then defines:

- ``train_features = df.columns.difference(cols_to_remove)``
- ``X = df[train_features]``

**Metadata produced**

Stores and returns a dict:

- ``clip_threshold``
- ``helper_col``
- ``train_features`` (list)
- ``n_samples``
- ``n_features``

time_train_test_split
^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.time_train_test_split(X: pandas.DataFrame, y: pandas.Series) -> (X_train, X_test, y_train, y_test)

Performs a chronological split without shuffling using ``self.test_size``.

Validation:

- Raises ``ValueError`` if ``len(y) != len(X)``.
- Raises ``ValueError`` if ``test_size`` is not between 0 and 1.

Split logic:

- ``n_test = floor(n * test_size)``
- ``n_train = n - n_test``
- Train set = first ``n_train`` samples
- Test set = remaining ``n_test`` samples

Sequence Building
-----------------

build_sequences_from_df
^^^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.build_sequences_from_df(df_clean: pandas.DataFrame, scale_features: bool = True) -> (X_seq, y_seq, y_seq_original)

Builds sliding-window sequences for deep models.

**Validation**

- Requires that ``cfg.time_col`` exists in ``df_clean``.
- Requires that ``cfg.time_col`` dtype is numeric or datetime64; otherwise raises ``ValueError``.
- If multi-entity, requires that ``cfg.entity_col`` exists in ``df_clean``.

**Sorting**

- If multi-entity: sorts by ``[cfg.entity_col, cfg.time_col]``.
- Else: sorts by ``cfg.time_col``.

**Feature/target construction**

- Calls :meth:`prepare_X_y` to get tabular features and target.
- Stores the original target values in ``y_original`` as float32.

**Scaling (when scale_features=True)**

- Fits ``self.feature_scaler_ = StandardScaler()`` on features and transforms them.
- Fits ``self.target_scaler_ = StandardScaler()`` on target and transforms it.
- Uses scaled arrays for sequence creation.

When scale_features=False:

- uses raw feature values and raw target values (float32),
- and returns ``y_seq_original = None`` (as implemented by the final return statement).

**Monotonic time validation**

- Uses :meth:`_is_monotonic_increasing` to ensure time values are monotonic increasing
  (allows equal consecutive values) after sorting.
- Raises ``ValueError`` if time is not monotonic increasing.

**Sliding window logic**

For each entity (multi-entity case) or globally (single-entity case):

- Requires ``len(samples) > lookback``; otherwise:
  - multi-entity: prints a warning and skips the entity
  - single-entity: raises ``ValueError``

For each index ``i`` from ``lookback`` to end:

- appends ``X[i-lookback : i, :]`` to sequence features
- appends ``y[i]`` to sequence target
- appends original-scale target ``y_original[i]`` to ``y_seq_original`` tracking

Outputs:

- ``X_seq`` shape: (n_samples, lookback, n_features)
- ``y_seq`` shape: (n_samples,)
- ``y_seq_original`` shape: (n_samples,) when returned (only when scale_features=True)

_is_monotonic_increasing
^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline._is_monotonic_increasing(arr: numpy.ndarray) -> bool
   :staticmethod:

Returns True if the array is monotonically increasing allowing equality:
``np.all(arr[1:] >= arr[:-1])``.

seq_train_test_split
^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.seq_train_test_split(X_seq: numpy.ndarray, y_seq: numpy.ndarray) -> (X_train, X_test, y_train, y_test)

Chronological split for sequence arrays using the same ``self.test_size``.

Raises ``ValueError`` if ``len(X_seq) != len(y_seq)``.

Tabular Model Builders
----------------------

build_random_forest
^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.build_random_forest() -> sklearn.ensemble.RandomForestRegressor
   :staticmethod:

Returns a ``RandomForestRegressor`` configured as:

- ``n_estimators=200``
- ``random_state=42``
- ``n_jobs=-1``

build_xgboost
^^^^^^^^^^^^^

.. method:: AutoMLPipeline.build_xgboost() -> xgboost.XGBRegressor
   :staticmethod:

Returns an ``XGBRegressor`` configured as:

- ``n_estimators=200``
- ``learning_rate=0.05``
- ``max_depth=6``
- ``subsample=0.8``
- ``colsample_bytree=0.8``
- ``random_state=42``
- ``n_jobs=-1``
- ``tree_method="hist"``
- ``objective="reg:squarederror"``

Metrics
-------

regression_metrics
^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.regression_metrics(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> dict
   :staticmethod:

Computes regression metrics:

- MAE (mean absolute error)
- RMSE (root mean squared error)
- R2 (coefficient of determination)
- MAPE (mean absolute percentage error) with denominator clipped by 1e-8
- sMAPE (symmetric MAPE) with denominator clipped by 1e-8

Returns a dictionary with keys:

``{"MAE", "RMSE", "R2", "MAPE", "sMAPE"}``

_metric_for_sort
^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline._metric_for_sort(metrics: dict) -> float

Returns a scalar value used for sorting models according to ``self.primary_metric``.

- For ``RMSE``, ``MAE``, ``MAPE``, ``sMAPE``: returns the metric directly (lower is better).
- For ``R2``: returns ``-R2`` so that ``min()`` still selects the best (highest R2).

Benchmarking
------------

benchmark_tabular_models
^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.benchmark_tabular_models(X_train, y_train, X_test, y_test) -> dict

Trains and evaluates each model listed in ``self.tabular_models`` using:

- ``model.fit(X_train, y_train)``
- ``preds = model.predict(X_test)``
- metrics computed by :meth:`regression_metrics`

For each model, stores:

- ``model``: fitted model object
- ``metrics``: metrics dict
- ``y_pred``: predictions array

Returns a dictionary ``results[name] = {...}``.

benchmark_seq_models
^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.benchmark_seq_models(X_seq: numpy.ndarray, y_seq: numpy.ndarray) -> dict

Trains and evaluates each model listed in ``self.seq_models``.

- Performs :meth:`seq_train_test_split` to obtain train and validation splits.
- Builds model using the registry builder with ``input_shape = X_train.shape[1:]``.
- Calls :meth:`train_deep_model` with:

  - epochs = ``automl_cfg.epochs``
  - batch_size = ``automl_cfg.batch_size``
  - target_scaler = ``self.target_scaler_``

Returns a mapping of model name to the result dict returned by :meth:`train_deep_model`.

Walk-Forward Validation
-----------------------

walk_forward_validation (tabular)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.walk_forward_validation(df_clean: pandas.DataFrame, model_name: str, n_splits: int = 5) -> (pandas.Series, pandas.DataFrame)

Walk-forward validation for tabular models using ``TimeSeriesSplit``.

Validation:

- ``model_name`` must be provided and must be in ``self.tabular_models``.
- ``model_name`` must exist in ``TABULAR_MODEL_BUILDERS``.

Logic:

- Builds X and y via :meth:`prepare_X_y`.
- Creates ``TimeSeriesSplit(n_splits=n_splits)``.
- For each fold:
  - fits a new model instance on the fold train split
  - predicts on the fold test split
  - stores out-of-fold predictions in a pandas Series aligned to y index
  - computes metrics via :meth:`regression_metrics`
  - appends per-fold metrics and sizes into a list

Returns:

- ``oof_pred``: pandas Series of out-of-fold predictions
- ``metrics_df``: DataFrame with per-fold metrics and sizes

walk_forward_validation_seq (sequence)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.walk_forward_validation_seq(X_seq: numpy.ndarray, y_seq: numpy.ndarray, model_name: str, n_splits: int = 5) -> (numpy.ndarray, pandas.DataFrame)

Walk-forward validation for sequence models (``LSTM``, ``TCN``, ``TFT``) using ``TimeSeriesSplit``.

Validation:

- ``model_name`` must be in ``self.seq_models`` and in ``SEQ_MODEL_BUILDERS``.
- ``len(X_seq)`` must equal ``len(y_seq)``.

Logic:

- Uses ``TimeSeriesSplit`` over indices ``np.arange(n_samples)``.
- For each fold:
  - builds a fresh model with the sequence builder
  - calls :meth:`train_deep_model` with fold train/val splits
  - uses ``res["y_pred"]`` as fold predictions (returned by train_deep_model)
  - stores predictions into a numpy array ``oof_pred`` at fold indices
  - stores per-fold metrics in a list

Returns:

- ``oof_pred``: numpy array of out-of-fold predictions
- ``metrics_df``: DataFrame with per-fold metrics and sizes

Visualization Utilities
-----------------------

The module defines multiple matplotlib plotting helpers. The matplotlib backend is set to
``Agg`` (non-GUI) via ``matplotlib.use("Agg")``. Plot functions create figures but do not call
``plt.show()`` (calls are commented out).

plot_metrics_bar
^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.plot_metrics_bar(results, metric: str = "RMSE", title: str = None)
   :staticmethod:

Creates a bar plot comparing a single metric across models.

plot_true_vs_pred_line
^^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.plot_true_vs_pred_line(y_test, pred_dict, max_points: int = None, title_prefix: str = "True vs Predicted")
   :staticmethod:

Creates one line plot per model (stacked subplots), comparing true vs predicted values.
Optionally down-samples using :meth:`_downsample_indices`.

plot_true_vs_pred_scatter
^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.plot_true_vs_pred_scatter(y_test, pred_dict, max_points: int = 500, title_prefix: str = "Actual vs Predicted")
   :staticmethod:

Creates scatter plots of true vs predicted values (one subplot per model), with optional downsampling.

plot_residuals
^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.plot_residuals(y_test, pred_dict, max_points: int = 500, title_prefix: str = "Residuals")
   :staticmethod:

Creates residual scatter plots (true - predicted) per model.

plot_full_diagnostics
^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.plot_full_diagnostics(y_test, pred_dict, max_points: int = 500)
   :staticmethod:

Wrapper calling:

- :meth:`plot_true_vs_pred_line`
- :meth:`plot_true_vs_pred_scatter`
- :meth:`plot_residuals`

full_diagnostics_tabular
^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.full_diagnostics_tabular(y_true, results, max_points: int = 500, metric: str = "RMSE")

High-level plotting wrapper for tabular benchmark results.

It:

1) plots a metrics bar chart using :meth:`plot_metrics_bar`
2) builds a prediction dict from ``results[name]["y_pred"]``
3) calls :meth:`plot_full_diagnostics`

full_diagnostics_seq
^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.full_diagnostics_seq(y_val, seq_results, max_points: int = 500, metric: str = "RMSE", show_history: bool = True)

High-level plotting wrapper for sequence benchmark results.

It:

1) plots a metrics bar chart across sequence models
2) optionally plots per-model training histories (via :meth:`plot_training_history`)
3) runs :meth:`plot_full_diagnostics` on the validation set

plot_walk_forward_metrics
^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.plot_walk_forward_metrics(metrics_df: pandas.DataFrame, metric: str = "RMSE", title: str = "Walk-forward metrics by fold")
   :staticmethod:

Plots a metric value over folds from the provided ``metrics_df``.

Validates that the metric exists as a column.

plot_walk_forward_diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.plot_walk_forward_diagnostics(y_full, oof_pred, model_name: str, max_points: int = 500)

Runs :meth:`plot_full_diagnostics` for walk-forward predictions using a single-model ``pred_dict``.

Deep Models
-----------

build_lstm
^^^^^^^^^^

.. method:: AutoMLPipeline.build_lstm(input_shape) -> tensorflow.keras.Model
   :staticmethod:

Builds a sequential LSTM model with:

- LSTM(128, return_sequences=True) → Dropout(0.3)
- LSTM(64)
- Dense(64, relu) → Dropout(0.3)
- Dense(1)

Compiled with Adam(1e-3) and loss "mse".

build_tcn
^^^^^^^^^

.. method:: AutoMLPipeline.build_tcn(input_shape) -> tensorflow.keras.Model
   :staticmethod:

Builds a causal Conv1D residual stack with dilation rates [1, 2, 4, 8], followed by:

- GlobalAveragePooling1D
- Dense(64, relu) → Dropout(0.3)
- Dense(1)

Compiled with Adam(1e-3) and loss "mse".

build_tft
^^^^^^^^^

.. method:: AutoMLPipeline.build_tft(input_shape) -> tensorflow.keras.Model
   :staticmethod:

Builds a model using:

- Dense(64, relu)
- MultiHeadAttention(num_heads=4, key_dim=16) with residual + LayerNorm
- Feed-forward block with residual + LayerNorm
- GlobalAveragePooling1D
- Dense(64, relu) → Dropout(0.3)
- Dense(1)

Compiled with Adam(1e-3) and loss "mse".

train_deep_model
^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.train_deep_model(model, X_train, y_train, X_val, y_val, epochs: int = 50, batch_size: int = 32, target_scaler=None) -> dict
   :staticmethod:

Trains a Keras model with early stopping:

- monitors ``val_loss``
- patience = 8
- restore_best_weights = True

After training:

- predicts on ``X_val`` (scaled space)
- if ``target_scaler`` is provided:
  - inverse-transforms both ``y_val`` and predictions to original scale
- computes metrics using :meth:`regression_metrics` on original scale values

Returns a dict with keys:

- ``model``: the trained model
- ``metrics``: metrics dict
- ``y_pred``: predictions (returned in original scale if inverse-transformed)
- ``history``: Keras history object
- ``curves``: dict containing stored history arrays (loss, val_loss, mae, val_mae)

plot_training_history
^^^^^^^^^^^^^^^^^^^^^

.. method:: AutoMLPipeline.plot_training_history(history, title: str = "Training History")
   :staticmethod:

Plots training and validation loss curves from the Keras History object.

Running the Pipeline
--------------------

run_pipeline
^^^^^^^^^^^^

.. method:: AutoMLPipeline.run_pipeline(df_clean: pandas.DataFrame, top_k: Optional[int] = None, n_splits: Optional[int] = None) -> dict

Executes the end-to-end AutoML process exactly as implemented.

**Defaults**

- ``top_k`` defaults to ``automl_cfg.top_k`` if not provided.
- ``n_splits`` defaults to ``automl_cfg.n_splits`` if not provided.

**Implemented steps**

1) Prepare tabular X, y using :meth:`prepare_X_y`.
2) Split tabular data chronologically using :meth:`time_train_test_split`.
3) Benchmark tabular models (if any) using :meth:`benchmark_tabular_models`.
   If tabular results exist, stores a plot payload under ``plot_payload_["tabular"]``
   containing y_test and per-model predictions/metrics (converted to lists).
4) If sequence models are selected:
   - builds sequences with :meth:`build_sequences_from_df(scale_features=True)`
   - performs a sequence train/val split with :meth:`seq_train_test_split`
   - calls :meth:`benchmark_seq_models`
   - stores per-model training curves under ``plot_payload_["training_curves"]``
   - stores sequence diagnostics payload under ``plot_payload_["seq"]`` using original-scale y_val
5) If ``do_plots`` is enabled:
   - produces tabular diagnostics using :meth:`full_diagnostics_tabular`
   - produces sequence diagnostics using :meth:`full_diagnostics_seq` (if seq results exist)
6) Builds a combined summary dict ``all_results`` with:
   - model type ("tabular" or "seq")
   - benchmark metrics
7) Ranks models by ``primary_metric`` using :meth:`_metric_for_sort`.
   Selects the top-K model names.
8) For each top-K candidate:
   - runs walk-forward validation:
     - tabular → :meth:`walk_forward_validation(df_clean, model_name, n_splits)`
     - seq → :meth:`walk_forward_validation_seq(X_seq, y_seq, model_name, n_splits)`
   - stores average primary metric in ``wf_results[name]["avg_metric"]``
   - optionally plots walk-forward metrics and diagnostics if ``do_plots`` is True
   - stores chart-ready fold metrics in ``plot_payload_["walk_forward"]``
9) Selects the best model by minimizing average metric (or maximizing for R2 via sign flip).
10) Retrains the best model on the full dataset:
    - tabular: fits on full X, y
    - seq: rebuilds model and trains using:
      - a final split with ``val_split = 0.1`` on sequence arrays
      - :meth:`train_deep_model` with ``target_scaler=self.target_scaler_``
      - stores best model training curves in ``plot_payload_["training_curves_best"]``
11) Returns a ``summary`` dict containing:
    - tabular_results, seq_results, all_results, wf_results
    - y_val_original (used for sequence plotting payload)
    - best_model_name, best_model_type, best_model, best_avg_metric
    - primary_metric, meta, plot_payload
    - feature_scaler and target_scaler only if best_type == "seq" and scalers exist

