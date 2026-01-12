PredictX Module
===============

Location
--------

``backend/modules/predict_x.py``

Overview
--------

This module defines a prediction utility that supports two model paths:

- **Tabular models** (e.g., scikit-learn regressors or XGBoost Booster objects)
- **Sequence models** (Keras/TensorFlow models using a lookback window)

The main entry point is :meth:`PredictX.run`, which validates inputs, chooses feature columns,
and produces predictions either for one entity or multiple entities depending on configuration.

Configuration
------------

PredictionConfig
^^^^^^^^^^^^^^^^

.. class:: PredictionConfig

A dataclass holding prediction settings.

**Fields**

Core dataset fields:

- ``time_col`` (str): Name of the time column (must exist in the input DataFrame).
- ``target_col`` (str): Name of the target column (must exist in the input DataFrame).
- ``is_multi_entity`` (bool): Whether the dataset contains multiple entities. Default: ``False``.
- ``entity_col`` (Optional[str]): Entity identifier column name. Required if ``is_multi_entity=True``.
- ``horizon`` (int): Prediction horizon. Default: ``1``.

Prediction mode fields:

- ``mode`` (str): ``"multi_step"`` or ``"one_step"``. Default: ``"multi_step"``.
- ``steps`` (int): Number of steps to output in multi-step mode. Used only if ``mode="multi_step"``.
  Default: ``1``.

Entity scope fields (multi-entity):

- ``entity_scope`` (str): ``"one"`` or ``"all"``. Default: ``"one"``.
- ``entity_value`` (Optional[Union[str, int, float]]): Required when ``entity_scope="one"``.

Feature control fields:

- ``feature_cols`` (Optional[List[str]]): If provided, used directly. If not, features are inferred.
- ``max_entities`` (int): Cap on the number of entities when predicting for all entities. Default: ``200``.

Sequence model fields:

- ``lookback`` (int): Window size used for sequence prediction. Default: ``30``.
- ``model_type`` (str): ``"tabular"`` or ``"seq"``. Default: ``"tabular"``.

PredictX
--------

.. class:: PredictX

Prediction runner that supports both tabular and sequence models.

Construction
^^^^^^^^^^^^

.. method:: PredictX.__init__(*, model: Any, cfg: PredictionConfig, training_report: Optional[dict] = None, feature_scaler: Optional[StandardScaler] = None, target_scaler: Optional[StandardScaler] = None)

Initializes the predictor with:

- ``model``: The fitted model object.
- ``cfg``: The :class:`PredictionConfig`.
- ``training_report``: Optional metadata dict (default: empty dict).
- ``feature_scaler``: Optional ``StandardScaler`` used to transform features for sequence models.
- ``target_scaler``: Optional ``StandardScaler`` used to inverse-transform predictions for sequence models.

Model type inference:

- If ``cfg.model_type`` is ``"tabular"``, the constructor checks whether the model looks like a Keras model
  using:

  - ``hasattr(model, "predict")`` and
  - ``'keras' in str(type(model)).lower()``

  If this condition is true, it updates ``cfg.model_type`` to ``"seq"``.

Validation
----------

validate
^^^^^^^^

.. method:: PredictX.validate(df: pandas.DataFrame) -> None

Validates that required columns and configuration are consistent.

Checks performed:

- Ensures ``time_col`` exists in ``df.columns``; otherwise raises ``ValueError``.
- Ensures ``target_col`` exists in ``df.columns``; otherwise raises ``ValueError``.

If ``is_multi_entity`` is True:

- Requires ``entity_col`` to be set; otherwise raises ``ValueError``.
- Ensures ``entity_col`` exists in ``df.columns``; otherwise raises ``ValueError``.

Normalizes horizon/steps:

- ``cfg.horizon`` is coerced to int (or 1 if falsy).
- If ``mode == "one_step"``, sets ``cfg.steps = 1``.
- Else clamps ``cfg.steps`` to ``[1, cfg.horizon]`` (and converts to int).

Entity existence check (multi-entity + entity_scope="one"):

- Requires ``entity_value`` to be non-empty; otherwise raises ``ValueError``.
- Checks whether ``entity_value`` exists among available entity IDs (as strings).
  If not found, raises ``ValueError`` listing up to the first 10 available entities.

Feature Selection
-----------------

_infer_feature_cols
^^^^^^^^^^^^^^^^^^^

.. method:: PredictX._infer_feature_cols(df: pandas.DataFrame) -> list[str]

Determines feature columns when ``cfg.feature_cols`` is not provided.

Priority order:

1) From ``training_report``:

- Reads either ``training_report["feature_columns"]`` or ``training_report["train_features"]``.
- If this value is a non-empty list, it keeps only items that exist in ``df.columns``.
- Returns the filtered list if non-empty.

2) Fallback:

- Uses numeric columns only: ``df.select_dtypes(include=[np.number])``.
- Excludes:

  - ``cfg.target_col``
  - ``cfg.time_col``
  - ``cfg.entity_col`` (if multi-entity)

Returns the resulting list.

Sorting
-------

_sort_df
^^^^^^^^

.. method:: PredictX._sort_df(df: pandas.DataFrame) -> pandas.DataFrame

Sorts the DataFrame by time.

- If multi-entity and ``entity_col`` is set: sorts by ``[entity_col, time_col]``.
- Otherwise sorts by ``time_col``.

Always resets the index (``reset_index(drop=True)``).

Tabular Prediction Internals
----------------------------

_predict_one_row_tabular
^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: PredictX._predict_one_row_tabular(Xrow: numpy.ndarray) -> float

Predicts for a single row with a tabular model.

Supported cases:

1) **XGBoost Booster**:

- If ``self.model.__class__.__name__.lower() == "booster"``:
  - builds ``xgboost.DMatrix(Xrow)``
  - calls ``self.model.predict(dmatrix)``
  - returns the first scalar

2) **Sklearn-like model**:

- If the model has a ``predict`` attribute:
  - calls ``self.model.predict(Xrow)``
  - returns the first scalar

Otherwise raises:

- ``ValueError("Unsupported model type: ...")``

Sequence Prediction Internals
-----------------------------

_build_sequence_window
^^^^^^^^^^^^^^^^^^^^^

.. method:: PredictX._build_sequence_window(df_sorted: pandas.DataFrame, feature_cols: list[str]) -> numpy.ndarray

Builds a single sequence window from the most recent data.

Behavior:

- Takes the last ``cfg.lookback`` rows using ``df_sorted.tail(lookback)``.
- If fewer than lookback rows exist, raises ``ValueError``.
- Extracts feature values:

  - ``df_window[feature_cols].apply(pd.to_numeric, errors="coerce").values``

- If ``feature_scaler`` is provided, applies ``feature_scaler.transform(X)``.

Returns:

- A float32 array of shape ``(1, lookback, n_features)``.

_predict_sequence
^^^^^^^^^^^^^^^^^

.. method:: PredictX._predict_sequence(X_seq: numpy.ndarray) -> float

Predicts a scalar value from a sequence window.

Supported case:

- If the model has ``predict`` and the model type string contains ``"keras"``:

  - computes ``y_scaled = model.predict(X_seq, verbose=0)``
  - takes the first scalar
  - if ``target_scaler`` is provided, applies:

    ``target_scaler.inverse_transform([[y_value]])``

Returns:

- The prediction as a Python float (inverse-transformed if scaler provided).

Otherwise raises:

- ``ValueError("Unsupported sequence model type: ...")``

Per-Entity Prediction
---------------------

predict_for_entity_df
^^^^^^^^^^^^^^^^^^^^^

.. method:: PredictX.predict_for_entity_df(df_ent: pandas.DataFrame, feature_cols: list[str]) -> dict

Predicts for a single entity slice (or for the entire dataset when not multi-entity).

The behavior depends on ``cfg.model_type``.

Sequence path (cfg.model_type == "seq")
"""""""""""""""""""""""""""""""""""""""

- Attempts to build a sequence window with :meth:`_build_sequence_window`.
- If window building fails (ValueError), returns:

  - ``{"error": "<message>", "y_pred": None}``

Prediction output:

- If ``mode == "one_step"``:
  - returns ``{"y_pred": <float>}``

- Else (multi-step mode):
  - loops ``cfg.steps`` times and repeatedly calls :meth:`_predict_sequence`
  - collects predictions into a list

  .. note::

     In this multi-step sequence path, the code does not update the input window between steps.
     A ``TODO`` comment indicates the window update is not implemented.

Tabular path (otherwise)
""""""""""""""""""""""""

- Takes the last row: ``df_ent.tail(1)``
- Builds ``X`` from ``feature_cols`` using numeric coercion:

  - ``apply(pd.to_numeric, errors="coerce")``
  - converts to numpy with ``to_numpy(dtype=float)``

Prediction output:

- If ``mode == "one_step"``:
  - returns ``{"y_pred": <float>}``

- Else (multi-step mode):
  - loops ``cfg.steps`` times
  - appends the same single-row prediction repeatedly
  - returns ``{"y_pred": [ ... ]}``

Running Predictions
-------------------

run
^^^

.. method:: PredictX.run(df: pandas.DataFrame) -> dict

Main entry point.

Steps:

1) Calls :meth:`validate`.
2) Sorts the DataFrame using :meth:`_sort_df`.
3) Determines feature columns:

   - uses ``cfg.feature_cols`` if provided
   - otherwise calls :meth:`_infer_feature_cols`

   Raises ``ValueError`` if no feature columns are found.

4) If not multi-entity (``cfg.is_multi_entity == False``):

   - calls :meth:`predict_for_entity_df` on the full DataFrame
   - returns a dict containing:

     - ``mode``
     - ``steps``
     - ``model_type``
     - ``feature_columns``
     - ``lookback`` (only if model_type == "seq", else None)
     - ``result`` (the per-entity prediction result)

5) If multi-entity:

   - requires ``cfg.entity_col`` (asserts non-None)
   - two modes:

   a) ``entity_scope == "one"``

      - filters the entity slice by string match:

        ``df[df[entity_col].astype(str) == str(cfg.entity_value)]``

      - if empty, raises ``ValueError("Entity '<value>' not found")``
      - calls :meth:`predict_for_entity_df`
      - returns a dict containing:

        - ``mode``
        - ``steps``
        - ``model_type``
        - ``entity_column``
        - ``entity_value``
        - ``feature_columns``
        - ``lookback`` (seq only)
        - ``result``

   b) ``entity_scope == "all"``

      - collects unique entity IDs from the entity column (as strings)
      - caps to ``cfg.max_entities``
      - for each entity, computes ``predict_for_entity_df``
      - returns a dict containing:

        - ``mode``
        - ``steps``
        - ``model_type``
        - ``entity_column``
        - ``feature_columns``
        - ``lookback`` (seq only)
        - ``results`` (mapping entity_id → result)
        - ``capped_entities`` (number of entities processed)
