DataForge Module
================

Location
--------

``backend/modules/data_forge.py``

Overview
--------

This module provides utilities for inspecting and preprocessing time-series or
tabular datasets using a configuration object (:class:`DataConfig`).

It includes:

- a configuration dataclass describing key dataset columns
- dataset overview utilities (counts, missingness, constant columns, dtype groups)
- datetime parsing and sorting utilities
- duplicate / constant-column / empty-row cleanup
- datetime regularity analysis and optional reindex+interpolation
- basic feature-engineering helpers (calendar, lag, rolling stats)
- a preprocessing pipeline method (:meth:`DataForge.preprocess`) that runs the
  implemented steps and returns both the processed DataFrame and a structured report

Configuration
------------

DataConfig
^^^^^^^^^^

.. class:: DataConfig

A dataclass used to describe the dataset structure and preprocessing context.

**Fields**

Mandatory:

- ``time_col`` (str): Name of the time axis column.
- ``target_col`` (str): Name of the target column to predict.
- ``is_multi_entity`` (bool): Whether the dataset contains multiple entities.
- ``sensor_cols`` (List[str]): List of sensor/feature column names.
  (Comment in code: sensor columns are dataset columns excluding time/target/entity columns.)

Optional:

- ``entity_col`` (Optional[str]): Entity identifier column. Required when
  ``is_multi_entity=True`` (validated in :meth:`get_data_overview`).
- ``horizon_steps`` (int): Steps ahead to predict. Default is ``1``.
- ``freq_tolerance`` (float): Threshold used for datetime regularity detection
  (see :meth:`analyze_datetime_regularity`). Default is ``0.95``.

DataForge
---------

.. class:: DataForge

Main utility class. Constructed with a :class:`DataConfig`:

.. code-block:: python

   forge = DataForge(cfg)

Public Methods
--------------

get_data_overview
^^^^^^^^^^^^^^^^^

.. method:: DataForge.get_data_overview(df: pandas.DataFrame) -> dict

Returns a dictionary containing key information about the dataset based on the
current :class:`DataConfig`.

**Outputs (keys in returned dict)**

Basic structure:

- ``n_rows``: number of rows
- ``n_cols``: number of columns
- ``columns``: list of column names
- ``dtypes``: mapping column → dtype string

Missing values:

- ``missing_counts``: mapping column → missing count
- ``missing_ratio``: mapping column → missing ratio (missing / n_rows), or empty if n_rows == 0

Constant columns:

- ``constant_columns``: list of columns with ``nunique(dropna=False) <= 1``

Column type groups:

- ``numeric_columns``: columns selected by ``df.select_dtypes(include=np.number)``
- ``datetime_columns``: columns selected by datetime64 dtypes
- ``categorical_columns``: columns selected by ``object`` or ``category`` dtype

Target statistics:

If ``cfg.target_col`` exists:

- ``target_dtype``: dtype string
- ``target_missing``: missing count for target
- ``target_describe``:
  - ``target_series.describe().to_dict()`` if target is numeric
  - otherwise ``None``

If ``cfg.target_col`` does not exist:

- ``target_dtype`` = ``None``
- ``target_missing`` = ``None``
- ``target_describe`` = ``None``

Sensor statistics:

- ``sensor_cols_effective``: subset of ``cfg.sensor_cols`` that exist in the DataFrame
- ``sensors_describe``:
  - ``df[existing_sensors].describe().to_dict()`` if any sensors exist
  - otherwise ``{}``

Entity-level information:

If ``cfg.is_multi_entity`` is True:

- Raises ``ValueError`` if ``cfg.entity_col`` is ``None``
- Raises ``ValueError`` if ``cfg.entity_col`` is not in the DataFrame
- Computes:
  - ``n_entities``: number of distinct entities (via value_counts)
  - ``entity_counts``: counts per entity value (sorted by index)
  - ``entity_min_length``: minimum entity sequence length
  - ``entity_max_length``: maximum entity sequence length

If ``cfg.is_multi_entity`` is False:

- ``n_entities`` = 1
- ``entity_counts`` = None
- ``entity_min_length`` = n_rows
- ``entity_max_length`` = n_rows

Time axis information:

If ``cfg.time_col`` exists:

- ``time_raw_dtype``: dtype string
- ``time_kind``:
  - ``"numeric"`` if dtype is numeric
  - ``"datetime"`` if dtype is datetime64
  - ``"string"`` otherwise
- ``time_min`` / ``time_max``: min/max of the time series if computable, else None
- ``time_inferred_freq``:
  - computed using ``pd.infer_freq(pd.DatetimeIndex(time_series))`` only when
    ``time_kind == "datetime"`` and ``cfg.is_multi_entity == False``
  - otherwise None
- ``time_needs_parsing``: True if ``time_kind == "string"``, else False

If ``cfg.time_col`` does not exist:

- all time-related keys are set to ``None``

_date_parser
^^^^^^^^^^^^

.. method:: DataForge._date_parser(df: pandas.DataFrame, col_name: str) -> pandas.DataFrame

Parses a date column without requiring an explicit user-provided format.

**Behavior**

- Converts the column values to stripped strings.
- Detects "day-first evidence" by checking whether any value starts with a day
  greater than 12 followed by a separator (regex pattern):
  ``^(?:1[3-9]|2[0-9]|3[01])[\/\-\.]``

- First pass parsing using:

  - ``pd.to_datetime(..., dayfirst=day_first_evidence, errors="coerce", format="mixed")``

- For values that still failed parsing (coerced to NaT), attempts a fallback parse
  using ``dateutil.parser.parse(..., fuzzy=True)`` on each unique failed string.

- Values that cannot be parsed in fallback are assigned ``pd.NaT``.

Returns a copy of the DataFrame with the parsed datetime column.

_parse_and_sort_datetime
^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: DataForge._parse_and_sort_datetime(df: pandas.DataFrame) -> pandas.DataFrame

Parses ``cfg.time_col`` using :meth:`_date_parser`, then sorts ascending by the
time column and resets the index.

Raises ``ValueError`` if ``cfg.time_col`` is not found in the DataFrame.

_drop_constant_columns
^^^^^^^^^^^^^^^^^^^^^^

.. method:: DataForge._drop_constant_columns(df: pandas.DataFrame) -> (pandas.DataFrame, list[str])
   :staticmethod:

Drops columns where ``nunique(dropna=False) <= 1``.

Returns:

- the DataFrame without those columns
- the list of dropped column names

drop_empty_rows
^^^^^^^^^^^^^^^

.. method:: DataForge.drop_empty_rows(df: pandas.DataFrame) -> pandas.DataFrame

Drops rows where all values are considered empty or null-like.

**Null-like patterns replaced with NaN**

- empty/whitespace-only: ``^\s*$``
- ``null`` (case-insensitive)
- ``none`` (case-insensitive)
- ``nan`` (case-insensitive) when written as a string

Implementation details:

- Replaces these patterns with NaN using ``df.replace(..., regex=True)``.
- Drops rows where all values are NaN: ``df.dropna(how="all")``.
- Resets index.

_drop_full_duplicates
^^^^^^^^^^^^^^^^^^^^^

.. method:: DataForge._drop_full_duplicates(df: pandas.DataFrame) -> (pandas.DataFrame, int)
   :staticmethod:

Drops fully duplicated rows (across all columns).

Returns:

- deduplicated DataFrame (index reset)
- number of duplicate rows detected and removed (``df.duplicated().sum()``)

analyze_datetime_regularity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: DataForge.analyze_datetime_regularity(df: pandas.DataFrame) -> dict

Analyzes whether datetime intervals on ``cfg.time_col`` are roughly regular.

Raises ``ValueError`` if ``cfg.time_col`` does not exist.

**Behavior**

- Sorts by time column.
- If fewer than 3 timestamps exist, returns:

  - ``is_regular`` = False
  - ``inferred_freq`` = None
  - ``mode_gap_ns`` = None
  - ``mode_ratio`` = 0.0

- Computes deltas between consecutive timestamps (nanoseconds).
- Determines the most common delta (mode) and the ratio of that mode to all deltas.
- Attempts ``pd.infer_freq`` on the full timestamp index.

A series is considered regular when:

- ``mode_ratio >= cfg.freq_tolerance``
- and ``inferred_freq`` is not None

**Returns keys**

- ``is_regular`` (bool)
- ``inferred_freq`` (str or None)
- ``mode_gap_ns`` (int or None)
- ``mode_ratio`` (float)

reindex_regular_and_interpolate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: DataForge.reindex_regular_and_interpolate(df: pandas.DataFrame) -> (pandas.DataFrame, int)

If a datetime frequency can be inferred on ``cfg.time_col``, reindexes the dataset
to a complete regular datetime grid and fills missing rows.

**Behavior**

1. Sorts by ``cfg.time_col``.
2. Attempts ``pd.infer_freq(df[cfg.time_col])``.
3. If frequency cannot be inferred, returns the original DataFrame and ``0``.
4. Builds a full datetime range using ``pd.date_range(start=..., end=..., freq=freq)``.
5. Reindexes using ``df.set_index(time_col).reindex(full_range)``.
6. Counts new rows added: ``n_after - n_before``.

Filling strategy:

- If ``cfg.target_col`` exists, fills it using:
  - ``interpolate(method="time")`` then ``ffill()`` then ``bfill()``
- For all other columns:
  - ``ffill()`` then ``bfill()``

Returns:

- the reindexed DataFrame with the time column restored
- number of rows added by reindexing

Feature Engineering Helpers
---------------------------

create_calendar_features
^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: DataForge.create_calendar_features(df: pandas.DataFrame) -> pandas.DataFrame

Adds two calendar features based on ``cfg.time_col`` after converting it with
``pd.to_datetime``:

- ``dow``: day of week (0=Monday, 6=Sunday)
- ``month``: month number (1–12)

create_lag_features
^^^^^^^^^^^^^^^^^^^

.. method:: DataForge.create_lag_features(df: pandas.DataFrame, max_lag: int = 12, lag_cols: list[str] | None = None) -> pandas.DataFrame

Creates lagged versions of specified columns.

- If ``lag_cols`` is falsy (None or empty), returns the DataFrame unchanged.
- For each ``col`` in ``lag_cols`` that exists in the DataFrame:
  - creates columns: ``{col}_lag_1`` ... ``{col}_lag_{max_lag}`` using ``shift(lag)``

create_rolling_features
^^^^^^^^^^^^^^^^^^^^^^^

.. method:: DataForge.create_rolling_features(df: pandas.DataFrame, windows: list[int] = [6, 12, 24], min_periods: int = 2, roll_cols: list[str] | None = None) -> pandas.DataFrame

Creates rolling mean and standard deviation features for specified columns.

- If ``roll_cols`` is falsy (None or empty), returns the DataFrame unchanged.
- For each ``col`` in ``roll_cols`` that exists:
  - computes rolling statistics on the 1-step shifted series (``shift(1)``)
  - for each window ``w`` in ``windows``:
    - ``{col}_roll_mean_{w}``
    - ``{col}_roll_std_{w}``

Rolling is computed with:

- ``rolling(window=w, min_periods=min_periods)``

Preprocessing Pipeline
----------------------

preprocess
^^^^^^^^^^

.. method:: DataForge.preprocess(df: pandas.DataFrame) -> (pandas.DataFrame, dict)

Runs the module’s preprocessing pipeline and returns:

- the processed DataFrame
- a structured report dictionary

**Steps (exactly as implemented)**

1. Compute ``overview_before`` using :meth:`get_data_overview`.
2. Copy the DataFrame.
3. If ``cfg.time_col`` exists:

   - If ``time_col`` dtype is object:
     - parse and sort using :meth:`_parse_and_sort_datetime`
   - Else if ``time_col`` dtype is datetime:
     - sort by time column and reset index

4. Drop full duplicates using :meth:`_drop_full_duplicates`.
5. Drop constant columns using :meth:`_drop_constant_columns`.
6. Drop fully empty rows using :meth:`drop_empty_rows`.
7. If ``cfg.time_col`` exists and is datetime:

   - compute ``time_regularity`` with :meth:`analyze_datetime_regularity`
   - reindex and fill with :meth:`reindex_regular_and_interpolate`
   - track ``n_rows_added_by_reindex``

8. Compute ``overview_after`` using :meth:`get_data_overview`.

**Report keys**

The report dictionary contains:

- ``overview_before``
- ``overview_after``
- ``n_full_duplicates_dropped``
- ``dropped_constant_columns``
- ``time_regularity`` (dict or None)
- ``n_rows_added_by_reindex`` (int)

Notes
-----

- Feature engineering helpers (calendar/lag/rolling) are not invoked by
  :meth:`preprocess`; they must be called explicitly if needed.
