Data Ingestion Module
=====================

Location
--------

``backend/modules/data_ingestion.py``

Overview
--------

This module provides a small ingestion utility class that reads dataset files into
a pandas DataFrame, validates basic structure, standardizes column types, and
generates a profiling report suitable for storage as JSON.

It supports common tabular formats and includes lightweight heuristics for:

- detecting datetime-like columns
- coercing numeric columns stored as strings
- coercing boolean-like string columns into booleans

Public API
----------

DataIngestionError
^^^^^^^^^^^^^^^^^^

.. class:: DataIngestionError

A custom exception raised when ingestion, validation, or profiling fails.

Supported file extensions
^^^^^^^^^^^^^^^^^^^^^^^^^

The module supports the following file extensions:

- ``csv``
- ``json``
- ``xlsx``
- ``xls``
- ``parquet``

This set is defined by ``SUPPORTED_EXT``.

DataIngestion
^^^^^^^^^^^^^

.. class:: DataIngestion

Provides ingestion, standardization, and profiling utilities.

The class exposes three main public methods:

- :meth:`read_to_dataframe`
- :meth:`standardize`
- :meth:`profile`

read_to_dataframe
^^^^^^^^^^^^^^^^^

.. method:: DataIngestion.read_to_dataframe(file_bytes: bytes, filename: str) -> pandas.DataFrame

Reads a file into a pandas DataFrame.

**Parameters**

- ``file_bytes``: Raw bytes of the uploaded file.
- ``filename``: Filename used to infer the extension.

**Behavior**

1. Extracts the file extension using :meth:`_get_ext`.
2. Verifies the extension is supported (``SUPPORTED_EXT``).
3. Uses pandas readers based on extension:

   - ``csv`` → ``pd.read_csv``
   - ``json`` → ``pd.read_json``
   - ``xlsx`` / ``xls`` → ``pd.read_excel``
   - ``parquet`` → ``pd.read_parquet``

4. Validates the resulting DataFrame with :meth:`_validate_dataframe`.

**Returns**

- A validated pandas DataFrame.

**Raises**

- :class:`DataIngestionError` if:
  - the extension is not supported
  - pandas fails to read the file
  - the DataFrame is empty, has no columns, or appears to have no proper header row

standardize
^^^^^^^^^^^

.. method:: DataIngestion.standardize(df: pandas.DataFrame) -> pandas.DataFrame

Standardizes a DataFrame by normalizing column names and attempting type coercion.

**Parameters**

- ``df``: The input DataFrame.

**Behavior**

The method returns a modified copy of the DataFrame and performs the following steps:

1) Normalize column names
"""""""""""""""""""""""""

- Column names are transformed using :meth:`_norm_col`:
  - Converts to string
  - Replaces ``/`` and ``-`` with ``_``
  - Strips whitespace and collapses spaces into underscores
  - Lowercases the result

2) Datetime detection and parsing (object columns)
"""""""""""""""""""""""""""""""""""""""""""""""""

For each column whose dtype is ``object`` (``"O"``):

- Takes up to 50 non-null values as strings.
- Attempts ``pd.to_datetime(sample, errors="raise")`` on the sample.
- If at least 70% of the sample parses successfully (``parsed.notna().mean() >= 0.7``),
  the full column is converted with ``pd.to_datetime(..., errors="coerce")``.

This conversion may introduce ``NaT`` values where parsing fails.

3) Numeric coercion (remaining object columns)
"""""""""""""""""""""""""""""""""""""""""""""

For each remaining ``object`` column:

- Tries ``pd.to_numeric(..., errors="coerce")``.
- If the number of non-null numeric values is at least:

  - 5, or
  - 70% of the number of non-null values in the original column

  then the column is replaced by the numeric series.

4) Boolean-like coercion (remaining object columns)
""""""""""""""""""""""""""""""""""""""""""""""""""

For each remaining ``object`` column:

- Collects unique lowercase string values from non-null entries.
- If all values are contained within:

  ``{"true", "false", "0", "1", "yes", "no"}``

  the column is mapped using:

- ``true``, ``1``, ``yes`` → ``True``
- ``false``, ``0``, ``no`` → ``False``

**Returns**

- A standardized pandas DataFrame copy.

profile
^^^^^^^

.. method:: DataIngestion.profile(df: pandas.DataFrame, target_column: str) -> dict

Computes a profiling report for a given DataFrame and target column.

**Parameters**

- ``df``: The input DataFrame.
- ``target_column``: Name of the target column (must exist in ``df.columns``).

**Validation**

- Raises :class:`DataIngestionError` if ``target_column`` is not present.

**Computed fields**

The returned dictionary includes:

- ``shape``: A tuple ``(rows, cols)`` as integers.
- ``columns``: List of column names.
- ``dtypes``: Mapping of column name → dtype string.
- ``time_columns``: Result of :meth:`detect_time_columns`.
- ``target_column``: The provided target column name.
- ``missing_values``: Mapping of column name → missing count (``isna().sum()``).
- ``duplicate_rows``: Number of duplicated rows (``df.duplicated().sum()``).
- ``numeric_summary``:
  - For numeric columns only (``df.select_dtypes(include=[np.number])``),
    the output of ``describe()`` converted to a dictionary.
  - If there are no numeric columns, this is an empty dictionary.
- ``categorical_summary``:
  - Computed for columns that are neither numeric nor datetime.
  - For each such column:
    - ``distinct``: number of unique non-null values (``nunique(dropna=True)``)
    - ``top_values``: up to 5 most frequent values with counts
- ``sample``:
  - Output of :meth:`_sample(df, n=5)` (first 5 rows),
    with datetime values formatted as strings.

**Returns**

- A JSON-serializable dictionary summarizing dataset characteristics.

Helper Methods
--------------

The following methods support the public API.

_get_ext
^^^^^^^^

.. method:: DataIngestion._get_ext(filename: str) -> str
   :staticmethod:

Extracts the file extension from ``filename`` and lowercases it.

- If there is no ``.`` in the filename, returns an empty string.

_validate_dataframe
^^^^^^^^^^^^^^^^^^^

.. method:: DataIngestion._validate_dataframe(df: pandas.DataFrame) -> None
   :staticmethod:

Validates basic DataFrame structure:

- Raises :class:`DataIngestionError` if the DataFrame is empty.
- Raises :class:`DataIngestionError` if there are no columns.
- Checks for missing/invalid header row by detecting column names like
  ``Unnamed: 0`` (case-insensitive):

  - Converts column labels to strings
  - Uses regex ``^unnamed[:\s]*\d*``
  - If at least 50% of columns match this pattern (minimum 1), raises an error:

    "The file seems to have no proper header row. Please include a first row with column names."

detect_time_columns
^^^^^^^^^^^^^^^^^^^

.. method:: DataIngestion.detect_time_columns(df: pandas.DataFrame) -> list[str]
   :staticmethod:

Returns a list of columns that look like time/datetime.

Detection logic:

1) Includes columns already typed as datetime64.
2) For remaining object columns:
   - Parses up to 50 non-null values using ``pd.to_datetime(errors="raise")``.
   - If at least 70% parse successfully, the column is included.

_norm_col
^^^^^^^^^

.. method:: DataIngestion._norm_col(name: str) -> str
   :staticmethod:

Normalizes a column name:

- Converts to string
- Replaces ``/`` and ``-`` with ``_``
- Splits on whitespace and joins with underscores
- Lowercases

_sample
^^^^^^^

.. method:: DataIngestion._sample(df: pandas.DataFrame, n: int = 5) -> list[dict]
   :staticmethod:

Returns the first ``n`` rows of the DataFrame as a list of records
(``orient="records"``). For datetime64 columns, values are formatted as:

``YYYY-MM-DD HH:MM:SS``

and converted to ``object`` dtype to remain JSON-serializable.
