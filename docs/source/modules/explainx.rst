ExplainX Module
===============

Location
--------

``backend/modules/explainx.py``

Overview
--------

This module provides a model explanation helper class, :class:`ExplainX`, that can compute:

- **SHAP global explanations** for:
  - tabular models in families ``"gbm"`` and ``"linear"``
  - deep/sequence models in family ``"deep"``
- **SHAP plots** (summary, bar, local explanation)
- **Permutation importance** for tabular and deep models

Type Alias
----------

.. data:: ArrayLike

   Alias for inputs that can be:

   - ``numpy.ndarray``
   - ``pandas.Series``
   - ``pandas.DataFrame``

ExplainX
--------

.. class:: ExplainX(model, model_family, X_train, X_val, y_val=None, feature_names=None, task_name="target", background_size=200)

A generic explainer wrapper around SHAP and permutation importance utilities.

Supported model families
^^^^^^^^^^^^^^^^^^^^^^^^

The parameter ``model_family`` is normalized to lowercase and must be one of:

- ``"gbm"``: uses ``shap.TreeExplainer``
- ``"linear"``: uses ``shap.LinearExplainer``
- ``"deep"``: uses ``shap.GradientExplainer`` on sequence data

Constructor
^^^^^^^^^^^

.. method:: ExplainX.__init__(model, model_family, X_train, X_val, y_val=None, feature_names=None, task_name="target", background_size=200)

**Parameters**

- ``model``: Trained model object.
- ``model_family``: One of ``"gbm"``, ``"linear"``, ``"deep"``.
- ``X_train``: Training data used to fit the explainer.
  - For ``"gbm"`` / ``"linear"``: 2D tabular (DataFrame or ndarray).
  - For ``"deep"``: ndarray with shape ``(n, L, F)``.
- ``X_val``: Validation data to explain.
  - For ``"gbm"`` / ``"linear"``: 2D tabular.
  - For ``"deep"``: ndarray with shape ``(n, L, F)``.
- ``y_val``: Optional validation targets. Stored as a 1D numpy array if provided.
- ``feature_names``: Optional list of feature names.
  - For deep models, names refer to feature channels (F), not time steps (L).
- ``task_name``: Label string used in plot titles. Default: ``"target"``.
- ``background_size``: Deep-model SHAP background sample size (subsample of ``X_train``). Default: ``200``.

**Stored attributes**

- ``self.shap_values``: SHAP values (computed by :meth:`compute_shap_global`).
- ``self.expected_value``: SHAP expected value (tabular), or mean background prediction (deep).
- ``self.explainer``: SHAP explainer instance (TreeExplainer / LinearExplainer / GradientExplainer).

**Validation / sanity checks**

- Raises ``ValueError`` if ``model_family`` is not one of ``{"gbm", "linear", "deep"}``.
- For ``model_family="deep"``:
  - requires ``X_train`` to be a ``numpy.ndarray`` with ``ndim == 3``
  - requires ``X_val`` to be a ``numpy.ndarray`` with ``ndim == 3``
  - otherwise raises ``ValueError``

SHAP Global Computation
-----------------------

compute_shap_global
^^^^^^^^^^^^^^^^^^^

.. method:: ExplainX.compute_shap_global(max_samples=1000) -> dict

Fits a SHAP explainer and computes SHAP values on ``X_val``.

**Parameters**

- ``max_samples`` (int): Maximum number of validation samples used for computation.
  If ``len(X_val) > max_samples`` in the tabular case, only the first
  ``max_samples`` samples are used. Default: ``1000``.

**Returns**

A dictionary with:

- ``"shap_values"``: numpy array of SHAP values
- ``"expected_value"``: expected value from the explainer (tabular) or mean prediction (deep)

Tabular (gbm / linear)
""""""""""""""""""""""

If ``model_family`` is ``"gbm"`` or ``"linear"``:

- Converts ``X_train`` and ``X_val`` to numpy arrays:
  - if DataFrame: uses ``.values``
  - otherwise: ``np.asarray(...)``
- Subsamples validation to the first ``max_samples`` rows if needed.
- Explainer choice:
  - ``"gbm"``: ``shap.TreeExplainer(self.model)``
  - ``"linear"``: ``shap.LinearExplainer(self.model, X_train_arr)``
- Computes SHAP values with ``explainer.shap_values(X_val_arr)``.
- If SHAP returns a list, the first element is used.
- Saves:
  - ``self.shap_values`` as an ``np.ndarray`` of shape ``(n_samples, n_features)``
  - ``self.expected_value`` from ``explainer.expected_value``

Deep (sequence)
"""""""""""""""

If ``model_family`` is ``"deep"``:

- Converts train/val to numpy arrays:
  - ``X_train_seq`` shape: ``(n_train, L, F)``
  - ``X_val_seq`` shape: ``(n_val, L, F)``
- Builds a background tensor of size ``min(background_size, n_train)``:
  - ``background = X_train_seq[:bg_size]``
- Uses ``shap.GradientExplainer(self.model, background)``.
- Subsamples validation to the first ``min(max_samples, n_val)`` samples.
- Calls ``explainer.shap_values(X_val_sub)``.
  - If a list is returned, uses the first element.
- Handles SHAP value shapes:
  - If SHAP values have shape ``(n, L, F, 1)``: squeezes last dim.
  - If SHAP values have shape ``(n, L, F, C)`` with ``C != 1``: averages over last dim.
  - Requires final shape to be 3D ``(n, L, F)``; otherwise raises ``ValueError``.
- Saves:
  - ``self.shap_values`` as an array of shape ``(n_use, L, F)``
  - ``self.expected_value`` as a float computed from the **mean model prediction**
    on the background tensor:

    ``np.mean(self.model.predict(background, verbose=0))``

Feature Names for Plots
-----------------------

_get_feature_names_for_global
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: ExplainX._get_feature_names_for_global() -> list[str]

Returns a feature name list used by global plots.

Tabular case (gbm / linear):

- If ``self.feature_names`` is provided, returns it.
- Else if ``X_val`` is a DataFrame, returns ``list(X_val.columns)``.
- Else returns default names: ``["f0", "f1", ...]`` using ``X_val.shape[1]``.

Deep case:

- Uses only the feature dimension (F) of ``X_val`` (ignores time dimension for naming).
- If ``self.feature_names`` exists and its length equals F, returns it.
- Else returns default: ``["feat_0", "feat_1", ...]``.

_prepare_global_shap_for_deep
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: ExplainX._prepare_global_shap_for_deep()

Deep-only helper that aggregates SHAP values over the time dimension for standard feature-level plots.

Requirements:

- ``model_family`` must be ``"deep"`` (assertion).
- ``self.shap_values`` must be set (raises ``RuntimeError`` otherwise).
- Expects SHAP values shape ``(n_samples, L, F)``; otherwise raises ``ValueError``.

Returns:

- ``shap_feat``: ``mean(abs(shap_values), axis=1)`` with shape ``(n_samples, F)``
- ``X_feat``: ``mean(X_val, axis=1)`` with shape ``(n_samples, F)``
  (using the same number of samples as in SHAP values)

SHAP Plots
----------

plot_shap_summary
^^^^^^^^^^^^^^^^^

.. method:: ExplainX.plot_shap_summary(max_display=15)

Produces a SHAP beeswarm (summary) plot.

- Requires ``self.shap_values`` to be computed (else raises ``RuntimeError``).
- Uses :meth:`_get_feature_names_for_global` for naming.

Tabular:

- Converts X_val to an array.
- Uses ``n = self.shap_values.shape[0]`` because SHAP may be computed on a subset.
- Calls:

  ``shap.summary_plot(self.shap_values, X_val_arr[:n], feature_names=..., max_display=...)``

Deep:

- Uses :meth:`_prepare_global_shap_for_deep` to aggregate.
- Calls ``shap.summary_plot`` on aggregated arrays.

plot_shap_bar
^^^^^^^^^^^^^

.. method:: ExplainX.plot_shap_bar(max_display=15)

Produces a SHAP bar plot.

Behavior mirrors :meth:`plot_shap_summary`, but uses:

- ``plot_type="bar"``

Local Explanation Plots
-----------------------

plot_local_instance
^^^^^^^^^^^^^^^^^^^

.. method:: ExplainX.plot_local_instance(idx=-1)

Produces a local force-plot style explanation for one validation sample.

Requirements:

- ``self.shap_values`` and ``self.expected_value`` must exist (else raises ``RuntimeError``).

Index selection:

- If ``idx < 0``, it is translated from the end:

  ``idx = self.shap_values.shape[0] + idx``

Tabular:

- Converts X_val to array.
- Calls ``shap.force_plot`` with:

  - expected value
  - SHAP values row ``self.shap_values[idx, :]``
  - instance row ``X_val_arr[idx, :]``
  - feature names
  - ``matplotlib=True``

Deep:

- Expects deep SHAP values to be 3D: ``(n, L, F)``.
- Selects one sequence SHAP matrix: ``seq_shap = sv[idx]`` with shape ``(L, F)``.
- Aggregates over time:

  - ``sv_feat = mean(seq_shap, axis=0)`` shape ``(F,)``
  - ``x_feat = mean(X_val[idx], axis=0)`` shape ``(F,)``

- Calls ``shap.force_plot`` with aggregated vectors and ``matplotlib=True``.

plot_deep_heatmap
^^^^^^^^^^^^^^^^^

.. method:: ExplainX.plot_deep_heatmap(idx=-1)

Deep-only visualization that plots a **time × feature** heatmap of SHAP values for one sequence window.

Requirements:

- ``model_family`` must be ``"deep"`` (else raises ``RuntimeError``).
- ``self.shap_values`` must be computed (else raises ``RuntimeError``).
- SHAP values must be 3D ``(n, L, F)`` (else raises ``ValueError``).

Behavior:

- Negative indices are supported (translated from end).
- Uses ``plt.imshow(seq_shap.T, aspect="auto", cmap="RdBu_r")`` so the y-axis corresponds to features.
- Adds a colorbar labeled ``"SHAP value"``.
- Uses feature names from :meth:`_get_feature_names_for_global` if length matches F;
  otherwise falls back to ``feat_0..feat_{F-1}``.
- Sets x-label: ``"Lookback time step"``.
- Title includes ``task_name`` and the sample index.
- Calls ``plt.show()``.

Permutation Importance
----------------------

compute_permutation_importance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. method:: ExplainX.compute_permutation_importance(n_repeats=10, random_state=42) -> pandas.DataFrame

Computes model-agnostic permutation importance on the validation set.

**Parameters**

- ``n_repeats`` (int): Number of shuffles per feature. Default: ``10``.
- ``random_state`` (int): Seed for reproducibility. Default: ``42``.

**Returns**

A DataFrame with columns:

- ``feature``
- ``importance_mean``
- ``importance_std``

Tabular (gbm / linear)
""""""""""""""""""""""

- Converts X_val to a DataFrame (preserving column names if already DataFrame).
- Requires ``y_val`` (else raises ``RuntimeError``).
- Defines a scoring function:

  ``scoring = lambda est, X, y: -mean_squared_error(y, est.predict(X))``

- Calls ``sklearn.inspection.permutation_importance`` with this scoring.
- Builds a DataFrame from ``r.importances_mean`` and ``r.importances_std``.
- Sorts by ``importance_mean`` descending.
- Returns the sorted DataFrame.

Deep (sequence)
"""""""""""""""

- Requires ``y_val`` (else raises ``RuntimeError``).
- Requires ``X_val`` to be 3D with shape ``(n, L, F)`` (else raises ``ValueError``).
- Requires ``len(y_val) == n`` (else raises ``ValueError``).

Baseline:

- Predicts on original sequences:

  ``y_pred_base = model.predict(X_val_seq, verbose=0).reshape(-1)``

- Computes baseline MSE:

  ``base_mse = mean_squared_error(y_val, y_pred_base)``

Feature naming:

- Uses ``self.feature_names`` if provided and length equals F.
- Else uses ``feat_0..feat_{F-1}``.

Permutation strategy:

- For each feature index ``j``:
  - repeats ``n_repeats`` times
  - creates ``X_perm = X_val_seq.copy()``
  - draws a permutation over samples: ``perm = rng.permutation(n_samples)``
  - replaces feature channel j across samples:

    ``X_perm[:, :, j] = X_perm[perm, :, j]``

  - predicts and computes new MSE
  - stores ``mse_perm - base_mse`` as the importance value

Outputs:

- Mean and std over repeats are computed per feature.
- Returns a DataFrame sorted by ``importance_mean`` descending.
