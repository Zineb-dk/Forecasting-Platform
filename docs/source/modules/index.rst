Backend Modules
===============

This section documents the core backend modules that implement the Forecasting-Platform
logic. Each module is designed to cover a distinct part of the workflow (data ingestion,
preprocessing, training, prediction, explainability).

Module Map
----------

- **data_ingestion**: Upload parsing, format validation, and ingestion utilities.
- **data_forge**: Preprocessing and dataset preparation utilities.
- **automl**: Training orchestration and model selection logic.
- **predict_x**: Inference/prediction utilities.
- **explainx**: Model explainability utilities and artifact generation.

.. note::

   These pages document the modules located under ``backend/modules/`` in the repository.

Contents
--------

.. toctree::
   :maxdepth: 1

   data_ingestion
   data_forge
   automl
   predict_x
   explainx
