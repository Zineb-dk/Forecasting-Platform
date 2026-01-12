Installation & Setup
====================

This guide describes how to install and run the Forecasting-Platform locally.

Project Introduction
--------------------

Forecasting-Platform is a full-stack predictive analytics workspace that helps you go from **raw datasets** to
**trained forecasting models**, **predictions**, and **explainable results** in a reproducible way.

Added Value
^^^^^^^^^^^

This platform is designed to reduce the time and complexity of building forecasting pipelines by providing:

- **Dataset ingestion + profiling** (upload datasets, generate reports, and keep a consistent dataset registry).
- **Data preprocessing pipelines** (cleaning, validation, and structured exports for modeling).
- **AutoML training** (benchmark multiple models, validate performance, and select the best candidate).
- **Forecast generation** (run batch predictions using trained artifacts).
- **Explainability** (feature attribution/importance and model interpretation outputs).
- **Versioned storage** (datasets + model artifacts stored in Supabase Storage with metadata tracked in Supabase tables).

Architecture Overview
---------------------

The project is split into two main services:

- **Frontend**: MakerKit (Next.js + Supabase SaaS Kit Lite) with project-specific UI routes and pages.
- **Backend API**: FastAPI service for ingestion, preprocessing, AutoML training, prediction, and explainability.

Supabase is used as the shared backend infrastructure for:

- Authentication (Supabase Auth)
- Database tables (datasets/models metadata)
- Storage buckets (datasets + model artifacts)

Repository Structure
--------------------

.. code-block:: text

   Forecasting-Platform/
   ├── backend/
   │   ├── db/
   │   │   └── schema.sql
   │   ├── modules/
   │   │   ├── automl.py
   │   │   ├── data_forge.py
   │   │   ├── data_ingestion.py
   │   │   ├── explainx.py
   │   │   └── predict_x.py
   │   ├── main.py
   │   ├── requirements.txt
   │   ├── Dockerfile
   │   └── .env
   ├── docs/
   ├── frontend_patch/
   │   └── apps/web/app/...
   ├── LICENSE
   ├── README.md
   └── readthedocs.yaml

Prerequisites
-------------

Install the following software:

- **Git**: Version control system.
- **Node.js & pnpm**: JavaScript runtime and package manager.
- **Python 3.10+**: Backend API runtime.
- **Docker Desktop**: Recommended for local Supabase and containerized backend runs.

Supabase Cloud Setup (Required)
-------------------------------

This project uses Supabase for:

- Authentication (frontend obtains JWT access tokens)
- Storage (dataset files + model artifacts)
- Database tables (datasets + models metadata)

Create a Supabase account + project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1) Open the Supabase dashboard:
   https://app.supabase.com

2) Create **New project**
3) Choose a name (example: ``forecasting-platform-v1``)
4) Choose a strong database password (**store it safely**)
5) Choose a region close to you
6) Wait until the project status becomes **Active**

Collect Supabase credentials (URL + keys)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From **Supabase dashboard → Project Settings → API keys**:

Copy and store:

- **Anon public key**
- **Service role key** (**KEEP THIS SECRET**; never expose it in the frontend)

From **Supabase dashboard → Project Settings → Data API**:

Copy and store:

- **Project URL** (example: ``https://xxxx.supabase.co``)

Store these in a safe place (you will use them in both frontend and backend configuration).

Create Storage buckets
^^^^^^^^^^^^^^^^^^^^^^

Open **Supabase dashboard → Storage → Buckets → Create bucket**.

Create at minimum:

1) ``datasets`` bucket
2) ``models`` bucket

Recommended bucket settings:

- Public: **OFF** (private)
- RLS: **ON** if you use it. If you do not configure Storage RLS policies, enforce access in the backend using the service role key.

Typical stored artifacts:

- Dataset files (CSV / Parquet) in ``datasets``
- Trained model artifacts (``.joblib``, ``.pkl``, ``.h5``) in ``models``
- Optional scalers/encoders/explainability artifacts alongside model artifacts

Create database tables
^^^^^^^^^^^^^^^^^^^^^^

The backend depends on specific tables (minimum: ``datasets`` and ``models``).

1) Open **Supabase dashboard → SQL Editor**
2) Paste the SQL migration from: ``backend/db/schema.sql``
3) Run it
4) Verify the tables exist in **Table Editor**

Minimum expected tables (typical v1):

- ``datasets``
- ``models``

Frontend Setup (MakerKit + Patch)
---------------------------------

The frontend is based on MakerKit's **Next.js + Supabase SaaS Kit Lite** template.

To keep this repository lightweight, the full MakerKit source code is **not included** here.

This repository does not contain the full MakerKit source code; it contains a **patch** (changed + added files).

You must :

1) Clone this repository (Forecasting-Platform)
2) Clone MakerKit **inside** it under ``fp-frontend/``
3) Copy this repository's patch files into ``fp-frontend/``

Step 1 — Install pnpm
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: powershell

   npm install -g pnpm

Step 2 — Clone this repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: powershell

   git clone https://github.com/Zineb-dk/Forecasting-Platform.git
   cd Forecasting-Platform

Step 3 — Clone MakerKit into ``fp-frontend/``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From inside the ``Forecasting-Platform`` folder:

.. code-block:: powershell

   git clone https://github.com/makerkit/next-supabase-saas-kit-lite.git fp-frontend
   cd fp-frontend
   pnpm install

Step 4 — Configure frontend environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Edit (or create) this file in the MakerKit project root:

- ``Forecasting-Platform/fp-frontend/.env.local``

Add:

.. code-block:: text

   # Supabase (public / browser-safe)
   NEXT_PUBLIC_SUPABASE_URL="https://xxxx.supabase.co"
   NEXT_PUBLIC_SUPABASE_ANON_KEY="your_anon_public_key"

   # Backend API
   NEXT_PUBLIC_API_URL="http://localhost:8000"

Important:

- Variables starting with ``NEXT_PUBLIC_`` are visible in the browser.
- **Never** place your Supabase service role key in the frontend env file.

Step 5 — Apply this repository's frontend patch (exact steps)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The patch is located in this repository at:

- ``Forecasting-Platform/frontend_patch/``

Copy the patch content into the MakerKit folder (``Forecasting-Platform/fp-frontend``), replacing files when prompted.

Patched areas
"""""""""""""

The patch includes:

1) Marketing page updates:

- ``apps/web/app/(marketing)/layout.tsx`` (modified)
- ``apps/web/app/(marketing)/page.tsx`` (modified)

2) Protected app pages under ``apps/web/app/home/``:

- New folders for the Forecasting Platform UI sections (examples: datasets, models, predictions, monitoring, automl)
- Updated routing pages to match the platform navigation and API integration

3) Environment override:

- ``.env.local`` (example template; you still must fill your own keys)

The patch only **adds new files** and **overwrites a small number of existing pages**.

It does **not delete** any MakerKit files.

Make sure you are inside the ``Forecasting-Platform`` directory before running the commands below.

How to copy the patch (Linux/macOS)
"""""""""""""""""""""""""""""""""""

From the ``Forecasting-Platform`` folder:

.. code-block:: bash

   cp -R frontend_patch/* fp-frontend/

Windows PowerShell (robocopy recommended)
"""""""""""""""""""""""""""""""""""""""""

From the ``Forecasting-Platform`` folder:

.. code-block:: powershell

   robocopy ".\frontend_patch" ".\fp-frontend" /E

Verify the patch was applied
""""""""""""""""""""""""""""
Notes:

- Existing MakerKit files are preserved.
- Only files with the same relative path are overwritten.
- No files are removed.

You should now have (inside ``Forecasting-Platform/fp-frontend``):

- ``apps/web/app/(marketing)/layout.tsx`` (patched)
- ``apps/web/app/(marketing)/page.tsx`` (patched)
- ``apps/web/app/home/`` (new folders + patched pages)

Optional cleanup
""""""""""""""""

After successfully applying the patch, you may delete the patch folder:

- ``Forecasting-Platform/frontend_patch/``

(This does not affect the frontend after patching.)

Step 6 — Start local Supabase services (Docker must be running)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ensure Docker Desktop is running.

In ``Forecasting-Platform/fp-frontend``:

.. code-block:: powershell

   pnpm run supabase:web:start

Wait for: ``Started supabase local development setup....``

Leave this terminal running.

Step 7 — Start the Next.js dev server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open a second terminal in ``Forecasting-Platform/fp-frontend`` and run:

.. code-block:: powershell

   pnpm run dev

Frontend available at: http://localhost:3000

Backend Setup (FastAPI)
-----------------------

The backend is a FastAPI application that uses Supabase to:

- Validate user identity (JWT Bearer token from Supabase Auth)
- Store and retrieve artifacts from buckets (datasets/models)
- Write and read datasets/models metadata in tables

Backend environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Edit:

- ``Forecasting-Platform/backend/.env``

Set:

.. code-block:: text

   SUPABASE_URL="https://xxxx.supabase.co"
   SUPABASE_SERVICE_KEY="your_service_role_key"

Important:

- The service role key is server-only.
- Never expose it in frontend environment files.

Run backend with Docker (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd Forecasting-Platform/backend

   # build + run
   docker build -t forecasting-api .
   docker run -d -p 8000:80 --name prediction-api-container \
     --env SUPABASE_URL="https://xxxx.supabase.co" \
     --env SUPABASE_SERVICE_KEY="your_service_role_key" \
     forecasting-api

Backend available at: http://localhost:8000

Run backend locally (no Docker)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd Forecasting-Platform/backend
   python -m venv .venv

   # Linux/macOS:
   source .venv/bin/activate

   # Windows:
   # .venv\Scripts\activate

   pip install -r requirements.txt
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

Backend available at: http://localhost:8000

Final Project Structure (After Setup)
-------------------------------------

After completing setup, your local structure should look like this:

.. code-block:: text

   Forecasting-Platform/
    ├── backend/                       # FastAPI backend
    │   ├── modules/
    │   │   ├── automl.py
    │   │   ├── data_forge.py
    │   │   ├── data_ingestion.py
    │   │   ├── explainx.py
    │   │   └── predict_x.py
    │   ├── db/
    │   │   └── schema.sql
    │   ├── main.py
    │   ├── requirements.txt
    │   ├── Dockerfile
    │   └── .env                      # Backend environment variables
    │   
    ├── docs/                    # Read the Docs sources
    │   
    ├── fp-frontend/             # MakerKit frontend (cloned by the user, patched)
    │   ├── apps/web/app/
    │   │   ├── (marketing)/
    │   │   ├── auth/
    │   │   └── home/
    │   │       ├── dashboard/
    │   │       ├── datasets/
    │   │       ├── automl/
    │   │       ├── models/
    │   │       ├── predictions/
    │   │       └── monitoring/
    │   └── .env.local
    │   
    ├── README.md
    ├── LICENSE
    └── readthedocs.yaml

Optional:

- You may delete ``frontend_patch/`` after applying it successfully.


End-to-End Workflow
-------------------

1) Upload a dataset
^^^^^^^^^^^^^^^^^^^

- Uploads the file to Storage (original + parquet copies)
- Stores metadata and ingestion report in the ``datasets`` table

2) EDA
^^^^^^

- Reads parquet and returns:
  - preview rows
  - numeric histograms
  - target series data (for visualization)

3) Preprocessing
^^^^^^^^^^^^^^^^

- Cleans, validates, standardizes, and produces a processed parquet
- Stores a preprocessing report

4) AutoML Training
^^^^^^^^^^^^^^^^^^

- Benchmarks selected models and metrics
- Validates (including walk-forward evaluation where applicable)
- Retrains the best model and stores:
  - model artifact in Storage (``models`` bucket)
  - evaluation report in ``models`` table
  - explainability artifacts (ExplainX)

5) Prediction
^^^^^^^^^^^^^

- Loads model artifact from Storage
- Generates predictions for new inputs or uploaded data

6) Explainability
^^^^^^^^^^^^^^^^^

- Returns precomputed ExplainX artifacts (if generated during training)
- Or computes explanation on-demand (depending on backend configuration)

Troubleshooting
---------------

- Docker must be running for local Supabase services (``pnpm run supabase:web:start``).
- If ``SUPABASE_URL`` or ``SUPABASE_SERVICE_KEY`` is missing, the backend will fail at startup.
- Ensure your Storage buckets exist: ``datasets`` and ``models``.
- Ensure tables exist: ``datasets`` and ``models``.
- If the frontend cannot reach the backend, verify:
  - backend is running on ``http://localhost:8000``
  - ``NEXT_PUBLIC_API_URL`` is set correctly in ``fp-frontend/.env.local``
