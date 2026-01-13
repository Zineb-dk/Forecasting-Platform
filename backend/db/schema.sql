
-- =========================================================
-- DATASETS
-- =========================================================
CREATE TABLE IF NOT EXISTS public.datasets (
  id                     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id                uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

  original_filename      text NOT NULL,
  file_extension         text,
  file_size_bytes        bigint,

  -- Storage paths (Supabase Storage)
  storage_original_path  text,
  storage_parquet_path   text,
  processed_parquet_path text,  -- cleaned dataset after DataForge.preprocess

  -- Shape / structure
  rows_count             integer,
  columns_count          integer,

  -- Integrity
  file_hash              text,

  -- Time-series metadata
  time_column            text,     -- user-selected time column
  target_column          text,     -- user-selected target
  forecast_horizon       integer,  -- horizon_steps (RUL etc.)

  -- Multi-entity
  is_multi_entity        boolean NOT NULL DEFAULT false,
  entity_column          text,     -- required if is_multi_entity = true
  sensor_columns         text[],   -- computed after user config

  -- Status
  status                 text NOT NULL DEFAULT 'uploaded'
    CHECK (status IN ('uploaded', 'processing', 'ready')),

  -- Saved metadata / reports
  ingestion_info         jsonb,
  processing_report      jsonb,
  raw_plots              jsonb,
  processed_plots        jsonb,

  created_at             timestamptz DEFAULT now(),
  updated_at             timestamptz DEFAULT now(),

  -- Enforce entity column if multi-entity
  CONSTRAINT datasets_multi_entity_entity_required
    CHECK (NOT is_multi_entity OR entity_column IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_datasets_user_id ON public.datasets(user_id);
CREATE INDEX IF NOT EXISTS idx_datasets_status  ON public.datasets(status);

-- Enable RLS
ALTER TABLE public.datasets ENABLE ROW LEVEL SECURITY;

-- RLS Policies
DROP POLICY IF EXISTS "Users can view their own datasets"   ON public.datasets;
DROP POLICY IF EXISTS "Users can insert their own datasets" ON public.datasets;
DROP POLICY IF EXISTS "Users can update their own datasets" ON public.datasets;
DROP POLICY IF EXISTS "Users can delete their own datasets" ON public.datasets;

CREATE POLICY "Users can view their own datasets"
  ON public.datasets
  FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own datasets"
  ON public.datasets
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own datasets"
  ON public.datasets
  FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own datasets"
  ON public.datasets
  FOR DELETE
  USING (auth.uid() = user_id);


-- =========================================================
-- MODELS
-- =========================================================
CREATE TABLE IF NOT EXISTS public.models (
  id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Ownership
  user_id              uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  dataset_id           uuid NOT NULL REFERENCES public.datasets(id) ON DELETE CASCADE,

  -- Identity
  model_name           text NOT NULL,
  algorithm            text,

  -- Lifecycle status
  status               text NOT NULL DEFAULT 'training'
    CHECK (status IN ('training', 'ready', 'deployed', 'failed')),

  -- Key/high-level metric shown in UI
  primary_metric       text,   -- e.g. 'RMSE: 1.84'

  -- Detailed metrics
  train_metrics        jsonb,
  test_metrics         jsonb,

  -- Reports & artifacts
  training_report      jsonb,
  evaluation_report    jsonb,
  model_plots          jsonb,
  artifacts_path       text,

  -- Optional denormalized info
  target_column        text,
  forecast_horizon     integer,

  -- Explainability + scalers (added later; included here for completeness)
  explain_status       text DEFAULT 'none',
  explain_report       jsonb NOT NULL DEFAULT '{}'::jsonb,
  explain_artifacts_path text,

  feature_scaler_path  text,
  target_scaler_path   text,

  created_at           timestamptz DEFAULT now(),
  updated_at           timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_models_user_id    ON public.models(user_id);
CREATE INDEX IF NOT EXISTS idx_models_dataset_id ON public.models(dataset_id);
CREATE INDEX IF NOT EXISTS idx_models_status     ON public.models(status);

-- Enable RLS
ALTER TABLE public.models ENABLE ROW LEVEL SECURITY;

-- RLS Policies
DROP POLICY IF EXISTS "Users can view their own models"   ON public.models;
DROP POLICY IF EXISTS "Users can insert their own models" ON public.models;
DROP POLICY IF EXISTS "Users can update their own models" ON public.models;
DROP POLICY IF EXISTS "Users can delete their own models" ON public.models;

CREATE POLICY "Users can view their own models"
  ON public.models
  FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own models"
  ON public.models
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own models"
  ON public.models
  FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own models"
  ON public.models
  FOR DELETE
  USING (auth.uid() = user_id);


