'use client';

import React, { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@kit/ui/card';
import { Button } from '@kit/ui/button';
import { Badge } from '@kit/ui/badge';

import {
  Sparkles,
  Database,
  Boxes,
  Wand2,
  LineChart as LineChartIcon,
  ChevronRight,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  Info,
} from 'lucide-react';

import { getSupabaseBrowserClient } from '@kit/supabase/browser-client';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
const supabase = getSupabaseBrowserClient();

// ------------------------ types ------------------------
type DatasetRow = {
  id: string;
  original_filename: string;
  created_at: string;
  status?: string | null;

  time_column?: string | null;
  target_column?: string | null;
  forecast_horizon?: number | null;

  is_multi_entity?: boolean | null;
  entity_column?: string | null;

  rows_count?: number | null;
  columns_count?: number | null;
};

type ModelRow = {
  id: string;
  dataset_id?: string | null;
  model_name?: string | null;
  algorithm?: string | null;
  status?: string | null;
  created_at?: string | null;

  primary_metric?: string | null;
  training_report?: any | null;
  evaluation_report?: any | null;
};

type PredictMode = 'one_step' | 'multi_step';
type EntityScope = 'one' | 'all';

type PredictResult = {
  mode: PredictMode;
  steps: number;
  horizon: number;
  dataset_id: string;
  model_id: string;
  entity_scope: EntityScope;
  entity_value?: string | number | null;
  payload: any;
};

// ------------------------ utils ------------------------
const fmtDate = (iso?: string | null) => (iso ? new Date(iso).toLocaleString() : '—');

const fmtNum = (n?: number | null) =>
  typeof n === 'number' && Number.isFinite(n) ? n.toLocaleString() : '—';

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

const pretty = (obj: any) => {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
};

async function authorizedFetch<T = any>(path: string, init?: RequestInit): Promise<T> {
  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session?.access_token) {
    throw new Error('Not authenticated. Please sign in again.');
  }

  const token = session.access_token;

  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.headers || {}),
      Authorization: `Bearer ${token}`,
      ...(init?.method && init.method !== 'GET' ? { 'Content-Type': 'application/json' } : {}),
    },
    cache: 'no-store',
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} ${txt}`);
  }

  if (res.status === 204) return {} as T;
  return res.json();
}

// ------------------------ backend calls ------------------------
async function loadDatasets(): Promise<DatasetRow[]> {
  const rows = await authorizedFetch<any[]>('/api/v1/datasets');
  return (rows || []).map((r) => ({
    id: String(r.id),
    original_filename: String(r.original_filename ?? 'Dataset'),
    created_at: String(r.created_at ?? ''),
    status: r.status ?? null,

    time_column: r.time_column ?? null,
    target_column: r.target_column ?? null,
    forecast_horizon: typeof r.forecast_horizon === 'number' ? r.forecast_horizon : null,

    is_multi_entity: Boolean(r.is_multi_entity),
    entity_column: r.entity_column ?? null,

    rows_count: typeof r.rows_count === 'number' ? r.rows_count : null,
    columns_count: typeof r.columns_count === 'number' ? r.columns_count : null,
  }));
}

async function loadModels(): Promise<ModelRow[]> {
  const rows = await authorizedFetch<any[]>('/api/v1/models');
  return (rows || []).map((r) => ({
    id: String(r.id),
    dataset_id: r.dataset_id ?? null,
    model_name: r.model_name ?? r.name ?? null,
    algorithm: r.algorithm ?? null,
    status: r.status ?? 'ready',
    created_at: r.created_at ?? null,
    primary_metric: r.primary_metric ?? (r.training_report?.primary_metric ?? null),
    training_report: r.training_report ?? null,
    evaluation_report: r.evaluation_report ?? null,
  }));
}

/**
 * MUST match backend response:
 * { entity_column: string | null, values: string[] }
 */
async function loadDatasetEntities(datasetId: string): Promise<{ entity_column: string | null; values: string[] }> {
  return authorizedFetch(`/api/v1/datasets/${encodeURIComponent(datasetId)}/entities`);
}

async function predictApi(body: any): Promise<any> {
  return authorizedFetch('/api/v1/predict', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

async function predictApiV2(body: any): Promise<any> {
  return authorizedFetch('/api/v2/predict', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// ------------------------ small UI helpers ------------------------
function StepBadge({ done, label }: { done: boolean; label: string }) {
  return (
    <span
      className={[
        'inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium',
        done
          ? 'border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900/50 dark:bg-emerald-900/20 dark:text-emerald-200'
          : 'border-neutral-200 bg-white/70 text-neutral-700 dark:border-neutral-800 dark:bg-neutral-950/60 dark:text-neutral-200',
      ].join(' ')}
    >
      {done ? <CheckCircle2 className="h-3.5 w-3.5" /> : <Info className="h-3.5 w-3.5 opacity-70" />}
      {label}
    </span>
  );
}

function SectionTitle({
  title,
  desc,
  right,
}: {
  title: string;
  desc?: string;
  right?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div>
        <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
        {desc ? <p className="text-sm text-neutral-600 dark:text-neutral-400">{desc}</p> : null}
      </div>
      {right ? <div className="shrink-0">{right}</div> : null}
    </div>
  );
}

// ------------------------ page ------------------------
export default function PredictionsPage() {
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [models, setModels] = useState<ModelRow[]>([]);

  const [datasetId, setDatasetId] = useState<string>('');
  const [modelId, setModelId] = useState<string>('');

  const [entities, setEntities] = useState<string[]>([]);
  const [entitiesLoading, setEntitiesLoading] = useState(false);

  const [entityScope, setEntityScope] = useState<EntityScope>('one');
  const [entityValue, setEntityValue] = useState<string>('');
  const [mode, setMode] = useState<PredictMode>('multi_step');
  const [steps, setSteps] = useState<number>(1);

  const [result, setResult] = useState<PredictResult | null>(null);

  const [loading, setLoading] = useState({
    datasets: true,
    models: true,
    predict: false,
  });
  const [error, setError] = useState<string | null>(null);
  const searchParams = useSearchParams();
  const [useV2, setUseV2] = useState(false);

  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        setError(null);
        setLoading((s) => ({ ...s, datasets: true }));
        const ds = await loadDatasets();
        if (!mounted) return;
        setDatasets(ds);
      } catch (e: any) {
        if (!mounted) return;
        setError(e?.message ?? 'Failed to load datasets.');
      } finally {
        if (mounted) setLoading((s) => ({ ...s, datasets: false }));
      }
    })();

    (async () => {
      try {
        setError(null);
        setLoading((s) => ({ ...s, models: true }));
        const md = await loadModels();
        if (!mounted) return;
        setModels(md);
      } catch (e: any) {
        if (!mounted) return;
        setError((prev) => prev ?? e?.message ?? 'Failed to load models.');
      } finally {
        if (mounted) setLoading((s) => ({ ...s, models: false }));
      }
    })();

    return () => {
      mounted = false;
    };
  }, []);

  const selectedDataset = useMemo(() => datasets.find((d) => d.id === datasetId) ?? null, [datasets, datasetId]);

  const modelsForDataset = useMemo(() => {
    if (!datasetId) return [];
    return models
      .filter((m) => String(m.dataset_id ?? '') === datasetId)
      .sort((a, b) => String(b.created_at ?? '').localeCompare(String(a.created_at ?? '')));
  }, [models, datasetId]);

  const selectedModel = useMemo(() => models.find((m) => m.id === modelId) ?? null, [models, modelId]);

  const horizon = useMemo(() => {
    const h = selectedDataset?.forecast_horizon;
    return typeof h === 'number' && Number.isFinite(h) && h > 0 ? h : 1;
  }, [selectedDataset]);

  useEffect(() => {
    setModelId('');
    setResult(null);
    setEntities([]);
    setEntityScope('one');
    setEntityValue('');
    setMode('multi_step');
    setSteps(1);
  }, [datasetId]);

  useEffect(() => {
    const dsId = searchParams.get('dataset_id') ?? '';
    const mId = searchParams.get('model_id') ?? '';

    if (dsId) setDatasetId(dsId);
    if (mId) setModelId(mId);
  }, [searchParams]);

  useEffect(() => {
    if (mode === 'one_step') {
      setSteps(1);
    } else {
      setSteps((prev) => {
        const next = prev <= 1 ? horizon : prev;
        return clamp(next, 1, Math.max(1, horizon));
      });
    }
  }, [mode, horizon]);

  useEffect(() => {
    let mounted = true;

    const shouldLoad =
      Boolean(selectedDataset?.is_multi_entity) &&
      Boolean(selectedDataset?.entity_column) &&
      Boolean(datasetId);

    if (!shouldLoad) return;

    (async () => {
      try {
        setEntitiesLoading(true);
        setError(null);
        const res = await loadDatasetEntities(datasetId);
        if (!mounted) return;
        const vals = Array.isArray(res?.values) ? res.values : [];
        setEntities(vals);
      } catch (e: any) {
        if (!mounted) return;
        setEntities([]);
        setError(e?.message ?? 'Failed to load entity values.');
      } finally {
        if (mounted) setEntitiesLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, [datasetId, selectedDataset?.is_multi_entity, selectedDataset?.entity_column]);

  const isMultiEntity = Boolean(selectedDataset?.is_multi_entity) && Boolean(selectedDataset?.entity_column);
  const canPickModel = Boolean(datasetId);
  const canPredict =
    Boolean(datasetId) &&
    Boolean(modelId) &&
    (!isMultiEntity || entityScope === 'all' || (entityScope === 'one' && String(entityValue).length > 0));

  const stepsEffective = mode === 'one_step' ? 1 : clamp(steps, 1, Math.max(1, horizon));

  async function handlePredict() {
    if (!selectedDataset || !selectedModel) return;

    setError(null);
    setResult(null);

    try {
      setLoading((s) => ({ ...s, predict: true }));

      const body = {
        dataset_id: selectedDataset.id,
        model_id: selectedModel.id,
        mode,
        steps: stepsEffective,
        horizon, 
        entity_scope: isMultiEntity ? entityScope : 'one',
        entity_value: isMultiEntity && entityScope === 'one' ? entityValue : null,
      };

      const payload = useV2 ? await predictApiV2(body) : await predictApi(body);

      setResult({
        mode,
        steps: stepsEffective,
        horizon,
        dataset_id: selectedDataset.id,
        model_id: selectedModel.id,
        entity_scope: isMultiEntity ? entityScope : 'one',
        entity_value: isMultiEntity && entityScope === 'one' ? entityValue : null,
        payload,
      });
    } catch (e: any) {
      setError(e?.message ?? 'Prediction failed.');
    } finally {
      setLoading((s) => ({ ...s, predict: false }));
    }
  }

  const step1Done = Boolean(datasetId);
  const step2Done = Boolean(modelId);
  const step3Done = canPredict;

  return (
    <div className="relative min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,#000_70%,transparent_110%)] dark:bg-[linear-gradient(to_right,#262626_1px,transparent_1px),linear-gradient(to_bottom,#262626_1px,transparent_1px)]" />

      <div className="relative mx-auto max-w-7xl px-6 py-10 space-y-10">
        <header className="space-y-6">
          <div className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-white/80 px-4 py-2 text-sm shadow-sm backdrop-blur dark:border-neutral-800 dark:bg-neutral-900/80">
            <Sparkles className="h-3 w-3 text-blue-600 dark:text-cyan-400" />
            <span className="font-medium text-gray-900 dark:text-neutral-50">Forecasting</span>
            <span className="text-gray-600 dark:text-neutral-400">Dataset → Model → Config → Predict</span>
          </div>

          <div className="flex flex-col justify-between gap-6 md:flex-row md:items-end">
            <div>
              <h1 className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-600 bg-clip-text text-4xl font-bold tracking-tight text-transparent md:text-5xl dark:from-neutral-50 dark:via-neutral-200 dark:to-neutral-400">
                Predictions
              </h1>
              <p className="mt-3 max-w-2xl text-neutral-600 dark:text-neutral-400">
                Choose a dataset, select one of its trained models, configure entity & horizon settings, then run a prediction.
              </p>

              <div className="mt-4 flex flex-wrap gap-2">
                <StepBadge done={step1Done} label="1) Dataset selected" />
                <StepBadge done={step2Done} label="2) Model selected" />
                <StepBadge done={step3Done} label="3) Ready to predict" />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Link href="/home/datasets/upload">
                <Button className="rounded-full">
                  <Database className="mr-2 h-4 w-4" />
                  Upload dataset
                </Button>
              </Link>
              <Link href="/home/automl">
                <Button variant="outline" className="rounded-full">
                  <Wand2 className="mr-2 h-4 w-4" />
                  Train AutoML
                </Button>
              </Link>
            </div>
          </div>
        </header>

        {error ? (
          <div className="rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-900/20 dark:text-red-200">
            {error}
          </div>
        ) : null}

        {/* STEP 1 */}
        <section className="space-y-4">
          <SectionTitle
            title="Step 1 — Choose a dataset"
            desc="Forecast context is defined by the dataset row."
            right={
              <Button
                variant="outline"
                size="sm"
                className="rounded-full"
                onClick={async () => {
                  try {
                    setError(null);
                    setLoading((s) => ({ ...s, datasets: true }));
                    const ds = await loadDatasets();
                    setDatasets(ds);
                  } catch (e: any) {
                    setError(e?.message ?? 'Failed to refresh datasets.');
                  } finally {
                    setLoading((s) => ({ ...s, datasets: false }));
                  }
                }}
              >
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
            }
          />

          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardHeader>
              <CardTitle className="text-base">Datasets</CardTitle>
              <CardDescription>Select the dataset you want to predict from.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {loading.datasets ? (
                <div className="text-sm text-neutral-600 dark:text-neutral-400">Loading datasets…</div>
              ) : datasets.length ? (
                <>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="space-y-2">
                      <div className="text-sm font-medium">Dataset</div>
                      <select
                        value={datasetId}
                        onChange={(e) => setDatasetId(e.target.value)}
                        className="h-10 w-full rounded-md border border-neutral-200 bg-white px-3 text-sm shadow-sm dark:border-neutral-800 dark:bg-neutral-950"
                      >
                        <option value="">— Select dataset —</option>
                        {datasets.map((d) => (
                          <option key={d.id} value={d.id}>
                            {d.original_filename} ({d.id.slice(0, 8)})
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="space-y-2">
                      <div className="text-sm font-medium">Status</div>
                      <div className="flex items-center gap-2">
                        <Badge variant="default">{selectedDataset?.status ?? '—'}</Badge>
                        <span className="text-xs text-neutral-500">Created: {fmtDate(selectedDataset?.created_at ?? null)}</span>
                      </div>
                      <div className="text-xs text-neutral-500">
                        Rows: {fmtNum(selectedDataset?.rows_count ?? null)} · Cols: {fmtNum(selectedDataset?.columns_count ?? null)}
                      </div>
                    </div>
                  </div>

                  {selectedDataset ? (
                    <div className="rounded-md border border-neutral-200 bg-white/70 p-4 text-sm dark:border-neutral-800 dark:bg-neutral-950/60">
                      <div className="grid gap-3 md:grid-cols-4">
                        <div>
                          <div className="text-xs text-neutral-500">Time column</div>
                          <div className="font-medium">{selectedDataset.time_column ?? '—'}</div>
                        </div>
                        <div>
                          <div className="text-xs text-neutral-500">Target column</div>
                          <div className="font-medium">{selectedDataset.target_column ?? '—'}</div>
                        </div>
                        <div>
                          <div className="text-xs text-neutral-500">Horizon</div>
                          <div className="font-medium">{fmtNum(selectedDataset.forecast_horizon ?? null)}</div>
                        </div>
                        <div>
                          <div className="text-xs text-neutral-500">Entity</div>
                          <div className="font-medium">
                            {selectedDataset.is_multi_entity ? `${selectedDataset.entity_column ?? 'entity'} (multi)` : 'single'}
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </>
              ) : (
                <div className="flex items-center gap-3 text-sm text-neutral-600 dark:text-neutral-400">
                  <AlertTriangle className="h-4 w-4" />
                  No datasets found. Upload one to begin.
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        {/* STEP 2 */}
        <section className="space-y-4">
          <SectionTitle title="Step 2 — Choose a trained model" desc="Models are filtered by dataset_id so you only see compatible models." />

          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardHeader>
              <CardTitle className="text-base">Models</CardTitle>
              <CardDescription>Select a model trained on the chosen dataset.</CardDescription>
            </CardHeader>

            <CardContent className="space-y-4">
              {!canPickModel ? (
                <div className="text-sm text-neutral-600 dark:text-neutral-400">Select a dataset first.</div>
              ) : loading.models ? (
                <div className="text-sm text-neutral-600 dark:text-neutral-400">Loading models…</div>
              ) : modelsForDataset.length ? (
                <div className="grid gap-3 md:grid-cols-2">
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Model</div>
                    <select
                      value={modelId}
                      onChange={(e) => setModelId(e.target.value)}
                      className="h-10 w-full rounded-md border border-neutral-200 bg-white px-3 text-sm shadow-sm dark:border-neutral-800 dark:bg-neutral-950"
                    >
                      <option value="">— Select model —</option>
                      {modelsForDataset.map((m) => (
                        <option key={m.id} value={m.id}>
                          {(m.model_name ?? m.algorithm ?? 'Model')} ({m.id.slice(0, 8)})
                        </option>
                      ))}
                    </select>
                    <div className="text-xs text-neutral-500">Found {modelsForDataset.length} model(s) for this dataset.</div>
                  </div>

                  <div className="space-y-2">
                    <div className="text-sm font-medium">Model meta</div>
                    <div className="rounded-md border border-neutral-200 bg-white/70 p-3 text-sm dark:border-neutral-800 dark:bg-neutral-950/60">
                      <div className="flex items-center justify-between gap-2">
                        <div className="font-medium">{selectedModel?.model_name ?? selectedModel?.algorithm ?? '—'}</div>
                        <Badge variant="default">{selectedModel?.status ?? '—'}</Badge>
                      </div>
                      <div className="mt-2 text-xs text-neutral-500">
                        Algorithm: {selectedModel?.algorithm ?? '—'} · Created: {fmtDate(selectedModel?.created_at ?? null)}
                      </div>
                      <div className="mt-1 text-xs text-neutral-500">Primary metric: {selectedModel?.primary_metric ?? '—'}</div>

                      {selectedModel ? (
                        <div className="mt-3 flex gap-2">
                          <Link href={`/home/models/${encodeURIComponent(selectedModel.id)}`}>
                            <Button variant="outline" size="sm" className="rounded-full">
                              <Info className="mr-2 h-4 w-4" />
                              View model
                            </Button>
                          </Link>

                          <Link href="/home/automl">
                            <Button variant="outline" size="sm" className="rounded-full">
                              <Wand2 className="mr-2 h-4 w-4" />
                              Train more
                            </Button>
                          </Link>
                        </div>
                      ) : null}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-3 text-sm text-neutral-600 dark:text-neutral-400">
                  <Boxes className="h-4 w-4" />
                  No models found for this dataset yet. Run AutoML.
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        {/* STEP 3 */}
        <section className="space-y-4">
          <SectionTitle title="Step 3 — Configure prediction" desc="Choose entity scope and prediction mode (one-step vs multi-step)." />

          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardHeader>
              <CardTitle className="text-base">Prediction config</CardTitle>
              <CardDescription>This decides what the predict endpoint will compute.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              {!selectedDataset || !selectedModel ? (
                <div className="text-sm text-neutral-600 dark:text-neutral-400">Select dataset + model first.</div>
              ) : (
                <>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <div className="text-sm font-medium">Entity scope</div>
                      <select
                        value={isMultiEntity ? entityScope : 'one'}
                        onChange={(e) => setEntityScope(e.target.value as EntityScope)}
                        disabled={!isMultiEntity}
                        className="h-10 w-full rounded-md border border-neutral-200 bg-white px-3 text-sm shadow-sm disabled:opacity-60 dark:border-neutral-800 dark:bg-neutral-950"
                      >
                        <option value="one">One entity</option>
                        <option value="all">All entities</option>
                      </select>
                      <div className="text-xs text-neutral-500">
                        {isMultiEntity ? `Multi-entity dataset (${selectedDataset.entity_column}).` : 'Single entity dataset — scope is fixed.'}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="text-sm font-medium">Entity value</div>
                      <select
                        value={entityValue}
                        onChange={(e) => setEntityValue(e.target.value)}
                        disabled={!isMultiEntity || entityScope !== 'one'}
                        className="h-10 w-full rounded-md border border-neutral-200 bg-white px-3 text-sm shadow-sm disabled:opacity-60 dark:border-neutral-800 dark:bg-neutral-950"
                      >
                        <option value="">
                          {entitiesLoading ? 'Loading entities…' : entityScope !== 'one' ? 'Disabled (scope = all)' : '— Select entity —'}
                        </option>
                        {entities.map((v) => (
                          <option key={String(v)} value={String(v)}>
                            {String(v)}
                          </option>
                        ))}
                      </select>
                      {isMultiEntity && entityScope === 'one' ? (
                        <div className="text-xs text-neutral-500">
                          Populated from distinct values of <b>{selectedDataset.entity_column}</b>.
                        </div>
                      ) : (
                        <div className="text-xs text-neutral-500">—</div>
                      )}
                    </div>
                  </div>

                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="space-y-2">
                      <div className="text-sm font-medium">Prediction mode</div>
                      <select
                        value={mode}
                        onChange={(e) => setMode(e.target.value as PredictMode)}
                        className="h-10 w-full rounded-md border border-neutral-200 bg-white px-3 text-sm shadow-sm dark:border-neutral-800 dark:bg-neutral-950"
                      >
                        <option value="one_step">One-step (t+1)</option>
                        <option value="multi_step">Multi-step (t+1 … t+K)</option>
                      </select>
                      <div className="text-xs text-neutral-500">
                        Horizon in dataset: <b>{horizon}</b>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="text-sm font-medium">Steps (K)</div>
                      <input
                        type="number"
                        min={1}
                        max={Math.max(1, horizon)}
                        value={stepsEffective}
                        onChange={(e) => setSteps(Number(e.target.value))}
                        disabled={mode === 'one_step'}
                        className="h-10 w-full rounded-md border border-neutral-200 bg-white px-3 text-sm shadow-sm disabled:opacity-60 dark:border-neutral-800 dark:bg-neutral-950"
                      />
                      <div className="text-xs text-neutral-500">
                        {mode === 'one_step' ? 'Fixed to 1.' : `Choose 1…${Math.max(1, horizon)} (recommended = horizon).`}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="text-sm font-medium">Safety</div>
                      <div className="rounded-md border border-neutral-200 bg-white/70 p-3 text-xs text-neutral-600 dark:border-neutral-800 dark:bg-neutral-950/60 dark:text-neutral-400">
                        {isMultiEntity && entityScope === 'all' ? (
                          <div className="flex items-start gap-2">
                            <AlertTriangle className="mt-0.5 h-4 w-4" />
                            <div>
                              Predicting for <b>all entities</b> can be heavy.
                            </div>
                          </div>
                        ) : (
                          <div className="flex items-start gap-2">
                            <CheckCircle2 className="mt-0.5 h-4 w-4 text-emerald-500" />
                            <div>Good default: one entity, multi-step = horizon.</div>
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      <input
                        type="checkbox"
                        checked={useV2}
                        onChange={(e) => setUseV2(e.target.checked)}
                      />
                      Use V2 predict
                    </div>

                  </div>

                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="text-xs text-neutral-500">
                      Ready:{' '}
                      {canPredict ? (
                        <b className="text-emerald-600 dark:text-emerald-400">Yes</b>
                      ) : (
                        <b className="text-red-600 dark:text-red-400">No</b>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        className="rounded-full"
                        onClick={() => {
                          setResult(null);
                          setError(null);
                          setEntityScope('one');
                          setEntityValue('');
                          setMode('multi_step');
                          setSteps(1);
                        }}
                      >
                        Reset config
                      </Button>

                      <Button className="rounded-full" onClick={handlePredict} disabled={!canPredict || loading.predict}>
                        <LineChartIcon className="mr-2 h-4 w-4" />
                        {loading.predict ? 'Predicting…' : 'Run prediction'}
                        <ChevronRight className="ml-2 h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </section>

        {/* RESULT */}
        <section className="space-y-4 pb-8">
          <SectionTitle title="Result" desc="Raw payload (for now)." />

          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardHeader>
              <CardTitle className="text-base">Prediction output</CardTitle>
              <CardDescription>We render the raw JSON so you can test the API contract.</CardDescription>
            </CardHeader>

            <CardContent className="space-y-4">
              {!result ? (
                <div className="text-sm text-neutral-600 dark:text-neutral-400">Run a prediction to see results here.</div>
              ) : (
                <>
                  <div className="rounded-md border border-neutral-200 bg-white/70 p-4 text-sm dark:border-neutral-800 dark:bg-neutral-950/60">
                    <div className="grid gap-3 md:grid-cols-4">
                      <div>
                        <div className="text-xs text-neutral-500">Mode</div>
                        <div className="font-medium">{result.mode}</div>
                      </div>
                      <div>
                        <div className="text-xs text-neutral-500">Steps</div>
                        <div className="font-medium">{result.steps}</div>
                      </div>
                      <div>
                        <div className="text-xs text-neutral-500">Entity</div>
                        <div className="font-medium">{result.entity_scope === 'all' ? 'all entities' : result.entity_value ?? '—'}</div>
                      </div>
                      <div>
                        <div className="text-xs text-neutral-500">IDs</div>
                        <div className="font-medium text-xs">
                          ds: {result.dataset_id.slice(0, 8)} · mdl: {result.model_id.slice(0, 8)}
                        </div>
                      </div>
                    </div>
                  </div>

                  <details className="rounded-md border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-950">
                    <summary className="cursor-pointer font-medium">Open raw JSON</summary>
                    <pre className="mt-3 text-xs whitespace-pre-wrap rounded-md border border-neutral-200 bg-white/70 p-4 dark:border-neutral-800 dark:bg-neutral-950/60">
                      {pretty(result.payload)}
                    </pre>
                  </details>
                </>
              )}
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  );
}
