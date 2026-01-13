'use client';

import { useEffect, useState, FormEvent } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@kit/ui/card';
import { Button } from '@kit/ui/button';
import { Input } from '@kit/ui/input';
import { Label } from '@kit/ui/label';
import { Badge } from '@kit/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@kit/ui/select';
import { Checkbox } from '@kit/ui/checkbox';

import {
  Activity,
  ArrowLeft,
  BarChart3,
  Brain,
  CheckCircle2,
  Database,
  LineChart as LineChartIcon,
  Loader2,
  Sparkles,
} from 'lucide-react';

import {
  Line as RechartsLine,
  LineChart as RechartsLineChart,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Bar,
  BarChart as RechartsBarChart,
  Legend,
} from 'recharts';

import { getSupabaseBrowserClient } from '@kit/supabase/browser-client';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
const supabase = getSupabaseBrowserClient();

type DatasetRow = {
  id: string;
  original_filename: string;
  status: string;
  created_at: string;
  rows_count?: number | null;
  columns_count?: number | null;
};

type WFFold = {
  fold: number;
  [metric: string]: number | undefined;
};

type AutoMLPlots = {
  tabular?: {
    y_test: number[];
    models: {
      [name: string]: {
        y_pred: number[];
        metrics: Record<string, number>;
      };
    };
  };
  seq?: {
    y_val: number[];
    models: {
      [name: string]: {
        y_pred: number[];
        metrics: Record<string, number>;
      };
    };
  };
  walk_forward?: {
    [name: string]: {
      type: 'tabular' | 'seq';
      avg_metric: number;
      folds: WFFold[];
    };
  };
};

type AutoMLModelSummary = {
  name: string;
  type: 'tabular' | 'seq';
  primary_metric_value: number;
  metrics: Record<string, number>;
  wf_avg_metric?: number;
};

type AutoMLStep = {
  key: string;
  label: string;
  done: boolean;
};

type AutoMLRunResponse = {
  dataset_id: string;
  primary_metric: string;
  steps?: AutoMLStep[];
  models: AutoMLModelSummary[];
  best_model_id: string | null;
  best_model_name: string | null;
  best_model_type: 'tabular' | 'seq' | null;
  best_avg_metric: number | null;
  plots?: AutoMLPlots;
};

async function authorizedFetch<T = any>(
  path: string,
  init?: RequestInit,
): Promise<T> {
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
      'Content-Type': 'application/json',
    },
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} ${txt}`);
  }

  return res.json();
}

async function loadDatasets(): Promise<DatasetRow[]> {
  const rows = await authorizedFetch<any[]>('/api/v1/datasets');

  return (rows || []).map(
    (r) =>
      ({
        id: r.id,
        original_filename: r.original_filename,
        status: r.status ?? 'ready',
        created_at: r.created_at,
        rows_count: r.rows_count ?? r.ingestion_info?.shape?.[0] ?? null,
        columns_count: r.columns_count ?? r.ingestion_info?.shape?.[1] ?? null,
      }) as DatasetRow,
  );
}

async function runAutoml(
  datasetId: string,
  body: any,
): Promise<AutoMLRunResponse> {
  return authorizedFetch<AutoMLRunResponse>(
    `/api/v1/datasets/${datasetId}/automl/run`,
    {
      method: 'POST',
      body: JSON.stringify(body),
    },
  );
}

async function runAutomlV2(
  datasetId: string,
  body: any,
): Promise<AutoMLRunResponse> {
  return authorizedFetch<AutoMLRunResponse>(
    `/api/v2/datasets/${datasetId}/automl/run`,
    {
      method: 'POST',
      body: JSON.stringify(body),
    },
  );
}

function buildTrueVsPredData(
  yTrue?: Array<number | null> | null,
  yPred?: Array<number | null> | null,
) {
  if (!yTrue || !yPred) return [];
  const n = Math.min(yTrue.length, yPred.length);
  const data: { idx: number; true: number; pred: number }[] = [];
  for (let i = 0; i < n; i++) {
    const t = yTrue[i];
    const p = yPred[i];
    if (
      typeof t !== 'number' ||
      typeof p !== 'number' ||
      !Number.isFinite(t) ||
      !Number.isFinite(p)
    ) {
      continue;
    }
    data.push({ idx: i, true: t, pred: p });
  }
  return data;
}

function buildWfFoldData(folds: WFFold[] | undefined, metric: string) {
  if (!folds) return [];
  return folds.map((f) => ({
    fold: f.fold,
    value: f[metric] ?? NaN,
  }));
}

const fmtMetric = (v: number | undefined | null) =>
  typeof v === 'number' && isFinite(v) ? v.toFixed(3) : '–';

type LineVisibility = 'both' | 'true' | 'pred';

function isHigherBetter(metric: string) {
  return metric === 'R2';
}

function rankModels(
  models: AutoMLModelSummary[],
  primaryMetric: string,
): AutoMLModelSummary[] {
  const higherBetter = isHigherBetter(primaryMetric);
  return [...models].sort((a, b) => {
    const av = a.primary_metric_value;
    const bv = b.primary_metric_value;
    if (Number.isNaN(av) || Number.isNaN(bv)) return 0;
    return higherBetter ? bv - av : av - bv;
  });
}

export default function AutoMLPage() {
  const router = useRouter();
  const [useV2, setUseV2] = useState(false);
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');

  const [selectedModels, setSelectedModels] = useState<string[]>([
    'Random Forest',
    'XGBoost',
  ]);
  const [primaryMetric, setPrimaryMetric] = useState<
    'RMSE' | 'MAE' | 'R2' | 'MAPE' | 'sMAPE'
  >('RMSE');
  const [testSize, setTestSize] = useState<number | ''>(0.2);
  const [useClip, setUseClip] = useState(false);
  const [clipThreshold, setClipThreshold] = useState<number | ''>(125);
  const [topK, setTopK] = useState<number | ''>(2);
  const [nSplits, setNSplits] = useState<number | ''>(5);
  const [lookback, setLookback] = useState<number | ''>(30);
  const [epochs, setEpochs] = useState<number | ''>(50);
  const [batchSize, setBatchSize] = useState<number | ''>(32);
  const [doPlots, setDoPlots] = useState(false);

  const [runStatus, setRunStatus] = useState<
    'idle' | 'running' | 'done' | 'error'
  >('idle');
  const [runError, setRunError] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<AutoMLRunResponse | null>(null);

  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  const isRunning = runStatus === 'running';
  const progressPercent = runStatus === 'done' ? 100 : isRunning ? 60 : 0;

  const [tabularVisibility, setTabularVisibility] = useState<
    Record<string, LineVisibility>
  >({});
  const [seqVisibility, setSeqVisibility] = useState<
    Record<string, LineVisibility>
  >({});

  const selectedDataset =
    datasets.find((d) => d.id === selectedDatasetId) ?? null;

 
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        setDatasetsLoading(true);
        const ds = await loadDatasets();
        if (!mounted) return;
        setDatasets(ds);
        const first = ds[0];
        if (first) {
          setSelectedDatasetId(first.id);
        }
      } catch (err: any) {
        if (!mounted) return;
        setRunError(err?.message ?? 'Failed to load datasets for AutoML page.');
      } finally {
        if (mounted) setDatasetsLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, []);

  /* ---------------- run automl ---------------- */

  async function onRun(e: FormEvent) {
    e.preventDefault();
    setRunError(null);
    setRunResult(null);
    setSaveMessage(null);

    if (!selectedDatasetId) {
      setRunError('Please select a dataset.');
      return;
    }

    if (selectedModels.length === 0) {
      setRunError('Please select at least one model to train.');
      return;
    }

    if (!testSize || testSize <= 0 || testSize >= 1) {
      setRunError('test_size must be between 0 and 1 (e.g. 0.2).');
      return;
    }

    if (!topK || topK <= 0) {
      setRunError('top_k must be a positive integer.');
      return;
    }

    if (!nSplits || nSplits <= 1) {
      setRunError('n_splits must be at least 2.');
      return;
    }

    if (!lookback || lookback <= 1) {
      setRunError('lookback must be at least 2.');
      return;
    }

    if (!epochs || epochs <= 0) {
      setRunError('epochs must be a positive integer.');
      return;
    }

    if (!batchSize || batchSize <= 0) {
      setRunError('batch_size must be a positive integer.');
      return;
    }

    try {
      setRunStatus('running');

      const body = {
        models_to_train: selectedModels,
        primary_metric: primaryMetric,
        test_size: Number(testSize),
        use_clip: useClip,
        clip_threshold: useClip
          ? clipThreshold === ''
            ? null
            : Number(clipThreshold)
          : null,
        top_k: Number(topK),
        n_splits: Number(nSplits),
        lookback: Number(lookback),
        epochs: Number(epochs),
        batch_size: Number(batchSize),
        do_plots: doPlots,
      };

      const resp = useV2
        ? await runAutomlV2(selectedDatasetId, body)
        : await runAutoml(selectedDatasetId, body);

      setRunResult(resp);
      setRunStatus('done');

      if (resp.best_model_id) {
        setSaveMessage(
          `Best model saved automatically (id: ${resp.best_model_id}).`,
        );
      } else {
        setSaveMessage(null);
      }
    } catch (err: any) {
      setRunStatus('error');
      setRunError(err?.message ?? 'AutoML run failed.');
    }
  }

  /* ---------------- best model actions ---------------- */

  function handleNewConfig() {
    setRunResult(null);
    setRunStatus('idle');
    setRunError(null);
    setSaveMessage(null);

    setSelectedModels(['Random Forest', 'XGBoost']);
    setPrimaryMetric('RMSE');
    setTestSize(0.2);
    setUseClip(false);
    setClipThreshold(125);
    setTopK(2);
    setNSplits(5);
    setLookback(30);
    setEpochs(50);
    setBatchSize(32);
  }

  function handleGoToPredictions() {
    if (!runResult?.best_model_id) return;

    router.push(
      `/home/predictions?dataset_id=${encodeURIComponent(
        runResult.dataset_id,
      )}&model_id=${encodeURIComponent(runResult.best_model_id)}`,
    );
  }

  const bestModelName = runResult?.best_model_name ?? null;
  const plots: AutoMLPlots = runResult?.plots ?? {};
  const tabularPlot = plots.tabular;
  const seqPlot = plots.seq;
  const wfPlot = plots.walk_forward ?? {};
  const primaryMetricKey = runResult?.primary_metric ?? primaryMetric;

  const sortedModels = runResult?.models
    ? rankModels(runResult.models, primaryMetricKey)
    : [];

  const getTabVis = (name: string): LineVisibility =>
    tabularVisibility[name] ?? 'both';
  const getSeqVis = (name: string): LineVisibility =>
    seqVisibility[name] ?? 'both';

  return (
    <div className="relative min-h-screen overflow-auto bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50">
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,#000_70%,transparent_110%)] dark:bg-[linear-gradient(to_right,#262626_1px,transparent_1px),linear-gradient(to_bottom,#262626_1px,transparent_1px)]" />

      <div className="relative mx-auto max-w-7xl px-6 py-8 space-y-8">
        <div className="flex items-center justify-between gap-4">
          <Button
            variant="outline"
            size="sm"
            className="flex items-center gap-2 rounded-full"
            onClick={() => router.push('/home')}
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Button>

          <div className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-white/80 px-4 py-1.5 text-xs shadow-sm backdrop-blur dark:border-neutral-800 dark:bg-neutral-900/80">
            <Sparkles className="h-3 w-3 text-blue-600 dark:text-cyan-400" />
            <span className="font-medium text-gray-900 dark:text-neutral-50">
              AutoML
            </span>
            <span className="text-gray-600 dark:text-neutral-400">
              Select a dataset, configure, and launch model search
            </span>
          </div>
        </div>

        <header className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
          <div>
            <h1 className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-600 bg-clip-text text-3xl font-bold tracking-tight text-transparent md:text-4xl dark:from-neutral-50 dark:via-neutral-200 dark:to-neutral-400">
              AutoML Pipeline
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
              Benchmark tabular and deep sequence models, run walk-forward
              validation, and inspect diagnostics in one place.
            </p>
          </div>

          {runResult && (
            <div className="flex items-center gap-4 text-sm">
              <div className="rounded-xl bg-emerald-50 px-4 py-2 text-emerald-700 shadow-sm ring-1 ring-emerald-100 dark:bg-emerald-900/40 dark:text-emerald-100 dark:ring-emerald-900/50">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4" />
                  <span className="font-medium">
                    Best model:&nbsp;
                    {bestModelName ?? '—'}
                  </span>
                </div>
                {runResult.best_avg_metric != null && (
                  <div className="text-xs">
                    Avg {runResult.primary_metric}:{' '}
                    {fmtMetric(runResult.best_avg_metric)}
                  </div>
                )}
              </div>
            </div>
          )}
        </header>

        {runStatus !== 'idle' && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-[11px] text-neutral-500 dark:text-neutral-400">
              <span>
                {isRunning
                  ? 'AutoML run in progress…'
                  : runStatus === 'done'
                    ? 'AutoML run completed.'
                    : 'AutoML encountered an error.'}
              </span>
              {runResult?.steps && runResult.steps.length > 0 && (
                <span>
                  Steps: {runResult.steps.filter((s) => s.done).length}/
                  {runResult.steps.length}
                </span>
              )}
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
              <div
                className={`h-full rounded-full bg-gradient-to-r from-blue-500 via-cyan-400 to-emerald-500 transition-all duration-500 ${
                  isRunning ? 'animate-pulse' : ''
                }`}
                style={{
                  width: `${progressPercent}%`,
                }}
              />
            </div>
          </div>
        )}

        <section className="space-y-6">
          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Brain className="h-4 w-4 text-blue-600 dark:text-cyan-400" />
                AutoML configuration
              </CardTitle>
              <CardDescription>
                Choose the dataset, models, and main hyperparameters.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form className="space-y-5" onSubmit={onRun}>
                <div className="space-y-2">
                  <Label>Dataset</Label>
                  {datasetsLoading ? (
                    <div className="flex items-center gap-2 text-xs text-neutral-500">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      Loading datasets…
                    </div>
                  ) : datasets.length > 0 ? (
                    <>
                      <Select
                        value={selectedDatasetId}
                        onValueChange={setSelectedDatasetId}
                        disabled={isRunning}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Choose a dataset" />
                        </SelectTrigger>
                        <SelectContent>
                          {datasets.map((d) => (
                            <SelectItem key={d.id} value={d.id}>
                              {d.original_filename} ({d.status})
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      {selectedDataset && (
                        <div className="mt-2 rounded-lg bg-neutral-50 px-3 py-2 text-[11px] text-neutral-600 dark:bg-neutral-900/50 dark:text-neutral-400">
                          <div className="flex justify-between gap-2">
                            <span>Rows</span>
                            <span>
                              {selectedDataset.rows_count != null
                                ? selectedDataset.rows_count.toLocaleString()
                                : '—'}
                            </span>
                          </div>
                          <div className="flex justify-between gap-2">
                            <span>Columns</span>
                            <span>
                              {selectedDataset.columns_count != null
                                ? selectedDataset.columns_count.toLocaleString()
                                : '—'}
                            </span>
                          </div>
                          <div className="flex justify-between gap-2">
                            <span>Status</span>
                            <span className="capitalize">
                              {selectedDataset.status}
                            </span>
                          </div>
                          <div className="pt-2">
                            <Link
                              href={`/home/datasets/${selectedDataset.id}/eda`}
                              className="text-[11px] text-blue-600 hover:underline dark:text-cyan-400"
                            >
                              Open EDA for this dataset
                            </Link>
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="text-xs text-neutral-500">
                      No datasets found.{' '}
                      <Link href="/home/datasets/upload" className="underline">
                        Upload one first.
                      </Link>
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <Label>Models to train</Label>
                  <div className="grid gap-2 text-xs sm:grid-cols-2 md:grid-cols-3">
                    {['Random Forest', 'XGBoost', 'LSTM', 'TCN', 'TFT'].map(
                      (name) => (
                        <label
                          key={name}
                          className="flex items-center gap-2 rounded-lg border border-gray-200 bg-gray-50/70 px-3 py-2 text-[11px] font-medium dark:border-neutral-800 dark:bg-neutral-900/60"
                        >
                          <Checkbox
                            checked={selectedModels.includes(name)}
                            onCheckedChange={(checked) => {
                              const isChecked = Boolean(checked);
                              setSelectedModels((prev) =>
                                isChecked
                                  ? [...prev, name]
                                  : prev.filter((m) => m !== name),
                              );
                            }}
                            disabled={isRunning}
                          />
                          <span>{name}</span>
                        </label>
                      ),
                    )}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Primary metric</Label>
                  <Select
                    value={primaryMetric}
                    onValueChange={(v) =>
                      setPrimaryMetric(v as typeof primaryMetric)
                    }
                    disabled={isRunning}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="RMSE">RMSE (lower better)</SelectItem>
                      <SelectItem value="MAE">MAE</SelectItem>
                      <SelectItem value="R2">R² (higher better)</SelectItem>
                      <SelectItem value="MAPE">MAPE</SelectItem>
                      <SelectItem value="sMAPE">sMAPE</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2 rounded-lg border border-gray-200 bg-neutral-50 px-3 py-2 text-xs dark:border-neutral-800 dark:bg-neutral-900/50">
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <Label className="text-xs">Target clipping</Label>
                      <p className="text-[10px] text-neutral-500 dark:text-neutral-400">
                        Optionally clip extreme targets (e.g. RUL &gt; 125).
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="use-clip"
                        checked={useClip}
                        onCheckedChange={(val) => setUseClip(Boolean(val))}
                        disabled={isRunning}
                      />
                      <Label
                        htmlFor="use-clip"
                        className="text-[11px] text-neutral-700 dark:text-neutral-300"
                      >
                        Enable clipping
                      </Label>
                    </div>
                  </div>

                  {useClip && (
                    <div className="mt-2 space-y-1">
                      <Label htmlFor="clip-threshold" className="text-xs">
                        Clip threshold
                      </Label>
                      <Input
                        id="clip-threshold"
                        type="number"
                        min={1}
                        value={clipThreshold}
                        onChange={(e) =>
                          setClipThreshold(
                            e.target.value === ''
                              ? ''
                              : Number(e.target.value),
                          )
                        }
                        disabled={isRunning}
                      />
                    </div>
                  )}
                </div>

                <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3">
                  <div className="space-y-1">
                    <Label>Test size</Label>
                    <Input
                      type="number"
                      step="0.05"
                      min={0.05}
                      max={0.5}
                      value={testSize}
                      onChange={(e) =>
                        setTestSize(
                          e.target.value === ''
                            ? ''
                            : Number(e.target.value),
                        )
                      }
                      disabled={isRunning}
                    />
                    <p className="text-[10px] text-neutral-500">
                      Fraction of last samples reserved for test (tabular / seq)
                    </p>
                  </div>
                  <div className="space-y-1">
                    <Label>Top-K for walk-forward</Label>
                    <Input
                      type="number"
                      min={1}
                      value={topK}
                      onChange={(e) =>
                        setTopK(
                          e.target.value === ''
                            ? ''
                            : Number(e.target.value),
                        )
                      }
                      disabled={isRunning}
                    />
                    <p className="text-[10px] text-neutral-500">
                      Only best K models go to walk-forward.
                    </p>
                  </div>
                  <div className="space-y-1">
                    <Label>WF splits (TimeSeriesSplit)</Label>
                    <Input
                      type="number"
                      min={2}
                      value={nSplits}
                      onChange={(e) =>
                        setNSplits(
                          e.target.value === ''
                            ? ''
                            : Number(e.target.value),
                        )
                      }
                      disabled={isRunning}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Lookback (seq models)</Label>
                    <Input
                      type="number"
                      min={4}
                      value={lookback}
                      onChange={(e) =>
                        setLookback(
                          e.target.value === ''
                            ? ''
                            : Number(e.target.value),
                        )
                      }
                      disabled={isRunning}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Epochs (deep)</Label>
                    <Input
                      type="number"
                      min={1}
                      value={epochs}
                      onChange={(e) =>
                        setEpochs(
                          e.target.value === ''
                            ? ''
                            : Number(e.target.value),
                        )
                      }
                      disabled={isRunning}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Batch size (deep)</Label>
                    <Input
                      type="number"
                      min={1}
                      value={batchSize}
                      onChange={(e) =>
                        setBatchSize(
                          e.target.value === ''
                            ? ''
                            : Number(e.target.value),
                        )
                      }
                      disabled={isRunning}
                    />
                  </div>
                </div>

                <div className="flex items-center justify-between gap-2 pt-1">
                  <div className="text-[11px] text-neutral-500 dark:text-neutral-400">
                    If enabled, the backend will also produce matplotlib plots
                    (useful in notebooks). For API usage you can keep this off.
                  </div>
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="do-plots"
                      checked={doPlots}
                      onCheckedChange={(val) => setDoPlots(Boolean(val))}
                      disabled={isRunning}
                    />
                    <Label
                      htmlFor="do-plots"
                      className="text-[11px] text-neutral-700 dark:text-neutral-300"
                    >
                      do_plots
                    </Label>
                  </div>
                </div>

                <div className="flex items-center justify-between gap-2">
                  <div className="text-[11px] text-neutral-500 dark:text-neutral-400">
                    Use the V2 AutoML endpoints (only if backend supports /api/v2).
                  </div>
                  <div className="flex items-center gap-2">
                    <Checkbox
                      id="use-v2"
                      checked={useV2}
                      onCheckedChange={(val) => setUseV2(Boolean(val))}
                      disabled={isRunning}
                    />
                    <Label
                      htmlFor="use-v2"
                      className="text-[11px] text-neutral-700 dark:text-neutral-300"
                    >
                      use_v2
                    </Label>
                  </div>
                </div>

                <div className="pt-2 space-y-2">
                  <Button
                    type="submit"
                    className="rounded-full"
                    disabled={
                      isRunning ||
                      datasetsLoading ||
                      !selectedDatasetId ||
                      selectedModels.length === 0
                    }
                  >
                    {isRunning && (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    )}
                    {isRunning ? 'Running AutoML…' : 'Run AutoML'}
                  </Button>

                  {runError && (
                    <p className="text-xs text-red-600 dark:text-red-400">
                      {runError}
                    </p>
                  )}
                  {!runError && runStatus === 'idle' && (
                    <p className="text-xs text-neutral-500 dark:text-neutral-400">
                      AutoML will benchmark all selected models, run
                      walk-forward on top-K, and pick the best one. Plots are
                      rendered below.
                    </p>
                  )}
                </div>
              </form>
            </CardContent>
          </Card>

          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Activity className="h-4 w-4 text-blue-600 dark:text-cyan-400" />
                  Models leaderboard & comparison
                </CardTitle>
                <CardDescription>
                  All trained models ranked by the primary metric (
                  {primaryMetricKey}).
                </CardDescription>
              </div>
              {runResult && (
                <Badge variant="outline" className="text-[11px]">
                  {runResult.models.length} models trained
                </Badge>
              )}
            </CardHeader>

            <CardContent className="max-h-[360px] overflow-auto p-0">
              {sortedModels.length ? (
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-white/90 text-[11px] uppercase tracking-wide text-neutral-500 dark:bg-neutral-950/90 dark:text-neutral-400">
                    <tr>
                      <th className="px-4 py-2 text-left">#</th>
                      <th className="px-4 py-2 text-left">Model</th>
                      <th className="px-4 py-2 text-left">Type</th>
                      <th className="px-4 py-2 text-left">
                        {primaryMetricKey}
                      </th>
                      <th className="px-4 py-2 text-left">WF avg</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedModels.map((m, idx) => {
                      const isBest = m.name === bestModelName;
                      return (
                        <tr
                          key={m.name}
                          className={
                            'border-t border-gray-100 text-[11px] dark:border-neutral-800 ' +
                            (isBest
                              ? 'bg-emerald-50/60 dark:bg-emerald-900/20'
                              : 'bg-transparent')
                          }
                        >
                          <td className="px-4 py-2 font-semibold">{idx + 1}</td>
                          <td className="px-4 py-2 font-medium">{m.name}</td>
                          <td className="px-4 py-2">
                            <Badge
                              variant="outline"
                              className="border-dashed px-2 py-0 text-[10px]"
                            >
                              {m.type}
                            </Badge>
                          </td>
                          <td className="px-4 py-2">
                            {fmtMetric(m.primary_metric_value)}
                          </td>
                          <td className="px-4 py-2">
                            {fmtMetric(m.wf_avg_metric ?? undefined)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              ) : (
                <div className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-xs text-neutral-500">
                  <LineChartIcon className="h-6 w-6 text-neutral-400" />
                  {runStatus === 'running'
                    ? 'AutoML is running… Models will appear here.'
                    : 'No AutoML run yet. Configure and click “Run AutoML”.'}
                </div>
              )}
            </CardContent>
          </Card>

          {runResult && (
            <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm">
                  <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                  Best model summary & actions
                </CardTitle>
                <CardDescription>
                  Review the winning model, its metric, and decide what to do
                  next.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 text-xs">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">Best model:</span>
                  <Badge>
                    {runResult.best_model_name ?? '—'} (
                    {runResult.best_model_type ?? 'n/a'})
                  </Badge>
                  {typeof runResult.best_avg_metric === 'number' && (
                    <span className="text-neutral-600 dark:text-neutral-300">
                      avg {primaryMetricKey}:{' '}
                      {fmtMetric(runResult.best_avg_metric)}
                    </span>
                  )}
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleNewConfig}
                    disabled={isRunning}
                  >
                    New config
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={handleGoToPredictions}
                    disabled={!runResult.best_model_id}
                  >
                    Go to predictions
                  </Button>
                </div>

                {saveMessage && (
                  <p className="text-[11px] text-emerald-700 dark:text-emerald-300">
                    {saveMessage}
                  </p>
                )}
              </CardContent>
            </Card>
          )}
        </section>

        {tabularPlot && Object.keys(tabularPlot.models ?? {}).length > 0 && (
          <section className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="flex items-center gap-2 text-base font-semibold tracking-tight">
                  <LineChartIcon className="h-4 w-4 text-blue-600 dark:text-cyan-400" />
                  Tabular models — True vs Predicted
                </h2>
                <p className="text-xs text-neutral-600 dark:text-neutral-400">
                  For each tabular model, we plot the test target vs its
                  predictions.
                </p>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {Object.entries(tabularPlot.models).map(([name, info]) => {
                const data = buildTrueVsPredData(
                  tabularPlot.y_test,
                  info.y_pred,
                );
                if (!data.length) return null;

                const metricVal =
                  info.metrics?.[
                    primaryMetricKey as keyof typeof info.metrics
                  ];

                const vis = getTabVis(name);

                return (
                  <Card
                    key={name}
                    className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80"
                  >
                    <CardHeader className="pb-1 space-y-2">
                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <CardTitle className="flex items-center gap-2 text-sm">
                            <BarChart3 className="h-4 w-4 text-blue-600 dark:text-cyan-400" />
                            {name}
                          </CardTitle>
                          <CardDescription className="text-[11px]">
                            {primaryMetricKey}:{' '}
                            {fmtMetric(metricVal as number | undefined)}
                          </CardDescription>
                        </div>
                        <div className="flex gap-1">
                          {(['true', 'pred', 'both'] as LineVisibility[]).map(
                            (mode) => (
                              <Button
                                key={mode}
                                type="button"
                                size="sm"
                                variant={vis === mode ? 'default' : 'outline'}
                                className="h-6 px-2 text-[10px]"
                                onClick={() =>
                                  setTabularVisibility((prev) => ({
                                    ...prev,
                                    [name]: mode,
                                  }))
                                }
                              >
                                {mode === 'both'
                                  ? 'Both'
                                  : mode === 'true'
                                    ? 'True'
                                    : 'Pred'}
                              </Button>
                            ),
                          )}
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="h-64 p-3">
                      <ResponsiveContainer width="100%" height="100%">
                        <RechartsLineChart
                          data={data}
                          margin={{
                            top: 10,
                            right: 10,
                            bottom: 10,
                            left: 0,
                          }}
                        >
                          <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
                          <XAxis
                            dataKey="idx"
                            tick={{ fontSize: 9 }}
                            minTickGap={12}
                          />
                          <YAxis tick={{ fontSize: 9 }} />
                          <Tooltip />
                          <Legend />
                          {vis !== 'pred' && (
                            <RechartsLine
                              type="monotone"
                              dataKey="true"
                              stroke="#2563eb"
                              strokeWidth={2}
                              dot={false}
                            />
                          )}
                          {vis !== 'true' && (
                            <RechartsLine
                              type="monotone"
                              dataKey="pred"
                              stroke="#f97316"
                              strokeWidth={2}
                              dot={false}
                            />
                          )}
                        </RechartsLineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </section>
        )}

        {seqPlot && Object.keys(seqPlot.models ?? {}).length > 0 && (
          <section className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="flex items-center gap-2 text-base font-semibold tracking-tight">
                  <LineChartIcon className="h-4 w-4 text-violet-600 dark:text-violet-400" />
                  Sequence models — True vs Predicted
                </h2>
                <p className="text-xs text-neutral-600 dark:text-neutral-400">
                  Validation target vs predictions from deep sequence models.
                </p>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {Object.entries(seqPlot.models).map(([name, info]) => {
                const data = buildTrueVsPredData(seqPlot.y_val, info.y_pred);
                if (!data.length) return null;

                const metricVal =
                  info.metrics?.[
                    primaryMetricKey as keyof typeof info.metrics
                  ];

                const vis = getSeqVis(name);

                return (
                  <Card
                    key={name}
                    className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80"
                  >
                    <CardHeader className="pb-1 space-y-2">
                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <CardTitle className="flex items-center gap-2 text-sm">
                            <Database className="h-4 w-4 text-violet-600 dark:text-violet-400" />
                            {name}
                          </CardTitle>
                          <CardDescription className="text-[11px]">
                            {primaryMetricKey}:{' '}
                            {fmtMetric(metricVal as number | undefined)}
                          </CardDescription>
                        </div>
                        <div className="flex gap-1">
                          {(['true', 'pred', 'both'] as LineVisibility[]).map(
                            (mode) => (
                              <Button
                                key={mode}
                                type="button"
                                size="sm"
                                variant={vis === mode ? 'default' : 'outline'}
                                className="h-6 px-2 text-[10px]"
                                onClick={() =>
                                  setSeqVisibility((prev) => ({
                                    ...prev,
                                    [name]: mode,
                                  }))
                                }
                              >
                                {mode === 'both'
                                  ? 'Both'
                                  : mode === 'true'
                                    ? 'True'
                                    : 'Pred'}
                              </Button>
                            ),
                          )}
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="h-64 p-3">
                      <ResponsiveContainer width="100%" height="100%">
                        <RechartsLineChart
                          data={data}
                          margin={{
                            top: 10,
                            right: 10,
                            bottom: 10,
                            left: 0,
                          }}
                        >
                          <CartesianGrid strokeDasharray="3 3" opacity={0.25} />
                          <XAxis
                            dataKey="idx"
                            tick={{ fontSize: 9 }}
                            minTickGap={12}
                          />
                          <YAxis tick={{ fontSize: 9 }} />
                          <Tooltip />
                          <Legend />
                          {vis !== 'pred' && (
                            <RechartsLine
                              type="monotone"
                              dataKey="true"
                              stroke="#8b5cf6"
                              strokeWidth={2}
                              dot={false}
                            />
                          )}
                          {vis !== 'true' && (
                            <RechartsLine
                              type="monotone"
                              dataKey="pred"
                              stroke="#22c55e"
                              strokeWidth={2}
                              dot={false}
                            />
                          )}
                        </RechartsLineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </section>
        )}

        {wfPlot && Object.keys(wfPlot).length > 0 && (
          <section className="space-y-3 pb-10">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="flex items-center gap-2 text-base font-semibold tracking-tight">
                  <BarChart3 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                  Walk-forward metrics
                </h2>
                <p className="text-xs text-neutral-600 dark:text-neutral-400">
                  {primaryMetricKey} across folds for each walk-forward
                  candidate.
                </p>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {Object.entries(wfPlot).map(([name, info]) => {
                const data = buildWfFoldData(info.folds, primaryMetricKey);
                if (!data.length) return null;

                return (
                  <Card
                    key={name}
                    className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80"
                  >
                    <CardHeader className="pb-2">
                      <CardTitle className="flex items-center gap-2 text-sm">
                        <Activity className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                        {name}
                      </CardTitle>
                      <CardDescription className="text-[11px]">
                        WF avg {primaryMetricKey}:{' '}
                        {fmtMetric(info.avg_metric ?? undefined)}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="h-52 p-3">
                      <ResponsiveContainer width="100%" height="100%">
                        <RechartsBarChart
                          data={data}
                          margin={{
                            top: 10,
                            right: 10,
                            bottom: 10,
                            left: 0,
                          }}
                        >
                          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                          <XAxis
                            dataKey="fold"
                            tick={{ fontSize: 9 }}
                            minTickGap={10}
                          />
                          <YAxis tick={{ fontSize: 9 }} />
                          <Tooltip />
                          <Bar dataKey="value" fill="#22c55e" />
                        </RechartsBarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
