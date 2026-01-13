'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';

import { getSupabaseBrowserClient } from '@kit/supabase/browser-client';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from '@kit/ui/card';
import { Button } from '@kit/ui/button';
import { Badge } from '@kit/ui/badge';
import { Table, TableBody, TableCell, TableRow } from '@kit/ui/table';
import { Checkbox } from '@kit/ui/checkbox';

import {
  ArrowLeft,
  Activity,
  Play,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Trophy,
  Check,
  Circle,
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
const supabase = getSupabaseBrowserClient();

type Json = Record<string, any>;

interface DatasetRow {
  id: string;
  original_filename: string;
  processed_parquet_path?: string | null;
  target_column: string | null;
  time_column: string | null;
  forecast_horizon: number | null;
  status?: string | null;
  created_at: string;
}

interface AutoMLStep {
  key: string;
  label: string;
  done: boolean;
}

interface MetricDict {
  MAE: number;
  RMSE: number;
  R2: number;
  MAPE: number;
  sMAPE: number;
}

interface AutoMLModelSummary {
  id: string | null;
  name: string;
  type: 'tabular' | 'seq' | string;
  status: string;
  primary_metric_name: string;
  primary_metric_value: number;
  metrics: MetricDict;
  wf_metrics: Array<Json>;
}

interface AutoMLRunSummary {
  dataset_id: string;
  primary_metric: string;
  steps: AutoMLStep[];
  models: AutoMLModelSummary[];
  best_model_id: string | null;
  best_model_name: string | null;
  best_model_type: string | null;
  best_avg_metric: number;
}

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

  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.headers || {}),
      Authorization: `Bearer ${session.access_token}`,
      'Content-Type': 'application/json',
    },
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} ${txt}`);
  }

  return res.json();
}

function DatasetAutoMLPage() {
  const params = useParams();
  const router = useRouter();
  const datasetId = params?.datasetId as string;

  const [dataset, setDataset] = useState<DatasetRow | null>(null);
  const [loadingDataset, setLoadingDataset] = useState(true);

  const [runStatus, setRunStatus] = useState<'idle' | 'running' | 'done' | 'failed'>(
    'idle',
  );
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<AutoMLRunSummary | null>(null);

  // AutoML config
  const [selectedModels, setSelectedModels] = useState<string[]>([
    'Random Forest',
    'XGBoost',
  ]);
  const [primaryMetric, setPrimaryMetric] = useState<string>('RMSE');
  const [testSize, setTestSize] = useState<number>(0.2);

  const [useClip, setUseClip] = useState<boolean>(false);
  const [clipThreshold, setClipThreshold] = useState<number | null>(null);

  const [topK, setTopK] = useState<number>(2);
  const [nSplits, setNSplits] = useState<number>(3);
  const [lookback, setLookback] = useState<number>(50);
  const [epochs, setEpochs] = useState<number>(10);
  const [batchSize, setBatchSize] = useState<number>(64);

  // -------- Load dataset ----------
  useEffect(() => {
    if (!datasetId) return;

    let mounted = true;
    (async () => {
      try {
        setLoadingDataset(true);
        setError(null);
        const row = await authorizedFetch<DatasetRow>(
          `/api/v1/datasets/${datasetId}`,
        );
        if (mounted) setDataset(row);
      } catch (e: any) {
        if (mounted) setError(e?.message ?? 'Failed to load dataset.');
      } finally {
        if (mounted) setLoadingDataset(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, [datasetId]);

  const canRun =
    !!dataset?.processed_parquet_path &&
    selectedModels.length > 0 &&
    runStatus !== 'running';

  // -------- Status badge ----------
  const statusBadge = (() => {
    if (runStatus === 'running') {
      return (
        <Badge
          variant="secondary"
          className="inline-flex items-center gap-1 text-xs"
        >
          <Activity className="h-3 w-3 animate-spin" />
          Running AutoML…
        </Badge>
      );
    }
    if (runStatus === 'done') {
      return (
        <Badge
          variant="default"
          className="inline-flex items-center gap-1 text-xs"
        >
          <CheckCircle2 className="h-3 w-3" />
          AutoML completed
        </Badge>
      );
    }
    if (runStatus === 'failed') {
      return (
        <Badge
          variant="outline"
          className="inline-flex items-center gap-1 text-xs"
        >
          <AlertTriangle className="h-3 w-3" />
          Failed
        </Badge>
      );
    }
    return (
      <Badge
        variant="outline"
        className="inline-flex items-center gap-1 text-xs"
      >
        <Clock className="h-3 w-3" />
        Idle
      </Badge>
    );
  })();

  // -------- Steps for UI ----------
  const steps: AutoMLStep[] = useMemo(() => {
    if (summary?.steps) return summary.steps;
    return [
      {
        key: 'prepare_data',
        label: 'Prepare data (X, y, sequences)',
        done: false,
      },
      {
        key: 'benchmark_tabular',
        label: 'Benchmark tabular models',
        done: false,
      },
      {
        key: 'benchmark_seq',
        label: 'Benchmark sequence models',
        done: false,
      },
      {
        key: 'rank_models',
        label: 'Rank models by primary metric',
        done: false,
      },
      {
        key: 'walk_forward',
        label: 'Walk-forward validation (top-K)',
        done: false,
      },
      {
        key: 'retrain_best',
        label: 'Retrain best model on full data',
        done: false,
      },
    ];
  }, [summary]);

  // -------- Leaderboard rows ----------
  const leaderboardRows = useMemo(() => {
    if (!summary) return [];
    const metricName = summary.primary_metric;
    const models = summary.models || [];

    return [...models].sort((a, b) => {
      // rank by primary_metric_value (WF avg)
      if (metricName === 'R2') {
        return (b.primary_metric_value ?? 0) - (a.primary_metric_value ?? 0);
      }
      return (
        (a.primary_metric_value ?? Number.POSITIVE_INFINITY) -
        (b.primary_metric_value ?? Number.POSITIVE_INFINITY)
      );
    });
  }, [summary]);

  // -------- Actions ----------
  const runAutoML = async () => {
    if (!datasetId || !canRun) return;

    setError(null);
    setRunStatus('running');
    setSummary(null);

    const body = {
      models_to_train: selectedModels,
      primary_metric: primaryMetric,
      test_size: testSize,
      use_clip: useClip,
      clip_threshold: useClip ? clipThreshold ?? 125 : null,
      top_k: topK,
      n_splits: nSplits,
      lookback,
      epochs,
      batch_size: batchSize,
      do_plots: false,
    };

    try {
      const res = await authorizedFetch<AutoMLRunSummary>(
        `/api/v1/datasets/${datasetId}/automl/run`,
        {
          method: 'POST',
          body: JSON.stringify(body),
        },
      );
      setSummary(res);
      setRunStatus('done');
    } catch (e: any) {
      setError(e?.message ?? 'AutoML run failed.');
      setRunStatus('failed');
    }
  };

  const handleViewPlots = (model: AutoMLModelSummary) => {
    if (!model.id) {
      console.warn('No model id, cannot navigate to plots page.');
      return;
    }
    router.push(`/home/models/${model.id}/plots`);
  };

  const handleUseModel = (model: AutoMLModelSummary) => {
    console.log('Use model as production:', model);
  };

  // -------- UI helpers ----------
  const fmtDate = (iso?: string | null) =>
    iso ? new Date(iso).toLocaleString() : '—';

  return (
    <div className="relative min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50">
      {/* background grid */}
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,#f5f5f5_1px,transparent_1px),linear-gradient(to_bottom,#f5f5f5_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-70 dark:bg-[linear-gradient(to_right,#262626_1px,transparent_1px),linear-gradient(to_bottom,#262626_1px,transparent_1px)]" />
      <div className="relative mx-auto max-w-6xl px-6 py-8 space-y-8">
        {/* Top bar */}
        <div className="flex items-center justify-between gap-4">
          <Link href="/home">
            <Button
              variant="outline"
              size="sm"
              className="flex items-center gap-2 rounded-full"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to dashboard
            </Button>
          </Link>

          {dataset && (
            <span className="text-xs text-neutral-500 dark:text-neutral-400">
              Dataset ID: {dataset.id}
            </span>
          )}
        </div>

        {/* Header */}
        <header className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-white/80 px-4 py-1.5 text-xs shadow-sm backdrop-blur dark:border-neutral-800 dark:bg-neutral-900/80">
              <Activity className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
              <span className="font-medium text-gray-900 dark:text-neutral-50">
                AutoML – model selection
              </span>
              <span className="text-gray-600 dark:text-neutral-400">
                Benchmark tabular & sequence models on your cleaned dataset.
              </span>
            </div>
            <h1 className="mt-3 bg-gradient-to-br from-gray-900 via-gray-800 to-gray-600 bg-clip-text text-3xl font-bold tracking-tight text-transparent md:text-4xl dark:from-neutral-50 dark:via-neutral-200 dark:to-neutral-400">
              AutoML run
            </h1>
            {dataset && (
              <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
                Dataset: <span className="font-mono">{dataset.original_filename}</span>{' '}
                · created {fmtDate(dataset.created_at)}
              </p>
            )}
          </div>
          <div className="flex items-center gap-3">{statusBadge}</div>
        </header>

        {/* Config + Steps + Run */}
        <section className="grid gap-6 md:grid-cols-[2fr_1.4fr]">
          {/* Config */}
          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardHeader>
              <CardTitle>Configuration</CardTitle>
              <CardDescription>
                Choose models and evaluation settings before running AutoML.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              {/* Models checklist */}
              <div>
                <div className="mb-1 text-xs font-semibold uppercase text-neutral-500">
                  Models to train
                </div>
                <div className="flex flex-wrap gap-3 text-xs">
                  {['Random Forest', 'XGBoost', 'LSTM', 'TCN', 'TFT'].map(
                    (name) => {
                      const checked = selectedModels.includes(name);
                      return (
                        <label
                          key={name}
                          className="inline-flex items-center gap-2 rounded-full border px-3 py-1 hover:bg-neutral-50 dark:hover:bg-neutral-900"
                        >
                          <Checkbox
                            checked={checked}
                            onCheckedChange={(v: any) => {
                              if (v) {
                                setSelectedModels((prev) =>
                                  prev.includes(name) ? prev : [...prev, name],
                                );
                              } else {
                                setSelectedModels((prev) =>
                                  prev.filter((m) => m !== name),
                                );
                              }
                            }}
                          />
                          <span>{name}</span>
                        </label>
                      );
                    },
                  )}
                </div>
              </div>

              {/* Core numeric params */}
              <div className="grid gap-3 md:grid-cols-3 text-xs">
                <div>
                  <div className="mb-1 text-neutral-500">Primary metric</div>
                  <select
                    className="w-full rounded-md border border-neutral-200 bg-white px-2 py-1 text-xs dark:border-neutral-700 dark:bg-neutral-900"
                    value={primaryMetric}
                    onChange={(e) => setPrimaryMetric(e.target.value)}
                  >
                    <option value="RMSE">RMSE</option>
                    <option value="MAE">MAE</option>
                    <option value="R2">R²</option>
                    <option value="MAPE">MAPE</option>
                    <option value="sMAPE">sMAPE</option>
                  </select>
                </div>
                <div>
                  <div className="mb-1 text-neutral-500">Test size</div>
                  <input
                    type="number"
                    step="0.05"
                    min={0.05}
                    max={0.5}
                    value={testSize}
                    onChange={(e) =>
                      setTestSize(parseFloat(e.target.value || '0.2'))
                    }
                    className="w-full rounded-md border border-neutral-200 bg-white px-2 py-1 text-xs dark:border-neutral-700 dark:bg-neutral-900"
                  />
                </div>
                <div>
                  <div className="mb-1 text-neutral-500">Top-K (WF)</div>
                  <input
                    type="number"
                    min={1}
                    max={5}
                    value={topK}
                    onChange={(e) =>
                      setTopK(parseInt(e.target.value || '2', 10))
                    }
                    className="w-full rounded-md border border-neutral-200 bg-white px-2 py-1 text-xs dark:border-neutral-700 dark:bg-neutral-900"
                  />
                </div>
              </div>

              <div className="grid gap-3 md:grid-cols-4 text-xs">
                <div>
                  <div className="mb-1 text-neutral-500">WF folds</div>
                  <input
                    type="number"
                    min={2}
                    max={10}
                    value={nSplits}
                    onChange={(e) =>
                      setNSplits(parseInt(e.target.value || '3', 10))
                    }
                    className="w-full rounded-md border border-neutral-200 bg-white px-2 py-1 text-xs dark:border-neutral-700 dark:bg-neutral-900"
                  />
                </div>
                <div>
                  <div className="mb-1 text-neutral-500">Lookback (seq)</div>
                  <input
                    type="number"
                    min={5}
                    max={200}
                    value={lookback}
                    onChange={(e) =>
                      setLookback(parseInt(e.target.value || '50', 10))
                    }
                    className="w-full rounded-md border border-neutral-200 bg-white px-2 py-1 text-xs dark:border-neutral-700 dark:bg-neutral-900"
                  />
                </div>
                <div>
                  <div className="mb-1 text-neutral-500">Epochs</div>
                  <input
                    type="number"
                    min={1}
                    max={200}
                    value={epochs}
                    onChange={(e) =>
                      setEpochs(parseInt(e.target.value || '10', 10))
                    }
                    className="w-full rounded-md border border-neutral-200 bg-white px-2 py-1 text-xs dark:border-neutral-700 dark:bg-neutral-900"
                  />
                </div>
                <div>
                  <div className="mb-1 text-neutral-500">Batch size</div>
                  <input
                    type="number"
                    min={8}
                    max={512}
                    value={batchSize}
                    onChange={(e) =>
                      setBatchSize(parseInt(e.target.value || '64', 10))
                    }
                    className="w-full rounded-md border border-neutral-200 bg-white px-2 py-1 text-xs dark:border-neutral-700 dark:bg-neutral-900"
                  />
                </div>
              </div>

              {/* Clip */}
              <div className="mt-2 flex items-center gap-3 text-xs">
                <Checkbox
                  checked={useClip}
                  onCheckedChange={(v: any) => setUseClip(Boolean(v))}
                />
                <span>
                  Clip target at{' '}
                  <input
                    type="number"
                    disabled={!useClip}
                    value={clipThreshold ?? ''}
                    placeholder="125"
                    onChange={(e) =>
                      setClipThreshold(
                        e.target.value === ''
                          ? null
                          : parseFloat(e.target.value),
                      )
                    }
                    className="ml-1 inline-block w-20 rounded-md border border-neutral-200 bg-white px-1 py-0.5 text-xs dark:border-neutral-700 dark:bg-neutral-900"
                  />{' '}
                  (if empty, defaults to 125 when enabled)
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Steps + Run */}
          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Activity className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                AutoML steps
              </CardTitle>
              <CardDescription>
                Pipeline stages. Once a step is completed, it is checked.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <ul className="space-y-2 text-xs">
                {steps.map((step) => {
                  const icon =
                    step.done && runStatus !== 'running' ? (
                      <Check className="h-3 w-3 text-emerald-500" />
                    ) : runStatus === 'running' ? (
                      <Activity className="h-3 w-3 animate-spin text-neutral-400" />
                    ) : (
                      <Circle className="h-3 w-3 text-neutral-400" />
                    );

                  return (
                    <li
                      key={step.key}
                      className="flex items-center gap-2 text-neutral-700 dark:text-neutral-300"
                    >
                      {icon}
                      <span>{step.label}</span>
                    </li>
                  );
                })}
              </ul>

              <Button
                className="mt-4 w-full rounded-full"
                size="sm"
                onClick={runAutoML}
                disabled={!canRun}
              >
                <Play className="mr-2 h-4 w-4" />
                {runStatus === 'running' ? 'Running AutoML…' : 'Run AutoML'}
              </Button>

              {!dataset?.processed_parquet_path && (
                <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">
                  Dataset is not preprocessed yet. Please run preprocessing first.
                </p>
              )}
            </CardContent>
          </Card>
        </section>

        {/* Leaderboard */}
        {summary && (
          <section className="space-y-3">
            <div className="flex items-center justify-between gap-2">
              <h2 className="flex items-center gap-2 text-xl font-semibold">
                <Trophy className="h-5 w-5 text-amber-500" />
                Model leaderboard
              </h2>
              <span className="text-xs text-neutral-500 dark:text-neutral-400">
                Ranked by {summary.primary_metric} (walk-forward average)
              </span>
            </div>
            <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
              <CardContent className="overflow-x-auto py-4 text-sm">
                {leaderboardRows.length === 0 ? (
                  <p className="text-xs text-neutral-500">
                    No models were benchmarked.
                  </p>
                ) : (
                  <Table>
                    <TableBody>
                      <TableRow>
                        <TableCell className="text-xs font-semibold uppercase">
                          #
                        </TableCell>
                        <TableCell className="text-xs font-semibold uppercase">
                          Model
                        </TableCell>
                        <TableCell className="text-xs font-semibold uppercase">
                          Type
                        </TableCell>
                        <TableCell className="text-xs font-semibold uppercase">
                          {summary.primary_metric} (WF avg)
                        </TableCell>
                        <TableCell className="text-xs font-semibold uppercase">
                          RMSE
                        </TableCell>
                        <TableCell className="text-xs font-semibold uppercase">
                          MAE
                        </TableCell>
                        <TableCell className="text-xs font-semibold uppercase">
                          R²
                        </TableCell>
                        <TableCell />
                      </TableRow>

                      {leaderboardRows.map((m, idx) => (
                        <TableRow key={m.name}>
                          <TableCell className="text-xs">
                            {idx + 1}
                            {idx === 0 && (
                              <span className="ml-1 text-[10px] text-amber-500">
                                (best)
                              </span>
                            )}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {m.name}
                          </TableCell>
                          <TableCell className="text-xs">
                            {m.type === 'seq' ? 'Sequence' : 'Tabular'}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {m.primary_metric_value != null
                              ? m.primary_metric_value.toFixed(3)
                              : '—'}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {m.metrics.RMSE.toFixed(3)}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {m.metrics.MAE.toFixed(3)}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {m.metrics.R2.toFixed(3)}
                          </TableCell>
                          <TableCell className="flex gap-2 text-xs">
                            <Button
                              size="sm"
                              variant="outline"
                              className="rounded-full"
                              onClick={() => handleViewPlots(m)}
                              disabled={!m.id}
                            >
                              View plots
                            </Button>
                            <Button
                              size="sm"
                              className="rounded-full"
                              variant={idx === 0 ? 'default' : 'secondary'}
                              onClick={() => handleUseModel(m)}
                              disabled={!m.id}
                            >
                              Use model
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </section>
        )}

        {error && (
          <div className="flex items-start gap-2 rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-900/20 dark:text-red-200">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default DatasetAutoMLPage;
export { DatasetAutoMLPage };
