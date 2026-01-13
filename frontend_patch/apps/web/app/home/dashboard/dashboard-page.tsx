'use client';

import { useEffect, useMemo, useState } from 'react';
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@kit/ui/table';
import { Badge } from '@kit/ui/badge';
import {
  FileText,
  Upload,
  Play,
  CheckCircle,
  Database,
  Sparkles,
  LineChart,
  Target,
  Activity,
  Trash2,
  Info,
} from 'lucide-react';

import { getSupabaseBrowserClient } from '@kit/supabase/browser-client';

// ---------- env & supabase client (browser) ----------
const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
const supabase = getSupabaseBrowserClient();

// ---------- types ----------
type DatasetRow = {
  id: string;
  original_filename: string;
  rows_count?: number | null;
  columns_count?: number | null;
  created_at: string;
  status: string;
  ingestion_info?: {
    shape?: [number, number];
  } | null;
};

type ModelRow = {
  id: string;
  name: string;
  algorithm: string;
  status: string;
  created_at: string;
  primary_metric?: string | null;
  dataset_id?: string | null;
  dataset_name?: string | null;
};

// ---------- small utils ----------
const fmtNum = (n?: number | null) =>
  typeof n === 'number' ? n.toLocaleString() : '—';

const fmtDate = (iso?: string) =>
  iso ? new Date(iso).toLocaleString() : '—';

const metricFromTest = (
  test?: Record<string, unknown> | null
): string | null => {
  if (!test) return null;
  const rmse = test['RMSE'];
  if (typeof rmse === 'number') return `RMSE: ${rmse.toFixed(3)}`;
  const mae = test['MAE'];
  if (typeof mae === 'number') return `MAE: ${mae.toFixed(3)}`;
  const r2 = test['R2'];
  if (typeof r2 === 'number') return `R2: ${r2.toFixed(3)}`;
  const mape = test['MAPE'];
  if (typeof mape === 'number') return `MAPE: ${mape.toFixed(2)}%`;
  return null;
};

// ---------- fetch helpers ----------
async function authorizedFetch<T = any>(
  path: string,
  init?: RequestInit
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

  if (res.status === 204) {
    return {} as T;
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
        rows_count:
          r.rows_count ?? r.ingestion_info?.shape?.[0] ?? null,
        columns_count:
          r.columns_count ?? r.ingestion_info?.shape?.[1] ?? null,
        created_at: r.created_at,
        status: r.status,
        ingestion_info: r.ingestion_info ?? null,
      }) as DatasetRow
  );
}

async function loadModels(): Promise<ModelRow[]> {
  const rows = await authorizedFetch<any[]>('/api/v1/models');

  return (rows || []).map(
    (r) =>
      ({
        id: r.id,
        name: r.model_name ?? 'Model',
        algorithm: r.algorithm ?? '—',
        status: r.status ?? 'ready',
        created_at: r.created_at,
        primary_metric:
          metricFromTest(r.test_metrics) ??
          (typeof r.composite_score === 'number'
            ? `Score: ${r.composite_score.toFixed(3)}`
            : null),
        dataset_id: r.dataset_id ?? null,
        dataset_name: null, 
      }) as ModelRow
  );
}

async function deleteDatasetApi(id: string): Promise<void> {
  await authorizedFetch(`/api/v1/datasets/${id}`, {
    method: 'DELETE',
  });
}

async function deleteModelApi(id: string): Promise<void> {
  await authorizedFetch(`/api/v1/models/${id}`, {
    method: 'DELETE',
  });
}

// ---------- component ----------
export default function DashboardPage() {
  const router = useRouter();

  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [models, setModels] = useState<ModelRow[]>([]);
  const [loading, setLoading] = useState({ ds: true, mdl: true });
  const [error, setError] = useState<string | null>(null);

  const systemHealth = useMemo(
    () => [
      {
        label: 'Datasets',
        value: fmtNum(datasets.length),
        status: 'healthy',
      },
      {
        label: 'Models',
        value: fmtNum(models.length),
        status: 'healthy',
      },
      { label: 'Active Alerts', value: '0', status: 'healthy' },
      { label: 'Avg Latency', value: '—', status: 'healthy' },
    ],
    [datasets.length, models.length]
  );

  const modelsWithDatasetNames = useMemo(() => {
    const map = new Map<string, string>();
    for (const d of datasets) {
      map.set(d.id, d.original_filename);
    }
    return models.map((m) => ({
      ...m,
      dataset_name:
        m.dataset_name ??
        (m.dataset_id ? map.get(m.dataset_id) ?? null : null),
    }));
  }, [datasets, models]);

  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        setLoading((s) => ({ ...s, ds: true }));
        const ds = await loadDatasets();
        if (mounted) setDatasets(ds);
      } catch (e: any) {
        if (mounted)
          setError(
            e?.message ??
              'Failed to load datasets (check authentication).'
          );
      } finally {
        if (mounted)
          setLoading((s) => ({ ...s, ds: false }));
      }
    })();

    (async () => {
      try {
        setLoading((s) => ({ ...s, mdl: true }));
        const md = await loadModels();
        if (mounted) setModels(md);
      } catch (e: any) {
        if (mounted)
          setError((prev) => prev ?? e?.message ?? 'Failed to load models');
      } finally {
        if (mounted)
          setLoading((s) => ({ ...s, mdl: false }));
      }
    })();

    return () => {
      mounted = false;
    };
  }, []);

  const recentDatasets = datasets;
  const recentModels = modelsWithDatasetNames;

  // ---------- handlers delete ----------
  async function handleDeleteDataset(id: string) {
    const ok = window.confirm(
      'Are you sure you want to delete this dataset? All related models will also be removed.'
    );
    if (!ok) return;

    try {
      await deleteDatasetApi(id);
      setDatasets((prev) => prev.filter((d) => d.id !== id));
      setModels((prev) => prev.filter((m) => m.dataset_id !== id));
    } catch (e: any) {
      setError(e?.message ?? 'Failed to delete dataset.');
    }
  }

  async function handleDeleteModel(id: string) {
    const ok = window.confirm(
      'Are you sure you want to delete this model from the database?'
    );
    if (!ok) return;

    try {
      await deleteModelApi(id);
      setModels((prev) => prev.filter((m) => m.id !== id));
    } catch (e: any) {
      setError(e?.message ?? 'Failed to delete model.');
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50">
      {/* Background grid */}
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,#000_70%,transparent_110%)] dark:bg-[linear-gradient(to_right,#262626_1px,transparent_1px),linear-gradient(to_bottom,#262626_1px,transparent_1px)]" />

      <div className="relative mx-auto max-w-7xl px-6 py-10 space-y-10">
        {/* Header / Hero */}
        <header className="space-y-6">
          <div className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-white/80 px-4 py-2 text-sm shadow-sm backdrop-blur dark:border-neutral-800 dark:bg-neutral-900/80">
            <Sparkles className="h-3 w-3 text-blue-600 dark:text-cyan-400" />
            <span className="font-medium text-gray-900 dark:text-neutral-50">
              Data &amp; AI
            </span>
            <span className="text-gray-600 dark:text-neutral-400">
              Your predictive control center
            </span>
          </div>

          <div className="flex flex-col justify-between gap-6 md:flex-row md:items-end">
            <div>
              <h1 className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-600 bg-clip-text text-4xl font-bold tracking-tight text-transparent md:text-5xl dark:from-neutral-50 dark:via-neutral-200 dark:to-neutral-400">
                Predictive Platform Dashboard
              </h1>
              <p className="mt-3 max-w-2xl text-neutral-600 dark:text-neutral-400">
                Monitor your data pipeline, AutoML runs, deployed models, and
                forecasts — all in one place.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Link href="/home/datasets/upload">
                <Button className="rounded-full">Upload Dataset</Button>
              </Link>
              <Link href="/home/automl">
                <Button variant="outline" className="rounded-full">
                  Start AutoML
                </Button>
              </Link>
            </div>
          </div>
        </header>

        {/* Action Cards */}
        <section className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <Link href="/home/datasets/upload">
            <Card className="group cursor-pointer border-gray-200 bg-gradient-to-br from-white to-gray-50/80 transition-all hover:-translate-y-1 hover:border-blue-500 hover:shadow-lg hover:shadow-blue-500/10 dark:border-neutral-800 dark:from-neutral-900 dark:to-neutral-950">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-green-100 group-hover:bg-green-200 dark:bg-green-900/40 dark:group-hover:bg-green-800/60">
                    <Upload className="h-6 w-6 text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <CardTitle className="text-base">
                      Upload Dataset
                    </CardTitle>
                    <CardDescription>
                      Import your data to begin analysis.
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
            </Card>
          </Link>

          <Link href="/home/automl">
            <Card className="group cursor-pointer border-gray-200 bg-gradient-to-br from-white to-gray-50/80 transition-all hover:-translate-y-1 hover:border-cyan-500 hover:shadow-lg hover:shadow-cyan-500/10 dark:border-neutral-800 dark:from-neutral-900 dark:to-neutral-950">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-blue-100 group-hover:bg-blue-200 dark:bg-blue-900/40 dark:group-hover:bg-blue-800/60">
                    <Play className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <CardTitle className="text-base">Start AutoML</CardTitle>
                    <CardDescription>
                      Launch model search and tuning.
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
            </Card>
          </Link>

          <Link href="/home/predictions">
            <Card className="group cursor-pointer border-gray-200 bg-gradient-to-br from-white to-gray-50/80 transition-all hover:-translate-y-1 hover:border-violet-500 hover:shadow-lg hover:shadow-violet-500/10 dark:border-neutral-800 dark:from-neutral-900 dark:to-neutral-950">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-violet-100 group-hover:bg-violet-200 dark:bg-violet-900/40 dark:group-hover:bg-violet-800/60">
                    <Target className="h-6 w-6 text-violet-600 dark:text-violet-400" />
                  </div>
                  <div>
                    <CardTitle className="text-base">
                      Real-Time Forecasts
                    </CardTitle>
                    <CardDescription>
                      View live forecasts &amp; make on-demand predictions.
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
            </Card>
          </Link>

          <Link href="/home/monitoring">
            <Card className="group cursor-pointer border-gray-200 bg-gradient-to-br from-white to-gray-50/80 transition-all hover:-translate-y-1 hover:border-emerald-500 hover:shadow-lg hover:shadow-emerald-500/10 dark:border-neutral-800 dark:from-neutral-900 dark:to-neutral-950">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-100 group-hover:bg-emerald-200 dark:bg-emerald-900/40 dark:group-hover:bg-emerald-800/60">
                    <Activity className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
                  </div>
                  <div>
                    <CardTitle className="text-base">
                      Monitoring &amp; Drift
                    </CardTitle>
                    <CardDescription>
                      Track health, drift, and performance over time.
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
            </Card>
          </Link>
        </section>

        {/* System Health */}
        <section className="space-y-4">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-xl font-semibold tracking-tight">
              System Health
            </h2>
            <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300">
              Live overview
            </span>
          </div>

          <div className="grid gap-4 md:grid-cols-4">
            {systemHealth.map((metric) => (
              <Card
                key={metric.label}
                className="border-gray-200 bg-gradient-to-br from-white to-gray-50/80 dark:border-neutral-800 dark:from-neutral-900 dark:to-neutral-950"
              >
                <CardHeader className="pb-3">
                  <CardDescription>{metric.label}</CardDescription>
                  <CardTitle className="mt-1 flex items-center gap-2 text-2xl">
                    {metric.value}
                    <CheckCircle className="h-5 w-5 text-emerald-500" />
                  </CardTitle>
                </CardHeader>
              </Card>
            ))}
          </div>
        </section>

        {/* Recent Datasets */}
        <section className="space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold tracking-tight">
                Recent Datasets
              </h2>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                Track the latest data ingested into DataForge.
              </p>
            </div>
            <Link href="/home/datasets/upload">
              <Button
                variant="outline"
                size="sm"
                className="rounded-full"
              >
                Upload
              </Button>
            </Link>
          </div>

          {loading.ds ? (
            <Card className="border-gray-200 dark:border-neutral-800">
              <CardContent className="p-6 text-sm text-neutral-500">
                Loading datasets…
              </CardContent>
            </Card>
          ) : recentDatasets.length > 0 ? (
            <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Filename</TableHead>
                      <TableHead>Rows</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">
                        Actions
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {recentDatasets.map((ds) => (
                      <TableRow key={ds.id}>
                        <TableCell className="font-medium">
                          <div className="flex items-center gap-2">
                            <FileText className="h-4 w-4 text-neutral-500" />
                            {ds.original_filename}
                          </div>
                        </TableCell>
                        <TableCell>
                          {fmtNum(
                            ds.rows_count ??
                              ds.ingestion_info?.shape?.[0] ??
                              0
                          )}
                        </TableCell>
                        <TableCell>{fmtDate(ds.created_at)}</TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              ds.status === 'ready'
                                ? 'default'
                                : 'secondary'
                            }
                          >
                            {ds.status}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Link
                              href={`/home/datasets/${ds.id}/eda`}
                            >
                              <Button variant="ghost" size="sm">
                                Open
                              </Button>
                            </Link>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="text-red-600 hover:text-red-700"
                              onClick={() => handleDeleteDataset(ds.id)}
                            >
                              <Trash2 className="mr-1 h-4 w-4" />
                              Delete
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : (
            <Card className="border-dashed border-gray-200 bg-white/70 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/60">
              <CardContent className="flex flex-col items-center justify-center py-12">
                <Database className="mb-4 h-12 w-12 text-neutral-400" />
                <p className="text-neutral-600 dark:text-neutral-400">
                  No datasets yet — upload your first one to begin.
                </p>
              </CardContent>
            </Card>
          )}
        </section>

        {/* Recent Models */}
        <section className="space-y-4 pb-8">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-semibold tracking-tight">
                Recent Models
              </h2>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                Review the latest AutoML runs and trained models.
              </p>
            </div>
            <Link href="/home/automl">
              <Button
                variant="outline"
                size="sm"
                className="rounded-full"
              >
                View AutoML
              </Button>
            </Link>
          </div>

          {loading.mdl ? (
            <Card className="border-gray-200 dark:border-neutral-800">
              <CardContent className="p-6 text-sm text-neutral-500">
                Loading models…
              </CardContent>
            </Card>
          ) : recentModels.length > 0 ? (
            <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Model</TableHead>
                      <TableHead>Algorithm</TableHead>
                      <TableHead>Dataset</TableHead>
                      <TableHead>Metric</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead className="text-right">
                        Actions
                      </TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {recentModels.map((model) => (
                      <TableRow key={model.id}>
                        <TableCell className="font-medium">
                          {model.name}
                        </TableCell>
                        <TableCell>{model.algorithm ?? '—'}</TableCell>
                        <TableCell>
                          {model.dataset_name ??
                            model.dataset_id ??
                            '—'}
                        </TableCell>
                        <TableCell>
                          {model.primary_metric ?? '—'}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              model.status === 'ready' ||
                              model.status === 'deployed'
                                ? 'default'
                                : 'secondary'
                            }
                          >
                            {model.status}
                          </Badge>
                        </TableCell>
                        <TableCell>{fmtDate(model.created_at)}</TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            <Link href={`/home/models/${model.id}`}>
                              <Button
                                variant="ghost"
                                size="sm"
                              >
                                <Info className="mr-1 h-4 w-4" />
                                View
                              </Button>
                            </Link>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="text-red-600 hover:text-red-700"
                              onClick={() => handleDeleteModel(model.id)}
                            >
                              <Trash2 className="mr-1 h-4 w-4" />
                              Delete
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : (
            <Card className="border-dashed border-gray-200 bg-white/70 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/60">
              <CardContent className="flex flex-col items-center justify-center py-12">
                <LineChart className="mb-4 h-12 w-12 text-neutral-400" />
                <p className="text-neutral-600 dark:text-neutral-400">
                  No models trained yet — run AutoML to create your first
                  one.
                </p>
              </CardContent>
            </Card>
          )}
        </section>

        {/* Error banner */}
        {error && (
          <div className="rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-900/20 dark:text-red-200">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

export { DashboardPage };
