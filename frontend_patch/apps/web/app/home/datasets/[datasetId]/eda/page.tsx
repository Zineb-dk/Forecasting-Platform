'use client';

import { useEffect, useState, useMemo } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';

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
  ArrowLeft,
  BarChart3,
  FileText,
  Info,
} from 'lucide-react';

import { getSupabaseBrowserClient } from '@kit/supabase/browser-client';

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
} from 'recharts';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
const supabase = getSupabaseBrowserClient();


type IngestionInfo = {
  shape?: [number, number];
  columns?: string[];
  dtypes?: Record<string, string>;
  numeric_columns?: string[];
  categorical_columns?: string[];
  datetime_columns?: string[];
  missing_values?: Record<string, number>;
} & Record<string, any>;

type NumericHistogram = {
  bin_edges: number[];
  counts: number[];
};

type TargetSeries =
  | {
      kind: 'time';
      time_col: string;
      target_col: string;
      time: string[];
      values: number[];
    }
  | {
      kind: 'index';
      target_col: string;
      index: number[];
      values: number[];
    };

type EdaResponse = {
  dataset: any;
  ingestion_info: IngestionInfo;
  preview: any[];
  numeric_histograms: Record<string, NumericHistogram>;
  target_series: TargetSeries | null;
};

type LoadingState = 'idle' | 'loading' | 'error';


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

// ---------- Page component ----------

export default function DatasetEdaPage() {
  const params = useParams();
  const router = useRouter();
  const datasetId = (params as { datasetId?: string })?.datasetId;

  const [state, setState] = useState<LoadingState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [eda, setEda] = useState<EdaResponse | null>(null);

  useEffect(() => {
    if (!datasetId) return;

    (async () => {
      try {
        setState('loading');
        setError(null);
        const data = await authorizedFetch<EdaResponse>(
          `/api/v1/datasets/${datasetId}/eda`,
        );
        setEda(data);
        setState('idle');
      } catch (err: any) {
        setError(err?.message ?? 'Failed to load EDA data.');
        setState('error');
      }
    })();
  }, [datasetId]);

  const columns = useMemo(
    () => eda?.ingestion_info?.columns ?? [],
    [eda?.ingestion_info],
  );

  const dtypes = useMemo(
    () => eda?.ingestion_info?.dtypes ?? {},
    [eda?.ingestion_info],
  );

  const previewRows = useMemo(
    () => eda?.preview ?? [],
    [eda?.preview],
  );

  const numericCols = useMemo(
    () => Object.keys(eda?.numeric_histograms ?? {}),
    [eda?.numeric_histograms],
  );

  const shape = eda?.ingestion_info?.shape;
  const nRows = shape?.[0] ?? eda?.dataset?.rows_count ?? null;
  const nCols = shape?.[1] ?? eda?.dataset?.columns_count ?? null;

  const targetSeries = eda?.target_series ?? null;

  return (
    <div className="relative min-h-screen overflow-auto bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50">
      {/* Background grid */}
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,#000_70%,transparent_110%)] dark:bg-[linear-gradient(to_right,#262626_1px,transparent_1px),linear-gradient(to_bottom,#262626_1px,transparent_1px)]" />

      <div className="relative mx-auto max-w-7xl px-6 py-8 space-y-8">
        {/* Top bar */}
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

          <div className="flex flex-wrap items-center gap-3">
            {datasetId && (
              <Link href={`/home/datasets/${datasetId}/processing`}>
                <Button className="rounded-full">Preprocess Data</Button>
              </Link>
            )}
          </div>
        </div>

        {/* Header */}
        <header className="space-y-3">
          <div className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-white/80 px-4 py-1.5 text-xs shadow-sm backdrop-blur dark:border-neutral-800 dark:bg-neutral-900/80">
            <BarChart3 className="h-3 w-3 text-blue-600 dark:text-cyan-400" />
            <span className="font-medium text-gray-900 dark:text-neutral-50">
              Dataset EDA
            </span>
            <span className="text-gray-600 dark:text-neutral-400">
              Explore your raw data before preprocessing
            </span>
          </div>

          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
            <div>
              <h1 className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-600 bg-clip-text text-3xl font-bold tracking-tight text-transparent md:text-4xl dark:from-neutral-50 dark:via-neutral-200 dark:to-neutral-400">
                {eda?.dataset?.original_filename ?? 'Dataset'}
              </h1>
              <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
                Inspect the structure, distributions, and target behavior over
                time before launching any preprocessing or AutoML pipeline.
              </p>
            </div>

            {nRows !== null && nCols !== null && (
              <div className="flex gap-4 text-sm">
                <div className="rounded-xl bg-white/80 px-4 py-2 shadow-sm ring-1 ring-gray-200 dark:bg-neutral-900/80 dark:ring-neutral-800">
                  <div className="text-xs text-neutral-500">Rows</div>
                  <div className="text-lg font-semibold">
                    {nRows.toLocaleString()}
                  </div>
                </div>
                <div className="rounded-xl bg-white/80 px-4 py-2 shadow-sm ring-1 ring-gray-200 dark:bg-neutral-900/80 dark:ring-neutral-800">
                  <div className="text-xs text-neutral-500">Columns</div>
                  <div className="text-lg font-semibold">
                    {nCols.toLocaleString()}
                  </div>
                </div>
              </div>
            )}
          </div>
        </header>

        {/* Loading / Error */}
        {state === 'loading' && (
          <Card className="border-gray-200 dark:border-neutral-800">
            <CardContent className="p-6 text-sm text-neutral-600 dark:text-neutral-400">
              Loading EDA data…
            </CardContent>
          </Card>
        )}

        {state === 'error' && error && (
          <Card className="border-red-300 bg-red-50/80 dark:border-red-900/60 dark:bg-red-900/20">
            <CardContent className="flex items-center gap-2 p-4 text-sm text-red-700 dark:text-red-200">
              <Info className="h-4 w-4" />
              <span>{error}</span>
            </CardContent>
          </Card>
        )}

        {/* Main content when EDA is loaded */}
        {eda && (
          <>
            {/* Dataset info + columns */}
            <section className="grid gap-6 md:grid-cols-3">
              <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80 md:col-span-1">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <Info className="h-4 w-4 text-blue-600 dark:text-cyan-400" />
                    Dataset details
                  </CardTitle>
                  <CardDescription>
                    Basic information about the uploaded dataset.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between gap-2">
                    <span className="text-neutral-500">Filename</span>
                    <span className="font-medium truncate max-w-[180px] text-right">
                      {eda.dataset.original_filename}
                    </span>
                  </div>
                  <div className="flex justify-between gap-2">
                    <span className="text-neutral-500">Status</span>
                    <Badge variant="secondary" className="capitalize">
                      {eda.dataset.status}
                    </Badge>
                  </div>
                  <div className="flex justify-between gap-2">
                    <span className="text-neutral-500">Target column</span>
                    <span className="font-medium">
                      {eda.dataset.target_column ?? '—'}
                    </span>
                  </div>
                  <div className="flex justify-between gap-2">
                    <span className="text-neutral-500">
                      Forecast horizon
                    </span>
                    <span className="font-medium">
                      {eda.dataset.forecast_horizon ?? '—'}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80 md:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <FileText className="h-4 w-4 text-blue-600 dark:text-cyan-400" />
                    Columns & dtypes
                  </CardTitle>
                  <CardDescription>
                    Overview of all columns and their detected types.
                  </CardDescription>
                </CardHeader>
                <CardContent className="max-h-64 overflow-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Column</TableHead>
                        <TableHead>Type</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {columns.map((col) => (
                        <TableRow key={col}>
                          <TableCell className="font-medium">{col}</TableCell>
                          <TableCell className="text-xs text-neutral-600 dark:text-neutral-400">
                            {dtypes[col] ?? '—'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </section>

            {/* Preview rows */}
            <section className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold tracking-tight">
                    Sample rows
                  </h2>
                  <p className="text-xs text-neutral-600 dark:text-neutral-400">
                    First 50 rows from the standardized dataset.
                  </p>
                </div>
              </div>

              <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
                <CardContent className="max-h-80 overflow-auto p-0">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {columns.map((col) => (
                          <TableHead key={col}>{col}</TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {previewRows.map((row, idx) => (
                        <TableRow key={idx}>
                          {columns.map((col) => (
                            <TableCell key={col}>
                              {String(row[col] ?? '')}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            </section>

            {/* Target over time */}
            {targetSeries && (
              <section className="space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h2 className="text-base font-semibold tracking-tight">
                      Target behavior
                    </h2>
                    <p className="text-xs text-neutral-600 dark:text-neutral-400">
                      Visualize how your target evolves over time (or index).
                    </p>
                  </div>
                </div>

                <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
                  <CardContent className="h-72 p-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <RechartsLineChart
                        data={buildTargetChartData(targetSeries)}
                        margin={{ top: 10, right: 20, bottom: 10, left: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                        <XAxis
                          dataKey="x"
                          tick={{ fontSize: 10 }}
                          minTickGap={16}
                        />
                        <YAxis tick={{ fontSize: 10 }} />
                        <Tooltip />
                        <RechartsLine
                          type="monotone"
                          dataKey="y"
                          stroke="currentColor"
                          strokeWidth={2}
                          dot={false}
                        />
                      </RechartsLineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </section>
            )}

            {/* Numeric histograms */}
            {numericCols.length > 0 && (
              <section className="space-y-3 pb-10">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h2 className="text-base font-semibold tracking-tight">
                      Numeric distributions
                    </h2>
                    <p className="text-xs text-neutral-600 dark:text-neutral-400">
                      Histograms for each numeric column in the dataset.
                    </p>
                  </div>
                </div>

                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                  {numericCols.map((col) => {
                    const hist = eda.numeric_histograms[col];
                    const chartData = buildHistogramChartData(hist);

                    if (!chartData.length) return null;

                    return (
                      <Card
                        key={col}
                        className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80"
                      >
                        <CardHeader className="pb-2">
                          <CardTitle className="flex items-center gap-2 text-sm">
                            <BarChart3 className="h-4 w-4 text-blue-600 dark:text-cyan-400" />
                            {col}
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="h-40 p-3">
                          <ResponsiveContainer width="100%" height="100%">
                            <RechartsBarChart data={chartData}>
                              <CartesianGrid
                                strokeDasharray="3 3"
                                opacity={0.15}
                              />
                              <XAxis
                                dataKey="x"
                                tick={{ fontSize: 9 }}
                                minTickGap={12}
                              />
                              <YAxis tick={{ fontSize: 9 }} />
                              <Tooltip />
                              <Bar dataKey="count" />
                            </RechartsBarChart>
                          </ResponsiveContainer>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ---------- Chart helpers ----------

function buildTargetChartData(target?: TargetSeries | null) {
  if (!target) return [];

  const data: { x: string; y: number }[] = [];

  if (target.kind === 'time') {
    const len = Math.min(target.time.length, target.values.length);
    for (let i = 0; i < len; i++) {
      const v = target.values[i];
      const t = target.time[i];
      if (v == null || t == null) continue;

      data.push({
        x: t,         
        y: v,
      });
    }
  } else {
    const len = Math.min(target.index.length, target.values.length);
    for (let i = 0; i < len; i++) {
      const v = target.values[i];
      const idx = target.index[i];
      if (v == null || idx == null) continue;

      data.push({
        x: String(idx),  
        y: v,
      });
    }
  }

  return data;
}


function buildHistogramChartData(hist?: NumericHistogram) {
  if (!hist) return [];

  const { bin_edges, counts } = hist;

  const data: { x: string; count: number }[] = [];
  const n = Math.min(counts.length, bin_edges.length - 1);

  for (let i = 0; i < n; i++) {
    const left = bin_edges[i] ?? 0;
    const right = bin_edges[i + 1] ?? left;
    const count = counts[i] ?? 0;

    const label = `${roundCompact(left)}–${roundCompact(right)}`;
    data.push({ x: label, count });
  }

  return data;
}

function roundCompact(x: number | undefined): string {
  if (x == null || !isFinite(x)) return String(x);

  const abs = Math.abs(x);

  if (abs >= 1000) {
    return x.toExponential(1);
  }
  if (abs >= 10) {
    return x.toFixed(1);
  }
  if (abs >= 1) {
    return x.toFixed(2);
  }
  return x.toFixed(3);
}
