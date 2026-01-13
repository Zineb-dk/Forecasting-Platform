'use client';

import React, { useEffect, useState, useMemo } from 'react';
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

import {
  ArrowLeft,
  Activity,
  CheckCircle2,
  Clock,
  AlertTriangle,
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
const supabase = getSupabaseBrowserClient();

type Json = Record<string, any>;

interface DatasetRow {
  id: string;
  original_filename: string;
  rows_count: number | null;
  columns_count: number | null;
  target_column: string | null;
  time_column: string | null;
  forecast_horizon: number | null;
  ingestion_info: Json | null;
  processed_parquet_path?: string | null;
  processing_report?: Json | null;
  status?: string | null;
  created_at: string;
}

interface PreprocessResponse {
  status: string;
  dataset_id: string;
  time_col: string;
  target_col: string;
  entity_col: string | null;
  is_multi_entity: boolean;
  sensor_cols: string[];
  processed_parquet_path: string;
  processing_report: Json;
  n_rows_clean: number;
  n_cols_clean: number;
}

// ---------- helper for authorized fetch ----------
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

export default function DatasetProcessingPage() {
  const params = useParams();
  const router = useRouter();
  const datasetId = params?.datasetId as string;

  const [dataset, setDataset] = useState<DatasetRow | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [processing, setProcessing] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [lastRun, setLastRun] = useState<PreprocessResponse | null>(null);

  // ---- load dataset row ----
  useEffect(() => {
    if (!datasetId) return;

    let mounted = true;
    (async () => {
      try {
        setLoading(true);
        setError(null);
        const row = await authorizedFetch<DatasetRow>(
          `/api/v1/datasets/${datasetId}`,
        );
        if (mounted) setDataset(row);
      } catch (e: any) {
        if (mounted) setError(e?.message ?? 'Failed to load dataset.');
      } finally {
        if (mounted) setLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, [datasetId]);

  const fmtNum = (n?: number | null) =>
    typeof n === 'number' ? n.toLocaleString() : '—';

  const fmtDate = (iso?: string) =>
    iso ? new Date(iso).toLocaleString() : '—';

  const timeCol =
    dataset?.time_column ||
    (dataset?.ingestion_info as any)?.time_columns?.[0] ||
    null;

  const hasProcessed = Boolean(dataset?.processed_parquet_path);

  const overallStatus = useMemo(() => {
    if (processing) return 'running';
    if (hasProcessed) return 'done';
    return 'idle';
  }, [processing, hasProcessed]);

  const processingReport = (dataset?.processing_report ||
    lastRun?.processing_report) as any | undefined;

  const rowsBefore =
    processingReport?.overview_before?.n_rows ??
    processingReport?.overview_before?.shape?.[0] ??
    null;

  const rowsAfter =
    processingReport?.overview_after?.n_rows ??
    processingReport?.overview_after?.shape?.[0] ??
    null;

  const duplicatesDropped =
    processingReport?.n_full_duplicates_dropped ?? null;

  const constantDropped =
    processingReport?.dropped_constant_columns?.length ?? null;

  const runPreprocess = async () => {
    if (!datasetId) return;
    if (hasProcessed) return; 

    try {
      setProcessing(true);
      setProgress(10);
      setError(null);
      setTimeout(() => setProgress(35), 200);
      setTimeout(() => setProgress(65), 600);

      const resp = await authorizedFetch<PreprocessResponse>(
        `/api/v1/datasets/${datasetId}/preprocess`,
        {
          method: 'POST',
          body: JSON.stringify({}),
        },
      );

      setLastRun(resp);
      setProgress(100);

      const updated = await authorizedFetch<DatasetRow>(
        `/api/v1/datasets/${datasetId}`,
      );
      setDataset(updated);
    } catch (e: any) {
      setError(e?.message ?? 'Preprocessing failed.');
      setProgress(0);
    } finally {
      setTimeout(() => {
        setProcessing(false);
      }, 300);
    }
  };

  const progressValue = processing
    ? progress
    : hasProcessed
    ? 100
    : 0;

  const canRunPreprocess = !loading && !processing && !hasProcessed;

  const statusBadge = (() => {
    if (overallStatus === 'running') {
      return (
        <Badge
          variant="secondary"
          className="inline-flex items-center gap-1 text-xs"
        >
          <Activity className="h-3 w-3 animate-spin" />
          Running…
        </Badge>
      );
    }
    if (overallStatus === 'done') {
      return (
        <Badge
          variant="default"
          className="inline-flex items-center gap-1 text-xs"
        >
          <CheckCircle2 className="h-3 w-3" />
          Processed
        </Badge>
      );
    }
    return (
      <Badge
        variant="outline"
        className="inline-flex items-center gap-1 text-xs"
      >
        <Clock className="h-3 w-3" />
        Not processed
      </Badge>
    );
  })();

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
        <header className="space-y-3">
          <div className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-white/80 px-4 py-1.5 text-xs shadow-sm backdrop-blur dark:border-neutral-800 dark:bg-neutral-900/80">
            <Activity className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
            <span className="font-medium text-gray-900 dark:text-neutral-50">
              DataForge Preprocessing
            </span>
            <span className="text-gray-600 dark:text-neutral-400">
              One generic cleaning pipeline for your PdM / time-series dataset.
            </span>
          </div>

          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
            <div>
              <h1 className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-600 bg-clip-text text-3xl font-bold tracking-tight text-transparent md:text-4xl dark:from-neutral-50 dark:via-neutral-200 dark:to-neutral-400">
                Dataset preprocessing
              </h1>
              <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
                This step parses dates, sorts by time, drops duplicates /
                constant columns, optionally regularizes the time grid, and
                stores a clean parquet artifact for the next stages (AutoML,
                explainability, etc.).
              </p>
            </div>
            <div className="flex items-center gap-3">
              {statusBadge}
            </div>
          </div>
        </header>

        {/* Dataset + Status cards */}
        <section className="grid gap-6 md:grid-cols-3">
          {/* Dataset basic info */}
          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80 md:col-span-2">
            <CardHeader>
              <CardTitle>Dataset</CardTitle>
              <CardDescription>
                Raw standardized dataset registered in Supabase.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading && (
                <p className="text-sm text-neutral-500">Loading…</p>
              )}
              {!loading && !dataset && (
                <p className="text-sm text-red-500">Dataset not found.</p>
              )}
              {dataset && (
                <Table>
                  <TableBody>
                    <TableRow>
                      <TableCell className="font-medium">Filename</TableCell>
                      <TableCell>{dataset.original_filename}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">Rows × Cols</TableCell>
                      <TableCell>
                        {fmtNum(dataset.rows_count)} ×{' '}
                        {fmtNum(dataset.columns_count)}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">
                        Target column
                      </TableCell>
                      <TableCell>{dataset.target_column ?? '—'}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">Time column</TableCell>
                      <TableCell>
                        {timeCol ?? (
                          <span className="text-xs text-neutral-500">
                            (from ingestion_info.time_columns[0])
                          </span>
                        )}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">
                        Forecast horizon
                      </TableCell>
                      <TableCell>
                        {dataset.forecast_horizon ?? '—'}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">Created at</TableCell>
                      <TableCell>{fmtDate(dataset.created_at)}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">
                        Processed parquet
                      </TableCell>
                      <TableCell className="text-xs break-all">
                        {dataset.processed_parquet_path ?? '—'}
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>

          {/* Preprocessing status + progress */}
          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Activity className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                Preprocessing status
              </CardTitle>
              <CardDescription>
                Run and monitor the generic cleaning pipeline.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-neutral-500 dark:text-neutral-400">
                    Pipeline progress
                  </span>
                  <span className="font-mono text-neutral-700 dark:text-neutral-200">
                    {progressValue}%
                  </span>
                </div>
                {/* custom progress bar */}
                <div className="h-2 w-full rounded-full bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
                  <div
                    className="h-2 rounded-full bg-emerald-500 transition-[width] duration-300 ease-out"
                    style={{ width: `${progressValue}%` }}
                  />
                </div>
              </div>

              <Button
                className="w-full rounded-full"
                size="sm"
                onClick={runPreprocess}
                disabled={!canRunPreprocess}
              >
                {processing
                  ? 'Running preprocessing…'
                  : hasProcessed
                  ? 'Already preprocessed'
                  : 'Run preprocessing'}
              </Button>

              {hasProcessed && !processing && dataset && (
                <div className="space-y-2">
                  <Button
                    className="w-full rounded-full"
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      router.push(
                        `/home/datasets/${dataset.id}/processing/report`,
                      )
                    }
                  >
                    View preprocessing report
                  </Button>
                  <Button
                    className="w-full rounded-full"
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      router.push(`/home/automl?datasetId=${dataset.id}`)
                    }
                  >
                    Go to AutoML
                  </Button>
                </div>
              )}

              {processingReport && (
                <div className="mt-2 space-y-1 text-xs text-neutral-600 dark:text-neutral-400">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">Summary:</span>
                    {rowsBefore != null && rowsAfter != null && (
                      <span>
                        rows {rowsBefore} → {rowsAfter}
                      </span>
                    )}
                  </div>
                  <div className="flex flex-col gap-1">
                    {duplicatesDropped != null && (
                      <span>
                        • Full duplicates dropped:{' '}
                        <span className="font-mono">
                          {duplicatesDropped}
                        </span>
                      </span>
                    )}
                    {constantDropped != null && (
                      <span>
                        • Constant columns removed:{' '}
                        <span className="font-mono">{constantDropped}</span>
                      </span>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        {/* Info block */}
        <section className="space-y-3">
          <h2 className="text-xl font-semibold">What happens in preprocessing?</h2>
          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardContent className="space-y-2 py-4 text-sm text-neutral-600 dark:text-neutral-400">
              <p>
                Under the hood, the backend calls{' '}
                <code className="rounded bg-neutral-100 px-1 py-0.5 text-xs dark:bg-neutral-800">
                  DataForge.preprocess(df)
                </code>{' '}
                with your dataset and:
              </p>
              <ul className="ml-4 list-disc space-y-1">
                <li>parses and sorts the time column,</li>
                <li>drops fully duplicated rows,</li>
                <li>drops constant columns and fully empty rows,</li>
                <li>
                  if the time axis is datetime and regular enough, reindexes to
                  a full grid and interpolates,
                </li>
                <li>
                  writes the cleaned dataframe to{' '}
                  <span className="font-mono">
                    datasets.processed_parquet_path
                  </span>{' '}
                  in Supabase Storage and stores a JSON report in{' '}
                  <span className="font-mono">datasets.processing_report</span>.
                </li>
              </ul>
            </CardContent>
          </Card>
        </section>

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

export { DatasetProcessingPage };
