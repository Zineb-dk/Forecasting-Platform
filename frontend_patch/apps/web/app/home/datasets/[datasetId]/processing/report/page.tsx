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
  AlertTriangle,
  Clock,
  GitBranch,
  BarChart2,
  SlidersHorizontal,
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

interface ReportResponse {
  dataset_id: string;
  report: Json | null;
  processed_parquet_path?: string | null;
  status?: string | null;
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

// ---------- helpers ----------
const fmtNum = (n?: number | null) =>
  typeof n === 'number' ? n.toLocaleString() : '—';

const fmtDate = (iso?: string) =>
  iso ? new Date(iso).toLocaleString() : '—';

function formatPercent(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(1)}%`;
}

function formatGapFromNs(ns?: number | null): string {
  if (ns == null || Number.isNaN(ns)) return '—';
  const seconds = ns / 1e9;
  if (seconds < 1) {
    return `${seconds.toFixed(3)} s`;
  }
  if (seconds < 60) {
    return `${seconds.toFixed(2)} s`;
  }
  const minutes = seconds / 60;
  if (minutes < 60) {
    return `${minutes.toFixed(2)} min`;
  }
  const hours = minutes / 60;
  return `${hours.toFixed(2)} h`;
}

function safeKeys(obj: any): string[] {
  if (!obj || typeof obj !== 'object') return [];
  return Object.keys(obj);
}

export default function DatasetPreprocessingReportPage() {
  const params = useParams();
  const router = useRouter();
  const datasetId = params?.datasetId as string;

  const [dataset, setDataset] = useState<DatasetRow | null>(null);
  const [report, setReport] = useState<Json | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // ---- load dataset row + report ----
  useEffect(() => {
    if (!datasetId) return;

    let mounted = true;
    (async () => {
      try {
        setLoading(true);
        setError(null);

        const [row, rep] = await Promise.all([
          authorizedFetch<DatasetRow>(`/api/v1/datasets/${datasetId}`),
          authorizedFetch<ReportResponse>(`/api/v1/datasets/${datasetId}/report`),
        ]);

        if (!mounted) return;

        setDataset(row);
        setReport(rep.report ?? null);
      } catch (e: any) {
        if (!mounted) return;
        setError(e?.message ?? 'Failed to load preprocessing report.');
      } finally {
        if (mounted) setLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, [datasetId]);

  const overviewBefore = (report as any)?.overview_before ?? null;
  const overviewAfter = (report as any)?.overview_after ?? null;
  const nDuplicates = (report as any)?.n_full_duplicates_dropped ?? null;
  const droppedConstant: string[] =
    (report as any)?.dropped_constant_columns ?? [];
  const timeReg = (report as any)?.time_regularity ?? null;
  const nRowsAdded = (report as any)?.n_rows_added_by_reindex ?? null;

  const timeCol =
    dataset?.time_column ||
    (dataset?.ingestion_info as any)?.time_columns?.[0] ||
    null;

  const isProcessed = Boolean(dataset?.processed_parquet_path);

  const columnDiffs = useMemo(() => {
    if (!overviewBefore || !overviewAfter) return [];
    const beforeDtypes = overviewBefore.dtypes || {};
    const afterDtypes = overviewAfter.dtypes || {};
    const allCols = new Set<string>([
      ...safeKeys(beforeDtypes),
      ...safeKeys(afterDtypes),
      ...droppedConstant,
    ]);

    const diffs: {
      col: string;
      before: string | null;
      after: string | null;
      note: string;
    }[] = [];

    allCols.forEach((col) => {
      const b = beforeDtypes[col] ?? null;
      const a = afterDtypes[col] ?? null;

      let note = '';
      if (droppedConstant.includes(col)) {
        note = 'Dropped (constant column)';
      } else if (b && !a) {
        note = 'Removed';
      } else if (!b && a) {
        note = 'Added';
      } else if (b !== a) {
        note = 'Converted dtype';
      }

      if (note) {
        diffs.push({ col, before: b, after: a, note });
      }
    });

    return diffs.sort((x, y) => x.col.localeCompare(y.col));
  }, [overviewBefore, overviewAfter, droppedConstant]);

  // ---- missing values before vs after ----
  const missingRows = useMemo(() => {
    const beforeMissing = (overviewBefore?.missing_counts ?? {}) as Record<
      string,
      number
    >;
    const afterMissing = (overviewAfter?.missing_counts ?? {}) as Record<
      string,
      number
    >;

    const allCols = new Set<string>([
      ...safeKeys(beforeMissing),
      ...safeKeys(afterMissing),
    ]);

    const rows: { col: string; before: number; after: number }[] = [];
    allCols.forEach((col) => {
      const b = beforeMissing[col] ?? 0;
      const a = afterMissing[col] ?? 0;
      if (b !== a || b > 0 || a > 0) {
        rows.push({ col, before: b, after: a });
      }
    });

    return rows.sort((x, y) => x.col.localeCompare(y.col));
  }, [overviewBefore, overviewAfter]);

  // ---- sensor columns after ----
  const sensorCols: string[] =
    (overviewAfter?.sensor_cols_effective as string[]) ?? [];

  // ---- multi-entity info ----
  const nEntities = overviewAfter?.n_entities ?? null;
  const entityCounts =
    (overviewAfter?.entity_counts as Record<string, number>) ?? null;
  const entityMin = overviewAfter?.entity_min_length ?? null;
  const entityMax = overviewAfter?.entity_max_length ?? null;

  const entityRows = useMemo(() => {
    if (!entityCounts) return [];
    return Object.entries(entityCounts)
      .map(([k, v]) => ({ entity: k, count: v }))
      .sort((a, b) => a.entity.localeCompare(b.entity));
  }, [entityCounts]);

  const statusBadge = (() => {
    if (loading) {
      return (
        <Badge
          variant="outline"
          className="inline-flex items-center gap-1 text-xs"
        >
          <Clock className="h-3 w-3 animate-pulse" />
          Loading…
        </Badge>
      );
    }
    if (!report) {
      return (
        <Badge
          variant="outline"
          className="inline-flex items-center gap-1 text-xs"
        >
          <AlertTriangle className="h-3 w-3" />
          No preprocessing report
        </Badge>
      );
    }
    return (
      <Badge
        variant="default"
        className="inline-flex items-center gap-1 text-xs"
      >
        <CheckCircle2 className="h-3 w-3" />
        Preprocessing completed
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
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              className="flex items-center gap-2 rounded-full"
              onClick={() => router.push(`/home/datasets/${datasetId}/processing`)}
            >
              <ArrowLeft className="h-4 w-4" />
              Back to preprocessing
            </Button>

            <Link href="/home">
              <Button
                variant="outline"
                size="sm"
                className="hidden items-center gap-2 rounded-full md:flex"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to dashboard
              </Button>
            </Link>
          </div>

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
              DataForge Preprocessing Report
            </span>
            <span className="text-gray-600 dark:text-neutral-400">
              Before/after stats, time regularity, and entity analysis.
            </span>
          </div>

          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
            <div>
              <h1 className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-600 bg-clip-text text-3xl font-bold tracking-tight text-transparent md:text-4xl dark:from-neutral-50 dark:via-neutral-200 dark:to-neutral-400">
                Preprocessing report
              </h1>
              <p className="mt-2 max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
                This page summarizes what happened when your dataset went
                through{' '}
                <code className="rounded bg-neutral-100 px-1 py-0.5 text-xs dark:bg-neutral-800">
                  DataForge.preprocess
                </code>
                : rows/columns changes, duplicates and constant columns removal,
                time axis regularity, and entity-level coverage.
              </p>
            </div>
            <div className="flex items-center gap-3">{statusBadge}</div>
          </div>
        </header>

        {dataset && (
          <section className="grid gap-6 md:grid-cols-3">
            {/* Dataset basic info */}
            <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80 md:col-span-2">
              <CardHeader>
                <CardTitle>Dataset</CardTitle>
                <CardDescription>
                  Raw standardized dataset and its cleaned artifact.
                </CardDescription>
              </CardHeader>
              <CardContent>
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
                      <TableCell className="font-medium">
                        Processed parquet
                      </TableCell>
                      <TableCell className="text-xs break-all">
                        {dataset.processed_parquet_path ?? '—'}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">Created at</TableCell>
                      <TableCell>{fmtDate(dataset.created_at)}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </CardContent>
            </Card>

            {/* Quick summary cards */}
            <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-sm">
                  <BarChart2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                  Summary
                </CardTitle>
                <CardDescription>
                  Rows, columns, duplicates, and reindexing impact.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span>Rows</span>
                    <span className="font-mono">
                      {overviewBefore?.n_rows ?? '—'} →{' '}
                      {overviewAfter?.n_rows ?? '—'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span>Columns</span>
                    <span className="font-mono">
                      {overviewBefore?.n_cols ?? '—'} →{' '}
                      {overviewAfter?.n_cols ?? '—'}
                    </span>
                  </div>
                </div>

                <div className="space-y-1 text-xs">
                  <div className="flex items-center justify-between">
                    <span>Duplicates removed</span>
                    <span className="font-mono">
                      {nDuplicates != null ? nDuplicates : '—'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Constant columns dropped</span>
                    <span className="font-mono">
                      {droppedConstant?.length ?? 0}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Rows added by reindex</span>
                    <span className="font-mono">
                      {nRowsAdded != null ? nRowsAdded : '—'}
                    </span>
                  </div>
                </div>

                {isProcessed && (
                  <Button
                    className="mt-2 w-full rounded-full"
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      router.push(`/home/datasets/${datasetId}/automl`)
                    }
                  >
                    Go to AutoML
                  </Button>
                )}
              </CardContent>
            </Card>
          </section>
        )}

        {/* Column changes */}
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-xl font-semibold">
            <SlidersHorizontal className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            Column changes
          </h2>
          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardContent className="space-y-4 py-4 text-sm">
              <div>
                <p className="mb-2 text-sm text-neutral-600 dark:text-neutral-400">
                  Columns that were dropped, converted, or added during
                  preprocessing.
                </p>
                {columnDiffs.length === 0 ? (
                  <p className="text-xs text-neutral-500">
                    No structural changes detected in columns.
                  </p>
                ) : (
                  <div className="overflow-x-auto">
                    <Table>
                      <TableBody>
                        <TableRow>
                          <TableCell className="text-xs font-semibold uppercase">
                            Column
                          </TableCell>
                          <TableCell className="text-xs font-semibold uppercase">
                            Before dtype
                          </TableCell>
                          <TableCell className="text-xs font-semibold uppercase">
                            After dtype
                          </TableCell>
                          <TableCell className="text-xs font-semibold uppercase">
                            Notes
                          </TableCell>
                        </TableRow>
                        {columnDiffs.map((c) => (
                          <TableRow key={c.col}>
                            <TableCell className="font-mono text-xs">
                              {c.col}
                            </TableCell>
                            <TableCell className="text-xs">
                              {c.before ?? '—'}
                            </TableCell>
                            <TableCell className="text-xs">
                              {c.after ?? '—'}
                            </TableCell>
                            <TableCell className="text-xs">
                              {c.note}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </div>

              {/* Constant columns list */}
              <div className="border-t border-dashed border-neutral-200 pt-3 text-xs dark:border-neutral-800">
                <span className="font-semibold">Constant columns removed: </span>
                {droppedConstant && droppedConstant.length > 0 ? (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {droppedConstant.map((c) => (
                      <span
                        key={c}
                        className="rounded-full bg-neutral-100 px-2 py-0.5 font-mono text-[11px] dark:bg-neutral-800"
                      >
                        {c}
                      </span>
                    ))}
                  </div>
                ) : (
                  <span className="text-neutral-500">
                    no constant columns were removed.
                  </span>
                )}
              </div>
            </CardContent>
          </Card>
        </section>

        {/* Time regularity */}
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-xl font-semibold">
            <Clock className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            Time regularity analysis
          </h2>
          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardContent className="space-y-3 py-4 text-sm">
              {timeReg ? (
                <>
                  <p className="text-neutral-600 dark:text-neutral-400">
                    DataForge examined the gaps between consecutive timestamps
                    to check if the time series is roughly on a regular grid.
                  </p>
                  <div className="grid gap-3 md:grid-cols-4 text-xs">
                    <div>
                      <div className="text-neutral-500">Regular grid?</div>
                      <div className="font-mono">
                        {timeReg.is_regular ? 'Yes' : 'No'}
                      </div>
                    </div>
                    <div>
                      <div className="text-neutral-500">Inferred freq</div>
                      <div className="font-mono">
                        {timeReg.inferred_freq ?? '—'}
                      </div>
                    </div>
                    <div>
                      <div className="text-neutral-500">Most common gap</div>
                      <div className="font-mono">
                        {formatGapFromNs(timeReg.mode_gap_ns)}
                      </div>
                    </div>
                    <div>
                      <div className="text-neutral-500">
                        Mode gap coverage
                      </div>
                      <div className="font-mono">
                        {formatPercent(timeReg.mode_ratio)}
                      </div>
                    </div>
                  </div>

                  {nRowsAdded != null && (
                    <p className="mt-2 text-xs text-neutral-500">
                      Rows added during reindexing:{' '}
                      <span className="font-mono">{nRowsAdded}</span>. When the
                      frequency is detected and coverage is high enough, the
                      time axis is expanded to a full grid and values are
                      interpolated/ffilled.
                    </p>
                  )}
                </>
              ) : (
                <p className="text-xs text-neutral-500">
                  Time regularity analysis was not applied (time column was not
                  datetime or frequency could not be inferred).
                </p>
              )}
            </CardContent>
          </Card>
        </section>

        {/* Missing values */}
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-xl font-semibold">
            <GitBranch className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            Missing values before vs after
          </h2>
          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardContent className="space-y-3 py-4 text-sm">
              {missingRows.length === 0 ? (
                <p className="text-xs text-neutral-500">
                  No missing values were detected before or after preprocessing.
                </p>
              ) : (
                <>
                  <p className="text-neutral-600 dark:text-neutral-400">
                    Columns where missing values were present or changed during
                    preprocessing.
                  </p>
                  <div className="overflow-x-auto">
                    <Table>
                      <TableBody>
                        <TableRow>
                          <TableCell className="text-xs font-semibold uppercase">
                            Column
                          </TableCell>
                          <TableCell className="text-xs font-semibold uppercase">
                            Missing (before)
                          </TableCell>
                          <TableCell className="text-xs font-semibold uppercase">
                            Missing (after)
                          </TableCell>
                        </TableRow>
                        {missingRows.map((r) => (
                          <TableRow key={r.col}>
                            <TableCell className="font-mono text-xs">
                              {r.col}
                            </TableCell>
                            <TableCell className="font-mono text-xs">
                              {r.before}
                            </TableCell>
                            <TableCell className="font-mono text-xs">
                              {r.after}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </section>

        {/* Sensor columns */}
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-xl font-semibold">
            <BarChart2 className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            Sensor columns in the cleaned dataset
          </h2>
          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardContent className="space-y-3 py-4 text-sm">
              {sensorCols.length === 0 ? (
                <p className="text-xs text-neutral-500">
                  No active sensor columns were detected in the cleaned
                  dataset.
                </p>
              ) : (
                <>
                  <p className="text-neutral-600 dark:text-neutral-400">
                    These columns are considered as sensor features after
                    preprocessing.
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {sensorCols.map((c) => (
                      <span
                        key={c}
                        className="rounded-full bg-neutral-100 px-2 py-0.5 font-mono text-[11px] dark:bg-neutral-800"
                      >
                        {c}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </section>

        {/* Multi-entity analysis (if applicable) */}
        {nEntities != null && nEntities > 1 && (
          <section className="space-y-3">
            <h2 className="flex items-center gap-2 text-xl font-semibold">
              <Activity className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
              Entity coverage (multi-entity dataset)
            </h2>
            <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
              <CardContent className="space-y-3 py-4 text-sm">
                <div className="grid gap-3 md:grid-cols-3 text-xs">
                  <div>
                    <div className="text-neutral-500">Number of entities</div>
                    <div className="font-mono">{nEntities}</div>
                  </div>
                  <div>
                    <div className="text-neutral-500">Min rows per entity</div>
                    <div className="font-mono">
                      {entityMin != null ? entityMin : '—'}
                    </div>
                  </div>
                  <div>
                    <div className="text-neutral-500">Max rows per entity</div>
                    <div className="font-mono">
                      {entityMax != null ? entityMax : '—'}
                    </div>
                  </div>
                </div>

                {entityRows.length > 0 && (
                  <div className="overflow-x-auto text-xs">
                    <Table>
                      <TableBody>
                        <TableRow>
                          <TableCell className="text-xs font-semibold uppercase">
                            Entity
                          </TableCell>
                          <TableCell className="text-xs font-semibold uppercase">
                            Rows
                          </TableCell>
                        </TableRow>
                        {entityRows.map((r) => (
                          <TableRow key={r.entity}>
                            <TableCell className="font-mono text-xs">
                              {r.entity}
                            </TableCell>
                            <TableCell className="font-mono text-xs">
                              {r.count}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </CardContent>
            </Card>
          </section>
        )}

        {/* Raw JSON (collapsible) */}
        <section className="space-y-3">
          <h2 className="flex items-center gap-2 text-xl font-semibold">
            <GitBranch className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            Raw preprocessing report (JSON)
          </h2>
          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardContent className="py-3 text-sm">
              <details className="space-y-2">
                <summary className="cursor-pointer text-xs text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100">
                  Show raw JSON (for debugging or export)
                </summary>
                <pre className="max-h-[420px] overflow-auto rounded-md bg-neutral-950/95 p-3 text-xs text-neutral-50 shadow-inner dark:bg-black">
{JSON.stringify(report ?? {}, null, 2)}
                </pre>
              </details>
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

export { DatasetPreprocessingReportPage };
