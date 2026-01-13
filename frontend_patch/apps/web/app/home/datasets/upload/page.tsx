'use client';

import { useState, ChangeEvent, FormEvent } from 'react';
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
  Upload,
  FileText,
  CheckCircle,
  ArrowLeft,
} from 'lucide-react';

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@kit/ui/select';

import { Checkbox } from '@kit/ui/checkbox';

import { getSupabaseBrowserClient } from '@kit/supabase/browser-client';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
const supabase = getSupabaseBrowserClient();

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

function normalizeColumnName(name: string): string {
  return name
    .trim()
    .replace(/\//g, '_')
    .replace(/-/g, '_')
    .split(/\s+/)
    .join('_')
    .toLowerCase();
}

async function uploadDatasetToApi(params: {
  file: File;
  targetFeature: string;
  timeColumn: string;
  forecastHorizon: number;
  isMultiEntity: boolean;
  entityColumn?: string | null;
}) {
  const {
    file,
    targetFeature,
    timeColumn,
    forecastHorizon,
    isMultiEntity,
    entityColumn,
  } = params;

  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session?.access_token) {
    throw new Error('Not authenticated. Please sign in again.');
  }

  const token = session.access_token;

  const formData = new FormData();
  formData.append('file', file);
  formData.append('target_feature', targetFeature);
  formData.append('time_column', timeColumn);
  formData.append('forecast_horizon', String(forecastHorizon));
  formData.append('is_multi_entity', String(isMultiEntity));
  if (isMultiEntity && entityColumn) {
    formData.append('entity_column', entityColumn);
  }

  const res = await fetch(`${API_URL}/api/v1/data/upload`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} ${txt}`);
  }

  return res.json();
}

export default function DatasetUploadPage() {
  const router = useRouter();

  const [file, setFile] = useState<File | null>(null);
  const [targetFeature, setTargetFeature] = useState('');
  const [timeColumn, setTimeColumn] = useState('');

  const [forecastHorizon, setForecastHorizon] = useState<number | ''>('');
  const [isMultiEntity, setIsMultiEntity] = useState(false);
  const [entityColumn, setEntityColumn] = useState('');

  const [status, setStatus] = useState<UploadStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [createdDatasetName, setCreatedDatasetName] = useState<string | null>(
    null,
  );

  const [columns, setColumns] = useState<string[]>([]);
  const [columnsLoading, setColumnsLoading] = useState(false);
  const [columnsError, setColumnsError] = useState<string | null>(null);

  const isUploading = status === 'uploading';

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setCreatedDatasetName(null);
    setStatus('idle');
    setError(null);

    setColumns([]);
    setColumnsError(null);
    setTargetFeature('');
    setTimeColumn('');
    setEntityColumn('');
    setIsMultiEntity(false);

    if (!f) return;

    setColumnsLoading(true);
    (async () => {
      try {
        const nameLower = (f.name || '').toLowerCase();
        if (nameLower.endsWith('.csv')) {
          const text = await f.text();
          const firstLine =
            text.split(/\r?\n/).find((line) => line.trim().length > 0) || '';
          if (!firstLine) {
            setColumnsError('Could not detect header row in CSV.');
            return;
          }
          const cols = firstLine
            .split(',')
            .map((c) => c.trim().replace(/^"|"$/g, ''))
            .filter(Boolean);

          if (!cols.length) {
            setColumnsError('No columns detected in header row.');
            return;
          }
          setColumns(cols);
        } else {
          setColumnsError(
            'Automatic column detection is only available for CSV. For Excel / Parquet, type the column names manually.',
          );
        }
      } catch (err: any) {
        setColumnsError(
          err?.message ?? 'Failed to read file header for columns.',
        );
      } finally {
        setColumnsLoading(false);
      }
    })();
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setCreatedDatasetName(null);

    if (!file) {
      setError('Please choose a dataset file.');
      return;
    }
    if (!targetFeature.trim()) {
      setError('Please specify the target feature (column to predict).');
      return;
    }
    if (!timeColumn.trim()) {
      setError('Please specify the time column.');
      return;
    }
    if (!forecastHorizon || forecastHorizon <= 0) {
      setError('Please provide a positive forecast horizon.');
      return;
    }
    if (isMultiEntity) {
      if (!entityColumn.trim()) {
        setError('Please select the entity column for a multi-entity dataset.');
        return;
      }
      if (
        entityColumn === targetFeature ||
        entityColumn === timeColumn
      ) {
        setError(
          'Entity column must be different from the time column and the target column.',
        );
        return;
      }
    }

    try {
      setStatus('uploading');

      const normalizedTarget = normalizeColumnName(targetFeature.trim());
      const normalizedTime = normalizeColumnName(timeColumn.trim());
      const normalizedEntity = entityColumn
        ? normalizeColumnName(entityColumn.trim())
        : undefined;

      const resp = await uploadDatasetToApi({
        file,
        targetFeature: normalizedTarget,
        timeColumn: normalizedTime,
        forecastHorizon: Number(forecastHorizon),
        isMultiEntity,
        entityColumn: isMultiEntity ? normalizedEntity ?? null : null,
      });

      setStatus('success');
      setCreatedDatasetName(
        resp?.dataset?.original_filename ||
          resp?.original_filename ||
          resp?.id ||
          file.name,
      );

      if (resp?.dataset_id) {
        router.push(`/home/datasets/${resp.dataset_id}/eda`);
      }
    } catch (err: any) {
      setStatus('error');
      setError(err?.message ?? 'Upload failed');
    }
  };

  const entityOptions = columns.filter(
    (c) => c !== targetFeature && c !== timeColumn,
  );

  return (
    <div className="relative min-h-screen overflow-auto bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50">

      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,#000_70%,transparent_110%)] dark:bg-[linear-gradient(to_right,#262626_1px,transparent_1px),linear-gradient(to_bottom,#262626_1px,transparent_1px)]" />

      <div className="relative mx-auto flex min-h-screen max-w-3xl flex-col items-center justify-center px-6 py-10">

        <div className="mb-6 flex w-full items-center justify-between">
          <Link href="/home">
            <Button
              variant="outline"
              size="sm"
              className="flex items-center gap-2 rounded-full"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Dashboard
            </Button>
          </Link>
        </div>

        <div className="w-full space-y-6">
          <header className="space-y-3 text-center">
            <h1 className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-600 bg-clip-text text-3xl font-bold tracking-tight text-transparent md:text-4xl dark:from-neutral-50 dark:via-neutral-200 dark:to-neutral-400">
              Upload a PdM / Time Series Dataset
            </h1>
            <p className="mx-auto max-w-2xl text-sm text-neutral-600 dark:text-neutral-400">
              Upload your historical data, choose the target, time axis, and
              whether it&apos;s a multi-entity dataset. This configuration will
              drive DataForge preprocessing and the AutoML pipeline.
            </p>
          </header>

          <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
            <CardHeader>
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-blue-100 dark:bg-blue-900/40">
                  <Upload className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <CardTitle>Upload Dataset</CardTitle>
                  <CardDescription>
                    Supported formats: CSV, Excel (XLSX), Parquet.
                  </CardDescription>
                </div>
              </div>
            </CardHeader>

            <CardContent>
              <form className="space-y-6" onSubmit={onSubmit}>

                <div className="space-y-2">
                  <Label htmlFor="dataset-file">Dataset file</Label>
                  <div className="flex flex-col gap-3 rounded-xl border border-dashed border-gray-300 bg-gray-50/70 p-4 dark:border-neutral-700 dark:bg-neutral-900/60">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-white dark:bg-neutral-800">
                        <FileText className="h-5 w-5 text-neutral-500" />
                      </div>
                      <div className="flex-1">
                        <Input
                          id="dataset-file"
                          type="file"
                          accept=".csv,.xlsx,.xls,.parquet"
                          onChange={onFileChange}
                          disabled={isUploading}
                          className="cursor-pointer"
                        />
                        <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
                          The file must include a header row with column names.
                        </p>
                      </div>
                    </div>

                    {file && (
                      <div className="flex items-center justify-between text-xs text-neutral-600 dark:text-neutral-400">
                        <span>Selected file:</span>
                        <span className="max-w-[220px] truncate font-medium">
                          {file.name}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Target feature */}
                <div className="space-y-2">
                  <Label htmlFor="target-feature">Target column</Label>

                  {columnsLoading ? (
                    <div className="text-xs text-neutral-500 dark:text-neutral-400">
                      Reading CSV header to detect columns…
                    </div>
                  ) : columns.length > 0 ? (
                    <Select
                      value={targetFeature}
                      onValueChange={(val) => {
                        setTargetFeature(val);

                        if (val === entityColumn) setEntityColumn('');
                      }}
                      disabled={isUploading}
                    >
                      <SelectTrigger id="target-feature">
                        <SelectValue placeholder="Select the target column" />
                      </SelectTrigger>
                      <SelectContent>
                        {columns.map((col) => (
                          <SelectItem key={col} value={col}>
                            {col}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : (
                    <Input
                      id="target-feature"
                      placeholder="e.g. RUL, time_to_failure, risk_score"
                      value={targetFeature}
                      onChange={(e) => setTargetFeature(e.target.value)}
                      disabled={isUploading}
                    />
                  )}

                  <p className="text-xs text-neutral-500 dark:text-neutral-400">
                    Column you want to predict (RUL, time_to_failure, risk_score,
                    etc.).
                  </p>

                  {columnsError && (
                    <p className="text-xs text-amber-600 dark:text-amber-400">
                      {columnsError}
                    </p>
                  )}
                </div>

                {/* Time column */}
                <div className="space-y-2">
                  <Label htmlFor="time-column">Time column</Label>

                  {columns.length > 0 ? (
                    <Select
                      value={timeColumn}
                      onValueChange={(val) => {
                        setTimeColumn(val);

                        if (val === entityColumn) setEntityColumn('');
                      }}
                      disabled={isUploading}
                    >
                      <SelectTrigger id="time-column">
                        <SelectValue placeholder="Select the time column" />
                      </SelectTrigger>
                      <SelectContent>
                        {columns.map((col) => (
                          <SelectItem key={col} value={col}>
                            {col}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : (
                    <Input
                      id="time-column"
                      placeholder="e.g. time_in_cycles, timestamp"
                      value={timeColumn}
                      onChange={(e) => setTimeColumn(e.target.value)}
                      disabled={isUploading}
                    />
                  )}

                  <p className="text-xs text-neutral-500 dark:text-neutral-400">
                    Time axis of the dataset (cycles, timestamps, or integer index).
                  </p>
                </div>

                {/* Forecast horizon */}
                <div className="space-y-2">
                  <Label htmlFor="forecast-horizon">Forecast horizon</Label>
                  <Input
                    id="forecast-horizon"
                    type="number"
                    min={1}
                    placeholder="e.g. 1, 14, 30"
                    value={forecastHorizon}
                    onChange={(e) =>
                      setForecastHorizon(
                        e.target.value === '' ? '' : Number(e.target.value),
                      )
                    }
                    disabled={isUploading}
                  />
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">
                    Number of steps ahead you want to predict (in cycles or time
                    steps).
                  </p>
                </div>

                {/* Multi-entity config */}
                <div className="space-y-3 rounded-xl border border-gray-200 bg-gray-50/70 p-4 dark:border-neutral-800 dark:bg-neutral-900/60">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <Label>Multi-entity dataset</Label>
                      <p className="text-xs text-neutral-500 dark:text-neutral-400">
                        Check this if your dataset contains multiple machines /
                        engines / units (e.g. unit_number).
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="is-multi-entity"
                        checked={isMultiEntity}
                        onCheckedChange={(val) => {
                          const checked = Boolean(val);
                          setIsMultiEntity(checked);
                          if (!checked) {
                            setEntityColumn('');
                          }
                        }}
                        disabled={isUploading}
                      />
                      <Label
                        htmlFor="is-multi-entity"
                        className="text-xs text-neutral-700 dark:text-neutral-300"
                      >
                        Is multi-entity
                      </Label>
                    </div>
                  </div>

                  {isMultiEntity && (
                    <div className="space-y-2 pt-2">
                      <Label htmlFor="entity-column">Entity column</Label>

                      {columns.length > 0 ? (
                        <Select
                          value={entityColumn}
                          onValueChange={setEntityColumn}
                          disabled={isUploading || entityOptions.length === 0}
                        >
                          <SelectTrigger id="entity-column">
                            <SelectValue
                              placeholder={
                                entityOptions.length
                                  ? 'Select the entity column'
                                  : 'No available columns (check time/target choices)'
                              }
                            />
                          </SelectTrigger>
                          <SelectContent>
                            {entityOptions.map((col) => (
                              <SelectItem key={col} value={col}>
                                {col}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      ) : (
                        <Input
                          id="entity-column"
                          placeholder="e.g. unit_number, machine_id"
                          value={entityColumn}
                          onChange={(e) => setEntityColumn(e.target.value)}
                          disabled={isUploading}
                        />
                      )}

                      <p className="text-xs text-neutral-500 dark:text-neutral-400">
                        Column that identifies each unit / engine / machine. DataForge
                        will later infer sensors as all numeric columns except
                        time / target / entity.
                      </p>
                    </div>
                  )}
                </div>

                {/* Actions + status */}
                <div className="flex flex-col gap-3 pt-2">
                  <div className="flex flex-wrap items-center gap-3">
                    <Button
                      type="submit"
                      className="rounded-full"
                      disabled={isUploading}
                    >
                      {isUploading
                        ? 'Uploading & registering dataset...'
                        : 'Upload & Register Dataset'}
                    </Button>

                    <Link href="/home">
                      <Button
                        type="button"
                        variant="ghost"
                        className="rounded-full"
                        disabled={isUploading}
                      >
                        Cancel
                      </Button>
                    </Link>
                  </div>

                  <div className="text-xs">
                    {status === 'success' && (
                      <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400">
                        <CheckCircle className="h-4 w-4" />
                        <span>
                          Dataset registered successfully
                          {createdDatasetName
                            ? `: ${createdDatasetName}`
                            : '!'}
                        </span>
                      </div>
                    )}

                    {status === 'error' && error && (
                      <span className="text-red-600 dark:text-red-400">
                        {error}
                      </span>
                    )}

                    {status === 'idle' && !error && !createdDatasetName && (
                      <span className="text-neutral-500 dark:text-neutral-400">
                        After upload, your dataset will appear on the dashboard,
                        then you can configure DataForge, run AutoML and generate
                        forecasts.
                      </span>
                    )}
                  </div>
                </div>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

export { DatasetUploadPage };
