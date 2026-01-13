'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@kit/ui/card';
import { Badge } from '@kit/ui/badge';
import { Button } from '@kit/ui/button';

import { Activity, Database, LineChart } from 'lucide-react';
import { getSupabaseBrowserClient } from '@kit/supabase/browser-client';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
const supabase = getSupabaseBrowserClient();

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
      ...(init?.method && init.method !== 'GET'
        ? { 'Content-Type': 'application/json' }
        : {}),
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

function countByStatus(rows: any[]) {
  const out: Record<string, number> = {};
  for (const r of rows) {
    const s = (r.status ?? 'unknown') as string;
    out[s] = (out[s] ?? 0) + 1;
  }
  return out;
}

export default function MonitoringPage() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        setLoading(true);
        setError(null);

        const [ds, md] = await Promise.all([
          authorizedFetch<any[]>('/api/v1/datasets'),
          authorizedFetch<any[]>('/api/v1/models'),
        ]);

        if (!mounted) return;
        setDatasets(ds || []);
        setModels(md || []);
      } catch (e: any) {
        if (mounted) setError(e?.message ?? 'Failed to load monitoring data.');
      } finally {
        if (mounted) setLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, []);

  const datasetsByStatus = useMemo(() => countByStatus(datasets), [datasets]);
  const modelsByStatus = useMemo(() => countByStatus(models), [models]);

  const lastDataset = datasets?.[0]?.created_at
    ? new Date(datasets[0].created_at).toLocaleString()
    : '—';
  const lastModel = models?.[0]?.created_at
    ? new Date(models[0].created_at).toLocaleString()
    : '—';

  return (
    <div className="mx-auto max-w-6xl px-6 py-8 space-y-6">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
          <Activity className="h-5 w-5" /> Monitoring &amp; Drift
        </h1>
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          Quick system overview. (Drift charts can be added later.)
        </p>
      </header>

      {error && (
        <div className="rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-900/20 dark:text-red-200">
          {error}
        </div>
      )}

      {loading ? (
        <Card>
          <CardContent className="p-6 text-sm text-neutral-500">
            Loading monitoring…
          </CardContent>
        </Card>
      ) : (
        <>
          <section className="grid gap-4 md:grid-cols-4">
            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Datasets</CardDescription>
                <CardTitle className="text-2xl">{datasets.length}</CardTitle>
              </CardHeader>
              <CardContent className="text-xs text-neutral-600 dark:text-neutral-400">
                Last: {lastDataset}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Models</CardDescription>
                <CardTitle className="text-2xl">{models.length}</CardTitle>
              </CardHeader>
              <CardContent className="text-xs text-neutral-600 dark:text-neutral-400">
                Last: {lastModel}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Dataset statuses</CardDescription>
                <CardTitle className="text-base">Breakdown</CardTitle>
              </CardHeader>
              <CardContent className="text-xs space-y-1">
                {Object.entries(datasetsByStatus).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between">
                    <span>{k}</span>
                    <Badge variant="secondary">{v}</Badge>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardDescription>Model statuses</CardDescription>
                <CardTitle className="text-base">Breakdown</CardTitle>
              </CardHeader>
              <CardContent className="text-xs space-y-1">
                {Object.entries(modelsByStatus).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between">
                    <span>{k}</span>
                    <Badge variant="secondary">{v}</Badge>
                  </div>
                ))}
              </CardContent>
            </Card>
          </section>

          <section className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Database className="h-4 w-4" /> Latest datasets
                </CardTitle>
                <CardDescription>Quick links.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {(datasets || []).slice(0, 6).map((d) => (
                  <div key={d.id} className="flex items-center justify-between text-sm">
                    <div className="truncate">
                      <span className="font-medium">
                        {d.original_filename ?? d.id}
                      </span>
                      <span className="ml-2 text-xs text-neutral-500">
                        {d.status ?? '—'}
                      </span>
                    </div>
                    <Link href={`/home/datasets/${d.id}/eda`}>
                      <Button size="sm" variant="ghost">Open</Button>
                    </Link>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <LineChart className="h-4 w-4" /> Latest models
                </CardTitle>
                <CardDescription>Quick links.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {(models || []).slice(0, 6).map((m) => (
                  <div key={m.id} className="flex items-center justify-between text-sm">
                    <div className="truncate">
                      <span className="font-medium">
                        {m.model_name ?? m.algorithm ?? 'Model'}
                      </span>
                      <span className="ml-2 text-xs text-neutral-500">
                        {m.status ?? '—'}
                      </span>
                    </div>
                    <Link href={`/home/models/${m.id}`}>
                      <Button size="sm" variant="ghost">View</Button>
                    </Link>
                  </div>
                ))}
              </CardContent>
            </Card>
          </section>
        </>
      )}
    </div>
  );
}
