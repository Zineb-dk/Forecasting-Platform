'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@kit/ui/card';
import { Button } from '@kit/ui/button';
import { Badge } from '@kit/ui/badge';

import {
  Sparkles,
  Info,
  ChevronLeft,
  LineChart,
  Wand2,
  CheckCircle,
  Activity,
  Target,
  Crown,
} from 'lucide-react';
import { getSupabaseBrowserClient } from '@kit/supabase/browser-client';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';
const supabase = getSupabaseBrowserClient();

// -------------------- auth fetch --------------------
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

// -------------------- utils --------------------
const pretty = (obj: any) => {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
};

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

type MinMax = { lo: number; hi: number };

function safeDen(hi: number, lo: number, eps = 1e-9) {
  const d = hi - lo;
  return Number.isFinite(d) && Math.abs(d) > eps ? d : 1;
}

function quantile(sortedFiniteAsc: number[], q: number): number {
  const xs = sortedFiniteAsc.filter(Number.isFinite);
  if (xs.length === 0) return 0;

  const qq = clamp(q, 0, 1);
  const pos = (xs.length - 1) * qq;
  const base = Math.floor(pos);
  const rest = pos - base;

  const baseIdx = Math.min(Math.max(base, 0), xs.length - 1);
  const nextIdx = Math.min(baseIdx + 1, xs.length - 1);

  const fallback = xs[0] ?? 0;
  const a = xs[baseIdx] ?? fallback;
  const b = xs[nextIdx] ?? fallback;

  return a + rest * (b - a);
}

function robustMinMax(arr: number[]): MinMax {
  const xs = arr.filter(Number.isFinite).sort((a, b) => a - b);
  if (xs.length === 0) return { lo: 0, hi: 1 };

  const lo = quantile(xs, 0.01);
  const hi = quantile(xs, 0.99);

  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { lo: 0, hi: 1 };
  if (lo === hi) return { lo, hi: lo + 1 };
  if (lo > hi) return { lo: hi, hi: lo };
  return { lo, hi };
}

function shapColor01(t: number) {
  t = clamp(t, 0, 1);
  const r = Math.round(30 + t * (255 - 30));
  const g = Math.round(80 + t * (20 - 80));
  const b = Math.round(255 + t * (90 - 255));
  return `rgb(${r},${g},${b})`;
}

function rdBuR(v: number) {
  v = clamp(v, -1, 1);
  if (v < 0) {
    const t = v + 1; 
    const r = Math.round(255 * t);
    const g = Math.round(255 * t);
    const b = 255;
    return `rgb(${r},${g},${b})`;
  } else {
    const t = v; 
    const r = 255;
    const g = Math.round(255 * (1 - t));
    const b = Math.round(255 * (1 - t));
    return `rgb(${r},${g},${b})`;
  }
}

const PALETTE = [
  '#2563eb',
  '#7c3aed',
  '#059669',
  '#ea580c',
  '#db2777',
  '#0ea5e9',
  '#16a34a',
  '#f97316',
  '#a855f7',
  '#ef4444',
];

type MetricName = 'RMSE' | 'MAE' | 'R2';

function rmse(yTrue: number[], yPred: number[]) {
  const n = Math.min(yTrue.length, yPred.length);
  if (!n) return Infinity;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const yt = yTrue[i] ?? 0;
    const yp = yPred[i] ?? 0;
    const e = yp - yt;
    s += e * e;
  }
  return Math.sqrt(s / n);
}

function mae(yTrue: number[], yPred: number[]) {
  const n = Math.min(yTrue.length, yPred.length);
  if (!n) return Infinity;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const yt = yTrue[i] ?? 0;
    const yp = yPred[i] ?? 0;
    s += Math.abs(yp - yt);
  }
  return s / n;
}

function r2(yTrue: number[], yPred: number[]) {
  const n = Math.min(yTrue.length, yPred.length);
  if (!n) return -Infinity;

  let mean = 0;
  for (let i = 0; i < n; i++) mean += yTrue[i] ?? 0;
  mean /= n;

  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    const yt = yTrue[i] ?? 0;
    const yp = yPred[i] ?? 0;

    const d = yt - mean;
    ssTot += d * d;

    const e = yt - yp;
    ssRes += e * e;
  }

  return ssTot > 0 ? 1 - ssRes / ssTot : -Infinity;
}

function scoreByMetric(metric: MetricName, yTrue: number[], yPred: number[]) {
  if (metric === 'RMSE') return { name: 'RMSE', value: rmse(yTrue, yPred), better: 'min' as const };
  if (metric === 'MAE') return { name: 'MAE', value: mae(yTrue, yPred), better: 'min' as const };
  return { name: 'R2', value: r2(yTrue, yPred), better: 'max' as const };
}


function PlotSurface({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
        {subtitle ? <CardDescription>{subtitle}</CardDescription> : null}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}


function JsonDetails({
  title,
  obj,
  defaultOpen = false,
}: {
  title: string;
  obj: any;
  defaultOpen?: boolean;
}) {
  return (
    <details
      open={defaultOpen}
      className="rounded-md border border-gray-200 bg-white/80 p-4 dark:border-neutral-800 dark:bg-neutral-950/60"
    >
      <summary className="cursor-pointer font-medium">{title}</summary>
      <pre className="mt-3 text-xs whitespace-pre-wrap rounded-md border border-gray-200 bg-white/70 p-4 dark:border-neutral-800 dark:bg-neutral-950/60">
        {pretty(obj ?? {})}
      </pre>
    </details>
  );
}

function SvgBarChart({
  title,
  items,
  valueKey,
  labelKey,
  height = 280,
}: {
  title?: string;
  items: any[];
  valueKey: string;
  labelKey: string;
  height?: number;
}) {
  const width = 860;
  const padL = 190;
  const padR = 20;
  const padT = 24;
  const padB = 24;

  const vals = items.map((x) => Number(x?.[valueKey] ?? 0)).filter(Number.isFinite);
  const maxV = Math.max(1e-9, ...vals);

  const innerW = width - padL - padR;
  const rowH = Math.max(16, Math.floor((height - padT - padB) / Math.max(1, items.length)));

  return (
    <div className="w-full overflow-x-auto">
      {title ? <div className="mb-2 text-sm font-medium">{title}</div> : null}
      <div className="rounded-md border border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
        <svg width={width} height={height} className="rounded-md">
          <line x1={padL} y1={padT} x2={padL} y2={height - padB} stroke="currentColor" opacity={0.25} />
          {items.map((it, i) => {
            const v = Number(it?.[valueKey] ?? 0);
            const w = (clamp(v, 0, maxV) / maxV) * innerW;
            const y = padT + i * rowH;
            const label = String(it?.[labelKey] ?? '');
            return (
              <g key={i}>
                <text
                  x={padL - 10}
                  y={y + rowH * 0.7}
                  textAnchor="end"
                  fontSize="12"
                  fill="currentColor"
                  opacity={0.85}
                >
                  {label.length > 30 ? label.slice(0, 30) + '…' : label}
                </text>

                <rect
                  x={padL}
                  y={y + rowH * 0.2}
                  width={w}
                  height={Math.max(6, rowH * 0.6)}
                  rx={8}
                  ry={8}
                  fill="currentColor"
                  opacity={0.22}
                />
                <text x={padL + w + 8} y={y + rowH * 0.7} fontSize="12" fill="currentColor" opacity={0.6}>
                  {Number.isFinite(v) ? v.toFixed(4) : '—'}
                </text>

                <line x1={padL} y1={y + rowH} x2={width - padR} y2={y + rowH} stroke="currentColor" opacity={0.07} />
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

function SvgInteractiveLineSeries({
  title,
  series,
  height = 280,
  bestName,
}: {
  title?: string;
  series: { name: string; y: number[] }[];
  height?: number;
  bestName?: string | null;
}) {
  const width = 900;
  const padL = 56;
  const padR = 18;
  const padT = 18;
  const padB = 44;

  const all = series.flatMap((s) => s.y).filter(Number.isFinite);
  const minY = Math.min(...all, 0);
  const maxY = Math.max(...all, 1);

  const innerW = width - padL - padR;
  const innerH = height - padT - padB;

  const maxN = Math.max(2, ...series.map((s) => s.y.length));
  const xAt = (i: number) => padL + (i / Math.max(1, maxN - 1)) * innerW;
  const yAt = (v: number) => padT + (1 - (v - minY) / Math.max(1e-9, maxY - minY)) * innerH;

  const pathFor = (arr: number[]) => {
    const pts = arr.map((v, i) => `${xAt(i)},${yAt(v)}`);
    return pts.length ? `M ${pts.join(' L ')}` : '';
  };

  const [hoverX, setHoverX] = useState<number | null>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const t = clamp((x - padL) / innerW, 0, 1);
    const idx = Math.round(t * (maxN - 1));
    setHoverX(xAt(idx));
    setHoverIdx(idx);
  };

  const onLeave = () => {
    setHoverX(null);
    setHoverIdx(null);
  };

  return (
    <div className="w-full overflow-x-auto">
      {title ? <div className="mb-2 text-sm font-medium">{title}</div> : null}
      <div className="rounded-md border border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
        <svg width={width} height={height} onMouseMove={onMove} onMouseLeave={onLeave} className="rounded-md">
          {/* grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = padT + t * innerH;
            return <line key={t} x1={padL} y1={y} x2={width - padR} y2={y} stroke="currentColor" opacity={0.07} />;
          })}

          {/* axes */}
          <line x1={padL} y1={padT} x2={padL} y2={height - padB} stroke="currentColor" opacity={0.18} />
          <line x1={padL} y1={height - padB} x2={width - padR} y2={height - padB} stroke="currentColor" opacity={0.18} />

          {/* series */}
          {series.map((s, idx) => {
            const color = PALETTE[idx % PALETTE.length];
            const isBest = Boolean(bestName && s.name === bestName);
            return (
              <path
                key={s.name}
                d={pathFor(s.y)}
                fill="none"
                stroke={color}
                strokeWidth={isBest ? 2.8 : 2.0}
                opacity={isBest ? 1 : 0.85}
              />
            );
          })}

          {/* crosshair */}
          {hoverX !== null ? (
            <line x1={hoverX} y1={padT} x2={hoverX} y2={height - padB} stroke="currentColor" opacity={0.25} />
          ) : null}

          {/* tooltip */}
          {hoverX !== null && hoverIdx !== null ? (
            <g transform={`translate(${Math.min(hoverX + 12, width - 270)},${padT + 10})`}>
              <rect width={260} height={Math.min(160, 30 + series.length * 18)} rx={10} fill="black" opacity={0.65} />
              <text x={12} y={20} fontSize={12} fill="white" opacity={0.95}>
                index = {hoverIdx}
              </text>

              {series.map((s, i) => {
                const idx = typeof hoverIdx === 'number' ? hoverIdx : 0;
                const v = s.y[idx] ?? NaN; 
                const color = PALETTE[i % PALETTE.length];
                const isBest = Boolean(bestName && s.name === bestName);
                return (
                  <g key={s.name} transform={`translate(0,${30 + i * 18})`}>
                    <rect x={12} y={-10} width={10} height={10} rx={3} fill={color} />
                    <text x={28} y={0} fontSize={12} fill="white" opacity={0.95}>
                      {s.name}: {Number.isFinite(v) ? v.toFixed(5) : '—'} {isBest ? '★' : ''}
                    </text>
                  </g>
                );
              })}
            </g>
          ) : null}

          <g transform={`translate(${padL},${height - 14})`}>
            {series.map((s, idx) => {
              const color = PALETTE[idx % PALETTE.length];
              const isBest = Boolean(bestName && s.name === bestName);
              return (
                <g key={s.name} transform={`translate(${idx * 180},0)`}>
                  <rect x={0} y={-10} width={14} height={3} fill={color} opacity={0.95} />
                  <text x={20} y={-7} fontSize="12" fill="currentColor" opacity={0.8}>
                    {s.name}
                  </text>
                  {isBest ? (
                    <g transform="translate(150,-18)">
                      <Crown className="h-4 w-4" />
                    </g>
                  ) : null}
                </g>
              );
            })}
          </g>
        </svg>
      </div>
    </div>
  );
}

function ShapBeeswarm({
  featureNames,
  shapSigned,
  xFeat,
  topK = 12,
}: {
  featureNames: string[];
  shapSigned: number[][];
  xFeat: number[][];
  topK: number;
}) {
  const stats = useMemo(() => {
    const n = shapSigned?.length ?? 0;
    const F = featureNames?.length ?? 0;
    if (!n || !F) return { order: [] as number[], meanAbs: [] as number[] };

    const meanAbs = new Array(F).fill(0);
    for (let j = 0; j < F; j++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += Math.abs(Number(shapSigned[i]?.[j] ?? 0));
      meanAbs[j] = s / n;
    }
    const order = Array.from({ length: F }, (_, j) => j).sort((a, b) => meanAbs[b] - meanAbs[a]);
    return { order, meanAbs };
  }, [shapSigned, featureNames]);

  const idxs: number[] = stats.order.slice(0, Math.min(topK, stats.order.length));

  const width = 900;
  const rowH = 26;
  const height = 28 + idxs.length * rowH + 28;
  const padL = 230;
  const padR = 70; 
  const padT = 18;
  const padB = 22;

  const xAll = useMemo(() => {
    const xs: number[] = [];
    for (const j of idxs) {
      for (let i = 0; i < (shapSigned?.length ?? 0); i++) {
        const row = shapSigned[i];
        if (!row) continue;
        const v = Number(row[j]);
        if (Number.isFinite(v)) xs.push(v);
      }
    }
    return robustMinMax(xs);
  }, [idxs, shapSigned]);

  const innerW = width - padL - padR;

  type BeePoint = { x: number; y: number; c: string; r: number; sv: number; fv: number; feat: string };

  const points = useMemo(() => {
    const n = shapSigned?.length ?? 0;
    const out: BeePoint[] = [];
    if (!n || !idxs.length) return out;

    const bins = 44;
    const binStacks: Record<string, number> = {};

    const xToPx = (v: number) => padL + ((v - xAll.lo) / safeDen(xAll.hi, xAll.lo)) * innerW;

    for (let row = 0; row < idxs.length; row++) {
      const j0 = idxs[row];
      if (!Number.isFinite(j0)) continue;
      const j = j0 as number;

      const featVals: number[] = [];
      for (let i = 0; i < n; i++) {
        const featRow = xFeat[i];
        if (!featRow) continue;
        const cell = featRow[j];
        if (cell === undefined) continue;
        const fv = Number(cell);
        if (Number.isFinite(fv)) featVals.push(fv);
      }
      const mm = robustMinMax(featVals);

      for (let i = 0; i < n; i++) {
        const shapRow = shapSigned[i];
        const featRow = xFeat[i];
        if (!shapRow || !featRow) continue;

        const svCell = shapRow[j];
        const fvCell = featRow[j];
        if (svCell === undefined || fvCell === undefined) continue;

        const sv = Number(svCell);
        const fv = Number(fvCell);
        if (!Number.isFinite(sv) || !Number.isFinite(fv)) continue;

        const bx = Math.floor(((sv - xAll.lo) / safeDen(xAll.hi, xAll.lo)) * bins);
        const bin = clamp(bx, 0, bins - 1);
        const key = `${row}-${bin}`;
        const stack = (binStacks[key] ?? 0) + 1;
        binStacks[key] = stack;

        const jitter = ((stack % 11) - 5) * 1.55;
        const yBase = padT + row * rowH + rowH * 0.55;
        const y = yBase + jitter;

        const t = (fv - mm.lo) / safeDen(mm.hi, mm.lo);
        const color = shapColor01(t);

        out.push({
          x: xToPx(sv),
          y,
          c: color,
          r: 2.2,
          sv,
          fv,
          feat: featureNames[j] ?? `f${j}`,
        });
      }
    }

    return out;
  }, [idxs, shapSigned, xFeat, xAll.lo, xAll.hi, innerW, padL, padT, rowH, featureNames]);

  const xZero = padL + ((0 - xAll.lo) / safeDen(xAll.hi, xAll.lo)) * innerW;

  const [hoverP, setHoverP] = useState<BeePoint | null>(null);

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let best: BeePoint | null = null;
    let bestD = 1e18;


    for (const p of points) {
      const dx = p.x - mx;
      const dy = p.y - my;
      const d = dx * dx + dy * dy;
      if (d < bestD && d < 90) {
        bestD = d;
        best = p;
      }
    }
    setHoverP(best);
  };

  const onLeave = () => setHoverP(null);

  return (
    <div className="w-full overflow-x-auto">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="text-sm font-medium">SHAP summary (beeswarm)</div>
        <div className="text-xs text-neutral-500">Color = feature value (low→high)</div>
      </div>

      <div className="rounded-md border border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
        <svg width={width} height={height} className="rounded-md" onMouseMove={onMove} onMouseLeave={onLeave}>
          <line x1={xZero} y1={padT - 6} x2={xZero} y2={height - padB + 6} stroke="currentColor" opacity={0.25} />

          {idxs.map((j, row) => {
            const y = padT + row * rowH + rowH * 0.6;
            const label = featureNames[j] ?? `f${j}`;
            return (
              <g key={j}>
                <text x={padL - 10} y={y} textAnchor="end" fontSize="12" fill="currentColor" opacity={0.85}>
                  {label.length > 32 ? label.slice(0, 32) + '…' : label}
                </text>
                <line
                  x1={padL}
                  y1={padT + (row + 1) * rowH}
                  x2={width - padR}
                  y2={padT + (row + 1) * rowH}
                  stroke="currentColor"
                  opacity={0.06}
                />
              </g>
            );
          })}

          {points.map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r={p.r} fill={p.c} opacity={0.92} />
          ))}

          {/* x ticks */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const v = xAll.lo + t * (xAll.hi - xAll.lo);
            const x = padL + t * innerW;
            return (
              <g key={t}>
                <line x1={x} y1={height - padB} x2={x} y2={height - padB + 5} stroke="currentColor" opacity={0.25} />
                <text x={x} y={height - 6} textAnchor="middle" fontSize="11" fill="currentColor" opacity={0.6}>
                  {Number.isFinite(v) ? v.toFixed(2) : '—'}
                </text>
              </g>
            );
          })}

          <text x={padL + innerW / 2} y={height - 18} textAnchor="middle" fontSize="12" fill="currentColor" opacity={0.75}>
            SHAP value (impact on model output)
          </text>

          {/* colorbar */}
          <g transform={`translate(${width - 44},${padT})`}>
            {Array.from({ length: 90 }, (_, i) => i).map((i) => {
              const t = i / 89;
              return <rect key={i} x={0} y={i * 2.05} width={12} height={2.05} fill={shapColor01(1 - t)} />;
            })}
            <text x={16} y={10} fontSize="11" fill="currentColor" opacity={0.65}>
              High
            </text>
            <text x={16} y={90 * 2.05 - 2} fontSize="11" fill="currentColor" opacity={0.65}>
              Low
            </text>
          </g>

          {/* hover tooltip */}
          {hoverP ? (
            <g transform={`translate(${Math.min(hoverP.x + 10, width - 250)},${Math.max(hoverP.y - 40, 20)})`}>
              <rect width={240} height={72} rx={10} fill="black" opacity={0.65} />
              <text x={12} y={18} fontSize={12} fill="white" opacity={0.95}>
                {hoverP.feat}
              </text>
              <text x={12} y={38} fontSize={12} fill="white" opacity={0.95}>
                SHAP: {hoverP.sv.toFixed(6)}
              </text>
              <text x={12} y={58} fontSize={12} fill="white" opacity={0.95}>
                Feature: {hoverP.fv.toFixed(6)}
              </text>
            </g>
          ) : null}
        </svg>
      </div>
    </div>
  );
}


function DeepHeatmap({
  heatmapLast,
  featureNames,
}: {
  heatmapLast: number[][];
  featureNames: string[];
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverCell, setHoverCell] = useState<{ i: number; j: number; v: number } | null>(null);
  const [mm, setMm] = useState<MinMax>({ lo: 0, hi: 1 });

  const w = 820;
  const h = 320;
  const padL = 200;
  const padT = 12;
  const padR = 18;
  const padB = 32;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const L = heatmapLast?.length ?? 0;
    const F = heatmapLast?.[0]?.length ?? 0;
    if (!L || !F) return;

    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const vals: number[] = [];
    for (let i = 0; i < L; i++) {
      for (let j = 0; j < F; j++) {
        const v = Number(heatmapLast[i]?.[j]);
        if (Number.isFinite(v)) vals.push(v);
      }
    }
    const localMm = robustMinMax(vals);
    setMm(localMm);

    const innerW = w - padL - padR;
    const innerH = h - padT - padB;

    const cellW = innerW / Math.max(1, L);
    const cellH = innerH / Math.max(1, F);

    ctx.clearRect(0, 0, w, h);

    for (let j = 0; j < F; j++) {
      for (let i = 0; i < L; i++) {
        const v = Number(heatmapLast[i]?.[j] ?? 0);
        const t = (v - localMm.lo) / safeDen(localMm.hi, localMm.lo);
        const signed = t * 2 - 1;
        ctx.fillStyle = rdBuR(signed);
        const x = padL + i * cellW;
        const y = padT + j * cellH;
        ctx.fillRect(x, y, Math.ceil(cellW), Math.ceil(cellH));
      }
    }
  }, [heatmapLast, featureNames]);

  const F = heatmapLast?.[0]?.length ?? 0;
  const L = heatmapLast?.length ?? 0;

  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  const cellW = innerW / Math.max(1, L);
  const cellH = innerH / Math.max(1, F);

  return (
    <div className="w-full overflow-x-auto">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-sm font-medium">Deep SHAP heatmap (last window)</div>
        <div className="text-xs text-neutral-500">time steps × features</div>
      </div>

      <div className="rounded-md border border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
        <div
          className="relative w-[820px]"
          onMouseMove={(e) => {
            const canvas = canvasRef.current;
            if (!canvas) return;

            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const i = Math.floor((x - padL) / cellW);
            const j = Math.floor((y - padT) / cellH);

            if (i < 0 || i >= L || j < 0 || j >= F) {
              setHoverCell(null);
              return;
            }

            const v = Number(heatmapLast[i]?.[j] ?? 0);
            if (!Number.isFinite(v)) {
              setHoverCell(null);
              return;
            }

            setHoverCell({ i, j, v });
          }}
          onMouseLeave={() => setHoverCell(null)}
        >
          <canvas ref={canvasRef} className="rounded-md" />

          {/* feature labels on left */}
          <div className="absolute left-0 top-0 h-full w-[190px] p-2 text-xs text-neutral-700 dark:text-neutral-300 overflow-hidden">
            {Array.from({ length: Math.min(F, 14) }, (_, j) => (
              <div key={j} className="truncate" style={{ lineHeight: '22px' }}>
                {featureNames?.[j] ?? `f${j}`}
              </div>
            ))}
            {F > 14 ? <div className="text-neutral-500 mt-1">… +{F - 14} more</div> : null}
          </div>

          {/* hover tooltip */}
          {hoverCell ? (
            <div className="absolute right-2 top-2 rounded-md bg-black/70 px-3 py-2 text-xs text-white">
              <div>
                <b>t</b>: {hoverCell.i}
              </div>
              <div>
                <b>feature</b>: {featureNames?.[hoverCell.j] ?? `f${hoverCell.j}`}
              </div>
              <div>
                <b>value</b>: {hoverCell.v.toFixed(6)}
              </div>
            </div>
          ) : null}
        </div>

        {/* legend */}
        <div className="px-4 pb-4 pt-2">
          <div className="flex items-center gap-3 text-xs text-neutral-500">
            <span>{mm.lo.toFixed(2)}</span>
            <div
              className="h-2 w-56 rounded"
              style={{
                background: 'linear-gradient(90deg, rgb(0,0,255), rgb(255,255,255), rgb(255,0,0))',
              }}
            />
            <span>{mm.hi.toFixed(2)}</span>
            <span className="ml-2">hover cells for exact values</span>
          </div>
        </div>
      </div>
    </div>
  );
}


export default function ModelViewPage() {
  const params = useParams<{ modelId: string }>();
  const modelId = params.modelId;
  const router = useRouter();

  const [model, setModel] = useState<any | null>(null);
  const [explain, setExplain] = useState<any | null>(null);

  const [topFeatures, setTopFeatures] = useState<number>(12);

  const [loading, setLoading] = useState({ model: true, explain: false });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        setError(null);
        setLoading((s) => ({ ...s, model: true }));
        const m = await authorizedFetch(`/api/v1/models/${modelId}`);
        if (!mounted) return;
        setModel(m);
      } catch (e: any) {
        if (!mounted) return;
        setError(e?.message ?? 'Failed to load model.');
      } finally {
        if (mounted) setLoading((s) => ({ ...s, model: false }));
      }
    })();

    return () => {
      mounted = false;
    };
  }, [modelId]);

  async function handleExplain() {
    try {
      setError(null);
      setLoading((s) => ({ ...s, explain: true }));
      const x = await authorizedFetch(`/api/v1/models/${modelId}/explainx`);
      setExplain(x);
    } catch (e: any) {
      setError(e?.message ?? 'Failed to generate explanation.');
    } finally {
      setLoading((s) => ({ ...s, explain: false }));
    }
  }

  const datasetId = model?.dataset_id ?? null;

  const plotPayload = useMemo(() => {
    return model?.model_plots ?? model?.training_report?.plot_payload ?? model?.training_report?.plot_payload ?? {};
  }, [model]);

  const explainArrays = explain?.arrays ?? null;
  const explainReport = explain?.explain_report ?? null;
  const explainKind = explain?.kind ?? '—';
  const explainStatus = explain?.explain_status ?? model?.explain_status ?? 'none';

  const featureNames: string[] = useMemo(() => {
    const fn = explainArrays?.feature_names ?? explainReport?.feature_names;
    if (Array.isArray(fn)) return fn.map(String);
    return [];
  }, [explainArrays, explainReport]);

  const shapSigned: number[][] = useMemo(() => {
    const a = explainArrays?.shap_feat ?? explainArrays?.shap_values;
    return Array.isArray(a) ? a : [];
  }, [explainArrays]);

  const xFeat: number[][] = useMemo(() => {
    const a = explainArrays?.x_feat ?? explainArrays?.x_values;
    return Array.isArray(a) ? a : [];
  }, [explainArrays]);

  const heatmapLast: number[][] = useMemo(() => {
    const a = explainArrays?.heatmap_last;
    return Array.isArray(a) ? a : [];
  }, [explainArrays]);

  const tabularBench = plotPayload?.tabular ?? null;
  const seqBench = plotPayload?.seq ?? null;
  const wf = plotPayload?.walk_forward ?? null;

  const primaryMetric: MetricName = (model?.training_report?.primary_metric ??
    model?.primary_metric ??
    'RMSE') as MetricName;

  const tabularSeries = useMemo(() => {
    if (!tabularBench?.y_test || !tabularBench?.models) return null;
    const yTrue: number[] = tabularBench.y_test;
    const models = tabularBench.models;
    const names = Object.keys(models ?? {});
    return [
      { name: 'True', y: yTrue.slice(0, 400) },
      ...names.slice(0, 3).map((n) => ({ name: n, y: (models[n]?.y_pred ?? []).slice(0, 400) })),
    ];
  }, [tabularBench]);

  const seqSeries = useMemo(() => {
    if (!seqBench?.y_val || !seqBench?.models) return null;
    const yTrue: number[] = seqBench.y_val;
    const models = seqBench.models;
    const names = Object.keys(models ?? {});
    return [
      { name: 'True', y: yTrue.slice(0, 400) },
      ...names.slice(0, 2).map((n) => ({ name: n, y: (models[n]?.y_pred ?? []).slice(0, 400) })),
    ];
  }, [seqBench]);

  const wfSeries = useMemo(() => {
    if (!wf || typeof wf !== 'object') return null;
    const names = Object.keys(wf);
    if (!names.length) return null;

    const series: { name: string; y: number[] }[] = [];
    for (const n of names.slice(0, 4)) {
      const folds = (wf as any)[n]?.folds ?? [];
      const metricKey = model?.training_report?.primary_metric ?? model?.primary_metric ?? 'RMSE';
      const ys = (folds as any[]).map((r) => Number(r?.[metricKey] ?? r?.RMSE ?? 0)).filter(Number.isFinite);
      if (ys.length) series.push({ name: n, y: ys });
    }
    return series.length ? series : null;
  }, [wf, model]);

  const bestTabular = useMemo(() => {
    if (!tabularBench?.y_test || !tabularBench?.models) return null;
    const yTrue: number[] = tabularBench.y_test;
    const names = Object.keys(tabularBench.models ?? {});
    let best: { name: string; v: number; better: 'min' | 'max' } | null = null;

    for (const n of names) {
      const yPred: number[] = tabularBench.models[n]?.y_pred ?? [];
      const s = scoreByMetric(primaryMetric, yTrue, yPred);
      if (!best) best = { name: n, v: s.value, better: s.better };
      else {
        const isBetter = s.better === 'min' ? s.value < best.v : s.value > best.v;
        if (isBetter) best = { name: n, v: s.value, better: s.better };
      }
    }
    return best?.name ?? null;
  }, [tabularBench, primaryMetric]);

  const bestSeq = useMemo(() => {
    if (!seqBench?.y_val || !seqBench?.models) return null;
    const yTrue: number[] = seqBench.y_val;
    const names = Object.keys(seqBench.models ?? {});
    let best: { name: string; v: number; better: 'min' | 'max' } | null = null;

    for (const n of names) {
      const yPred: number[] = seqBench.models[n]?.y_pred ?? [];
      const s = scoreByMetric(primaryMetric, yTrue, yPred);
      if (!best) best = { name: n, v: s.value, better: s.better };
      else {
        const isBetter = s.better === 'min' ? s.value < best.v : s.value > best.v;
        if (isBetter) best = { name: n, v: s.value, better: s.better };
      }
    }
    return best?.name ?? null;
  }, [seqBench, primaryMetric]);

  const systemHealth = useMemo(() => {
    const hasExplain = Boolean(explain);
    return [
      { label: 'Model', value: model ? 'Loaded' : '—' },
      { label: 'ExplainX', value: hasExplain ? 'Ready' : 'Not loaded' },
      { label: 'Plots', value: tabularSeries || seqSeries || wfSeries ? 'Available' : '—' },
      { label: 'Status', value: String(model?.status ?? '—') },
    ];
  }, [model, explain, tabularSeries, seqSeries, wfSeries]);

  const essentials = useMemo(() => {
    return [
      { label: 'Algorithm', value: String(model?.algorithm ?? '—') },
      { label: 'Primary metric', value: String(primaryMetric ?? '—') },
      { label: 'Target', value: String(model?.target_column ?? '—') },
      { label: 'Horizon', value: String(model?.forecast_horizon ?? '—') },
      { label: 'Created', value: model?.created_at ? new Date(model.created_at).toLocaleString() : '—' },
      { label: 'Explain status', value: String(explainStatus ?? '—') },
    ];
  }, [model, primaryMetric, explainStatus]);

  return (
    <div className="relative min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-50">
      {/* Background grid */}
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_60%_at_50%_0%,#000_70%,transparent_110%)] dark:bg-[linear-gradient(to_right,#262626_1px,transparent_1px),linear-gradient(to_bottom,#262626_1px,transparent_1px)]" />

      <div className="relative mx-auto max-w-7xl px-6 py-10 space-y-10">
        {/* Hero */}
        <header className="space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="space-y-4">
              <div className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-white/80 px-4 py-2 text-sm shadow-sm backdrop-blur dark:border-neutral-800 dark:bg-neutral-900/80">
                <Sparkles className="h-3 w-3 text-blue-600 dark:text-cyan-400" />
                <span className="font-medium text-gray-900 dark:text-neutral-50">ExplainX</span>
                <span className="text-gray-600 dark:text-neutral-400">Model inspection & interpretability</span>
              </div>

              <div>
                <h1 className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-600 bg-clip-text text-4xl font-bold tracking-tight text-transparent md:text-5xl dark:from-neutral-50 dark:via-neutral-200 dark:to-neutral-400">
                  Model Details
                </h1>
                <p className="mt-3 max-w-2xl text-neutral-600 dark:text-neutral-400">
                  Inspect training, evaluation, and ExplainX artifacts — plots are rendered from saved arrays.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Button variant="ghost" onClick={() => router.back()} className="rounded-full">
                <ChevronLeft className="mr-2 h-4 w-4" />
                Back
              </Button>

              {datasetId ? (
                <Link
                  href={`/home/predictions?modelId=${encodeURIComponent(modelId)}&datasetId=${encodeURIComponent(
                    datasetId
                  )}`}
                >
                  <Button className="rounded-full">
                    <LineChart className="mr-2 h-4 w-4" />
                    Predict
                  </Button>
                </Link>
              ) : null}

              <Button variant="outline" className="rounded-full" onClick={handleExplain} disabled={loading.explain}>
                <Wand2 className="mr-2 h-4 w-4" />
                {loading.explain ? 'Generating…' : 'Explain'}
              </Button>
            </div>
          </div>
        </header>

        {/* Error */}
        {error ? (
          <div className="rounded-md border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-900/20 dark:text-red-200">
            {error}
          </div>
        ) : null}

        {/* Loading / Missing */}
        {loading.model ? (
          <Card className="border-gray-200 dark:border-neutral-800">
            <CardContent className="p-6 text-sm text-neutral-500">Loading model…</CardContent>
          </Card>
        ) : !model ? (
          <Card className="border-dashed border-gray-200 bg-white/70 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/60">
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Info className="mb-4 h-12 w-12 text-neutral-400" />
              <p className="text-neutral-600 dark:text-neutral-400">Model not found.</p>
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Quick health cards */}
            <section className="space-y-4">
              <div className="flex items-center justify-between gap-3">
                <h2 className="text-xl font-semibold tracking-tight">Overview</h2>
                <span className="rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300">
                  Live overview
                </span>
              </div>

              <div className="grid gap-4 md:grid-cols-4">
                {systemHealth.map((m) => (
                  <Card
                    key={m.label}
                    className="border-gray-200 bg-gradient-to-br from-white to-gray-50/80 dark:border-neutral-800 dark:from-neutral-900 dark:to-neutral-950"
                  >
                    <CardHeader className="pb-3">
                      <CardDescription>{m.label}</CardDescription>
                      <CardTitle className="mt-1 flex items-center gap-2 text-2xl">
                        {m.value}
                        <CheckCircle className="h-5 w-5 text-emerald-500" />
                      </CardTitle>
                    </CardHeader>
                  </Card>
                ))}
              </div>
            </section>

            {/* Essentials */}
            <section className="grid gap-6 md:grid-cols-3">
              <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80 md:col-span-2">
                <CardHeader className="pb-2">
                  <CardDescription>Model essentials</CardDescription>
                  <CardTitle className="text-lg">{model.model_name ?? model.algorithm ?? '—'}</CardTitle>
                </CardHeader>
                <CardContent className="grid gap-3 sm:grid-cols-2 text-sm text-neutral-700 dark:text-neutral-300">
                  {essentials.map((x) => (
                    <div key={x.label} className="flex items-center justify-between gap-3 rounded-md border border-neutral-200/60 bg-white/70 px-3 py-2 dark:border-neutral-800/60 dark:bg-neutral-950/60">
                      <div className="text-xs text-neutral-500">{x.label}</div>
                      <div className="font-medium truncate max-w-[240px] text-right">{x.value}</div>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
                <CardHeader className="pb-2">
                  <CardDescription>Status</CardDescription>
                  <CardTitle className="text-lg">{String(model.status ?? 'ready')}</CardTitle>
                </CardHeader>
                <CardContent className="text-sm text-neutral-600 dark:text-neutral-400 space-y-2">
                  <div className="flex items-center gap-2">
                    <Badge variant="default">{model.status ?? 'ready'}</Badge>
                    <span className="text-xs">ID: {String(model.id ?? '—')}</span>
                  </div>
                  <div className="text-xs">ExplainX: {String(explainKind)} • {String(explainStatus)}</div>
                </CardContent>
              </Card>
            </section>

            {/* JSON reports in dropdowns */}
            <section className="grid gap-6 md:grid-cols-2">
              <JsonDetails title="Training report (JSON)" obj={model.training_report ?? {}} />
              <JsonDetails title="Evaluation report (JSON)" obj={model.evaluation_report ?? {}} />
            </section>

            {/* TRAINING plots */}
            <section className="space-y-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-xl font-semibold tracking-tight">Training plots</h2>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">Benchmarks & training curves (if present).</p>
                </div>
                <div className="flex items-center gap-2 text-xs text-neutral-500">
                  <Activity className="h-4 w-4" />
                  training
                </div>
              </div>

              {seqSeries ? (
                <PlotSurface title="Sequence benchmark — True vs Pred" subtitle={`Primary metric: ${primaryMetric} (best model highlighted)`}>
                  <SvgInteractiveLineSeries title={undefined} series={seqSeries} bestName={bestSeq} />
                </PlotSurface>
              ) : (
                <Card className="border-dashed border-gray-200 bg-white/70 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/60">
                  <CardContent className="flex flex-col items-center justify-center py-10">
                    <Target className="mb-4 h-10 w-10 text-neutral-400" />
                    <p className="text-neutral-600 dark:text-neutral-400">No training plot series found.</p>
                  </CardContent>
                </Card>
              )}
            </section>

            {/* EVALUATION plots */}
            <section className="space-y-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-xl font-semibold tracking-tight">Evaluation plots</h2>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    Walk-forward evaluation & holdout comparisons.
                  </p>
                </div>
                <div className="flex items-center gap-2 text-xs text-neutral-500">
                  <Target className="h-4 w-4" />
                  evaluation
                </div>
              </div>

              <div className="grid gap-6">
                {tabularSeries ? (
                  <PlotSurface
                    title="Tabular benchmark — True vs Pred"
                    subtitle={`First 400 points. Primary metric: ${primaryMetric} (best model crowned)`}
                  >
                    <SvgInteractiveLineSeries series={tabularSeries} bestName={bestTabular} />
                  </PlotSurface>
                ) : null}

                {wfSeries ? (
                  <PlotSurface title="Walk-forward — metric per fold" subtitle="Fold-by-fold metric values (hover for exact numbers).">
                    <SvgInteractiveLineSeries series={wfSeries} bestName={null} />
                  </PlotSurface>
                ) : null}

                {!tabularSeries && !wfSeries ? (
                  <Card className="border-dashed border-gray-200 bg-white/70 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/60">
                    <CardContent className="flex flex-col items-center justify-center py-10">
                      <Target className="mb-4 h-10 w-10 text-neutral-400" />
                      <p className="text-neutral-600 dark:text-neutral-400">
                        No evaluation plot payload found in the model row.
                      </p>
                    </CardContent>
                  </Card>
                ) : null}
              </div>

              {/* Plot payload JSON (dropdown) */}
              <JsonDetails title="Plot payload (JSON)" obj={plotPayload ?? {}} />
            </section>

            {/* ExplainX */}
            <section className="space-y-4 pb-10">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-xl font-semibold tracking-tight">ExplainX</h2>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400">
                    From <code>/api/v1/models/:id/explainx</code>. Arrays are used to render plots.
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <div className="text-sm text-neutral-600 dark:text-neutral-400">Top features</div>
                  <select
                    value={topFeatures}
                    onChange={(e) => setTopFeatures(Number(e.target.value))}
                    className="h-9 rounded-full border border-neutral-200 bg-white px-4 text-sm shadow-sm dark:border-neutral-800 dark:bg-neutral-950"
                  >
                    {[8, 12, 15, 20].map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {!explain ? (
                <Card className="border-dashed border-gray-200 bg-white/70 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/60">
                  <CardContent className="flex flex-col items-center justify-center py-10">
                    <Wand2 className="mb-4 h-12 w-12 text-neutral-400" />
                    <p className="text-neutral-600 dark:text-neutral-400">
                      Click <b>Explain</b> to load ExplainX (SHAP + permutation importance).
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid gap-6">
                  {Array.isArray(explainReport?.global_importance) && explainReport.global_importance.length ? (
                    <PlotSurface title="Global SHAP importance" subtitle="Mean(|SHAP|) per feature. Top-K features.">
                      <SvgBarChart
                        items={explainReport.global_importance.slice(0, topFeatures)}
                        valueKey="mean_abs_shap"
                        labelKey="feature"
                        height={340}
                      />
                    </PlotSurface>
                  ) : (
                    <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
                      <CardContent className="p-6 text-sm text-neutral-600 dark:text-neutral-400">No global_importance found.</CardContent>
                    </Card>
                  )}

                  {featureNames.length && shapSigned.length && xFeat.length ? (
                    <PlotSurface title="SHAP summary (beeswarm)" subtitle="Hover points for exact SHAP + feature value.">
                      <ShapBeeswarm featureNames={featureNames} shapSigned={shapSigned} xFeat={xFeat} topK={topFeatures} />
                    </PlotSurface>
                  ) : (
                    <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
                      <CardContent className="p-6 text-sm text-neutral-600 dark:text-neutral-400">
                        No SHAP arrays found (expected arrays.shap_feat/shap_values and arrays.x_feat/x_values).
                      </CardContent>
                    </Card>
                  )}

                  {Array.isArray(heatmapLast) && heatmapLast.length ? (
                    <PlotSurface title="Deep SHAP heatmap" subtitle="Hover cells for exact values.">
                      <DeepHeatmap heatmapLast={heatmapLast} featureNames={featureNames} />
                    </PlotSurface>
                  ) : (
                    <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
                      <CardContent className="p-6 text-sm text-neutral-600 dark:text-neutral-400">
                        No deep heatmap found (expected arrays.heatmap_last for deep models).
                      </CardContent>
                    </Card>
                  )}

                  {Array.isArray(explainReport?.permutation_importance) && explainReport.permutation_importance.length ? (
                    <PlotSurface title="Permutation importance" subtitle="ΔMSE (importance_mean) per feature.">
                      <SvgBarChart
                        items={explainReport.permutation_importance.slice(0, topFeatures)}
                        valueKey="importance_mean"
                        labelKey="feature"
                        height={340}
                      />
                    </PlotSurface>
                  ) : (
                    <Card className="border-gray-200 bg-white/90 backdrop-blur dark:border-neutral-800 dark:bg-neutral-950/80">
                      <CardContent className="p-6 text-sm text-neutral-600 dark:text-neutral-400">
                        No permutation_importance found.
                      </CardContent>
                    </Card>
                  )}

                  <JsonDetails title="ExplainX payload (JSON)" obj={explain ?? {}} />
                </div>
              )}
            </section>
          </>
        )}
      </div>
    </div>
  );
}
