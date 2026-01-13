'use client';

import Image from "next/image";
import Link from "next/link";

import {
  ArrowRightIcon,
  Database,
  Sparkles,
  Bot,
  Gauge,
  LineChart,
  Target,
  Zap,
} from "lucide-react";

export default function Home() {
  return (
    <div className="relative overflow-hidden bg-white">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)]" />

      <div className="relative mt-4 flex flex-col space-y-32 py-14">
        <div className="container mx-auto px-4">
          <div className="flex flex-col items-center space-y-12 text-center">
            <div className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-white px-4 py-2 shadow-sm">
              <Sparkles className="h-3 w-3 text-blue-600" />
              <span className="text-sm font-medium text-gray-900">Data & AI</span>
              <span className="text-sm text-gray-600">The unified predictive platform</span>
            </div>

            <h1 className="max-w-4xl bg-gradient-to-br from-gray-900 via-gray-800 to-gray-600 bg-clip-text text-6xl font-bold leading-tight tracking-tight text-transparent md:text-7xl">
              <span className="block">From raw data to</span>
              <span className="block">actionable prediction</span>
            </h1>

            <p className="max-w-2xl text-lg leading-relaxed text-gray-600">
              Discover patterns, forecast trends, and optimize outcomes — all in
              one intelligent workspace. Transform your entire predictive workflow
              into a seamless process, from exploration to deployment.
            </p>

            <MainCallToActionButton />
          </div>
        </div>

        <div className="container mx-auto px-4">
          <div className="space-y-8">
            <div className="space-y-4 text-center">
              <div className="inline-flex items-center gap-3 rounded-full border border-gray-200 bg-white px-4 py-2 shadow-sm">
                <Database className="h-4 w-4 text-blue-600" />
                <span className="text-sm font-medium text-gray-900">DataForge</span>
              </div>
              <h2 className="text-4xl font-bold tracking-tight text-gray-900 md:text-5xl">
                Transforming Raw Data into Knowledge
              </h2>
              <p className="mx-auto max-w-2xl text-xl text-gray-600">
                Prepare, explore, and validate your datasets before modeling.
              </p>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <div className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-8 transition-all hover:shadow-lg hover:shadow-blue-500/10 md:col-span-2">
                <Zap className="mb-4 h-8 w-8 text-blue-600" />
                <h3 className="mb-2 text-xl font-semibold text-gray-900">
                  End-to-end preprocessing
                </h3>
                <p className="text-gray-600">
                  Cleaning, feature engineering, scaling, encoding, validation — everything your data needs to be model-ready.
                </p>
              </div>

              <div className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-8 transition-all hover:shadow-lg hover:shadow-blue-500/10">
                <Sparkles className="mb-4 h-8 w-8 text-cyan-600" />
                <h3 className="mb-2 text-xl font-semibold text-gray-900">
                  Understand your data
                </h3>
                <p className="text-gray-600">
                  Visualize distributions, correlations, missingness, and trends to build intuition before training.
                </p>
              </div>

              <div className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-8 transition-all hover:shadow-lg hover:shadow-blue-500/10">
                <Target className="mb-4 h-8 w-8 text-blue-600" />
                <h3 className="mb-2 text-xl font-semibold text-gray-900">
                  Consistent & structured
                </h3>
                <p className="text-gray-600">
                  Standardized pipelines ensure reproducible, trustworthy datasets across projects and teams.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="container mx-auto px-4">
          <div className="space-y-8">
            <div className="space-y-4 text-center">
              <div className="inline-flex items-center gap-3 rounded-full border border-gray-200 bg-white px-4 py-2 shadow-sm">
                <Bot className="h-4 w-4 text-cyan-600" />
                <span className="text-sm font-medium text-gray-900">AutoML</span>
              </div>
              <h2 className="text-4xl font-bold tracking-tight text-gray-900 md:text-5xl">
                Smarter Model Training
              </h2>
              <p className="mx-auto max-w-2xl text-xl text-gray-600">
                Test, compare, and tune multiple algorithms automatically.
              </p>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <div className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-8 transition-all hover:shadow-lg hover:shadow-cyan-500/10 md:col-span-2">
                <Sparkles className="mb-4 h-8 w-8 text-cyan-600" />
                <h3 className="mb-2 text-xl font-semibold text-gray-900">
                  One click to best model
                </h3>
                <p className="text-gray-600">
                  AutoML explores multiple models and hyperparameters to surface the most accurate candidate.
                </p>
              </div>

              <div className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-8 transition-all hover:shadow-lg hover:shadow-cyan-500/10">
                <Gauge className="mb-4 h-8 w-8 text-blue-600" />
                <h3 className="mb-2 text-xl font-semibold text-gray-900">
                  Progress & metrics
                </h3>
                <p className="text-gray-600">
                  Monitor training runs, compare validation scores, and review clear visual summaries.
                </p>
              </div>

              <div className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-8 transition-all hover:shadow-lg hover:shadow-cyan-500/10">
                <Zap className="mb-4 h-8 w-8 text-cyan-600" />
                <h3 className="mb-2 text-xl font-semibold text-gray-900">
                  No boilerplate
                </h3>
                <p className="text-gray-600">
                  Skip manual model wiring — focus on outcomes, not plumbing.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="container mx-auto px-4">
          <div className="space-y-8">
            <div className="space-y-4 text-center">
              <div className="inline-flex items-center gap-3 rounded-full border border-gray-200 bg-white px-4 py-2 shadow-sm">
                <Gauge className="h-4 w-4 text-blue-600" />
                <span className="text-sm font-medium text-gray-900">Monitoring</span>
              </div>
              <h2 className="text-4xl font-bold tracking-tight text-gray-900 md:text-5xl">
                Model Monitoring & Explainability
              </h2>
              <p className="mx-auto max-w-2xl text-xl text-gray-600">
                Trust, trace, and improve your models continuously.
              </p>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <div className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-8 transition-all hover:shadow-lg hover:shadow-blue-500/10 md:col-span-2">
                <LineChart className="mb-4 h-8 w-8 text-blue-600" />
                <h3 className="mb-2 text-xl font-semibold text-gray-900">
                  Error analysis
                </h3>
                <p className="text-gray-600">
                  Drill into residuals, drift, and stability to spot weak signals and biased segments.
                </p>
              </div>

              <div className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-8 transition-all hover:shadow-lg hover:shadow-blue-500/10">
                <Sparkles className="mb-4 h-8 w-8 text-cyan-600" />
                <h3 className="mb-2 text-xl font-semibold text-gray-900">
                  Explainability
                </h3>
                <p className="text-gray-600">
                  Understand feature importance and how each variable influences predictions.
                </p>
              </div>

              <div className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-8 transition-all hover:shadow-lg hover:shadow-blue-500/10">
                <Database className="mb-4 h-8 w-8 text-blue-600" />
                <h3 className="mb-2 text-xl font-semibold text-gray-900">
                  Lifecycle tracking
                </h3>
                <p className="text-gray-600">
                  Version models, compare runs, and keep a transparent audit trail.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="container mx-auto px-4">
          <div className="space-y-8">
            <div className="space-y-4 text-center">
              <div className="inline-flex items-center gap-3 rounded-full border border-gray-200 bg-white px-4 py-2 shadow-sm">
                <LineChart className="h-4 w-4 text-cyan-600" />
                <span className="text-sm font-medium text-gray-900">Forecasting</span>
              </div>
              <h2 className="text-4xl font-bold tracking-tight text-gray-900 md:text-5xl">
                Forecasting & Real-Time Prediction
              </h2>
              <p className="mx-auto max-w-2xl text-xl text-gray-600">
                Predict at any horizon and validate against reality.
              </p>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <div className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-8 transition-all hover:shadow-lg hover:shadow-cyan-500/10 md:col-span-2">
                <Target className="mb-4 h-8 w-8 text-cyan-600" />
                <h3 className="mb-2 text-xl font-semibold text-gray-900">
                  Flexible horizons
                </h3>
                <p className="text-gray-600">
                  Generate forecasts instantly at the horizon you choose — short- or long-term.
                </p>
              </div>

              <div className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-8 transition-all hover:shadow-lg hover:shadow-cyan-500/10">
                <Zap className="mb-4 h-8 w-8 text-blue-600" />
                <h3 className="mb-2 text-xl font-semibold text-gray-900">
                  Batch or real-time
                </h3>
                <p className="text-gray-600">
                  Upload new inputs on demand or stream events for live predictions.
                </p>
              </div>

              <div className="group relative overflow-hidden rounded-2xl border border-gray-200 bg-gradient-to-br from-white to-gray-50/50 p-8 transition-all hover:shadow-lg hover:shadow-cyan-500/10">
                <LineChart className="mb-4 h-8 w-8 text-cyan-600" />
                <h3 className="mb-2 text-xl font-semibold text-gray-900">
                  Track outcomes
                </h3>
                <p className="text-gray-600">
                  Compare predicted vs. actuals and measure performance over time.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="container mx-auto px-4">
          <div className="relative overflow-hidden rounded-3xl border border-gray-200 bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 p-12 text-center shadow-2xl md:p-16">
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#ffffff0a_1px,transparent_1px),linear-gradient(to_bottom,#ffffff0a_1px,transparent_1px)] bg-[size:4rem_4rem]" />
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 via-transparent to-cyan-500/10" />

            <div className="relative">
              <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-500 shadow-lg">
                <Target className="h-8 w-8 text-white" />
              </div>
              <h2 className="text-4xl font-bold text-white md:text-5xl">
                Ready to unlock your data's potential?
              </h2>
              <p className="mx-auto mb-8 mt-4 max-w-2xl text-lg text-gray-300">
                Stop waiting months for insights. Start building and deploying your
                first predictive model today.
              </p>
              <MainCallToActionButton variant="dark" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MainCallToActionButton({ variant = "light" }: { variant?: "light" | "dark" }) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-4">
      <Link href="/auth/sign-up">
        <button className="group relative inline-flex items-center gap-2 overflow-hidden rounded-full bg-gradient-to-r from-blue-600 to-cyan-600 px-8 py-4 font-semibold text-white shadow-lg transition-all hover:shadow-xl hover:shadow-blue-500/50">
          <span className="relative z-10">Start Building Now</span>
          <ArrowRightIcon className="relative z-10 h-4 w-4 transition-transform group-hover:translate-x-1" />
          <div className="absolute inset-0 bg-gradient-to-r from-blue-700 to-cyan-700 opacity-0 transition-opacity group-hover:opacity-100" />
        </button>
      </Link>
    </div>
  );
}
