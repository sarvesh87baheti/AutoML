"use client"

import { useEffect, useMemo, useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, Download, Share2 } from "lucide-react"
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
} from "recharts"

// ---- Fetch Latest Results ----
type ResultsPayload = {
  dataset: string
  problem_type?: string
  results: Record<string, any>
}

function useLatestResults() {
  const [data, setData] = useState<ResultsPayload | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let mounted = true
    ;(async () => {
      try {
        const res = await fetch("/api/results", { cache: "no-store" })
        const json = await res.json()
        if (!res.ok) throw new Error(json?.error || "Failed to fetch results")
        if (mounted) setData(json)
      } catch (e: any) {
        if (mounted) setError(e?.message)
      } finally {
        if (mounted) setLoading(false)
      }
    })()
    return () => {
      mounted = false
    }
  }, [])

  return { data, loading, error }
}

export default function MetricsPage() {
  const { data, loading, error } = useLatestResults()

  // ---- Computed values with hooks BEFORE render return ----

  const modelMetrics = useMemo(() => {
    const arr: { name: string; metrics: any }[] = []
    const r = data?.results || {}
    for (const [name, value] of Object.entries<any>(r)) {
      const m = value?.metrics?.val || value?.metrics?.train
      if (m) arr.push({ name, metrics: m })
    }
    return arr
  }, [data])

  const accuracyData = useMemo(() => {
    if (!modelMetrics.length) return []
    const isRegression = data?.problem_type === "regression"
    return modelMetrics
      .map(({ name, metrics }) => ({
        name,
        value: isRegression ? metrics.r2 : metrics.accuracy,
      }))
      .filter((d) => typeof d.value === "number")
      .sort((a, b) => b.value - a.value)
  }, [modelMetrics, data?.problem_type])

  const bestModelName = useMemo(
    () => accuracyData[0]?.name ?? "N/A",
    [accuracyData]
  )

  const avgMetric = useMemo(
    () => (accuracyData.length ? accuracyData.reduce((s, m) => s + m.value, 0) / accuracyData.length : 0),
    [accuracyData]
  )

  const regressionSeries = useMemo(() => {
    if (data?.problem_type !== "regression") return []
    const best = data?.results?.[bestModelName]
    const preds = best?.val_predictions
    const actual = best?.val_actual
    if (!preds || !actual) return []
    const out: any[] = []
    for (let i = 0; i < Math.min(preds.length, actual.length); i++) {
      out.push({ index: i, actual: actual[i], predicted: preds[i] })
    }
    return out
  }, [data?.results, bestModelName])

  const handleShare = () => {
    const text = `I trained models using EasyFlow ML! Best model: ${bestModelName}`
    navigator.clipboard.writeText(text)
    alert("Link copied to clipboard!")
  }

  const handleExportReport = () => {
    const blob = new Blob([JSON.stringify({ dataset: data?.dataset, bestModelName, accuracyData }, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `easyflow-report-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  // ---- UI Return ----
  if (loading) return <p className="text-center mt-10 text-muted-foreground">Loading results...</p>
  if (error) return <p className="text-center mt-10 text-red-500">{error}</p>

  return (
    <div className="container py-12">
      <div className="mx-auto max-w-6xl space-y-8">
        
        {/* HEADER */}
        <div className="flex items-center justify-between">
          <Link href="/build">
            <Button variant="outline">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Build
            </Button>
          </Link>

          <div className="flex gap-2">
            <Button variant="outline" onClick={handleExportReport}>
              <Download className="mr-2 h-4 w-4" />
              Export Report
            </Button>
            <Button variant="outline" onClick={handleShare}>
              <Share2 className="mr-2 h-4 w-4" />
              Share
            </Button>
          </div>
        </div>

        <h1 className="text-3xl font-bold">Model Performance Metrics</h1>
        <p className="text-muted-foreground">Evaluation of all trained models with comparison charts</p>

        {/* SUMMARY CARDS */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardHeader><CardTitle>Best Model</CardTitle></CardHeader>
            <CardContent><p className="text-2xl font-bold">{bestModelName}</p></CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>{data?.problem_type === "regression" ? "Avg R²" : "Avg Accuracy"}</CardTitle></CardHeader>
            <CardContent><p className="text-2xl font-bold">{avgMetric.toFixed(3)}</p></CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle>Models Trained</CardTitle></CardHeader>
            <CardContent><p className="text-2xl font-bold">{modelMetrics.length}</p></CardContent>
          </Card>
        </div>

        {/* BAR CHART: ACCURACY / R2 */}
        <Card>
          <CardHeader>
            <CardTitle>{data?.problem_type === "regression" ? "Model R² Comparison" : "Model Accuracy Comparison"}</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={accuracyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* LINES CHART: Actual vs Predicted */}
        {data?.problem_type === "regression" && (
          <Card>
            <CardHeader><CardTitle>Actual vs Predicted (Validation)</CardTitle></CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={regressionSeries}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="actual" stroke="#0ea5e9" strokeWidth={2} />
                  <Line type="monotone" dataKey="predicted" stroke="#22c55e" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* CHART: Precision & Recall Comparison (Classification Only) */}
        {data?.problem_type === "classification" && (
  <Card>
    <CardHeader>
      <CardTitle>Precision & Recall Comparison</CardTitle>
      <CardDescription>Trade-offs between true positives and false positives</CardDescription>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={modelMetrics.map(({ name, metrics }) => ({
          model: name,
          precision: metrics?.precision,
          recall: metrics?.recall,
        }))}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="model" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="precision" stroke="#3b82f6" strokeWidth={2} />
          <Line type="monotone" dataKey="recall" stroke="#10b981" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
        )}

        {/* NAVIGATION */}
        <div className="flex justify-between">
          <Link href="/build">
            <Button variant="outline">Train Another Model</Button>
          </Link>
          <Link href="/results/predictions">
            <Button>Make Predictions</Button>
          </Link>
        </div>
      </div>
    </div>
  )
}
