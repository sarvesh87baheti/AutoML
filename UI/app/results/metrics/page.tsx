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

  const problemType = useMemo(() => {
    if (data?.problem_type) return data.problem_type
    if (data?.results?.kmeans?.metrics) return "kmeans_clustering"
    return undefined
  }, [data])

  const isKMeans = problemType === "kmeans_clustering"
  const isClassification = problemType === "classification" || problemType === "image_classification"
  const isImageClassification = problemType === "image_classification"

  // ---- Computed values with hooks BEFORE render return ----

  const imageClassificationMetrics = useMemo(() => {
    if (!isImageClassification) return null
    const raw = data?.results?.image_classification?.metrics
    return raw?.val || raw?.train || raw?.test || null
  }, [data, isImageClassification])

  const modelMetrics = useMemo(() => {
    const arr: { name: string; metrics: any }[] = []
    const r = data?.results || {}
    for (const [name, value] of Object.entries<any>(r)) {
      const m = value?.metrics?.val || value?.metrics?.train
      if (m) arr.push({ name, metrics: m })
    }
    if (!arr.length && imageClassificationMetrics) {
      arr.push({ name: "image_classification", metrics: imageClassificationMetrics })
    }
    return arr
  }, [data, imageClassificationMetrics])

  const kmeansTrainMetrics = useMemo(() => {
    if (!isKMeans) return null
    const raw = data?.results?.kmeans?.metrics
    return raw?.train || raw?.val || null
  }, [data, isKMeans])

  const accuracyData = useMemo(() => {
    if (!modelMetrics.length) return []
    const isRegression = problemType === "regression"
    return modelMetrics
      .map(({ name, metrics }) => ({
        name,
        value: isRegression ? metrics.r2 : metrics.accuracy,
      }))
      .filter((d) => typeof d.value === "number")
      .sort((a, b) => b.value - a.value)
  }, [modelMetrics, problemType])

  const bestModelName = useMemo(() => {
    if (isKMeans) return data?.results?.best_model ?? modelMetrics[0]?.name ?? "kmeans"
    return accuracyData[0]?.name ?? data?.results?.best_model ?? "N/A"
  }, [accuracyData, data?.results, isKMeans, modelMetrics])

  const avgMetric = useMemo(
    () => (accuracyData.length ? accuracyData.reduce((s, m) => s + m.value, 0) / accuracyData.length : 0),
    [accuracyData]
  )

  const featureImportance = useMemo(() => {
    const raw = data?.results?.feature_importance
    if (!Array.isArray(raw)) return []
    return raw
      .filter(
        (item) =>
          typeof item?.percentage === "number" &&
          typeof item?.feature === "string" &&
          item.feature.length > 0
      )
      .slice(0, 10)
  }, [data?.results])

  const regressionSeries = useMemo(() => {
    if (problemType !== "regression") return []
    const best = data?.results?.[bestModelName]
    const preds = best?.val_predictions
    const actual = best?.val_actual
    if (!preds || !actual) return []
    const out: any[] = []
    for (let i = 0; i < Math.min(preds.length, actual.length); i++) {
      out.push({ index: i, actual: actual[i], predicted: preds[i] })
    }
    return out
  }, [data?.results, bestModelName, problemType])

  const confusionMatrixData = useMemo(() => {
    if (!isClassification) return null
    if (isImageClassification) {
      return data?.results?.image_classification?.confusion_matrix
    }
    // For regular classification, confusion matrix may be in model results
    const modelResult = data?.results?.[bestModelName]
    return modelResult?.confusion_matrix
  }, [data?.results, bestModelName, isClassification, isImageClassification])

  const handleShare = () => {
    const text = `I trained models using EasyFlow ML! Best model: ${bestModelName}`
    navigator.clipboard.writeText(text)
    alert("Link copied to clipboard!")
  }

  const handleExportReport = () => {
    const payload = {
      dataset: data?.dataset,
      problem_type: problemType,
      bestModelName,
      comparison: isKMeans ? undefined : accuracyData,
      clustering_metrics: isKMeans ? kmeansTrainMetrics : undefined,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })
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
        <p className="text-muted-foreground">
          {isKMeans
            ? "Clustering quality metrics for the trained model"
            : isImageClassification
            ? "Image classification model performance and evaluation metrics"
            : "Evaluation of all trained models with comparison charts"}
        </p>

        {/* SUMMARY CARDS */}
        {!isKMeans ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader><CardTitle>Best Model</CardTitle></CardHeader>
              <CardContent><p className="text-2xl font-bold">{bestModelName}</p></CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>{problemType === "regression" ? "Avg R²" : "Avg Accuracy"}</CardTitle></CardHeader>
              <CardContent><p className="text-2xl font-bold">{avgMetric.toFixed(3)}</p></CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>Models Trained</CardTitle></CardHeader>
              <CardContent><p className="text-2xl font-bold">{modelMetrics.length}</p></CardContent>
            </Card>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader><CardTitle>Model</CardTitle></CardHeader>
              <CardContent><p className="text-2xl font-bold">{bestModelName}</p></CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Silhouette</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">
                  {typeof kmeansTrainMetrics?.silhouette === "number" ? kmeansTrainMetrics.silhouette.toFixed(3) : "N/A"}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Inertia</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">
                  {typeof kmeansTrainMetrics?.inertia === "number" ? kmeansTrainMetrics.inertia.toFixed(3) : "N/A"}
                </p>
              </CardContent>
            </Card>
          </div>
        )}

        {/* BAR CHART: ACCURACY / R2 (not for kmeans) */}
        {!isKMeans && (
          <Card>
            <CardHeader>
              <CardTitle>{problemType === "regression" ? "Model R² Comparison" : "Model Accuracy Comparison"}</CardTitle>
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
        )}

        {/* KMEANS METRICS (single model, no comparison) */}
        {isKMeans && (
          <Card>
            <CardHeader>
              <CardTitle>Clustering Metrics</CardTitle>
              <CardDescription>Relevant evaluation metrics for k-means clustering</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex items-center justify-between rounded-md border p-3">
                  <span className="text-sm text-muted-foreground">Silhouette</span>
                  <span className="font-medium">
                    {typeof kmeansTrainMetrics?.silhouette === "number" ? kmeansTrainMetrics.silhouette.toFixed(4) : "N/A"}
                  </span>
                </div>
                <div className="flex items-center justify-between rounded-md border p-3">
                  <span className="text-sm text-muted-foreground">Calinski-Harabasz</span>
                  <span className="font-medium">
                    {typeof kmeansTrainMetrics?.calinski_harabasz === "number" ? kmeansTrainMetrics.calinski_harabasz.toFixed(4) : "N/A"}
                  </span>
                </div>
                <div className="flex items-center justify-between rounded-md border p-3">
                  <span className="text-sm text-muted-foreground">Davies-Bouldin</span>
                  <span className="font-medium">
                    {typeof kmeansTrainMetrics?.davies_bouldin === "number" ? kmeansTrainMetrics.davies_bouldin.toFixed(4) : "N/A"}
                  </span>
                </div>
                <div className="flex items-center justify-between rounded-md border p-3">
                  <span className="text-sm text-muted-foreground">Inertia</span>
                  <span className="font-medium">
                    {typeof kmeansTrainMetrics?.inertia === "number" ? kmeansTrainMetrics.inertia.toFixed(4) : "N/A"}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* LINES CHART: Actual vs Predicted */}
        {problemType === "regression" && (
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
        {isClassification && (
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

        {/* CONFUSION MATRIX (Classification Only) */}
        {isClassification && confusionMatrixData && (
          <Card>
            <CardHeader>
              <CardTitle>Confusion Matrix</CardTitle>
              <CardDescription>Predicted vs Actual classifications</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-sm">
                  <thead>
                    <tr>
                      <th className="border p-2 bg-gray-100 text-left font-semibold">Actual \ Predicted</th>
                      {confusionMatrixData.class_names?.map((className: string, idx: number) => (
                        <th key={idx} className="border p-2 bg-gray-100 text-center font-semibold">
                          {className}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {confusionMatrixData.matrix?.map((row: number[], rowIdx: number) => {
                      const maxVal = Math.max(...row)
                      return (
                        <tr key={rowIdx}>
                          <td className="border p-2 bg-gray-100 font-semibold">
                            {confusionMatrixData.class_names?.[rowIdx] || `Class ${rowIdx}`}
                          </td>
                          {row.map((val: number, colIdx: number) => {
                            const intensity = maxVal > 0 ? val / maxVal : 0
                            const bgColor = intensity > 0.7 ? 'bg-green-600' : intensity > 0.4 ? 'bg-green-400' : intensity > 0 ? 'bg-yellow-300' : 'bg-gray-50'
                            const textColor = intensity > 0.5 ? 'text-white' : 'text-gray-900'
                            return (
                              <td key={colIdx} className={`border p-2 text-center font-semibold ${bgColor} ${textColor}`}>
                                {val}
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
              <div className="mt-4 text-xs text-muted-foreground">
                <p>Color intensity indicates cell values: darker green = higher values</p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* FEATURE IMPORTANCE */}
        {!isKMeans && featureImportance.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Top Feature Importance</CardTitle>
              <CardDescription>Most influential columns by relative importance (% of total)</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={featureImportance} layout="vertical" margin={{ left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, "dataMax"]} tickFormatter={(value) => `${value}%`} />
                  <YAxis type="category" dataKey="feature" width={140} />
                  <Tooltip formatter={(value: number) => `${value.toFixed(2)}%`} />
                  <Bar dataKey="percentage" fill="#f97316" radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {/* NAVIGATION */}
        <div className="flex justify-between">
          <Link href="/build">
            <Button variant="outline">Train Another Model</Button>
          </Link>
          {/* <Link href="/results/predictions">
            <Button>Make Predictions</Button>
          </Link> */}
        </div>
      </div>
    </div>
  )
}
