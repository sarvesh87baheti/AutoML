"use client"

import { useEffect, useMemo, useState } from "react"
import Link from "next/link"
import { ArrowLeft, BarChart3, Download, ImageIcon, Layers3, Share2, Trophy } from "lucide-react"
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  CartesianGrid,
  Tooltip,
  XAxis,
  YAxis,
  LineChart,
  Line,
  Legend,
} from "recharts"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

type ResultsPayload = {
  dataset: string
  problem_type?: string
  metadata?: Record<string, any>
  results: Record<string, any>
}

type ComparisonModel = {
  name: string
  value?: number
  loss?: number
  time?: number
  size_mb?: number
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
        if (mounted) setError(e?.message || "Failed to fetch results")
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

function prettyModelName(name: string) {
  return name
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (match) => match.toUpperCase())
}

function formatMetric(value: unknown, digits = 4) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "N/A"
}

function formatCompactNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : "N/A"
}

function buildExportPayload(args: {
  dataset?: string
  problemType?: string
  bestModelName: string
  metadata?: Record<string, any>
  accuracyData: any[]
  isKMeans: boolean
  isImageClassification: boolean
  kmeansTrainMetrics: any
  imageClassificationData: any
}) {
  const {
    dataset,
    problemType,
    bestModelName,
    metadata,
    accuracyData,
    isKMeans,
    isImageClassification,
    kmeansTrainMetrics,
    imageClassificationData,
  } = args

  if (isImageClassification) {
    const failedModels = Array.isArray(imageClassificationData?.failedModels)
      ? imageClassificationData.failedModels.map((item: { name: string; error: string }) => ({
          model: prettyModelName(item.name),
          error: item.error,
        }))
      : []

    return {
      dataset,
      problemType,
      status: imageClassificationData?.hasSuccessfulModels ? "success" : "failed",
      imageMode: metadata?.image_mode,
      bestModel: imageClassificationData?.hasSuccessfulModels ? prettyModelName(bestModelName) : null,
      metrics: imageClassificationData?.hasSuccessfulModels
        ? {
            accuracy: imageClassificationData?.accuracy ?? null,
            loss: imageClassificationData?.loss ?? null,
            precision: imageClassificationData?.weightedAvg?.precision ?? null,
            recall: imageClassificationData?.weightedAvg?.recall ?? null,
            f1Score: imageClassificationData?.weightedAvg?.["f1-score"] ?? null,
            macroF1Score: imageClassificationData?.macroAvg?.["f1-score"] ?? null,
            weightedAverage: imageClassificationData?.weightedAvg ?? null,
            macroAverage: imageClassificationData?.macroAvg ?? null,
          }
        : null,
      classes: imageClassificationData?.classNames ?? [],
      datasetSnapshot: imageClassificationData?.datasetSnapshot ?? {},
      modelComparison: Array.isArray(imageClassificationData?.comparison)
        ? imageClassificationData.comparison.map((item: ComparisonModel) => ({
            model: prettyModelName(item.name),
            accuracy: item.value ?? null,
            loss: item.loss ?? null,
            trainingTimeSeconds: item.time ?? null,
            modelSizeMb: item.size_mb ?? null,
          }))
        : [],
      errors: failedModels,
    }
  }

  return {
    dataset,
    problemType,
    bestModel: bestModelName,
    metadata,
    comparison: !isKMeans ? accuracyData : undefined,
    clusteringMetrics: isKMeans ? kmeansTrainMetrics : undefined,
  }
}

export default function MetricsPage() {
  const { data, loading, error } = useLatestResults()

  const problemType = useMemo(() => {
    if (data?.problem_type) return data.problem_type
    if (data?.results?.kmeans?.metrics) return "kmeans_clustering"
    if (data?.results?.image_classification?.metrics) return "image_classification"
    return undefined
  }, [data])

  const isRegression = problemType === "regression"
  const isClassification = problemType === "classification"
  const isKMeans = problemType === "kmeans_clustering"
  const isImageClassification = problemType === "image_classification"
  const imageMode = isImageClassification
    ? data?.metadata?.image_mode ?? data?.results?.image_mode ?? "standard"
    : undefined
  const isLightMode = imageMode === "light"

  const modelMetrics = useMemo(() => {
    const arr: { name: string; metrics: any }[] = []
    const resultMap = data?.results || {}
    for (const [name, value] of Object.entries<any>(resultMap)) {
      const metrics = value?.metrics?.val || value?.metrics?.train
      if (metrics) arr.push({ name, metrics })
    }
    return arr
  }, [data])

  const accuracyData = useMemo(() => {
    if (isImageClassification) return []
    if (!modelMetrics.length) return []
    return modelMetrics
      .map(({ name, metrics }) => ({
        name,
        label: prettyModelName(name),
        value: isRegression ? metrics.r2 : metrics.accuracy,
      }))
      .filter((item) => typeof item.value === "number")
      .sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
  }, [isImageClassification, isRegression, modelMetrics])

  const bestModelName = useMemo(() => {
    if (isImageClassification) {
      return data?.results?.best_model ?? "No successful models"
    }
    if (isKMeans) return data?.results?.best_model ?? modelMetrics[0]?.name ?? "kmeans"
    return accuracyData[0]?.name ?? data?.results?.best_model ?? "N/A"
  }, [accuracyData, data?.results, isImageClassification, isKMeans, modelMetrics])

  const avgMetric = useMemo(() => {
    if (!accuracyData.length) return 0
    return accuracyData.reduce((sum, item) => sum + (item.value ?? 0), 0) / accuracyData.length
  }, [accuracyData])

  const kmeansTrainMetrics = useMemo(() => {
    if (!isKMeans) return null
    const raw = data?.results?.kmeans?.metrics
    return raw?.train || raw?.val || null
  }, [data, isKMeans])

  const kmeansClusterUrl = useMemo(() => {
    return data?.dataset ? `/api/results/kmeans-clusters?dataset=${encodeURIComponent(data.dataset)}` : null
  }, [data?.dataset])

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

  const classificationBestMetrics = useMemo(() => {
    if (!isClassification) return null
    const best = data?.results?.[bestModelName]
    return best?.metrics?.val || best?.metrics?.train || null
  }, [bestModelName, data?.results, isClassification])

  const regressionSeries = useMemo(() => {
    if (!isRegression) return []
    const best = data?.results?.[bestModelName]
    const preds = best?.val_predictions
    const actual = best?.val_actual
    if (!Array.isArray(preds) || !Array.isArray(actual)) return []

    return actual.slice(0, Math.min(actual.length, preds.length)).map((value: number, index: number) => ({
      index,
      actual: value,
      predicted: preds[index],
    }))
  }, [bestModelName, data?.results, isRegression])

  const classificationComparisonRows = useMemo(() => {
    if (!isClassification) return []
    return modelMetrics.map(({ name, metrics }) => ({
      name,
      label: prettyModelName(name),
      accuracy: metrics?.accuracy,
      precision: metrics?.precision,
      recall: metrics?.recall,
      f1: metrics?.f1,
    }))
  }, [isClassification, modelMetrics])

  const imageClassificationData = useMemo(() => {
    if (!isImageClassification) return null

    const raw = data?.results?.image_classification
    const comparison: ComparisonModel[] = Array.isArray(raw?.models_comparison) ? raw.models_comparison : []
    const allModels = raw?.all_models || {}
    const classNames = Array.isArray(data?.metadata?.class_names)
      ? data?.metadata?.class_names
      : Array.isArray(data?.results?.class_names)
      ? data?.results?.class_names
      : []

    const failedModels = Object.entries<any>(allModels)
      .filter(([, value]) => value?.error)
      .map(([name, value]) => ({
        name,
        error: value.error as string,
      }))

    const perClassRows = classNames
      .map((className: string) => ({
        className,
        metrics: raw?.classification_report?.[className],
      }))
      .filter((item: any) => item.metrics)

    const confusionMatrixUrl = data?.dataset ? `/api/results/confusion-matrix?dataset=${encodeURIComponent(data.dataset)}` : null

    return {
      accuracy:
        raw?.classification_report?.accuracy ??
        raw?.metrics?.val?.accuracy,
      loss: raw?.metrics?.val?.loss,
      comparison,
      allModels,
      classNames,
      classificationReport: raw?.classification_report,
      perClassRows,
      failedModels,
      confusionMatrixUrl,
      datasetSnapshot: {
        trainSize: data?.metadata?.train_size,
        valSize: data?.metadata?.val_size,
        testSize: data?.metadata?.test_size,
        numClasses: data?.metadata?.num_classes ?? data?.results?.num_classes,
        batchSize: data?.metadata?.batch_size,
        imageSize: Array.isArray(data?.metadata?.img_size) ? data?.metadata?.img_size.join(" x ") : null,
      },
      hasSuccessfulModels: comparison.length > 0,
      weightedAvg: raw?.classification_report?.["weighted avg"],
      macroAvg: raw?.classification_report?.["macro avg"],
    }
  }, [data, isImageClassification])

  const handleShare = async () => {
    const message = `EasyFlow ML result: ${prettyModelName(bestModelName)} on ${data?.dataset || "dataset"}`
    await navigator.clipboard.writeText(message)
    window.alert("Summary copied to clipboard.")
  }

  const handleExportReport = () => {
    const payload = buildExportPayload({
      dataset: data?.dataset,
      problemType,
      bestModelName,
      metadata: data?.metadata,
      accuracyData,
      isKMeans,
      isImageClassification,
      kmeansTrainMetrics,
      imageClassificationData,
    })

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = `easyflow-report-${Date.now()}.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  if (loading) return <p className="mt-10 text-center text-muted-foreground">Loading results...</p>
  if (error) return <p className="mt-10 text-center text-red-500">{error}</p>

  return (
    <div className="container py-12">
      <div className="mx-auto max-w-6xl space-y-8">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <Link href="/build">
            <Button variant="outline">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Build
            </Button>
          </Link>

          <div className="flex flex-wrap gap-2">
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

        <div
          className={isImageClassification
            ? "rounded-3xl border border-orange-500/30 bg-gradient-to-br from-orange-500/10 via-amber-500/5 to-background p-6"
            : isClassification
            ? "rounded-3xl border border-violet-500/20 bg-gradient-to-br from-violet-500/10 via-background to-blue-500/5 p-6"
            : isRegression
            ? "rounded-3xl border border-blue-500/20 bg-gradient-to-br from-blue-500/10 via-background to-cyan-500/5 p-6"
            : isKMeans
            ? "rounded-3xl border border-teal-500/20 bg-gradient-to-br from-teal-500/10 via-background to-emerald-500/5 p-6"
            : ""}
        >
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline">{data?.dataset || "Latest dataset"}</Badge>
                <Badge variant="outline">
                  {isImageClassification
                    ? "Image Classification"
                    : isKMeans
                    ? "K-Means Clustering"
                    : isRegression
                    ? "Regression"
                    : "Classification"}
                </Badge>
              </div>

              <div>
                <h1 className="text-3xl font-bold tracking-tight md:text-4xl">
                  {isImageClassification
                    ? "Image Classification Results"
                    : isClassification
                    ? "Classification Results"
                    : isRegression
                    ? "Regression Results"
                    : isKMeans
                    ? "K-Means Results"
                    : "Model Performance Metrics"}
                </h1>
                <p className="mt-2 max-w-3xl text-muted-foreground">
                  {isImageClassification
                    ? "Review the selected backbone, compare transfer-learning candidates, and inspect class-wise performance before deployment."
                    : isKMeans
                    ? "Inspect cluster compactness and separation metrics to judge whether the chosen k produced meaningful segments."
                    : isRegression
                    ? "Compare regressors, review fit quality, and inspect how closely predictions track the true validation values."
                    : isClassification
                    ? "Compare candidate classifiers and review precision, recall, F1-score, and category-prediction quality."
                    : "Evaluation of the trained models with comparison charts and summary metrics."}
                </p>
              </div>
            </div>

            {isImageClassification && (
              <div className="grid min-w-[220px] gap-3 rounded-2xl border border-orange-500/20 bg-background/80 p-4">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Trophy className="h-4 w-4 text-orange-500" />
                  {isLightMode ? "Model used" : "Best backbone"}
                </div>
                <p className="text-2xl font-semibold">{prettyModelName(bestModelName)}</p>
                <p className="text-sm text-muted-foreground">
                  Accuracy {formatMetric(imageClassificationData?.accuracy)} with loss {formatMetric(imageClassificationData?.loss)}
                </p>
              </div>
            )}

            {isClassification && classificationBestMetrics && (
              <div className="grid min-w-[250px] gap-3 rounded-2xl border border-violet-500/20 bg-background/80 p-4">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Trophy className="h-4 w-4 text-violet-600" />
                  Best classifier
                </div>
                <p className="text-2xl font-semibold">{prettyModelName(bestModelName)}</p>
                <p className="text-sm text-muted-foreground">
                  Accuracy {formatMetric(classificationBestMetrics.accuracy, 3)} and F1 {formatMetric(classificationBestMetrics.f1, 3)}
                </p>
              </div>
            )}

            {isRegression && (
              <div className="grid min-w-[250px] gap-3 rounded-2xl border border-blue-500/20 bg-background/80 p-4">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Trophy className="h-4 w-4 text-blue-500" />
                  Best regressor
                </div>
                <p className="text-2xl font-semibold">{prettyModelName(bestModelName)}</p>
                <p className="text-sm text-muted-foreground">
                  Top validation fit selected from the trained regression models.
                </p>
              </div>
            )}

            {isKMeans && (
              <div className="grid min-w-[250px] gap-3 rounded-2xl border border-teal-500/20 bg-background/80 p-4">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Layers3 className="h-4 w-4 text-teal-500" />
                  Cluster profile
                </div>
                <p className="text-2xl font-semibold">{prettyModelName(bestModelName)}</p>
                <p className="text-sm text-muted-foreground">
                  Use silhouette and Davies-Bouldin together to judge whether the current k is a good fit.
                </p>
              </div>
            )}
          </div>
        </div>

        {!isKMeans && !isImageClassification && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle>Best Model</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{prettyModelName(bestModelName)}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>{isRegression ? "Avg R²" : "Avg Accuracy"}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{avgMetric.toFixed(3)}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Models Trained</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{modelMetrics.length}</p>
              </CardContent>
            </Card>
          </div>
        )}

        {isKMeans && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <Card>
              <CardHeader>
                <CardTitle>Model</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-2 text-2xl font-bold">
                  <Layers3 className="h-6 w-6 text-teal-500" />
                  {prettyModelName(bestModelName)}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Silhouette</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{formatMetric(kmeansTrainMetrics?.silhouette, 3)}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Inertia</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{formatMetric(kmeansTrainMetrics?.inertia, 3)}</p>
              </CardContent>
            </Card>
          </div>
        )}

        {isClassification && classificationBestMetrics && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Card>
              <CardHeader>
                <CardTitle>Best Classifier</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{prettyModelName(bestModelName)}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Validation Accuracy</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{formatMetric(classificationBestMetrics.accuracy, 3)}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Precision</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{formatMetric(classificationBestMetrics.precision, 3)}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>F1-Score</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{formatMetric(classificationBestMetrics.f1, 3)}</p>
              </CardContent>
            </Card>
          </div>
        )}

        {isRegression && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <Card>
              <CardHeader>
                <CardTitle>Best Regressor</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{prettyModelName(bestModelName)}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Average R²</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{avgMetric.toFixed(3)}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Models Compared</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{accuracyData.length}</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Validation Samples</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{regressionSeries.length || "N/A"}</p>
              </CardContent>
            </Card>
          </div>
        )}

        {isImageClassification && (
          <>
            <div className={isLightMode ? "grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4" : "grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4"}>
              <Card>
                <CardHeader>
                  <CardTitle>{isLightMode ? "Model Used" : "Best Backbone"}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2 text-2xl font-bold">
                    <ImageIcon className="h-6 w-6 text-orange-500" />
                    {prettyModelName(bestModelName)}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Test Accuracy</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{formatMetric(imageClassificationData?.accuracy)}</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Test Loss</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{formatMetric(imageClassificationData?.loss)}</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>{isLightMode ? "Weighted F1" : "Successful Models"}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">
                    {isLightMode
                      ? formatMetric(imageClassificationData?.weightedAvg?.["f1-score"], 3)
                      : imageClassificationData?.comparison?.length ?? 0}
                  </p>
                </CardContent>
              </Card>
            </div>

            {isLightMode && (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                <Card>
                  <CardHeader>
                    <CardTitle>Weighted Precision</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold">{formatMetric(imageClassificationData?.weightedAvg?.precision, 3)}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Weighted Recall</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold">{formatMetric(imageClassificationData?.weightedAvg?.recall, 3)}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Macro F1</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold">{formatMetric(imageClassificationData?.macroAvg?.["f1-score"], 3)}</p>
                  </CardContent>
                </Card>
              </div>
            )}

            <div className="grid gap-6 lg:grid-cols-[1.4fr_0.9fr]">
              {!isLightMode ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Backbone Leaderboard</CardTitle>
                    <CardDescription>Accuracy across the trained transfer-learning models.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {imageClassificationData?.comparison?.length ? (
                      <ResponsiveContainer width="100%" height={320}>
                        <BarChart data={imageClassificationData.comparison.map((item) => ({ ...item, label: prettyModelName(item.name) }))}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="label" interval={0} angle={-15} textAnchor="end" height={70} />
                          <YAxis domain={[0, 1]} />
                          <Tooltip formatter={(value: number) => value.toFixed(4)} />
                          <Bar dataKey="value" fill="#f97316" radius={[10, 10, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <p className="text-sm text-muted-foreground">No successful image models were recorded for this run.</p>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardHeader>
                    <CardTitle>Evaluation Summary</CardTitle>
                    <CardDescription>Single-model report for light mode.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm text-muted-foreground">
                    <div className="rounded-xl border p-4">
                      <p className="font-medium text-foreground">Mode</p>
                      <p className="mt-1">Light mode runs a single fast baseline model: {prettyModelName(bestModelName)}.</p>
                    </div>
                    <div className="rounded-xl border p-4">
                      <p className="font-medium text-foreground">Evaluation metrics</p>
                      <p className="mt-1">Test accuracy and test loss are appropriate here because they show how the trained model performs on unseen data.</p>
                    </div>
                    <div className="rounded-xl border p-4">
                      <p className="font-medium text-foreground">Per-class insight</p>
                      <p className="mt-1">Use the per-class table and confusion matrix to understand where the model confuses similar categories.</p>
                    </div>
                  </CardContent>
                </Card>
              )}

              <Card>
                <CardHeader>
                  <CardTitle>Dataset Snapshot</CardTitle>
                  <CardDescription>Quick context about the image dataset used for this run.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between rounded-lg border p-3">
                    <span className="text-sm text-muted-foreground">Classes</span>
                    <span className="font-medium">{formatCompactNumber(imageClassificationData?.datasetSnapshot.numClasses)}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-lg border p-3">
                    <span className="text-sm text-muted-foreground">Train Images</span>
                    <span className="font-medium">{formatCompactNumber(imageClassificationData?.datasetSnapshot.trainSize)}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-lg border p-3">
                    <span className="text-sm text-muted-foreground">Validation Images</span>
                    <span className="font-medium">{formatCompactNumber(imageClassificationData?.datasetSnapshot.valSize)}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-lg border p-3">
                    <span className="text-sm text-muted-foreground">Test Images</span>
                    <span className="font-medium">{formatCompactNumber(imageClassificationData?.datasetSnapshot.testSize)}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-lg border p-3">
                    <span className="text-sm text-muted-foreground">Image Size</span>
                    <span className="font-medium">{imageClassificationData?.datasetSnapshot.imageSize || "N/A"}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-lg border p-3">
                    <span className="text-sm text-muted-foreground">Batch Size</span>
                    <span className="font-medium">{formatCompactNumber(imageClassificationData?.datasetSnapshot.batchSize)}</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {!isLightMode && (
              <Card>
                <CardHeader>
                  <CardTitle>Model Comparison Table</CardTitle>
                  <CardDescription>Accuracy, loss, training time, and saved model size for each successful backbone.</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse text-sm">
                      <thead>
                        <tr className="border-b text-left">
                          <th className="p-3">Model</th>
                          <th className="p-3 text-right">Accuracy</th>
                          <th className="p-3 text-right">Loss</th>
                          <th className="p-3 text-right">Training Time (s)</th>
                          <th className="p-3 text-right">Model Size (MB)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(imageClassificationData?.comparison || []).map((model) => (
                          <tr key={model.name} className="border-b hover:bg-muted/40">
                            <td className="p-3 font-medium">{prettyModelName(model.name)}</td>
                            <td className="p-3 text-right">{formatMetric(model.value)}</td>
                            <td className="p-3 text-right">{formatMetric(model.loss)}</td>
                            <td className="p-3 text-right">{formatMetric(model.time, 2)}</td>
                            <td className="p-3 text-right">{formatMetric(model.size_mb, 2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            )}

            {imageClassificationData?.perClassRows?.length ? (
              <Card>
                <CardHeader>
                  <CardTitle>Per-Class Performance</CardTitle>
                  <CardDescription>Precision, recall, F1-score, and support for each class.</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse text-sm">
                      <thead>
                        <tr className="border-b text-left">
                          <th className="p-3">Class</th>
                          <th className="p-3 text-right">Precision</th>
                          <th className="p-3 text-right">Recall</th>
                          <th className="p-3 text-right">F1-Score</th>
                          <th className="p-3 text-right">Support</th>
                        </tr>
                      </thead>
                      <tbody>
                        {imageClassificationData.perClassRows.map((row: any) => (
                          <tr key={row.className} className="border-b hover:bg-muted/40">
                            <td className="p-3 font-medium">{row.className}</td>
                            <td className="p-3 text-right">{formatMetric(row.metrics?.precision, 3)}</td>
                            <td className="p-3 text-right">{formatMetric(row.metrics?.recall, 3)}</td>
                            <td className="p-3 text-right">{formatMetric(row.metrics?.["f1-score"], 3)}</td>
                            <td className="p-3 text-right">{formatMetric(row.metrics?.support, 0)}</td>
                          </tr>
                        ))}
                        <tr className="bg-muted/30 font-semibold">
                          <td className="p-3">Weighted Avg</td>
                          <td className="p-3 text-right">{formatMetric(imageClassificationData.classificationReport?.["weighted avg"]?.precision, 3)}</td>
                          <td className="p-3 text-right">{formatMetric(imageClassificationData.classificationReport?.["weighted avg"]?.recall, 3)}</td>
                          <td className="p-3 text-right">{formatMetric(imageClassificationData.classificationReport?.["weighted avg"]?.["f1-score"], 3)}</td>
                          <td className="p-3 text-right">{formatMetric(imageClassificationData.classificationReport?.["weighted avg"]?.support, 0)}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            ) : null}

            <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
              <Card>
                <CardHeader>
                  <CardTitle>Confusion Matrix</CardTitle>
                  <CardDescription>Visual inspection of class-wise prediction quality for the best model.</CardDescription>
                </CardHeader>
                <CardContent>
                  {imageClassificationData?.confusionMatrixUrl ? (
                    <img
                      src={imageClassificationData.confusionMatrixUrl}
                      alt="Confusion matrix"
                      className="w-full rounded-xl border bg-white p-2"
                    />
                  ) : (
                    <p className="text-sm text-muted-foreground">Confusion matrix image is not available for this run.</p>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Run Notes</CardTitle>
                  <CardDescription>Useful context about the trained backbones.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-muted-foreground">
                  <div className="rounded-xl border p-4">
                    <p className="font-medium text-foreground">{isLightMode ? "Model used" : "Best model"}</p>
                    <p className="mt-1">
                      {isLightMode
                        ? `${prettyModelName(bestModelName)} was trained as the fast baseline model for this run.`
                        : `${prettyModelName(bestModelName)} delivered the top recorded accuracy for this dataset.`}
                    </p>
                  </div>
                  <div className="rounded-xl border p-4">
                    <p className="font-medium text-foreground">Class coverage</p>
                    <p className="mt-1">
                      {imageClassificationData?.classNames?.length
                        ? `${imageClassificationData.classNames.length} class labels were detected: ${imageClassificationData.classNames.join(", ")}.`
                        : "Class labels were not found in metadata."}
                    </p>
                  </div>
                  <div className="rounded-xl border p-4">
                    <p className="font-medium text-foreground">Export ready</p>
                    <p className="mt-1">
                      Use the export action above if you want to attach this run summary to your report or presentation.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {imageClassificationData?.failedModels?.length ? (
              <Card>
                <CardHeader>
                  <CardTitle>Skipped or Failed Backbones</CardTitle>
                  <CardDescription>These models did not finish successfully during the run.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {imageClassificationData.failedModels.map((item: any) => (
                    <div key={item.name} className="rounded-xl border border-red-500/20 bg-red-500/5 p-4">
                      <p className="font-medium text-foreground">{prettyModelName(item.name)}</p>
                      <p className="mt-1 text-sm text-muted-foreground">{item.error}</p>
                    </div>
                  ))}
                </CardContent>
              </Card>
            ) : null}
          </>
        )}

        {!isKMeans && !isImageClassification && (
          <Card>
            <CardHeader>
              <CardTitle>{isRegression ? "Model R² Comparison" : "Model Accuracy Comparison"}</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={accuracyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}

        {isKMeans && (
          <Card>
            <CardHeader>
              <CardTitle>Clustering Metrics</CardTitle>
              <CardDescription>Relevant evaluation metrics for k-means clustering.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="flex items-center justify-between rounded-md border p-3">
                  <span className="text-sm text-muted-foreground">Silhouette</span>
                  <span className="font-medium">{formatMetric(kmeansTrainMetrics?.silhouette)}</span>
                </div>
                <div className="flex items-center justify-between rounded-md border p-3">
                  <span className="text-sm text-muted-foreground">Calinski-Harabasz</span>
                  <span className="font-medium">{formatMetric(kmeansTrainMetrics?.calinski_harabasz)}</span>
                </div>
                <div className="flex items-center justify-between rounded-md border p-3">
                  <span className="text-sm text-muted-foreground">Davies-Bouldin</span>
                  <span className="font-medium">{formatMetric(kmeansTrainMetrics?.davies_bouldin)}</span>
                </div>
                <div className="flex items-center justify-between rounded-md border p-3">
                  <span className="text-sm text-muted-foreground">Inertia</span>
                  <span className="font-medium">{formatMetric(kmeansTrainMetrics?.inertia)}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {isKMeans && (
          <Card>
            <CardHeader>
              <CardTitle>Cluster Visualization</CardTitle>
              <CardDescription>2D projection of the clustered training samples.</CardDescription>
            </CardHeader>
            <CardContent>
              {kmeansClusterUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={kmeansClusterUrl} alt="KMeans clusters" className="w-full rounded-xl border bg-white p-2" />
              ) : (
                <p className="text-sm text-muted-foreground">Cluster visualization is not available for this run.</p>
              )}
            </CardContent>
          </Card>
        )}

        {isClassification && classificationComparisonRows.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Classification Scorecard</CardTitle>
              <CardDescription>All candidate classifiers with their validation metrics.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-sm">
                  <thead>
                    <tr className="border-b text-left">
                      <th className="p-3">Model</th>
                      <th className="p-3 text-right">Accuracy</th>
                      <th className="p-3 text-right">Precision</th>
                      <th className="p-3 text-right">Recall</th>
                      <th className="p-3 text-right">F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {classificationComparisonRows.map((row) => (
                      <tr key={row.name} className="border-b hover:bg-muted/40">
                        <td className="p-3 font-medium">{row.label}</td>
                        <td className="p-3 text-right">{formatMetric(row.accuracy, 3)}</td>
                        <td className="p-3 text-right">{formatMetric(row.precision, 3)}</td>
                        <td className="p-3 text-right">{formatMetric(row.recall, 3)}</td>
                        <td className="p-3 text-right">{formatMetric(row.f1, 3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}

        {isRegression && (
          <div className="grid gap-6 lg:grid-cols-[1.3fr_0.9fr]">
            <Card>
              <CardHeader>
                <CardTitle>Regression Model Scorecard</CardTitle>
                <CardDescription>Ranked comparison of validation R² across trained regressors.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse text-sm">
                    <thead>
                      <tr className="border-b text-left">
                        <th className="p-3">Model</th>
                        <th className="p-3 text-right">R²</th>
                      </tr>
                    </thead>
                    <tbody>
                      {accuracyData.map((row) => (
                        <tr key={row.name} className="border-b hover:bg-muted/40">
                          <td className="p-3 font-medium">{row.label}</td>
                          <td className="p-3 text-right">{formatMetric(row.value, 3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Regression Notes</CardTitle>
                <CardDescription>How to read the current regression run.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-muted-foreground">
                <div className="rounded-xl border p-4">
                  <p className="font-medium text-foreground">Higher R² is better</p>
                  <p className="mt-1">R² estimates how much of the target variation the model explains on validation data.</p>
                </div>
                <div className="rounded-xl border p-4">
                  <p className="font-medium text-foreground">Use the line chart</p>
                  <p className="mt-1">When predicted and actual curves stay close together, the selected model is behaving consistently.</p>
                </div>
                <div className="rounded-xl border p-4">
                  <p className="font-medium text-foreground">Feature importance</p>
                  <p className="mt-1">The top-feature panel helps explain which columns are driving the predictions most strongly.</p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {isKMeans && (
          <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
            <Card>
              <CardHeader>
                <CardTitle>K-Means Interpretation Guide</CardTitle>
                <CardDescription>Quick heuristics for deciding if the current k looks healthy.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-muted-foreground">
                <div className="rounded-xl border p-4">
                  <p className="font-medium text-foreground">Silhouette</p>
                  <p className="mt-1">Closer to 1 means tighter, better-separated clusters. Values near 0 often indicate overlap.</p>
                </div>
                <div className="rounded-xl border p-4">
                  <p className="font-medium text-foreground">Davies-Bouldin</p>
                  <p className="mt-1">Lower is better. Use it to spot when clusters are too diffuse or too similar to one another.</p>
                </div>
                <div className="rounded-xl border p-4">
                  <p className="font-medium text-foreground">Inertia</p>
                  <p className="mt-1">Lower inertia often means tighter clusters, but compare it across different k values to find a useful tradeoff.</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Clustering Snapshot</CardTitle>
                <CardDescription>What this run tells you at a glance.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-muted-foreground">
                <div className="rounded-xl border p-4">
                  <p className="font-medium text-foreground">Current algorithm</p>
                  <p className="mt-1">{prettyModelName(bestModelName)} was used for unsupervised grouping.</p>
                </div>
                <div className="rounded-xl border p-4">
                  <p className="font-medium text-foreground">Next step</p>
                  <p className="mt-1">If silhouette is weak, rerun with a different k and compare the metrics to refine your segmentation.</p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {isRegression && regressionSeries.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Actual vs Predicted (Validation)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={320}>
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

        {isClassification && (
          <Card>
            <CardHeader>
              <CardTitle>Precision & Recall Comparison</CardTitle>
              <CardDescription>Trade-offs between true positives and false positives.</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart
                  data={modelMetrics.map(({ name, metrics }) => ({
                    model: prettyModelName(name),
                    precision: metrics?.precision,
                    recall: metrics?.recall,
                  }))}
                >
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

        {!isKMeans && featureImportance.length > 0 && !isImageClassification && (
          <Card>
            <CardHeader>
              <CardTitle>Top Feature Importance</CardTitle>
              <CardDescription>Most influential columns by relative importance.</CardDescription>
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

        <div className="flex justify-between">
          <Link href="/build">
            <Button variant="outline">
              <BarChart3 className="mr-2 h-4 w-4" />
              Train Another Model
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}
