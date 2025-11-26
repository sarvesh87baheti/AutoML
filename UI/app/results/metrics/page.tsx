// FILE PURPOSE: Model Metrics & Evaluation Page - Display comprehensive performance charts
// Shows: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve, etc.
// BACKEND INTEGRATION: Fetch metrics data from trained model

"use client"

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
  PieChart,
  Pie,
  Cell,
} from "recharts"

// SAMPLE DATA - Replace with real metrics from backend API
const accuracyData = [
  { name: "Random Forest", value: 0.92 },
  { name: "Gradient Boost", value: 0.89 },
  { name: "SVM", value: 0.85 },
  { name: "Logistic Reg", value: 0.82 },
]

const precisionRecallData = [
  { model: "Random Forest", precision: 0.91, recall: 0.93 },
  { model: "Gradient Boost", precision: 0.88, recall: 0.9 },
  { model: "SVM", precision: 0.84, recall: 0.86 },
]

const confusionMatrixData = [
  { name: "True Neg", value: 850, fill: "#10b981" },
  { name: "False Pos", value: 50, fill: "#f59e0b" },
  { name: "False Neg", value: 40, fill: "#f97316" },
  { name: "True Pos", value: 60, fill: "#3b82f6" },
]

const handleExportReport = () => {
  const reportData = {
    timestamp: new Date().toISOString(),
    bestModel: "Random Forest",
    accuracy: "92%",
    trainingTime: "2m 34s",
    modelsCount: 4,
    accuracyData,
    precisionRecallData,
    confusionMatrix: confusionMatrixData,
  }
  const dataStr = JSON.stringify(reportData, null, 2)
  const dataBlob = new Blob([dataStr], { type: "application/json" })
  const url = URL.createObjectURL(dataBlob)
  const link = document.createElement("a")
  link.href = url
  link.download = `easyflow-metrics-report-${Date.now()}.json`
  link.click()
  URL.revokeObjectURL(url)
}

const handleShare = () => {
  const shareText = `I trained 4 ML models with EasyFlow ML! Best Model: Random Forest with 92% accuracy. Try it yourself at ${window.location.origin}`
  if (navigator.share) {
    navigator.share({
      title: "EasyFlow ML Results",
      text: shareText,
    })
  } else {
    navigator.clipboard.writeText(shareText)
    alert("Results copied to clipboard!")
  }
}

export default function MetricsPage() {
  return (
    <div className="container py-12 bg-gradient-to-b from-background to-background">
      <div className="mx-auto max-w-6xl space-y-8">
        {/* PAGE HEADER WITH BACK BUTTON */}
        <div className="flex items-center justify-between">
          <div>
            <Link href="/build">
              <Button variant="outline" size="sm" className="mb-4 bg-transparent">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Build
              </Button>
            </Link>
            <h1 className="text-3xl font-bold tracking-tighter text-foreground">Model Performance Metrics</h1>
            <p className="text-muted-foreground mt-2">
              Comprehensive evaluation of all trained models with detailed performance comparisons
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleExportReport}>
              <Download className="h-4 w-4 mr-2" />
              Export Report
            </Button>
            <Button variant="outline" size="sm" onClick={handleShare}>
              <Share2 className="h-4 w-4 mr-2" />
              Share
            </Button>
          </div>
        </div>

        {/* GRID OF METRICS CARDS */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-foreground">Best Model</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold text-foreground">Random Forest</p>
              <p className="text-xs text-muted-foreground mt-1">92% Accuracy</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-foreground">Avg Accuracy</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold text-foreground">87%</p>
              <p className="text-xs text-muted-foreground mt-1">Across all models</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-foreground">Training Time</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold text-foreground">2m 34s</p>
              <p className="text-xs text-muted-foreground mt-1">For all models</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-foreground">Models Trained</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold text-foreground">4</p>
              <p className="text-xs text-muted-foreground mt-1">Different algorithms</p>
            </CardContent>
          </Card>
        </div>

        {/* CHART 1: MODEL ACCURACY COMPARISON */}
        <Card>
          <CardHeader>
            <CardTitle className="text-foreground">Model Accuracy Comparison</CardTitle>
            <CardDescription>Performance ranking of all trained models</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={accuracyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--muted)" />
                <XAxis dataKey="name" stroke="var(--foreground)" />
                <YAxis stroke="var(--foreground)" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--card)",
                    border: "1px solid var(--border)",
                    color: "var(--foreground)",
                  }}
                />
                <Bar dataKey="value" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* CHART 2: PRECISION & RECALL */}
        <Card>
          <CardHeader>
            <CardTitle className="text-foreground">Precision & Recall Analysis</CardTitle>
            <CardDescription>Trade-off between true positives and false positives</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={precisionRecallData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--muted)" />
                <XAxis dataKey="model" stroke="var(--foreground)" />
                <YAxis stroke="var(--foreground)" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--card)",
                    border: "1px solid var(--border)",
                    color: "var(--foreground)",
                  }}
                />
                <Legend wrapperStyle={{ color: "var(--foreground)" }} />
                <Line type="monotone" dataKey="precision" stroke="#3b82f6" strokeWidth={2} />
                <Line type="monotone" dataKey="recall" stroke="#10b981" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* CHART 3: CONFUSION MATRIX */}
        <Card>
          <CardHeader>
            <CardTitle className="text-foreground">Confusion Matrix (Best Model)</CardTitle>
            <CardDescription>Distribution of predictions vs actual values</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={confusionMatrixData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {confusionMatrixData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "var(--card)",
                      border: "1px solid var(--border)",
                      color: "var(--foreground)",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* ACTION BUTTONS */}
        <div className="flex gap-4 justify-between">
          <Link href="/build">
            <Button variant="outline">Build Another Model</Button>
          </Link>
          <Link href="/results/predictions">
            <Button className="bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 hover:from-violet-700 hover:via-blue-600 hover:to-teal-500 text-white font-semibold">
              Make Predictions
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}
