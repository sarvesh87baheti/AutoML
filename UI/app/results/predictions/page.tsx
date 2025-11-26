// FILE PURPOSE: Model Prediction Page - Use best trained model to make predictions
// Shows: Model comparison, data input, and prediction results
// BACKEND INTEGRATION: Send new data to best model for predictions
// HOW RESULTS ARE OBTAINED: Backend trains multiple models during build, returns metrics/predictions

"use client"

import Link from "next/link"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, Zap, Download } from "lucide-react"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

// SAMPLE MODEL COMPARISON DATA
// NOTE: These values would come from backend after model training
const modelComparison = [
  { name: "Random Forest", accuracy: "92%", training: "45s", inference: "0.2ms" },
  { name: "Gradient Boost", accuracy: "89%", training: "60s", inference: "0.3ms" },
  { name: "SVM", accuracy: "85%", training: "30s", inference: "0.15ms" },
]

const handleExportResults = (predictions: any) => {
  const resultsData = {
    timestamp: new Date().toISOString(),
    prediction: predictions.prediction,
    confidence: predictions.confidence,
    topPredictions: predictions.topPredictions,
  }
  const dataStr = JSON.stringify(resultsData, null, 2)
  const dataBlob = new Blob([dataStr], { type: "application/json" })
  const url = URL.createObjectURL(dataBlob)
  const link = document.createElement("a")
  link.href = url
  link.download = `easyflow-prediction-${Date.now()}.json`
  link.click()
  URL.revokeObjectURL(url)
}

export default function PredictionsPage() {
  const [inputData, setInputData] = useState("")
  const [predictions, setPredictions] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)

  // BACKEND INTEGRATION POINT: Make prediction
  // TODO: Send input data to best model API endpoint (/api/predict)
  // TODO: Payload format: { inputData: string, modelId: string, dataFormat: 'csv' | 'json' }
  // TODO: Handle prediction results and confidence scores from backend
  // EXPLANATION: The backend receives cleaned data from build page, trains multiple models,
  // stores them, and this endpoint uses the best-performing model to generate predictions
  const handlePredict = async () => {
    setIsLoading(true)

    // TODO: Backend - POST /api/predict
    // - Parse CSV/JSON input and validate against training features
    // - Pass through best model pipeline (same preprocessing as training)
    // - Return prediction + confidence + class probabilities

    console.log("[v0] Making prediction with input:", inputData)

    // Simulate prediction (replace with actual API call)
    setTimeout(() => {
      setPredictions({
        prediction: "Class A",
        confidence: 0.94,
        topPredictions: [
          { class: "Class A", probability: 0.94 },
          { class: "Class B", probability: 0.05 },
          { class: "Class C", probability: 0.01 },
        ],
      })
      setIsLoading(false)
    }, 1500)
  }

  return (
    <div className="container py-12 bg-gradient-to-b from-background to-background">
      <div className="mx-auto max-w-6xl space-y-8">
        {/* PAGE HEADER */}
        <div>
          <Link href="/build">
            <Button variant="outline" size="sm" className="mb-4 bg-transparent">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Build
            </Button>
          </Link>
          <h1 className="text-3xl font-bold tracking-tighter text-foreground">Make Predictions</h1>
          <p className="text-muted-foreground mt-2">
            Use your trained model to make predictions on new data. The model learned patterns from your training data.
          </p>
        </div>

        {/* LAYOUT: Model Comparison on left, Prediction input on right */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* MODEL COMPARISON SECTION */}
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle className="text-foreground text-lg">Model Comparison</CardTitle>
              <CardDescription>Performance of trained models</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* NOTE: In production, these values come from backend training results */}
              {modelComparison.map((model, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-lg border transition-colors ${
                    idx === 0 ? "border-green-600/50 bg-green-50 dark:bg-green-950/30" : "border-border bg-card"
                  }`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <p className="font-semibold text-sm text-foreground">{model.name}</p>
                      {idx === 0 && <p className="text-xs text-green-700 dark:text-green-400">✓ Best Model</p>}
                    </div>
                  </div>
                  <div className="space-y-1 text-xs text-muted-foreground">
                    <p>Accuracy: {model.accuracy}</p>
                    <p>Training: {model.training}</p>
                    <p>Inference: {model.inference}</p>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* PREDICTION INPUT SECTION */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="text-foreground">New Data for Prediction</CardTitle>
              <CardDescription>Enter or paste data for the best model to make predictions</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Tabs defaultValue="single" className="w-full">
                <TabsList className="grid w-full grid-cols-2 bg-muted">
                  <TabsTrigger value="single" className="text-foreground">
                    Single Record
                  </TabsTrigger>
                  <TabsTrigger value="batch" className="text-foreground">
                    Batch Predictions
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="single" className="space-y-4">
                  {/* TODO: Backend - Generate dynamic form fields based on training features */}
                  {/* The form fields should match the columns from the uploaded training dataset */}
                  <Textarea
                    placeholder="Enter feature values (e.g., feature1=5.2, feature2=3.1, feature3=1.5)..."
                    className="min-h-[120px] text-foreground bg-background"
                    value={inputData}
                    onChange={(e) => setInputData(e.target.value)}
                  />
                  <Button
                    onClick={handlePredict}
                    disabled={!inputData || isLoading}
                    className="w-full bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 hover:from-violet-700 hover:via-blue-600 hover:to-teal-500 text-white font-semibold"
                  >
                    <Zap className="h-4 w-4 mr-2" />
                    {isLoading ? "Making Prediction..." : "Get Prediction"}
                  </Button>
                </TabsContent>

                <TabsContent value="batch" className="space-y-4">
                  {/* TODO: Backend - Accept CSV with multiple records for batch prediction */}
                  {/* Returns array of predictions with confidence scores for each record */}
                  <Textarea
                    placeholder="Paste CSV with multiple records (with headers)..."
                    className="min-h-[120px] text-foreground bg-background"
                  />
                  <Button className="w-full bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 hover:from-violet-700 hover:via-blue-600 hover:to-teal-500 text-white font-semibold">
                    Batch Predict
                  </Button>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>

        {/* PREDICTION RESULTS */}
        {predictions && (
          <Card className="border-green-600/30 bg-green-50 dark:bg-green-950/20">
            <CardHeader>
              <CardTitle className="text-green-700 dark:text-green-400">✓ Prediction Result</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* MAIN PREDICTION DISPLAY */}
              <div className="bg-white dark:bg-slate-950 rounded-lg p-6 border border-green-600/30">
                <p className="text-sm text-muted-foreground mb-2">Predicted Value</p>
                <p className="text-3xl font-bold text-green-600 dark:text-green-400">{predictions.prediction}</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Confidence: {(predictions.confidence * 100).toFixed(2)}%
                </p>
              </div>

              {/* PROBABILITY DISTRIBUTION */}
              <div className="space-y-3">
                <p className="font-semibold text-foreground">Prediction Probabilities</p>
                {predictions.topPredictions.map((pred: any, idx: number) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-sm text-foreground">
                      <span>{pred.class}</span>
                      <span className="font-semibold">{(pred.probability * 100).toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 h-2 rounded-full transition-all"
                        style={{ width: `${pred.probability * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* ACTION BUTTONS */}
              <div className="flex gap-2 justify-between pt-4">
                <Button
                  variant="outline"
                  onClick={() => setPredictions(null)}
                  className="text-foreground border-border hover:bg-muted"
                >
                  Make Another Prediction
                </Button>
                <Button
                  onClick={() => handleExportResults(predictions)}
                  className="bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 hover:from-violet-700 hover:via-blue-600 hover:to-teal-500 text-white font-semibold"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Export Results
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* NAVIGATION BUTTONS */}
        <div className="flex gap-4 justify-between">
          <Link href="/results/metrics">
            <Button variant="outline" className="text-foreground border-border hover:bg-muted bg-transparent">
              View Metrics
            </Button>
          </Link>
          <Link href="/build">
            <Button className="bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 hover:from-violet-700 hover:via-blue-600 hover:to-teal-500 text-white font-semibold">
              Train New Model
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}
