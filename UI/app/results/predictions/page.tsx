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
