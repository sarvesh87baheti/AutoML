// FILE PURPOSE: Model Building Page - Main interface for users to configure and train ML models
// Handles: data upload/input, problem type selection, target column specification
// BACKEND INTEGRATION: All TODO comments mark API integration points

"use client"

import type React from "react"
import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { FileText, Upload, BarChart3, Wand2 } from "lucide-react"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

// COMPONENT STATE: Manages all form data, file uploads, and training state
export default function BuildPage() {
  // DATA SOURCE STATE - Determines if user uploads file or pastes CSV
  const [dataSource, setDataSource] = useState("upload") // "upload" or "paste"
  const [file, setFile] = useState<File | null>(null)
  const [csvData, setCsvData] = useState("")

  // PROBLEM TYPE STATE - User selects ML task (classification, regression, clustering)
  const [problemType, setProblemType] = useState<"classification" | "regression" | "clustering" | null>(null)

  // TARGET COLUMN STATE - Only shown for supervised learning (classification/regression)
  const [targetColumn, setTargetColumn] = useState("")
  const [showTargetInput, setShowTargetInput] = useState(false)

  // TRAINING STATE - Tracks if model is currently training and stores results
  const [isProcessing, setIsProcessing] = useState(false)
  const [trainingComplete, setTrainingComplete] = useState(false)

  // BACKEND INTEGRATION POINT: File upload handler
  // TODO: Add file validation (size, format), extract column names from CSV
  // TODO: Display extracted columns for target column selection dropdown
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      console.log("[v0] File selected:", e.target.files[0].name)
    }
  }

  // BACKEND INTEGRATION POINT: Model training trigger
  // TODO: Send request to backend API at POST /api/train-model
  // TODO: Include payload structure shown below in training request
  // TODO: Handle streaming progress updates during training
  const handleBuildModel = async () => {
    setIsProcessing(true)

    // PAYLOAD FOR BACKEND - Structure for API call
    const payload = {
      // Data source information
      dataSource: dataSource,
      file: file ? file.name : null,
      csvData: csvData || null,

      // Problem configuration
      problemType: problemType, // "classification", "regression", or "clustering"

      // Target column (null for clustering since it's unsupervised)
      targetColumn: problemType !== "clustering" ? targetColumn : null,
    }

    console.log("[v0] Sending to backend:", payload)

    // TODO: Backend API call - Replace with actual endpoint
    // Example:
    // const response = await fetch('/api/train-model', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(payload)
    // })
    // const result = await response.json()
    // Handle errors and store model ID, training metrics

    // Simulate API response
    setTimeout(() => {
      setTrainingComplete(true)
      setIsProcessing(false)
    }, 2000)
  }

  return (
    <div className="container py-12">
      <div className="mx-auto max-w-5xl space-y-8">
        {/* PAGE HEADER */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Build Your ML Model</h1>
          <p className="text-muted-foreground">
            Upload your dataset and configure your learning task. We'll handle the rest!
          </p>
        </div>

        {/* TRAINING WORKFLOW - Only show this section until training is complete */}
        {!trainingComplete && (
          <Card>
            <CardHeader>
              <CardTitle>Model Configuration</CardTitle>
              <CardDescription>
                Select your data source, choose a problem type, and optionally specify a target column for supervised
                learning.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* SECTION 1: DATA SOURCE SELECTION - Upload file or paste CSV data */}
              <div className="space-y-3">
                <Label>Data Source</Label>
                <Tabs defaultValue="upload" value={dataSource} onValueChange={setDataSource} className="w-full">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="upload">
                      <Upload className="h-4 w-4 mr-2" />
                      Upload File
                    </TabsTrigger>
                    <TabsTrigger value="paste">
                      <FileText className="h-4 w-4 mr-2" />
                      Paste CSV
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="upload" className="mt-4">
                    {/* FILE UPLOAD AREA */}
                    <div className="flex flex-col items-center justify-center border-2 border-dashed rounded-md p-12 text-center">
                      <Upload className="h-8 w-8 mb-4 text-muted-foreground" />
                      <div className="space-y-2">
                        <p className="text-sm font-medium">
                          Drag and drop your CSV or Excel file here, or click to browse
                        </p>
                        <p className="text-xs text-muted-foreground">Supported: CSV, XLSX, XLS (Max 100MB)</p>
                      </div>
                      <Input
                        type="file"
                        accept=".csv,.xlsx,.xls"
                        className="hidden"
                        id="data-upload"
                        onChange={handleFileChange}
                      />
                      <Button
                        variant="outline"
                        className="mt-4 bg-transparent"
                        onClick={() => document.getElementById("data-upload")?.click()}
                      >
                        Select File
                      </Button>
                      {file && <p className="mt-2 text-sm font-medium text-green-600">Selected: {file.name}</p>}
                    </div>
                  </TabsContent>

                  <TabsContent value="paste" className="mt-4">
                    {/* CSV DATA PASTE AREA */}
                    <Textarea
                      placeholder="Paste your CSV data here (comma-separated values with headers)..."
                      className="min-h-[200px]"
                      value={csvData}
                      onChange={(e) => setCsvData(e.target.value)}
                    />
                  </TabsContent>
                </Tabs>
              </div>

              {/* SECTION 2: PROBLEM TYPE SELECTION - Choose ML task type */}
              <div className="space-y-3">
                <Label>Problem Type</Label>
                <RadioGroup
                  value={problemType || ""}
                  onValueChange={(val) => {
                    setProblemType(val as "classification" | "regression" | "clustering")
                    // Show target column input ONLY for supervised learning tasks
                    setShowTargetInput(val !== "clustering")
                  }}
                >
                  <div className="space-y-3">
                    {/* CLASSIFICATION OPTION - Supervised learning: predict categories */}
                    <div
                      className={cn(
                        "flex items-start space-x-3 rounded-md border p-4 cursor-pointer",
                        problemType === "classification" &&
                          "border-violet-600 bg-gradient-to-br from-violet-600/5 via-blue-500/5 to-teal-400/5",
                      )}
                    >
                      <RadioGroupItem value="classification" id="classification" className="mt-1" />
                      <Label htmlFor="classification" className="cursor-pointer font-normal flex-1">
                        <div className="font-medium">Classification</div>
                        <p className="text-sm text-muted-foreground">
                          Predict categorical outcomes (e.g., spam/not spam, cat/dog/bird). Requires target column.
                        </p>
                      </Label>
                    </div>

                    {/* REGRESSION OPTION - Supervised learning: predict continuous values */}
                    <div
                      className={cn(
                        "flex items-start space-x-3 rounded-md border p-4 cursor-pointer",
                        problemType === "regression" &&
                          "border-blue-500 bg-gradient-to-br from-violet-600/5 via-blue-500/5 to-teal-400/5",
                      )}
                    >
                      <RadioGroupItem value="regression" id="regression" className="mt-1" />
                      <Label htmlFor="regression" className="cursor-pointer font-normal flex-1">
                        <div className="font-medium">Regression</div>
                        <p className="text-sm text-muted-foreground">
                          Predict continuous numerical values (e.g., price, temperature). Requires target column.
                        </p>
                      </Label>
                    </div>

                    {/* CLUSTERING OPTION - Unsupervised learning: group similar data points */}
                    <div
                      className={cn(
                        "flex items-start space-x-3 rounded-md border p-4 cursor-pointer",
                        problemType === "clustering" &&
                          "border-teal-500 bg-gradient-to-br from-violet-600/5 via-blue-500/5 to-teal-400/5",
                      )}
                    >
                      <RadioGroupItem value="clustering" id="clustering" className="mt-1" />
                      <Label htmlFor="clustering" className="cursor-pointer font-normal flex-1">
                        <div className="font-medium">Clustering</div>
                        <p className="text-sm text-muted-foreground">
                          Group similar data points together (unsupervised). No target column needed.
                        </p>
                      </Label>
                    </div>
                  </div>
                </RadioGroup>
              </div>

              {/* SECTION 3: TARGET COLUMN INPUT - Only appears for classification/regression */}
              {showTargetInput && (
                <div className="space-y-3 p-4 rounded-lg bg-gradient-to-br from-violet-600/5 via-blue-500/5 to-teal-400/5 border">
                  <Label htmlFor="target-column">
                    Target Column
                    <span className="text-red-500"> *</span>
                  </Label>
                  <Input
                    id="target-column"
                    placeholder={
                      problemType === "classification"
                        ? "e.g., 'category', 'label', 'class'"
                        : "e.g., 'price', 'temperature', 'score'"
                    }
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Enter the column name from your dataset that contains the values you want to predict.
                    {problemType === "classification" && " This should be a categorical column."}
                    {problemType === "regression" && " This should be a numerical column."}
                  </p>
                  {/* TODO: Backend - Dynamically populate column names from uploaded file */}
                  {/* TODO: Add dropdown selector for target column from file headers */}
                </div>
              )}

              {/* BUILD & TRAIN BUTTON - Triggers model training API call */}
              <Button
                onClick={handleBuildModel}
                disabled={
                  isProcessing ||
                  !problemType ||
                  (dataSource === "upload" && !file) ||
                  (dataSource === "paste" && !csvData) ||
                  (showTargetInput && !targetColumn)
                }
                className="w-full bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 text-white"
              >
                {isProcessing ? "Building Model..." : "Build & Train Model"}
              </Button>
            </CardContent>
          </Card>
        )}

        {/* TRAINING COMPLETE - Show options for viewing results */}
        {trainingComplete && (
          <Card className="border-green-600/30 bg-gradient-to-br from-green-600/5 to-green-500/5">
            <CardHeader>
              <CardTitle className="text-green-700 dark:text-green-400">Training Complete!</CardTitle>
              <CardDescription>
                Your model has been successfully trained and evaluated. View the results below.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* OPTIONS FOR VIEWING RESULTS */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* OPTION 1: VIEW METRIC CHARTS - Performance metrics and comparisons */}
                <Link href="/results/metrics">
                  <Button variant="outline" className="w-full h-full flex flex-col items-center justify-center p-6 bg-transparent bg-gradient-to-br from-violet-600/10 via-blue-500/10 to-teal-400/10 border border-violet-600/30 hover:border-violet-600/60 text-foreground hover:bg-gradient-to-br hover:from-violet-600/20 hover:via-blue-500/20 hover:to-teal-400/20">
                    <BarChart3 className="h-8 w-8 mb-2" />
                    <span className="font-semibold">View Metric Charts</span>
                    <span className="text-xs text-muted-foreground mt-1">
                      Compare performance metrics and model accuracy
                    </span>
                  </Button>
                </Link>

                {/* OPTION 2: MAKE PREDICTIONS - Use best model for predictions */}
                <Link href="/results/predictions">
                  <Button variant="outline" className="w-full h-full flex flex-col items-center justify-center p-6 bg-transparent bg-gradient-to-br from-violet-600/10 via-blue-500/10 to-teal-400/10 border border-violet-600/30 hover:border-violet-600/60 text-foreground hover:bg-gradient-to-br hover:from-violet-600/20 hover:via-blue-500/20 hover:to-teal-400/20">
                    <Wand2 className="h-8 w-8 mb-2" />
                    <span className="font-semibold">Make Predictions</span>
                    <span className="text-xs text-muted-foreground mt-1">
                      Use the best model to make predictions on new data
                    </span>
                  </Button>
                </Link>
              </div>

              {/* ACTION BUTTONS */}
              <div className="flex gap-2 justify-end">
                <Button
                  variant="outline"
                  onClick={() => {
                    setTrainingComplete(false)
                    setFile(null)
                    setCsvData("")
                    setProblemType(null)
                    setTargetColumn("")
                  }}
                >
                  Train Another Model
                </Button>
                <Button className="bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 text-white">
                  Export Model
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
