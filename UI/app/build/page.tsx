
"use client"

import type React from "react"
import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import {
  FileText,
  Upload,
  BarChart3,
  Image,
  Sparkles,
  Layers3,
  CheckCircle2,
  FolderTree,
  ScanSearch,
  BrainCircuit,
} from "lucide-react"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import { postUpload } from "@/lib/api"
import { toast } from "@/hooks/use-toast"

type ProblemType = "classification" | "regression" | "kmeans_clustering" | "image_classification"

const problemTypeCards: {
  value: ProblemType
  title: string
  description: string
  accent: string
  icon: typeof BarChart3
}[] = [
  {
    value: "classification",
    title: "Classification",
    description: "Predict labeled categories from structured tabular data.",
    accent: "border-violet-600 bg-violet-500/5",
    icon: CheckCircle2,
  },
  {
    value: "regression",
    title: "Regression",
    description: "Estimate continuous values like score, price, or demand.",
    accent: "border-blue-500 bg-blue-500/5",
    icon: BarChart3,
  },
  {
    value: "kmeans_clustering",
    title: "K-Means Clustering",
    description: "Group unlabeled data into meaningful clusters.",
    accent: "border-teal-500 bg-teal-500/5",
    icon: Layers3,
  },
  {
    value: "image_classification",
    title: "Image Classification",
    description: "Train CNN backbones from folder-based image datasets packed as a ZIP.",
    accent: "border-orange-500 bg-orange-500/5",
    icon: Image,
  },
]

const imageWorkflowSteps = [
  {
    title: "Organize images by class",
    description: "Create one folder per class such as cats, dogs, or flowers.",
    icon: FolderTree,
  },
  {
    title: "Zip the dataset",
    description: "Upload a single ZIP file that preserves your folder structure.",
    icon: Upload,
  },
  {
    title: "Train multiple vision models",
    description: "EasyFlow compares transfer-learning backbones and chooses the strongest one.",
    icon: BrainCircuit,
  },
]

const supportedImageModels = [
  "MobileNetV2",
  "MobileNetV3Small",
  "EfficientNet",
  "ResNet50V2",
]

const tabularWorkflowContent: Record<
  Exclude<ProblemType, "image_classification">,
  {
    badge: string
    title: string
    description: string
    tips: string[]
    sideTitle: string
    sideBody: string
    sideAccent: string
  }
> = {
  classification: {
    badge: "Classification workflow",
    title: "Train a category prediction pipeline from tabular data",
    description:
      "Upload a labeled dataset, choose the target column, and let EasyFlow compare supervised classifiers for category prediction.",
    tips: [
      "Use a target column with clear class labels like churn, approval, or species.",
      "Keep identifiers like customer IDs out of the target column.",
      "Review class balance if one label appears much more often than the others.",
    ],
    sideTitle: "Best for",
    sideBody: "Fraud detection, disease class prediction, email spam detection, student outcome prediction.",
    sideAccent: "text-violet-600",
  },
  regression: {
    badge: "Regression workflow",
    title: "Estimate continuous values from structured features",
    description:
      "Point EasyFlow at your numeric outcome column and it will benchmark multiple regressors, compare fit quality, and surface the strongest model.",
    tips: [
      "Use a numeric target such as price, score, sales, or demand.",
      "Missing values and mixed categorical columns are handled automatically.",
      "The results page will highlight R² and actual-vs-predicted behavior.",
    ],
    sideTitle: "Best for",
    sideBody: "Price forecasting, exam-score prediction, demand estimation, energy or sales modeling.",
    sideAccent: "text-blue-500",
  },
  kmeans_clustering: {
    badge: "Clustering workflow",
    title: "Discover hidden groupings in unlabeled tabular data",
    description:
      "Choose the number of clusters and EasyFlow will preprocess your features, fit k-means, and report clustering quality metrics.",
    tips: [
      "No target column is needed for clustering.",
      "Choose a starting k based on domain knowledge, then compare silhouette and inertia.",
      "Works well for segmentation, grouping, and exploratory analysis.",
    ],
    sideTitle: "Best for",
    sideBody: "Customer segmentation, product grouping, anomaly exploration, or discovering latent dataset structure.",
    sideAccent: "text-teal-500",
  },
}

export default function BuildPage() {
  const [dataSource, setDataSource] = useState("upload")
  const [file, setFile] = useState<File | null>(null)
  const [csvData, setCsvData] = useState("")
  const [imageTrainingMode, setImageTrainingMode] = useState<"light" | "standard">("light")

  const [problemType, setProblemType] = useState<ProblemType | null>(null)

  const [targetColumn, setTargetColumn] = useState("")
  const [showTargetInput, setShowTargetInput] = useState(false)
  const [kValue, setKValue] = useState("3")

  const [isProcessing, setIsProcessing] = useState(false)
  const [trainingComplete, setTrainingComplete] = useState(false)

  const [fileColumns, setFileColumns] = useState<string[]>([])
  const [columnsError, setColumnsError] = useState<string | null>(null)
  const isImageClassification = problemType === "image_classification"
  const selectedProblemCard = problemTypeCards.find((item) => item.value === problemType)
  const isTabularTask = problemType === "classification" || problemType === "regression" || problemType === "kmeans_clustering"
  const tabularContent = isTabularTask ? tabularWorkflowContent[problemType] : null

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setFileColumns([])
      setColumnsError(null)

      const f = e.target.files[0]
      if (f.name.toLowerCase().endsWith(".csv")) {
        const reader = new FileReader()
        reader.onerror = () => setColumnsError("Failed to read file")
        reader.onload = () => {
          try {
            const text = reader.result as string
            const firstLine = text.split(/\r?\n/)[0]
            const headers = firstLine
              .split(",")
              .map((h) => h.trim().replace(/^"|"$/g, ""))
              .filter((h) => h.length > 0)
            if (headers.length > 0) setFileColumns(headers)
            else setColumnsError("No headers found")
          } catch (err: any) {
            setColumnsError(err?.message || "Error parsing CSV header")
          }
        }
        reader.readAsText(f)
      }
      // For ZIP files (image classification), skip header parsing
    }
  }

  const handleBuildModel = async () => {
    try {
      setIsProcessing(true)

      if (dataSource === "upload") {
        if (!file || !problemType) {
          toast({
            title: "Missing Inputs",
            description: "Please select a file and choose a problem type.",
            variant: "destructive",
          })
          return
        }

        // Validate file type based on problem type
        if (isImageClassification && !file.name.toLowerCase().endsWith(".zip")) {
          toast({
            title: "Invalid File Format",
            description: "Please upload a ZIP file containing image folders for image classification.",
            variant: "destructive",
          })
          return
        }

        if (!isImageClassification) {
          const fileName = file.name.toLowerCase()
          if (!fileName.endsWith(".csv") && !fileName.endsWith(".xlsx") && !fileName.endsWith(".xls") && !fileName.endsWith(".zip")) {
            toast({
              title: "Invalid File Format",
              description: "Please upload a CSV, Excel, or ZIP file.",
              variant: "destructive",
            })
            return
          }
        }

        if (showTargetInput && targetColumn && fileColumns.length > 0) {
          if (!fileColumns.includes(targetColumn)) {
            toast({
              title: "Target Column Missing",
              description: `Column '${targetColumn}' not found in file headers.`,
              variant: "destructive",
            })
            return
          }
        }

        if (problemType === "kmeans_clustering" && (!kValue || Number(kValue) < 2)) {
          toast({
            title: "Invalid k",
            description: "Please provide k as a number greater than or equal to 2.",
            variant: "destructive",
          })
          return
        }

        const formData = new FormData()
        formData.append("dataset", file)
        formData.append("problem_type", problemType)
        if (problemType === "classification" || problemType === "regression") {
          formData.append("target_col", targetColumn)
        }
        if (problemType === "kmeans_clustering") {
          formData.append("k", kValue)
        }
        if (problemType === "image_classification") {
          formData.append("image_mode", imageTrainingMode)
        }

        const result = await postUpload(formData)
        console.log("Upload result:", result)

        toast({
          title: "Training Started",
          description: "Dataset uploaded successfully. Training in progress...",
        })
      }

      setTrainingComplete(true)
      toast({
        title: "Training Complete",
        description: "Your model has been trained successfully!",
      })
    } catch (err: any) {
      console.error("UPLOAD ERROR:", err)

      const code = err?.code || err?.response?.data?.code
      let friendly = err?.message || err?.response?.data?.message || "Failed to build model."
      let title = "Upload Error"

      if (code === "TARGET_COLUMN_NOT_FOUND") {
        title = "Target Column Missing"
        friendly = `Column '${targetColumn}' not found. Check name and try again.`
      } else if (code === "TARGET_COLUMN_REQUIRED") {
        title = "Target Column Required"
        friendly = "Please enter a target column for this task."
      } else if (code === "K_REQUIRED" || code === "K_INVALID") {
        title = "Invalid k"
        friendly = "Please enter a valid number of clusters (k ≥ 2)."
      } else if (code === "UNSUPPORTED_FORMAT") {
        title = "Unsupported Format"
        friendly = "Use CSV, XLS/XLSX for tabular data, or ZIP for image classification."
      } else if (code === "EMPTY_ZIP") {
        title = "Empty ZIP"
        friendly = "ZIP contained no usable files or folders."
      } else if (code === "INVALID_IMAGE_ZIP") {
        title = "Invalid Image ZIP"
        friendly = "ZIP should contain folders for each class with images inside."
      } else if (code === "INVALID_IMAGE_MODE") {
        title = "Invalid Training Mode"
        friendly = "Choose a valid image training mode and try again."
      }

      toast({
        title,
        description: friendly,
        variant: "destructive",
      })
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="container py-12">
      <div className="mx-auto max-w-5xl space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
            Build Your ML Model
          </h1>
          <p className="text-muted-foreground">
            Upload your dataset and configure your learning task.
          </p>
        </div>

        {!trainingComplete && (
          <Card>
            <CardHeader>
              <CardTitle>Model Configuration</CardTitle>
              <CardDescription>
                Pick a workflow, provide your dataset, and let EasyFlow ML prepare the training pipeline.
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-6">
              <div className="space-y-3">
                <Label>Problem Type</Label>
                <RadioGroup
                  value={problemType || ""}
                  onValueChange={(val) => {
                    setProblemType(val as ProblemType)
                    setShowTargetInput(val === "classification" || val === "regression")
                    setFile(null)
                    setFileColumns([])
                    setColumnsError(null)
                    setDataSource("upload")
                  }}
                >
                  <div className="grid gap-3 md:grid-cols-2">
                    {problemTypeCards.map(({ value, title, description, accent, icon: Icon }) => (
                      <label
                        key={value}
                        htmlFor={value}
                        className={cn(
                          "flex cursor-pointer gap-4 rounded-xl border p-4 transition-all hover:border-primary/40 hover:bg-muted/40",
                          problemType === value && accent
                        )}
                      >
                        <RadioGroupItem value={value} id={value} className="mt-1" />
                        <div className="space-y-1">
                          <div className="flex items-center gap-2 font-medium">
                            <Icon className="h-4 w-4" />
                            <span>{title}</span>
                          </div>
                          <p className="text-sm text-muted-foreground">{description}</p>
                        </div>
                      </label>
                    ))}
                  </div>
                </RadioGroup>
              </div>

              {selectedProblemCard && (
                <div
                  className={cn(
                    "rounded-2xl border px-5 py-4",
                    isImageClassification
                      ? "border-orange-500/40 bg-gradient-to-br from-orange-500/10 via-amber-500/5 to-background"
                      : "border-border bg-muted/30"
                  )}
                >
                  <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                    <div className="space-y-2">
                      <div className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium uppercase tracking-[0.2em] text-muted-foreground">
                        <Sparkles className="h-3.5 w-3.5" />
                        {selectedProblemCard.title} workflow
                      </div>
                      <h2 className="text-2xl font-semibold">
                        {isImageClassification ? "Build an image classifier from a zipped folder dataset" : "Configure your dataset and training inputs"}
                      </h2>
                      <p className="max-w-2xl text-sm text-muted-foreground">
                        {isImageClassification
                          ? "Upload a ZIP where each class lives in its own folder. EasyFlow will preprocess the images, fine-tune multiple transfer-learning models, and rank the candidates for you."
                          : "Choose the right file, provide the required columns, and EasyFlow will handle preprocessing, model training, and evaluation automatically."}
                      </p>
                    </div>

                    {isImageClassification && (
                      <div className="rounded-xl border border-orange-500/30 bg-background/80 px-4 py-3 text-sm shadow-sm">
                        <p className="font-medium">Recommended ZIP structure</p>
                        <pre className="mt-2 text-xs text-muted-foreground">
{`dataset.zip
  cats/
    cat-001.jpg
  dogs/
    dog-001.jpg`}
                        </pre>
                      </div>
                    )}
                  </div>

                  {isImageClassification && (
                    <div className="mt-5 grid gap-4 lg:grid-cols-[1.4fr_1fr]">
                      <div className="grid gap-3 sm:grid-cols-3">
                        {imageWorkflowSteps.map(({ title, description, icon: Icon }) => (
                          <div key={title} className="rounded-xl border border-orange-500/20 bg-background/80 p-4">
                            <Icon className="h-5 w-5 text-orange-500" />
                            <p className="mt-3 font-medium">{title}</p>
                            <p className="mt-1 text-sm text-muted-foreground">{description}</p>
                          </div>
                        ))}
                      </div>

                      <div className="rounded-xl border border-orange-500/20 bg-background/80 p-4">
                        <div className="flex items-center gap-2 font-medium">
                          <ScanSearch className="h-4 w-4 text-orange-500" />
                          Vision backbones available
                        </div>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {supportedImageModels.map((model) => (
                            <span
                              key={model}
                              className="rounded-full border border-orange-500/20 bg-orange-500/10 px-3 py-1 text-xs font-medium"
                            >
                              {model}
                            </span>
                          ))}
                        </div>
                        <p className="mt-3 text-sm text-muted-foreground">
                          Best for student projects, offline demos, and small datasets where fast comparison matters more than manual tuning.
                        </p>
                      </div>
                    </div>
                  )}

                  {tabularContent && (
                    <div className="mt-5 grid gap-4 lg:grid-cols-[1.35fr_0.95fr]">
                      <div className="rounded-xl border bg-background/80 p-5">
                        <div className="inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium uppercase tracking-[0.18em] text-muted-foreground">
                          {tabularContent.badge}
                        </div>
                        <h3 className="mt-3 text-xl font-semibold">{tabularContent.title}</h3>
                        <p className="mt-2 text-sm text-muted-foreground">{tabularContent.description}</p>

                        <div className="mt-4 grid gap-3">
                          {tabularContent.tips.map((tip) => (
                            <div key={tip} className="flex gap-3 rounded-xl border p-3">
                              <CheckCircle2 className={cn("mt-0.5 h-4 w-4 shrink-0", tabularContent.sideAccent)} />
                              <p className="text-sm text-muted-foreground">{tip}</p>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="rounded-xl border bg-background/80 p-5">
                        <p className={cn("text-sm font-medium", tabularContent.sideAccent)}>{tabularContent.sideTitle}</p>
                        <p className="mt-2 text-sm text-muted-foreground">{tabularContent.sideBody}</p>

                        <div className="mt-5 rounded-xl border bg-muted/30 p-4">
                          <p className="text-sm font-medium text-foreground">Expected file format</p>
                          <p className="mt-1 text-sm text-muted-foreground">CSV, XLSX, XLS, or ZIP</p>
                        </div>

                        <div className="mt-3 rounded-xl border bg-muted/30 p-4">
                          <p className="text-sm font-medium text-foreground">
                            {problemType === "kmeans_clustering" ? "Required input" : "Required input"}
                          </p>
                          <p className="mt-1 text-sm text-muted-foreground">
                            {problemType === "classification" || problemType === "regression"
                              ? "Select a dataset and provide the target column."
                              : "Select a dataset and choose the number of clusters k."}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div className="space-y-3">
                <Label>Data Source</Label>
                {isImageClassification ? (
                  <div className="rounded-2xl border-2 border-dashed border-orange-500/30 bg-gradient-to-br from-orange-500/10 via-background to-amber-500/5 p-8 text-center">
                    <div className="mx-auto flex max-w-xl flex-col items-center space-y-4">
                      <div className="rounded-full bg-orange-500/10 p-4">
                        <Image className="h-8 w-8 text-orange-500" />
                      </div>
                      <div className="space-y-2">
                        <h3 className="text-xl font-semibold">Upload image dataset ZIP</h3>
                        <p className="text-sm text-muted-foreground">
                          Use one folder per class inside the ZIP. JPG, JPEG and PNG images work best for training.
                        </p>
                      </div>

                      <Button variant="outline" size="lg" onClick={() => document.getElementById("data-upload")?.click()}>
                        <Upload className="mr-2 h-4 w-4" />
                        Select ZIP File
                      </Button>
                      <Input id="data-upload" type="file" accept=".zip" className="hidden" onChange={handleFileChange} />

                      {file ? (
                        <div className="rounded-xl border border-green-600/30 bg-green-500/5 px-4 py-3 text-sm text-green-700">
                          Ready for training: <span className="font-medium">{file.name}</span>
                        </div>
                      ) : (
                        <p className="text-xs text-muted-foreground">A single ZIP keeps the upload portable and offline-friendly.</p>
                      )}
                    </div>
                  </div>
                ) : (
                  <Tabs value={dataSource} onValueChange={setDataSource}>
                    <TabsList className="grid grid-cols-2">
                      <TabsTrigger value="upload">
                        <Upload className="mr-2 h-4 w-4" />
                        Upload File
                      </TabsTrigger>
                      <TabsTrigger value="paste">
                        <FileText className="mr-2 h-4 w-4" />
                        Paste CSV
                      </TabsTrigger>
                    </TabsList>

                    <TabsContent value="upload">
                      <div className="border-2 border-dashed p-8 rounded-lg text-center">
                        <Button
                          variant="outline"
                          onClick={() => document.getElementById("data-upload")?.click()}
                        >
                          Select File
                        </Button>
                        <Input id="data-upload" type="file" accept=".csv,.xlsx,.xls,.zip"
                          className="hidden" onChange={handleFileChange}
                        />

                        {file && <p className="mt-3 text-sm text-green-600">Selected: {file.name}</p>}
                        {fileColumns.length > 0 && (
                          <p className="text-xs text-muted-foreground mt-2">
                            Detected columns: {fileColumns.slice(0, 8).join(", ")}
                            {fileColumns.length > 8 && " ..."}
                          </p>
                        )}
                      </div>
                    </TabsContent>

                    <TabsContent value="paste">
                      <Textarea
                        placeholder="Paste CSV data here..."
                        value={csvData}
                        onChange={(e) => setCsvData(e.target.value)}
                        className="min-h-[200px]"
                      />
                    </TabsContent>
                  </Tabs>
                )}
                {columnsError && !isImageClassification && (
                  <p className="text-sm text-red-500">{columnsError}</p>
                )}
              </div>

              {isImageClassification && (
                <div className="rounded-xl border border-orange-500/20 bg-orange-500/5 p-4">
                  <div className="space-y-3">
                    <Label>Training Mode</Label>
                    <RadioGroup
                      value={imageTrainingMode}
                      onValueChange={(value) => setImageTrainingMode(value as "light" | "standard")}
                      className="grid gap-3 md:grid-cols-2"
                    >
                      <label
                        htmlFor="image-mode-light"
                        className={cn(
                          "flex cursor-pointer gap-4 rounded-xl border p-4 transition-all",
                          imageTrainingMode === "light" && "border-orange-500 bg-background"
                        )}
                      >
                        <RadioGroupItem value="light" id="image-mode-light" className="mt-1" />
                        <div>
                          <p className="font-medium">Light Mode</p>
                          <p className="text-sm text-muted-foreground">
                            Trains only `MobileNetV2` first for the fastest result.
                          </p>
                        </div>
                      </label>

                      <label
                        htmlFor="image-mode-standard"
                        className={cn(
                          "flex cursor-pointer gap-4 rounded-xl border p-4 transition-all",
                          imageTrainingMode === "standard" && "border-orange-500 bg-background"
                        )}
                      >
                        <RadioGroupItem value="standard" id="image-mode-standard" className="mt-1" />
                        <div>
                          <p className="font-medium">Standard Mode</p>
                          <p className="text-sm text-muted-foreground">
                            Runs the recommended multi-backbone comparison after preprocessing.
                          </p>
                        </div>
                      </label>
                    </RadioGroup>
                  </div>
                </div>
              )}

              {showTargetInput && (
                <div className="p-4 border rounded-lg space-y-2 bg-muted/30">
                  <Label>Target Column *</Label>
                  <Input
                    placeholder="Enter target column name"
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                  />
                </div>
              )}

              {problemType === "kmeans_clustering" && (
                <div className="p-4 border rounded-lg space-y-2 bg-muted/30">
                  <Label>Number of Clusters (k) *</Label>
                  <Input
                    type="number"
                    min={2}
                    placeholder="Enter k"
                    value={kValue}
                    onChange={(e) => setKValue(e.target.value)}
                  />
                </div>
              )}

              <Button
                onClick={handleBuildModel}
                disabled={
                  isProcessing ||
                  !problemType ||
                  (dataSource === "upload" && !file && !isImageClassification) ||
                  (!isImageClassification && dataSource === "upload" && !file) ||
                  (isImageClassification && !file) ||
                  (dataSource === "paste" && !csvData) ||
                  (showTargetInput && !targetColumn) ||
                  (problemType === "kmeans_clustering" && (!kValue || Number(kValue) < 2))
                }
                className={cn(
                  "w-full text-white",
                  isImageClassification
                    ? "bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500"
                    : "bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400"
                )}
              >
                {isProcessing
                  ? isImageClassification
                    ? imageTrainingMode === "light"
                      ? "Training MobileNetV2…"
                      : "Training Image Models…"
                    : "Building Model…"
                  : isImageClassification
                  ? imageTrainingMode === "light"
                    ? "Build & Train MobileNetV2"
                    : "Build & Train Image Classifier"
                  : "Build & Train Model"}
              </Button>
            </CardContent>
          </Card>
        )}

        {trainingComplete && (
          <Card className="border-green-600/30 bg-green-500/5">
            <CardHeader>
              <CardTitle className="text-green-700">
                {isImageClassification ? "Image Classification Training Complete!" : "Training Complete!"}
              </CardTitle>
              <CardDescription>
                {isImageClassification
                  ? "Your image dataset has been processed and the trained backbones are ready for evaluation."
                  : "Your model has been trained successfully."}
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link href="/results/metrics">
                  <Button className="w-full" variant="outline">
                    <BarChart3 className="mr-2" /> View Metric Charts
                  </Button>
                </Link>
                {/* <Link href="/results/predictions">
                  <Button className="w-full" variant="outline">
                    <Wand2 className="mr-2" /> Make Predictions
                  </Button>
                </Link> */}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
