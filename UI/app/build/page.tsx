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
import { postUpload } from "@/lib/api"
import { toast } from "@/hooks/use-toast"

export default function BuildPage() {
  const [dataSource, setDataSource] = useState("upload")
  const [file, setFile] = useState<File | null>(null)
  const [csvData, setCsvData] = useState("")

  const [problemType, setProblemType] =
    useState<"classification" | "regression" | "clustering" | null>(null)

  const [targetColumn, setTargetColumn] = useState("")
  const [showTargetInput, setShowTargetInput] = useState(false)

  const [isProcessing, setIsProcessing] = useState(false)
  const [trainingComplete, setTrainingComplete] = useState(false)

  const [fileColumns, setFileColumns] = useState<string[]>([])
  const [columnsError, setColumnsError] = useState<string | null>(null)

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

        const formData = new FormData()
        formData.append("dataset", file)
        formData.append("problem_type", problemType)
        if (problemType !== "clustering") {
          formData.append("target_col", targetColumn)
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
      } else if (code === "UNSUPPORTED_FORMAT") {
        title = "Unsupported Format"
        friendly = "Use CSV, XLS/XLSX or ZIP containing one."
      } else if (code === "EMPTY_ZIP") {
        title = "Empty ZIP"
        friendly = "ZIP contained no usable file."
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
                Select your data source, choose a problem type, and optionally specify a target column.
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-6">
              <div className="space-y-3">
                <Label>Data Source</Label>
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
                      <Input id="data-upload" type="file" accept=".csv,.xlsx,.xls"
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
              </div>

              <div className="space-y-3">
                <Label>Problem Type</Label>
                <RadioGroup
                  value={problemType || ""}
                  onValueChange={(val) => {
                    setProblemType(val as any)
                    setShowTargetInput(val !== "clustering")
                  }}
                >
                  <div className="space-y-3">
                    <div className={cn("flex gap-3 p-4 border rounded-md cursor-pointer",
                      problemType === "classification" && "border-violet-600")}
                    >
                      <RadioGroupItem value="classification" id="classification" />
                      <Label htmlFor="classification" className="cursor-pointer">Classification</Label>
                    </div>

                    <div className={cn("flex gap-3 p-4 border rounded-md cursor-pointer",
                      problemType === "regression" && "border-blue-500")}
                    >
                      <RadioGroupItem value="regression" id="regression" />
                      <Label htmlFor="regression" className="cursor-pointer">Regression</Label>
                    </div>

                    <div className={cn("flex gap-3 p-4 border rounded-md cursor-pointer",
                      problemType === "clustering" && "border-teal-500")}
                    >
                      <RadioGroupItem value="clustering" id="clustering" />
                      <Label htmlFor="clustering" className="cursor-pointer">Clustering</Label>
                    </div>
                  </div>
                </RadioGroup>
              </div>

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
                {isProcessing ? "Building Model…" : "Build & Train Model"}
              </Button>
            </CardContent>
          </Card>
        )}

        {trainingComplete && (
          <Card className="border-green-600/30 bg-green-500/5">
            <CardHeader>
              <CardTitle className="text-green-700">Training Complete!</CardTitle>
              <CardDescription>Your model has been trained successfully.</CardDescription>
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
