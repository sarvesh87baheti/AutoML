// FILE PURPOSE: About page - Information about EasyFlow ML, mission, and how it works

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { CheckCircle2 } from "lucide-react"
import Link from "next/link"

export default function AboutPage() {
  return (
    <div className="container py-12">
      <div className="mx-auto max-w-4xl space-y-12">
        {/* PAGE HEADER */}
        <div className="text-center space-y-4">
          <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
            About{" "}
            <span className="bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 bg-clip-text text-transparent">
              EasyFlow ML
            </span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Democratizing machine learning by making it accessible to everyone, regardless of technical expertise.
          </p>
        </div>

        {/* MISSION & TECHNOLOGY CARDS */}
        <div className="grid gap-8 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Our Mission</CardTitle>
            </CardHeader>
            <CardContent>
              <p>
                At EasyFlow ML, we believe machine learning should be accessible to everyone. Our mission is to
                eliminate the barriers to building predictive models by providing an automated platform that handles
                data preprocessing, model selection, hyperparameter tuning, and evaluation—all without requiring coding
                or deep ML expertise.
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Our Technology</CardTitle>
            </CardHeader>
            <CardContent>
              <p>
                We leverage advanced machine learning algorithms, ensemble methods, and automated machine learning
                (AutoML) techniques to analyze your data and build optimal models. Our intelligent pipeline
                automatically handles feature engineering, model selection across multiple algorithms, and
                hyperparameter optimization to deliver the best results for your specific problem.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* HOW IT WORKS SECTION */}
        <div className="space-y-6">
          <h2 className="text-2xl font-bold tracking-tighter sm:text-3xl text-center">How EasyFlow ML Works</h2>
          <div className="grid gap-6">
            {/* STEP 1: DATA UPLOAD */}
            <div className="flex gap-4">
              <div className="rounded-full bg-gradient-to-br from-violet-600/20 via-blue-500/20 to-teal-400/20 p-2 h-fit">
                <CheckCircle2 className="h-5 w-5 text-violet-600" />
              </div>
              <div>
                <h3 className="text-xl font-medium mb-2">Upload Your Data</h3>
                <p className="text-muted-foreground">
                  Upload your raw dataset in CSV or Excel format. Our system automatically validates and prepares your
                  data for modeling.
                </p>
              </div>
            </div>

            {/* STEP 2: SELECT PROBLEM TYPE */}
            <div className="flex gap-4">
              <div className="rounded-full bg-gradient-to-br from-violet-600/20 via-blue-500/20 to-teal-400/20 p-2 h-fit">
                <CheckCircle2 className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <h3 className="text-xl font-medium mb-2">Choose Your Problem Type</h3>
                <p className="text-muted-foreground">
                  Select from classification, regression, or clustering based on your business goal. Specify your target
                  column for supervised learning tasks.
                </p>
              </div>
            </div>

            {/* STEP 3: AUTOMATIC PREPROCESSING AND TRAINING */}
            <div className="flex gap-4">
              <div className="rounded-full bg-gradient-to-br from-violet-600/20 via-blue-500/20 to-teal-400/20 p-2 h-fit">
                <CheckCircle2 className="h-5 w-5 text-teal-400" />
              </div>
              <div>
                <h3 className="text-xl font-medium mb-2">Automatic Training & Evaluation</h3>
                <p className="text-muted-foreground">
                  Our pipeline automatically handles preprocessing, trains multiple models, performs hyperparameter
                  tuning, and evaluates performance to deliver the best model for your data.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* FINAL CTA SECTION */}
        <div className="bg-gradient-to-br from-violet-600/10 via-blue-500/10 to-teal-400/10 rounded-lg p-8 text-center">
          <h2 className="text-2xl font-bold tracking-tighter sm:text-3xl mb-4">Start Building ML Models Today</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto mb-6">
            No machine learning experience required. Upload your data and let EasyFlow ML do the heavy lifting. Get your
            first model in minutes.
          </p>
          <Link
            href="/build"
            className="inline-flex h-10 items-center justify-center rounded-md bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 px-8 text-sm font-medium text-white shadow transition-colors hover:opacity-90"
          >
            Try It Now
          </Link>
        </div>
      </div>
    </div>
  )
}
