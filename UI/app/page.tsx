// FILE PURPOSE: Homepage for EasyFlow ML - Landing page with hero, features, and CTA sections
// Showcases the main value proposition and directs users to start building models

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { BarChart3, Zap, Database, Settings } from "lucide-react"

export default function Home() {
  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      {/* HERO SECTION - Main landing area with headline and CTA buttons */}
      <section className="relative">
        <div className="absolute inset-0 bg-gradient-to-br from-violet-600/20 via-blue-500/20 to-teal-400/20 pointer-events-none" />
        <div className="container relative flex flex-col items-center justify-center space-y-8 py-24 text-center md:py-32">
          <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl md:text-6xl lg:text-7xl">
            Turn Raw Data into{" "}
            <span className="bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 bg-clip-text text-transparent">
              Smart Predictions — Without Code
            </span>
          </h1>

          <p className="max-w-[700px] text-lg text-muted-foreground md:text-xl">
            EasyFlow ML is an automated machine learning platform built for everyone. Simply upload your dataset, choose
            your problem type — classification, regression, or clustering — and EasyFlow ML automatically cleans your
            data, trains multiple models, compares performance, and delivers the best model in minutes. No coding. No ML
            expertise. Just results.
          </p>

          <div className="flex flex-col sm:flex-row gap-4">
            <Button asChild size="lg" className="bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 text-white">
              <Link href="/build">Start Building Models</Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <Link href="/about">Learn More</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* FEATURES SECTION - Showcase key capabilities of EasyFlow ML */}
      <section className="container py-16 md:py-24">
        <h2 className="text-3xl font-bold tracking-tighter text-center mb-12 sm:text-4xl">Powerful AutoML Features</h2>
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
          {/* FEATURE 1: Smart Model Selection */}
          <div className="flex flex-col items-center text-center p-6 rounded-lg border bg-card">
            <div className="rounded-full bg-gradient-to-br from-violet-600/20 via-blue-500/20 to-teal-400/20 p-3 mb-4">
              <BarChart3 className="h-6 w-6 text-violet-600" />
            </div>
            <h3 className="text-xl font-bold mb-2">Smart Model Selection</h3>
            <p className="text-muted-foreground">
              Automatically trains and compares multiple machine learning algorithms and selects the best-performing
              model based on accuracy, F1-score, RMSE, and other metrics.
            </p>
          </div>

          {/* FEATURE 2: Automatic Data Preprocessing */}
          <div className="flex flex-col items-center text-center p-6 rounded-lg border bg-card">
            <div className="rounded-full bg-gradient-to-br from-violet-600/20 via-blue-500/20 to-teal-400/20 p-3 mb-4">
              <Settings className="h-6 w-6 text-blue-500" />
            </div>
            <h3 className="text-xl font-bold mb-2">Automatic Data Preprocessing</h3>
            <p className="text-muted-foreground">
              Handles missing values, feature encoding, normalization, scaling, and train-test splits — fully automated
              with zero manual effort.
            </p>
          </div>

          {/* FEATURE 3: Classification, Regression & Clustering */}
          <div className="flex flex-col items-center text-center p-6 rounded-lg border bg-card">
            <div className="rounded-full bg-gradient-to-br from-violet-600/20 via-blue-500/20 to-teal-400/20 p-3 mb-4">
              <Database className="h-6 w-6 text-blue-400" />
            </div>
            <h3 className="text-xl font-bold mb-2">Classification, Regression & Clustering</h3>
            <p className="text-muted-foreground">
              Supports all major machine learning tasks — classification for categories, regression for continuous
              values, and clustering for discovering hidden patterns in unlabeled data.
            </p>
          </div>

          {/* FEATURE 4: Model Evaluation & Leaderboard */}
          <div className="flex flex-col items-center text-center p-6 rounded-lg border bg-card">
            <div className="rounded-full bg-gradient-to-br from-violet-600/20 via-blue-500/20 to-teal-400/20 p-3 mb-4">
              <Zap className="h-6 w-6 text-teal-400" />
            </div>
            <h3 className="text-xl font-bold mb-2">Model Evaluation & Leaderboard</h3>
            <p className="text-muted-foreground">
              Visual performance comparison of all trained models using accuracy, precision, recall, F1-score, confusion
              matrix, and error metrics.
            </p>
          </div>
        </div>
      </section>

      {/* CTA SECTION - Final call-to-action before footer */}
      <section className="bg-gradient-to-br from-violet-600 via-blue-500 to-teal-400 py-16 md:py-24 mt-auto">
        <div className="container flex flex-col items-center justify-center space-y-6 text-center">
          <h2 className="text-3xl font-bold tracking-tighter text-white sm:text-4xl">
            Ready to Build Your First ML Model?
          </h2>
          <p className="max-w-[600px] text-white/90 md:text-lg">
            Upload your dataset today and discover the power of automated machine learning. Get started in minutes.
          </p>
          <Button asChild size="lg" className="bg-white text-violet-600 hover:bg-white/90">
            <Link href="/build">Try EasyFlow ML Now</Link>
          </Button>
        </div>
      </section>
    </div>
  )
}
