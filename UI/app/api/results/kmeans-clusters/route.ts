import { NextResponse } from "next/server";
import path from "path";
import fs from "fs/promises";

export const runtime = "nodejs";

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const dataset = searchParams.get("dataset");
    const model = searchParams.get("model");

    if (!dataset) {
      return NextResponse.json({ error: "Dataset query parameter is required" }, { status: 400 });
    }

    const workspaceRoot = path.resolve(process.cwd(), "..");
    const resultsDir = path.join(workspaceRoot, "main", "model_results", dataset);

    // If a specific model image requested, prefer that
    if (model) {
      const modelPath = path.join(resultsDir, `${model}_clusters.png`);
      try {
        const imageBuffer = await fs.readFile(modelPath);
        return new NextResponse(imageBuffer, {
          status: 200,
          headers: { "Content-Type": "image/png", "Cache-Control": "no-store" },
        });
      } catch {}
    }

    // Otherwise, look up best model from summary
    const summaryPath = path.join(resultsDir, "training_summary.json");
    try {
      const raw = await fs.readFile(summaryPath, "utf-8");
      const summary = JSON.parse(raw);
      const best = summary?.best_model;
      if (best) {
        const candidate = path.join(resultsDir, `${best}_clusters.png`);
        try {
          const imageBuffer = await fs.readFile(candidate);
          return new NextResponse(imageBuffer, {
            status: 200,
            headers: { "Content-Type": "image/png", "Cache-Control": "no-store" },
          });
        } catch {}
      }
    } catch {}

    // Fallback to any kmeans_clusters.png file
    try {
      const fallback = path.join(resultsDir, `kmeans_clusters.png`);
      const imageBuffer = await fs.readFile(fallback);
      return new NextResponse(imageBuffer, {
        status: 200,
        headers: { "Content-Type": "image/png", "Cache-Control": "no-store" },
      });
    } catch (err: any) {
      return NextResponse.json({ error: err?.message || "KMeans cluster image not found" }, { status: 404 });
    }
  } catch (err: any) {
    return NextResponse.json({ error: err?.message || String(err) }, { status: 500 });
  }
}
