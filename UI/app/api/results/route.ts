import { NextResponse } from "next/server";
import path from "path";
import fs from "fs/promises";

export const runtime = "nodejs";

async function findLatestSummary(root: string) {
  const entries = await fs.readdir(root, { withFileTypes: true });
  let latest: { dataset: string; file: string; mtime: number } | null = null;
  for (const ent of entries) {
    if (!ent.isDirectory()) continue;
    const dataset = ent.name;
    const file = path.join(root, dataset, "training_summary.json");
    try {
      const stat = await fs.stat(file);
      const mtime = stat.mtimeMs;
      if (!latest || mtime > latest.mtime) latest = { dataset, file, mtime };
    } catch {
      continue;
    }
  }
  return latest;
}

export async function GET() {
  try {
    const workspaceRoot = path.resolve(process.cwd(), "..");
    const resultsRoot = path.join(workspaceRoot, "main", "model_results");
    const latest = await findLatestSummary(resultsRoot);
    if (!latest) return NextResponse.json({ error: "No training results found" }, { status: 404 });

    const summaryRaw = await fs.readFile(latest.file, "utf-8");
    const summary = JSON.parse(summaryRaw);
    const payload = summary?.results ?? summary;

    // Prefer explicit problem_type from the stored summary, fallback to metadata from processed data
    let problem_type: string | undefined = summary?.problem_type;
    const metaPath = path.join(workspaceRoot, "main", "processed_data", latest.dataset, "metadata.json");
    let metadata: Record<string, any> | undefined;
    try {
      const metaRaw = await fs.readFile(metaPath, "utf-8");
      metadata = JSON.parse(metaRaw);
      if (metadata?.problem_type) problem_type = metadata.problem_type;
    } catch {}

    if (!problem_type && Object.prototype.hasOwnProperty.call(payload, "image_classification")) {
      problem_type = "image_classification"
    }

    const results = {
      ...payload,
      best_model: summary?.best_model ?? payload?.best_model,
      model_scores: summary?.model_scores ?? payload?.model_scores,
      feature_importance: summary?.feature_importance ?? payload?.feature_importance,
      problem_type: problem_type ?? payload?.problem_type,
      dataset_name: summary?.dataset_name ?? payload?.dataset_name,
      image_mode: summary?.image_mode ?? payload?.image_mode,
    };

    return NextResponse.json({ dataset: latest.dataset, problem_type, metadata, results });
  } catch (err: any) {
    return NextResponse.json({ error: err?.message || String(err) }, { status: 500 });
  }
}
