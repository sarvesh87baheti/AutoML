import { NextResponse } from "next/server";
import path from "path";
import fs from "fs/promises";

export const runtime = "nodejs";

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url);
    const dataset = searchParams.get("dataset");

    if (!dataset) {
      return NextResponse.json({ error: "Dataset query parameter is required" }, { status: 400 });
    }

    const workspaceRoot = path.resolve(process.cwd(), "..");
    const imagePath = path.join(workspaceRoot, "main", "model_results", dataset, "confusion_matrix.png");

    const imageBuffer = await fs.readFile(imagePath);
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "no-store",
      },
    });
  } catch (err: any) {
    return NextResponse.json({ error: err?.message || "Confusion matrix not found" }, { status: 404 });
  }
}
