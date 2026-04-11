import { NextResponse } from "next/server";
import path from "path";
import fs from "fs/promises";
import { spawn } from "child_process";

export const runtime = "nodejs";

export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const file = form.get("dataset") as File | null;
    const problem_type = (form.get("problem_type") as string | null) || undefined;
    const target_col = (form.get("target_col") as string | null) || undefined;
    const k_value = (form.get("k") as string | null) || undefined;

    if (!file) {
      return NextResponse.json({ success: false, error: { code: "NO_FILE", message: "No file uploaded" } }, { status: 400 });
    }

    if ((problem_type === "regression" || problem_type === "classification") && !target_col) {
      return NextResponse.json({ success: false, error: { code: "TARGET_COLUMN_REQUIRED", message: "Target column required for supervised tasks" } }, { status: 400 });
    }

    if (problem_type === "kmeans_clustering" && !k_value) {
      return NextResponse.json({ success: false, error: { code: "K_REQUIRED", message: "k is required for kmeans clustering" } }, { status: 400 });
    }

    // Validate problem type
    const validProblemTypes = ["regression", "classification", "kmeans_clustering", "image_classification"];
    if (!validProblemTypes.includes(problem_type || "")) {
      return NextResponse.json({ success: false, error: { code: "INVALID_PROBLEM_TYPE", message: "Invalid problem type specified" } }, { status: 400 });
    }

    // Save uploaded file to workspace `uploaded_files/`
    const workspaceRoot = path.resolve(process.cwd(), "..");
    const uploadDir = path.join(workspaceRoot, "uploaded_files");
    await fs.mkdir(uploadDir, { recursive: true });

    const fileName = (file as File).name || `upload_${Date.now()}.csv`;
    const savePath = path.join(uploadDir, fileName);
    const arrayBuffer = await file.arrayBuffer();
    await fs.writeFile(savePath, Buffer.from(arrayBuffer));

    // Invoke Python runner.py with args
    const pythonPath = "python"; // assumes python is on PATH
    const runnerPath = path.join(workspaceRoot, "runner.py");

    const args: string[] = [runnerPath, "--file", savePath, "--problem", problem_type || "regression", "--json"];
    if (target_col) {
      args.push("--target", target_col);
    }
    if (k_value) {
      args.push("--k", k_value);
    }

    const result = await runPython(pythonPath, args);

    if (result && result.success === false) {
      const userErrorCodes = new Set([
        "TARGET_COLUMN_NOT_FOUND",
        "TARGET_COLUMN_REQUIRED",
        "DATASET_NOT_FOUND",
        "UNSUPPORTED_FORMAT",
        "EMPTY_ZIP",
        "INVALID_IMAGE_ZIP",
        "NO_FILE",
        "K_REQUIRED",
        "K_INVALID",
        "INVALID_PROBLEM_TYPE"
      ]);
      const status = userErrorCodes.has(result.error?.code) ? 400 : 500;
      return NextResponse.json(result, { status });
    }

    return NextResponse.json({ success: true, data: result }, { status: 200 });
  } catch (err: any) {
    return NextResponse.json({ success: false, error: { code: "SERVER_ERROR", message: err?.message || String(err) } }, { status: 500 });
  }
}

function parsePythonError(stderr: string) {
  const text = stderr || "";
  const lower = text.toLowerCase();
  let code = "PYTHON_ERROR";
  let message = text.trim() || "Unknown Python error";

  if (text.includes("Target column") && text.includes("not found")) {
    code = "TARGET_COLUMN_NOT_FOUND";
    const match = text.match(/Target column '([^']+)'/);
    if (match) {
      message = `Target column \"${match[1]}\" was not found in the uploaded dataset.`;
    } else {
      message = "Specified target column was not found in the uploaded dataset.";
    }
  } else if (text.includes("Target column must be provided")) {
    code = "TARGET_COLUMN_REQUIRED";
    message = "Please provide a target column name for this supervised task.";
  } else if (text.includes("Dataset not found")) {
    code = "DATASET_NOT_FOUND";
    message = "Uploaded dataset could not be located on the server.";
  } else if (text.includes("Unsupported format")) {
    code = "UNSUPPORTED_FORMAT";
    message = "Unsupported file format. Use CSV, XLS/XLSX, or ZIP for images.";
  } else if (text.includes("Number of clusters (k) must be provided")) {
    code = "K_REQUIRED";
    message = "Please provide k (number of clusters) for k-means clustering.";
  } else if (text.includes("Number of clusters (k) must be >= 2")) {
    code = "K_INVALID";
    message = "k must be greater than or equal to 2 for k-means clustering.";
  } else if (text.includes("ZIP contains no CSV/XLSX") || text.includes("ZIP contains no image")) {
    code = "EMPTY_ZIP";
    message = "ZIP file does not contain valid data or images.";
  } else if (text.includes("Invalid image ZIP") || text.includes("image classification") && text.includes("invalid")) {
    code = "INVALID_IMAGE_ZIP";
    message = "ZIP structure invalid. Expected: folders for each class with images inside.";
  } else if (text.includes("Unsupported problem type")) {
    code = "INVALID_PROBLEM_TYPE";
    message = "Invalid problem type specified.";
  }

  return { success: false, error: { code, message, raw: text } };
}

function runPython(pythonCmd: string, args: string[]): Promise<any> {
  return new Promise((resolve) => {
    let proc = spawn(pythonCmd, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";

    const attach = () => {
      proc.stdout.on("data", (d) => (stdout += d.toString()));
      proc.stderr.on("data", (d) => (stderr += d.toString()));
      proc.on("error", (e: any) => {
        if (e?.code === "ENOENT" && pythonCmd === "python") {
          proc.removeAllListeners();
          proc.kill();
          proc = spawn("py", args, { stdio: ["ignore", "pipe", "pipe"] });
          attach();
        } else {
          resolve(parsePythonError(e?.message || String(e)));
        }
      });
      proc.on("close", (code) => {
        if (code !== 0) {
          return resolve(parsePythonError(stderr || `runner.py exited with code ${code}`));
        }
        try {
          const parsed = JSON.parse(stdout.trim());
          resolve(parsed);
        } catch {
          resolve({ success: true, data: { message: "AutoML process completed", raw: stdout } });
        }
      });
    };
    attach();
  });
}
