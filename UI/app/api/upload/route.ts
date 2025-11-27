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

    if (!file) {
      return NextResponse.json({ success: false, error: { code: "NO_FILE", message: "No file uploaded" } }, { status: 400 });
    }

    if ((problem_type === "regression" || problem_type === "classification") && !target_col) {
      return NextResponse.json({ success: false, error: { code: "TARGET_COLUMN_REQUIRED", message: "Target column required for supervised tasks" } }, { status: 400 });
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

    const result = await runPython(pythonPath, args);

    if (result && result.success === false) {
      const userErrorCodes = new Set([
        "TARGET_COLUMN_NOT_FOUND",
        "TARGET_COLUMN_REQUIRED",
        "DATASET_NOT_FOUND",
        "UNSUPPORTED_FORMAT",
        "EMPTY_ZIP",
        "NO_FILE"
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
    message = "Unsupported file format. Use CSV, XLS/XLSX or ZIP containing a CSV/XLSX.";
  } else if (text.includes("ZIP contains no CSV/XLSX")) {
    code = "EMPTY_ZIP";
    message = "ZIP file does not contain a CSV or XLS/XLSX file.";
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
