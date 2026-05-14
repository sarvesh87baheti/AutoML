/**
 * /api/validate-scripts
 * ---------------------
 * Lightweight pre-validation endpoint.  The frontend calls this immediately
 * when the user selects custom script files, before clicking "Build & Train".
 * It runs runner.py --validate-only and returns per-file results as JSON so
 * the UI can show green/red status on each pill without starting a training run.
 *
 * Request (multipart/form-data):
 *   problem_type          string   — "classification" | "regression" | "kmeans_clustering"
 *   custom_script_0       File     — first  .py file (required)
 *   custom_script_1       File     — second .py file (optional)
 *   custom_script_2       File     — third  .py file (optional)
 *
 * Response 200:
 *   {
 *     success: true,
 *     results: [
 *       { filename: "my_model.py", valid: true,  reason: "ok"           },
 *       { filename: "broken.py",   valid: false, reason: "missing MODEL_NAME" },
 *     ]
 *   }
 *
 * Response 400: missing/invalid inputs.
 * Response 500: Python could not be spawned.
 *
 * Temp files are deleted after the response is built regardless of outcome.
 */

import { NextResponse } from "next/server";
import path from "path";
import fs from "fs/promises";
import { spawn } from "child_process";
import os from "os";

export const runtime = "nodejs";

const MAX_CUSTOM_SCRIPTS = 3;

// ---------------------------------------------------------------------------
// POST handler
// ---------------------------------------------------------------------------

export async function POST(req: Request) {
  // Temp dir for this validation run — cleaned up in finally block.
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "easyflow-validate-"));

  try {
    const form = await req.formData();
    const problem_type = (form.get("problem_type") as string | null) || "";

    if (!problem_type) {
      return NextResponse.json(
        {
          success: false,
          error: { code: "PROBLEM_TYPE_REQUIRED", message: "problem_type is required" },
        },
        { status: 400 }
      );
    }

    const validProblemTypes = ["regression", "classification", "kmeans_clustering"];
    if (!validProblemTypes.includes(problem_type)) {
      return NextResponse.json(
        {
          success: false,
          error: {
            code: "INVALID_PROBLEM_TYPE",
            message: `Custom scripts are not supported for '${problem_type}'.`,
          },
        },
        { status: 400 }
      );
    }

    // Collect uploaded script files
    const scriptFiles: File[] = [];
    for (let i = 0; i < MAX_CUSTOM_SCRIPTS; i++) {
      const f = form.get(`custom_script_${i}`) as File | null;
      if (f && f.size > 0) scriptFiles.push(f);
    }

    if (scriptFiles.length === 0) {
      return NextResponse.json(
        {
          success: false,
          error: { code: "NO_SCRIPTS", message: "At least one script file is required." },
        },
        { status: 400 }
      );
    }

    // Validate extensions before touching disk
    for (const sf of scriptFiles) {
      if (!sf.name.toLowerCase().endsWith(".py")) {
        return NextResponse.json(
          {
            success: false,
            error: {
              code: "INVALID_SCRIPT_EXTENSION",
              message: `All custom scripts must be .py files (got '${sf.name}').`,
            },
          },
          { status: 400 }
        );
      }
    }

    // Save to temp dir
    const savedPaths: string[] = [];
    for (const sf of scriptFiles) {
      const dest = path.join(tmpDir, sf.name);
      const buf  = await sf.arrayBuffer();
      await fs.writeFile(dest, Buffer.from(buf));
      savedPaths.push(dest);
    }

    // Build runner.py --validate-only args
    const workspaceRoot = path.resolve(process.cwd(), "..");
    const runnerPath    = path.join(workspaceRoot, "runner.py");

    const args: string[] = [
      runnerPath,
      "--validate-only",
      "--problem", problem_type,
      "--custom-scripts", ...savedPaths,
    ];

    // Spawn Python
    const raw = await runPython(args);

    // If Python itself failed to spawn or runner exited with error
    if (raw?.success === false) {
      return NextResponse.json(
        {
          success: false,
          error: raw.error ?? { code: "PYTHON_ERROR", message: "Validation subprocess failed." },
        },
        { status: 500 }
      );
    }

    // runner.py --validate-only prints: { "results": [...] }
    // We surface it as-is with success: true
    return NextResponse.json({ success: true, results: raw.results ?? [] }, { status: 200 });
  } catch (err: any) {
    return NextResponse.json(
      {
        success: false,
        error: { code: "SERVER_ERROR", message: err?.message || String(err) },
      },
      { status: 500 }
    );
  } finally {
    // Always clean up temp files — fire-and-forget, don't await in the response path.
    fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {
      // Non-fatal — OS will clean up on restart anyway.
    });
  }
}

// ---------------------------------------------------------------------------
// Python subprocess helpers
// (Identical to upload/route.ts — keeps each route self-contained so they
// can be changed independently without risk of shared-module coupling.)
// ---------------------------------------------------------------------------

function parsePythonError(stderr: string) {
  const text = stderr || "";

  if (
    text.includes("spawn python ENOENT") ||
    text.includes("spawn python3 ENOENT") ||
    text.includes("spawn py ENOENT")
  ) {
    return {
      success: false,
      error: {
        code:    "PYTHON_NOT_FOUND",
        message: "Python was not found on the server.",
        raw:     text,
      },
    };
  }

  return {
    success: false,
    error: { code: "PYTHON_ERROR", message: text.trim() || "Unknown Python error", raw: text },
  };
}

function spawnOnce(pythonCmd: string, args: string[]): Promise<any> {
  return new Promise((resolve) => {
    const proc = spawn(pythonCmd, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stdout  = "";
    let stderr  = "";
    let settled = false;

    proc.stdout.on("data", (d) => (stdout += d.toString()));
    proc.stderr.on("data", (d) => (stderr += d.toString()));

    proc.on("error", (e: any) => {
      if (settled) return;
      settled = true;
      resolve({
        success: false,
        error: {
          code:    e?.code    || "SPAWN_ERROR",
          message: e?.message || String(e),
          raw:     e?.stack   || String(e),
        },
      });
    });

    proc.on("close", (code, signal) => {
      if (settled) return;
      settled = true;

      if (code !== 0) {
        const failureText =
          stderr ||
          (signal
            ? `runner.py was terminated by signal ${signal}`
            : `runner.py exited with code ${String(code)}`);
        return resolve(parsePythonError(failureText));
      }

      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        // --validate-only always prints JSON; if parsing fails something went wrong
        resolve({
          success: false,
          error: { code: "PARSE_ERROR", message: "Could not parse validator output.", raw: stdout },
        });
      }
    });
  });
}

async function runPython(args: string[]): Promise<any> {
  const pythonCommands = ["python", "python3", "py"];
  let lastError: any = null;

  for (const pythonCmd of pythonCommands) {
    const result = await spawnOnce(pythonCmd, args);
    if (result?.success === false && result?.error?.code === "ENOENT") {
      lastError = result;
      continue;
    }
    return result;
  }

  return parsePythonError(
    lastError?.error?.message || "Python executable not found"
  );
}