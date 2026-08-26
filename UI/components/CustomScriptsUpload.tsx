"use client"

/**
 * CustomScriptsUpload
 * -------------------
 * Renders a collapsible "Custom Model Scripts (optional)" section below the
 * Data Source block on the Build Model page.
 *
 * Props
 * -----
 * problemType   — current selection.  When "image_classification" the entire
 *                 section is hidden (custom scripts not supported for images).
 * onFilesChange — callback fired whenever the validated file list changes.
 *                 The parent receives only the File objects that are currently
 *                 shown in the list (valid or pending); filtering to only valid
 *                 ones before submission is the parent's responsibility via the
 *                 `validationStatuses` snapshot it can read through the ref, or
 *                 more simply by checking the status map exposed via the callback.
 * onStatusChange — callback fired with the current per-filename status map so
 *                  the parent can gate the submit button or show the invalid-
 *                  script modal.
 */

import type React from "react"
import { useRef, useState } from "react"
import { Upload, X, CheckCircle2, XCircle, Loader2, ChevronDown, ChevronUp, FileCode2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input }  from "@/components/ui/input"
import { cn } from "@/lib/utils"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ValidationStatus =
  | { state: "idle"       }
  | { state: "validating" }
  | { state: "valid";   reason: string }
  | { state: "invalid"; reason: string }

export type ScriptEntry = {
  file:   File
  status: ValidationStatus
}

export type StatusMap = Record<string, ValidationStatus>  // keyed by filename

interface CustomScriptsUploadProps {
  problemType:    string | null
  onFilesChange:  (files: File[])    => void
  onStatusChange: (map: StatusMap)   => void
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_SCRIPTS = 3

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CustomScriptsUpload({
  problemType,
  onFilesChange,
  onStatusChange,
}: CustomScriptsUploadProps) {
  const [open,    setOpen]    = useState(false)
  const [entries, setEntries] = useState<ScriptEntry[]>([])
  const inputRef = useRef<HTMLInputElement>(null)

  // Hide entirely for image classification
  if (problemType === "image_classification") return null

  // ── Helpers ───────────────────────────────────────────────────────────────

  /** Push a status update for one file and propagate to parent. */
  function updateStatus(filename: string, status: ValidationStatus, currentEntries: ScriptEntry[]) {
    const next = currentEntries.map((e) =>
      e.file.name === filename ? { ...e, status } : e
    )
    setEntries(next)
    onFilesChange(next.map((e) => e.file))
    onStatusChange(Object.fromEntries(next.map((e) => [e.file.name, e.status])))
    return next
  }

  /** Call the validate-scripts API for a single file entry. */
  async function validateFile(file: File, currentEntries: ScriptEntry[]) {
    if (!problemType) return

    // Mark as validating
    const withValidating = updateStatus(file.name, { state: "validating" }, currentEntries)

    try {
      const fd = new FormData()
      fd.append("problem_type", problemType)
      fd.append("custom_script_0", file)

      const res  = await fetch("/api/validate-scripts", { method: "POST", body: fd })
      const json = await res.json()

      if (!json.success) {
        updateStatus(
          file.name,
          { state: "invalid", reason: json.error?.message ?? "Validation failed." },
          withValidating
        )
        return
      }

      // json.results is [{ filename, valid, reason }]
      const entry = (json.results as { filename: string; valid: boolean; reason: string }[])
        .find((r) => r.filename === file.name)

      if (!entry) {
        updateStatus(file.name, { state: "invalid", reason: "No validation result returned." }, withValidating)
        return
      }

      updateStatus(
        file.name,
        entry.valid
          ? { state: "valid",   reason: entry.reason }
          : { state: "invalid", reason: entry.reason },
        withValidating
      )
    } catch {
      updateStatus(file.name, { state: "invalid", reason: "Network error during validation." }, withValidating)
    }
  }

  /** Handle file input change event. */
  async function handleFilePick(e: React.ChangeEvent<HTMLInputElement>) {
    const picked = Array.from(e.target.files ?? [])
    // Reset input so the same file can be re-selected after removal
    if (inputRef.current) inputRef.current.value = ""

    if (picked.length === 0) return

    // Enforce .py extension
    const nonPy = picked.filter((f) => !f.name.toLowerCase().endsWith(".py"))
    if (nonPy.length > 0) {
      // Just skip non-.py silently — the input accept=".py" should already
      // prevent this, but browsers aren't always consistent.
      return
    }

    // Enforce 3-file cap (including already-loaded entries)
    const available = MAX_SCRIPTS - entries.length
    const toAdd = picked.slice(0, available)

    // Avoid duplicates by filename
    const existing = new Set(entries.map((e) => e.file.name))
    const fresh = toAdd.filter((f) => !existing.has(f.name))

    if (fresh.length === 0) return

    const newEntries: ScriptEntry[] = fresh.map((f) => ({
      file:   f,
      status: { state: "idle" },
    }))

    const nextEntries = [...entries, ...newEntries]
    setEntries(nextEntries)
    onFilesChange(nextEntries.map((e) => e.file))
    onStatusChange(Object.fromEntries(nextEntries.map((e) => [e.file.name, e.status])))

    // Kick off validation for each new file
    for (const entry of newEntries) {
      // Pass the latest snapshot so each async call doesn't stomp the others.
      // We validate sequentially to avoid race conditions in the state updater.
      await validateFile(entry.file, nextEntries)
    }
  }

  /** Remove a file from the list. */
  function removeFile(filename: string) {
    const next = entries.filter((e) => e.file.name !== filename)
    setEntries(next)
    onFilesChange(next.map((e) => e.file))
    onStatusChange(Object.fromEntries(next.map((e) => [e.file.name, e.status])))
  }

  // ── Derived state ─────────────────────────────────────────────────────────

  const hasAny      = entries.length > 0
  const atCap       = entries.length >= MAX_SCRIPTS
  const anyInvalid  = entries.some((e) => e.status.state === "invalid")
  const anyPending  = entries.some((e) => e.status.state === "validating" || e.status.state === "idle")

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="rounded-xl border bg-muted/20">
      {/* ── Header / toggle ──────────────────────────────────────────────── */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-5 py-4 text-left"
      >
        <div className="flex items-center gap-3">
          <FileCode2 className="h-5 w-5 text-muted-foreground" />
          <div>
            <p className="font-medium text-sm">Custom Model Scripts</p>
            <p className="text-xs text-muted-foreground">
              {hasAny
                ? `${entries.length} / ${MAX_SCRIPTS} script${entries.length > 1 ? "s" : ""} added`
                : "Optional — upload up to 3 .py model scripts"}
            </p>
          </div>

          {/* Status indicators in collapsed state */}
          {!open && hasAny && (
            <div className="flex items-center gap-1.5 ml-2">
              {anyInvalid && (
                <span className="rounded-full bg-red-500/10 px-2 py-0.5 text-xs font-medium text-red-600">
                  {entries.filter((e) => e.status.state === "invalid").length} invalid
                </span>
              )}
              {anyPending && (
                <span className="rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
                  checking…
                </span>
              )}
              {!anyInvalid && !anyPending && (
                <span className="rounded-full bg-green-500/10 px-2 py-0.5 text-xs font-medium text-green-700">
                  all valid
                </span>
              )}
            </div>
          )}
        </div>
        {open ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
      </button>

      {/* ── Expanded body ────────────────────────────────────────────────── */}
      {open && (
        <div className="border-t px-5 pb-5 pt-4 space-y-4">
          {/* Description */}
          <p className="text-sm text-muted-foreground">
            Upload your own sklearn-compatible model scripts. Each script must subclass{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-xs">ModelScript</code> and implement{" "}
            <code className="rounded bg-muted px-1 py-0.5 text-xs">train_model()</code>. Your
            models will be trained alongside the built-in ones and appear on the results page with
            a <span className="font-medium text-teal-600">Custom</span> badge.{" "}
            <a
              href="https://github.com/sarvesh87baheti/AutoML/blob/main/main/model_scripts/lasso.py"
              target="_blank"
              rel="noreferrer"
              className="underline underline-offset-2 text-primary"
            >
              See example script ↗
            </a>
          </p>

          {/* File list */}
          {hasAny && (
            <div className="space-y-2">
              {entries.map(({ file, status }) => (
                <ScriptPill
                  key={file.name}
                  filename={file.name}
                  status={status}
                  onRemove={() => removeFile(file.name)}
                />
              ))}
            </div>
          )}

          {/* Drop zone / picker — hidden when at cap */}
          {!atCap && (
            <div
              className={cn(
                "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-6 text-center transition-colors",
                "hover:border-primary/40 hover:bg-muted/30"
              )}
              onDragOver={(e) => e.preventDefault()}
              onDrop={async (e) => {
                e.preventDefault()
                const dt = e.dataTransfer
                if (!dt) return
                const synth = { target: { files: dt.files } } as unknown as React.ChangeEvent<HTMLInputElement>
                await handleFilePick(synth)
              }}
            >
              <Upload className="h-6 w-6 text-muted-foreground" />
              <div className="space-y-1">
                <p className="text-sm font-medium">
                  Drag &amp; drop .py files here, or{" "}
                  <button
                    type="button"
                    className="text-primary underline underline-offset-2"
                    onClick={() => inputRef.current?.click()}
                  >
                    browse
                  </button>
                </p>
                <p className="text-xs text-muted-foreground">
                  {MAX_SCRIPTS - entries.length} slot{MAX_SCRIPTS - entries.length !== 1 ? "s" : ""} remaining · .py files only
                </p>
              </div>
              <Input
                ref={inputRef}
                type="file"
                accept=".py"
                multiple
                className="hidden"
                onChange={handleFilePick}
              />
            </div>
          )}

          {atCap && (
            <p className="text-center text-xs text-muted-foreground">
              Maximum of {MAX_SCRIPTS} scripts reached. Remove one to add another.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// ScriptPill — one row per uploaded script
// ---------------------------------------------------------------------------

function ScriptPill({
  filename,
  status,
  onRemove,
}: {
  filename: string
  status:   ValidationStatus
  onRemove: () => void
}) {
  const isValid    = status.state === "valid"
  const isInvalid  = status.state === "invalid"
  const isChecking = status.state === "validating"

  return (
    <div
      className={cn(
        "flex items-center gap-3 rounded-lg border px-3 py-2.5 text-sm transition-colors",
        isValid   && "border-green-600/30 bg-green-500/5",
        isInvalid && "border-red-500/30   bg-red-500/5",
        isChecking && "border-border       bg-muted/30"
      )}
    >
      {/* Status icon */}
      <span className="shrink-0">
        {isChecking && <Loader2  className="h-4 w-4 animate-spin text-muted-foreground" />}
        {isValid    && <CheckCircle2 className="h-4 w-4 text-green-600" />}
        {isInvalid  && <XCircle  className="h-4 w-4 text-red-500" />}
        {status.state === "idle" && <FileCode2 className="h-4 w-4 text-muted-foreground" />}
      </span>

      {/* Filename + reason */}
      <div className="min-w-0 flex-1">
        <p className="truncate font-medium">{filename}</p>
        {isInvalid && (
          <p className="truncate text-xs text-red-500 mt-0.5">{status.reason}</p>
        )}
        {isValid && (
          <p className="text-xs text-green-700 mt-0.5">Valid — ready to train</p>
        )}
        {isChecking && (
          <p className="text-xs text-muted-foreground mt-0.5">Validating…</p>
        )}
      </div>

      {/* Remove button */}
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="h-7 w-7 shrink-0 text-muted-foreground hover:text-destructive"
        onClick={onRemove}
      >
        <X className="h-3.5 w-3.5" />
        <span className="sr-only">Remove {filename}</span>
      </Button>
    </div>
  )
}
