"use client"

/**
 * InvalidScriptModal
 * ------------------
 * Shown when the user clicks "Build & Train Model" while one or more uploaded
 * custom scripts have failed validation.  Offers two clear exits:
 *
 *   1. "Proceed with default scripts only" — strips invalid scripts and submits.
 *   2. "Upload again"                      — closes the modal so the user can fix
 *                                            or replace the offending files.
 */

import { XCircle, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface InvalidScriptEntry {
  filename: string
  reason:   string
}

interface InvalidScriptModalProps {
  /** Whether the modal is visible. */
  open: boolean
  /** Scripts that failed validation. */
  invalidScripts: InvalidScriptEntry[]
  /** Whether ANY scripts are valid (gates the "Proceed" label). */
  hasValidScripts: boolean
  /** Called when the user chooses to proceed without the invalid scripts. */
  onProceed: () => void
  /** Called when the user wants to go back and fix / replace scripts. */
  onUploadAgain: () => void
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function InvalidScriptModal({
  open,
  invalidScripts,
  hasValidScripts,
  onProceed,
  onUploadAgain,
}: InvalidScriptModalProps) {
  if (!open) return null

  const proceedLabel = hasValidScripts
    ? "Proceed with default scripts only"
    : "Proceed without custom scripts"

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm"
        aria-hidden="true"
        onClick={onUploadAgain}  // clicking outside = "go back"
      />

      {/* Dialog */}
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="invalid-scripts-title"
        className={cn(
          "fixed left-1/2 top-1/2 z-50 w-full max-w-md -translate-x-1/2 -translate-y-1/2",
          "rounded-2xl border bg-background p-6 shadow-xl",
          "flex flex-col gap-5"
        )}
      >
        {/* Header */}
        <div className="flex items-start gap-3">
          <div className="rounded-full bg-red-500/10 p-2 shrink-0">
            <AlertTriangle className="h-5 w-5 text-red-500" />
          </div>
          <div>
            <h2 id="invalid-scripts-title" className="text-base font-semibold leading-tight">
              {invalidScripts.length === 1
                ? "1 script failed validation"
                : `${invalidScripts.length} scripts failed validation`}
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              The following scripts don&apos;t meet the required format and cannot be
              trained. You can proceed using only the built-in models, or go back
              to fix or replace them.
            </p>
          </div>
        </div>

        {/* Script list */}
        <ul className="space-y-2">
          {invalidScripts.map(({ filename, reason }) => (
            <li
              key={filename}
              className="flex items-start gap-2.5 rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-2.5"
            >
              <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />
              <div className="min-w-0">
                <p className="truncate text-sm font-medium">{filename}</p>
                <p className="text-xs text-red-500 mt-0.5">{reason}</p>
              </div>
            </li>
          ))}
        </ul>

        {/* Actions */}
        <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          {/* Secondary — go back and fix */}
          <Button
            variant="outline"
            onClick={onUploadAgain}
            className="sm:w-auto w-full"
          >
            Upload again
          </Button>

          {/* Primary — strip invalid and submit */}
          <Button
            onClick={onProceed}
            className="sm:w-auto w-full bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 text-white"
          >
            {proceedLabel}
          </Button>
        </div>
      </div>
    </>
  )
}
