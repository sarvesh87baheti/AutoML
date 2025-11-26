// FILE PURPOSE: Results index page - redirects users to appropriate results view
// This prevents users from accessing /results directly

"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"

export default function ResultsPage() {
  const router = useRouter()

  useEffect(() => {
    router.push("/build")
  }, [router])

  return null
}
