"use client"

import React, { useRef, useState } from "react"

export default function ImageCompressor() {
  const [colors, setColors] = useState(8)
  const [originalUrl, setOriginalUrl] = useState<string | null>(null)
  const [processing, setProcessing] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const resultRef = useRef<HTMLCanvasElement | null>(null)

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files && e.target.files[0]
    if (!f) return
    const url = URL.createObjectURL(f)
    setOriginalUrl(url)
    setTimeout(() => runCompress(url, colors), 50)
  }

  async function runCompress(url: string, k: number) {
    setProcessing(true)
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.src = url
    await new Promise((res) => (img.onload = res))

    const maxW = 600
    const scale = Math.min(1, maxW / img.width)
    const w = Math.round(img.width * scale)
    const h = Math.round(img.height * scale)

    const canvas = canvasRef.current!
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext("2d")!
    ctx.drawImage(img, 0, 0, w, h)
    const imgData = ctx.getImageData(0, 0, w, h)

    // prepare pixel array
    const pixels: number[][] = []
    for (let i = 0; i < imgData.data.length; i += 4) {
      const r = imgData.data[i]
      const g = imgData.data[i + 1]
      const b = imgData.data[i + 2]
      pixels.push([r, g, b])
    }

    const { centers, labels } = kmeans(pixels, k, 20)

    // create result image
    const out = new ImageData(w, h)
    for (let i = 0; i < labels.length; i++) {
      const c = centers[labels[i]]
      out.data[i * 4 + 0] = c[0]
      out.data[i * 4 + 1] = c[1]
      out.data[i * 4 + 2] = c[2]
      out.data[i * 4 + 3] = 255
    }

    const rctx = resultRef.current!.getContext("2d")!
    resultRef.current!.width = w
    resultRef.current!.height = h
    rctx.putImageData(out, 0, 0)

    setProcessing(false)
  }

  function downloadResult() {
    const canvas = resultRef.current
    if (!canvas) return
    const url = canvas.toDataURL("image/png")
    const a = document.createElement("a")
    a.href = url
    a.download = "compressed.png"
    a.click()
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-4 items-center">
        <input type="file" accept="image/*" onChange={handleFile} />
        <div className="flex items-center gap-2">
          <label className="text-sm">Colors</label>
          <input
            aria-label="colors"
            type="range"
            min={2}
            max={64}
            value={colors}
            onChange={(e) => setColors(Number(e.target.value))}
          />
          <span className="w-10 text-right">{colors}</span>
        </div>
        <button
          className="bg-primary text-white px-3 py-1 rounded"
          onClick={() => originalUrl && runCompress(originalUrl, colors)}
          disabled={processing || !originalUrl}
        >
          {processing ? "Processing..." : "Apply"}
        </button>
        <button
          className="border px-3 py-1 rounded"
          onClick={downloadResult}
          disabled={!resultRef.current}
        >
          Download
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="border p-2">
          <div className="text-sm mb-2">Original</div>
          <canvas ref={canvasRef} className="w-full h-auto" />
        </div>
        <div className="border p-2">
          <div className="text-sm mb-2">Compressed</div>
          <canvas ref={resultRef} className="w-full h-auto" />
        </div>
      </div>
      <div className="text-xs text-muted-foreground">Try our new image compressor — change number of colors and download the result.</div>
    </div>
  )
}

// Simple k-means for RGB pixels
function kmeans(data: number[][], k: number, maxIter = 10) {
  if (data.length === 0) return { centers: [], labels: [] }
  // init centers by sampling
  const centers: number[][] = []
  const taken = new Set<number>()
  while (centers.length < k) {
    const idx = Math.floor(Math.random() * data.length)
    if (!taken.has(idx)) {
      centers.push([...data[idx]])
      taken.add(idx)
    }
  }

  const labels = new Array(data.length).fill(0)
  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false
    // assign
    for (let i = 0; i < data.length; i++) {
      let best = 0
      let bestDist = Infinity
      for (let c = 0; c < centers.length; c++) {
        const d = dist2(data[i], centers[c])
        if (d < bestDist) {
          bestDist = d
          best = c
        }
      }
      if (labels[i] !== best) {
        labels[i] = best
        changed = true
      }
    }
    // recompute
    const sums = Array.from({ length: k }, () => [0, 0, 0])
    const counts = new Array(k).fill(0)
    for (let i = 0; i < data.length; i++) {
      const c = labels[i]
      sums[c][0] += data[i][0]
      sums[c][1] += data[i][1]
      sums[c][2] += data[i][2]
      counts[c]++
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) continue
      centers[c][0] = Math.round(sums[c][0] / counts[c])
      centers[c][1] = Math.round(sums[c][1] / counts[c])
      centers[c][2] = Math.round(sums[c][2] / counts[c])
    }
    if (!changed) break
  }
  return { centers, labels }
}

function dist2(a: number[], b: number[]) {
  const dr = a[0] - b[0]
  const dg = a[1] - b[1]
  const db = a[2] - b[2]
  return dr * dr + dg * dg + db * db
}
