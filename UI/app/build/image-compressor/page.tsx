import ImageCompressor from "@/components/image-compressor"

export const metadata = {
  title: "Image Compressor - Build",
}

export default function Page() {
  return (
    <main className="container py-12">
      <h1 className="text-3xl font-bold mb-6">Image Compressor</h1>
      <p className="mb-6 text-muted-foreground">Try our new image compressor: reduce colors with a slider and download the result.</p>
      <div className="prose">
        <ImageCompressor />
      </div>
    </main>
  )
}
