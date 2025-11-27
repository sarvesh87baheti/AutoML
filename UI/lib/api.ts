export async function postUpload(formData: FormData) {
  const res = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });
  const data = await res.json();

  // New unified success/error contract
  if (!res.ok || data.success === false) {
    const code = data?.error?.code || "UPLOAD_FAILED";
    const message = data?.error?.message || data?.error || "Upload failed";
    const err = new Error(message);
    (err as any).code = code;
    (err as any).raw = data?.error?.raw;
    throw err;
  }

  return data;
}
