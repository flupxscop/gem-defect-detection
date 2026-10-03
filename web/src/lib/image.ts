const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
// The models run at 640 px, so anything much larger only costs upload time.
const MAX_SIDE = 1280;
const JPEG_QUALITY = 0.9;

export class UploadError extends Error {}

/**
 * Validate a picked file and shrink large photos before upload.
 * EXIF rotation is baked in, so the server sees the image the way the browser shows it.
 */
export async function prepareUpload(file: File): Promise<File> {
  if (!ACCEPTED_TYPES.includes(file.type)) {
    throw new UploadError(`"${file.name}" isn't a JPG, PNG or WEBP image.`);
  }

  const resized = await downscale(file).catch(() => file);
  if (resized.size > MAX_UPLOAD_BYTES) {
    const mb = (resized.size / 1024 / 1024).toFixed(1);
    throw new UploadError(`"${file.name}" is ${mb} MB. The limit is 10 MB.`);
  }
  return resized;
}

async function downscale(file: File): Promise<File> {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  try {
    const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
    if (scale === 1) return file;

    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext("2d")!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY));
    if (!blob) return file;
    return new File([blob], file.name.replace(/\.\w+$/, ".jpg"), { type: "image/jpeg" });
  } finally {
    bitmap.close();
  }
}
