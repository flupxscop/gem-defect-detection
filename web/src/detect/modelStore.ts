const CACHE_NAME = "gemscan-models-v1";

export type Progress = (loaded: number, total: number) => void;

/**
 * Download a model file, reporting progress, and keep it in Cache Storage.
 * URLs are pinned to a model-repo commit, so a cached file never goes stale.
 */
export async function fetchModel(url: string, onProgress: Progress): Promise<ArrayBuffer> {
  const cache = await openCache();
  const cached = await cache?.match(url);
  if (cached) {
    const buffer = await cached.arrayBuffer();
    onProgress(buffer.byteLength, buffer.byteLength);
    return buffer;
  }

  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`Download failed (${response.status}): ${url}`);
  const total = Number(response.headers.get("content-length")) || 0;

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total || loaded);
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  await cache?.put(url, new Response(buffer)).catch(() => undefined); // quota errors just skip caching
  return buffer.buffer;
}

async function openCache(): Promise<Cache | null> {
  try {
    return "caches" in self ? await caches.open(CACHE_NAME) : null;
  } catch {
    return null; // e.g. private browsing
  }
}
