import { RgbaImage } from "./types";

export const LETTERBOX_FILL = 114;

/**
 * Bilinear resize with OpenCV INTER_LINEAR semantics (pixel centres, edge clamping, no antialiasing),
 * which is what Ultralytics and the Python server use. Pillow-style antialiasing would shift scores.
 */
export function resizeBilinear(src: RgbaImage, width: number, height: number): RgbaImage {
  if (width === src.width && height === src.height) return src;

  const xs = axisWeights(src.width, width);
  const ys = axisWeights(src.height, height);
  const out = new Uint8ClampedArray(width * height * 4);
  const s = src.data;
  const rowStride = src.width * 4;

  for (let y = 0; y < height; y++) {
    const r0 = ys.lo[y] * rowStride;
    const r1 = ys.hi[y] * rowStride;
    const wy = ys.weight[y];
    for (let x = 0; x < width; x++) {
      const c0 = xs.lo[x] * 4;
      const c1 = xs.hi[x] * 4;
      const wx = xs.weight[x];
      const o = (y * width + x) * 4;
      for (let c = 0; c < 3; c++) {
        const top = s[r0 + c0 + c] * (1 - wx) + s[r0 + c1 + c] * wx;
        const bottom = s[r1 + c0 + c] * (1 - wx) + s[r1 + c1 + c] * wx;
        out[o + c] = Math.round(top * (1 - wy) + bottom * wy);
      }
      out[o + 3] = 255;
    }
  }
  return { data: out, width, height };
}

function axisWeights(srcSize: number, dstSize: number) {
  const scale = srcSize / dstSize;
  const lo = new Int32Array(dstSize);
  const hi = new Int32Array(dstSize);
  const weight = new Float32Array(dstSize);
  for (let i = 0; i < dstSize; i++) {
    const pos = (i + 0.5) * scale - 0.5;
    let left = Math.floor(pos);
    let w = pos - left;
    if (left < 0) {
      left = 0;
      w = 0;
    }
    if (left >= srcSize - 1) {
      left = srcSize - 1;
      w = 0;
    }
    lo[i] = left;
    hi[i] = Math.min(left + 1, srcSize - 1);
    weight[i] = w;
  }
  return { lo, hi, weight };
}

export interface Letterbox {
  tensor: Float32Array;
  scale: number;
  left: number;
  top: number;
}

/** Fit the image inside a size×size square, centred, padded with grey. Returns a 1×3×size×size tensor. */
export function letterbox(image: RgbaImage, size: number): Letterbox {
  const scale = Math.min(size / image.width, size / image.height);
  const newW = Math.round(image.width * scale);
  const newH = Math.round(image.height * scale);
  const left = Math.round((size - newW) / 2 - 0.1);
  const top = Math.round((size - newH) / 2 - 0.1);

  const resized = resizeBilinear(image, newW, newH);
  const tensor = new Float32Array(3 * size * size).fill(LETTERBOX_FILL / 255);
  writeChw(resized, tensor, size, left, top);
  return { tensor, scale, left, top };
}

/** Stretch the image to size×size (RT-DETR's preprocessing). */
export function stretch(image: RgbaImage, size: number): Float32Array {
  const tensor = new Float32Array(3 * size * size);
  writeChw(resizeBilinear(image, size, size), tensor, size, 0, 0);
  return tensor;
}

function writeChw(image: RgbaImage, tensor: Float32Array, size: number, left: number, top: number) {
  const plane = size * size;
  const { data, width, height } = image;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const t = (y + top) * size + (x + left);
      tensor[t] = data[i] / 255;
      tensor[plane + t] = data[i + 1] / 255;
      tensor[2 * plane + t] = data[i + 2] / 255;
    }
  }
}
