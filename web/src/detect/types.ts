export interface ModelSpec {
  name: string;
  file: string;
  label: string;
  architecture: string;
  family: "yolo" | "rtdetr";
  imgsz: number;
  classes: string[];
}

/** RGBA pixels, as returned by ImageData. */
export interface RgbaImage {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

export interface RawDetection {
  classIndex: number;
  confidence: number;
  box: [number, number, number, number]; // x1, y1, x2, y2 in source-image pixels
}
