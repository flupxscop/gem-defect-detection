import { describe, expect, it } from "vitest";
import fixtures from "./fixtures.json";
import { decodeRtDetr, decodeYolo } from "./postprocess";
import { letterbox, resizeBilinear, stretch } from "./preprocess";
import { RgbaImage } from "./types";

// Fixtures come from the Python server pipeline (scripts/make_web_fixtures.py).

function rgba(rgb: number[][][]): RgbaImage {
  const height = rgb.length;
  const width = rgb[0].length;
  const data = new Uint8ClampedArray(width * height * 4);
  rgb.flat().forEach((px, i) => data.set([...px, 255], i * 4));
  return { data, width, height };
}

const image = rgba(fixtures.image.rgb);

describe("preprocessing matches OpenCV", () => {
  it.each(fixtures.resize)("resizes to $width×$height within 1 level", ({ width, height, expected }) => {
    const out = resizeBilinear(image, width, height);
    const want = (expected as number[][][]).flat(2);
    const got = Array.from(out.data).filter((_, i) => i % 4 !== 3);
    got.forEach((v, i) => expect(Math.abs(v - want[i])).toBeLessThanOrEqual(1));
  });

  it("letterboxes like the YOLO server path", () => {
    const { tensor } = letterbox(image, fixtures.letterbox.size);
    tensor.forEach((v, i) => expect(Math.abs(v - fixtures.letterbox.tensor[i])).toBeLessThanOrEqual(1 / 255 + 1e-6));
  });

  it("stretches like the RT-DETR server path", () => {
    const tensor = stretch(image, fixtures.stretch.size);
    tensor.forEach((v, i) => expect(Math.abs(v - fixtures.stretch.tensor[i])).toBeLessThanOrEqual(1 / 255 + 1e-6));
  });
});

describe("postprocessing matches the server", () => {
  it("decodes YOLO output with NMS", () => {
    const { output, classes, conf, expected } = fixtures.yolo;
    const got = decodeYolo(Float32Array.from(output), classes, conf);
    expect(got).toHaveLength(expected.length);
    got.forEach((d, i) => {
      expect(d.classIndex).toBe(expected[i].class);
      expect(d.confidence).toBeCloseTo(expected[i].score, 5);
      d.box.forEach((v, j) => expect(v).toBeCloseTo(expected[i].box[j], 3));
    });
  });

  it("decodes RT-DETR output", () => {
    const { output, conf, expected } = fixtures.rtdetr;
    const got = decodeRtDetr(Float32Array.from(output), conf);
    expect(got).toHaveLength(expected.length);
    got.forEach((d, i) => {
      expect(d.classIndex).toBe(expected[i].class);
      expect(d.confidence).toBeCloseTo(expected[i].score, 5);
      d.box.forEach((v, j) => expect(v).toBeCloseTo(expected[i].box[j], 5));
    });
  });
});
