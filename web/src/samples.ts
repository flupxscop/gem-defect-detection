// Test-split images from the Diamond Inclusion dataset (CC BY 4.0), served from public/samples.
export const SAMPLES = ["sample-1.jpg", "sample-2.jpg", "sample-3.jpg", "sample-4.jpg"].map(
  (name) => `${import.meta.env.BASE_URL}samples/${name}`,
);
