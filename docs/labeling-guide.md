# Relabelling guide: Inclusion boxes

The current labels mix two styles: some flaws get their own tight box, others are grouped into one large box.
Models trained on them find about 91% of the labelled inclusions at a loose match (IoU 0.1) but only reach
mAP50 ≈ 0.19, because their boxes rarely line up with the grouped ones (see `docs/benchmarks.md`). Consistent
labels are the main lever for accuracy.

## Rules

1. **One box per inclusion.** Never draw one box around a group. If two flaws touch, draw two boxes unless they
   are a single continuous feature (one feather, one needle).
2. **Tight boxes.** The box edges touch the visible extent of the flaw. Include the dark core of a pinpoint, not
   the glare around it.
3. **Label every inclusion you can see at 100% zoom.** Missing labels count as false positives against the
   model. If you are unsure whether something is an inclusion, label it; if you are unsure whether it is a
   reflection, check whether it moves in neighbouring facets (reflections repeat, inclusions don't).
4. **Do not label facet reflections, dust on the surface, or the girdle edge.**
5. **Minimum size: about 8 px on the short side at full resolution (2584×1936).** Smaller specks are skipped
   consistently rather than labelled sometimes.
6. **Class:** keep a single class, `Inclusion`. The `Diamond` class (whole stone) can stay; `src/download.py`
   drops it by default.

## Order of work

1. **Test split first (16 images).** Without consistent test labels no metric is trustworthy. This takes about
   20–30 minutes.
2. **Validation split (33 images)**, then **train (116 images)**.
3. In Roboflow, generate a new version with the same settings as version 2 (auto-orient, no resize, same split).

## Retraining

```bash
python src/download.py --version <new> --out dataset-relabelled
python src/train.py --model yolo11s --data data/dataset-relabelled/data_fixed.yaml --name yolo11s-relabelled
python scripts/evaluate.py runs/yolo11s-relabelled/weights/best.pt \
    --data data/dataset-relabelled/data_fixed.yaml --imgsz 640
```

Compare against the current baseline (test mAP50 0.187) on the relabelled test split, and re-evaluate the
current model on the new labels too, so the gain from labels and from retraining can be told apart.
