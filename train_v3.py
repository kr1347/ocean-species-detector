"""
Fine-tune v3: YOLOv8m + expanded dataset + targeted fixes for weak classes.

Changes from v2:
  1. YOLOv8m backbone    — 25M params (vs 11M). Biggest accuracy lever remaining.
  2. Expanded dataset    — 600/species + all Ptereleotris/Ophiuroidea images
  3. imgsz=896           — up from 800; captures more fine detail
  4. copy_paste=0.5      — up from 0.3; harder push for weak small-fish classes
  5. degrees=10          — rotation aug; fish appear at any orientation
  6. scale=0.7           — random scale; helps small-object generalization
  7. cls=0.7             — up from 0.5; penalizes misclassification harder
                           (helps distinguish similar-looking species)
  8. 150 epochs + patience=40 — more room to converge on larger dataset
  9. Cosine LR + warmup  — smoother convergence

Expected: mAP@50 > 0.82, mAP@50-95 > 0.58

Run:
  PYTORCH_ENABLE_MPS_FALLBACK=1 python train_v3.py
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DATA_DIR   = Path(__file__).parent / "data" / "data_v3"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def verify_dataset(data_yaml: Path) -> bool:
    if not data_yaml.exists():
        print(f"Dataset not found: {data_yaml}")
        print("Run: python data/download_v3.py  first.")
        return False
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg["path"])
    train_imgs = list((base / "images" / "train").glob("*.jpg"))
    val_imgs   = list((base / "images" / "val").glob("*.jpg"))
    print(f"Dataset verified — train: {len(train_imgs)}, val: {len(val_imgs)} images")
    return True


def main():
    data_yaml = DATA_DIR / "dataset.yaml"
    if not verify_dataset(data_yaml):
        return

    # YOLOv8m — 25M params. The jump from s→m is larger than n→s in practice.
    # YOLOv8l (43M) is the next step but needs more VRAM; m is the sweet spot
    # for M1 Pro 16GB.
    model = YOLO("yolov8m.pt")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    results = model.train(
        data=str(data_yaml),
        project=str(MODELS_DIR),
        name="fathomnet_v3",
        device="mps",

        # ── Core ──────────────────────────────────────────────────────────────
        epochs=150,
        batch=8,            # YOLOv8m at imgsz=896 needs smaller batch on 16GB
        imgsz=896,          # up from 800; better small-object coverage
        patience=40,
        save_period=10,

        # ── Optimizer ─────────────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.0008,         # slightly lower for larger model — more stable
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,        # cosine LR schedule — smoother than linear decay

        # ── Loss weights ──────────────────────────────────────────────────────
        # box: weight on bounding box regression loss
        # cls: weight on classification loss — increasing this penalizes
        #      misclassifying species harder (helps Ptereleotris vs Chromis confusion)
        # dfl: distribution focal loss weight
        box=7.5,
        cls=0.7,            # up from 0.5 — punish misclassification more
        dfl=1.5,

        # ── Transfer learning ─────────────────────────────────────────────────
        freeze=5,           # freeze backbone for first 5 warmup epochs

        # ── Augmentation ──────────────────────────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.45,         # slightly more brightness variation
        flipud=0.3,
        fliplr=0.5,
        degrees=10,         # NEW: rotation ±10° — fish at any orientation
        scale=0.7,          # NEW: random scale 0.3x–1.7x — key for small fish
        mosaic=1.0,
        mixup=0.15,         # slightly more mixup
        copy_paste=0.5,     # up from 0.3 — harder push for Ptereleotris/Ophiuroidea
        erasing=0.4,        # NEW: random rectangular erasing — improves occlusion robustness

        # ── Logging ───────────────────────────────────────────────────────────
        verbose=True,
        plots=True,
    )

    map50   = results.results_dict.get("metrics/mAP50(B)",    "N/A")
    map5095 = results.results_dict.get("metrics/mAP50-95(B)", "N/A")

    print("\n── v3 Training Complete ───────────────────────────────────────")
    print(f"Best weights: {MODELS_DIR}/fathomnet_v3/weights/best.pt")
    print(f"mAP@50:       {map50}")
    print(f"mAP@50-95:    {map5095}")
    print("\nNext: python evaluate.py --weights models/fathomnet_v3/weights/best.pt")


if __name__ == "__main__":
    main()
