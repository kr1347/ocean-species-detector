"""
Fine-tune v2: targeting mAP@50 > 0.80 on the FathomNet test set.

Changes from v1 (train.py):
  1. yolov8s backbone  — 11M params vs 3M. Biggest single lever.
  2. imgsz=800         — better small-object detection (Ptereleotris fix).
  3. 100 epochs        — v1 was still improving at epoch 50.
  4. copy_paste=0.3    — synthesizes rare species into new scenes; helps
                         underperforming classes like Ptereleotris (0.130 mAP).
  5. freeze=5          — freeze backbone for first 5 epochs so the new head
                         stabilizes before we unfreeze and fine-tune everything.
  6. Wider HSV augmentation — underwater imagery has heavy color shift by depth.

Run:
  PYTORCH_ENABLE_MPS_FALLBACK=1 python train_v2.py
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def verify_dataset(data_yaml: Path) -> bool:
    if not data_yaml.exists():
        print(f"Dataset config not found: {data_yaml}")
        print("Run:  python data/download_fathomnet.py  first.")
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

    model = YOLO("yolov8s.pt")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    results = model.train(
        data=str(data_yaml),
        project=str(MODELS_DIR),
        name="fathomnet_v2",
        device="mps",

        # ── Core ──────────────────────────────────────────────────────────────
        epochs=100,
        batch=16,
        imgsz=800,          # was 640; helps Ptereleotris and other small species
        patience=30,        # more room to improve before early-stopping
        save_period=10,

        # ── Optimizer ─────────────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,    # gradual LR ramp instead of cold start

        # ── Transfer learning strategy ────────────────────────────────────────
        # Freeze backbone (layers 0-5) for first 5 epochs so the new head
        # stabilizes, then unfreeze for end-to-end fine-tuning.
        freeze=5,

        # ── Augmentation ──────────────────────────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.7,          # was 0.5 — underwater color heavily depth-dependent
        hsv_v=0.4,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,     # NEW: pastes objects from other images into scene
                            # most effective fix for rare/hard classes

        # ── Logging ───────────────────────────────────────────────────────────
        verbose=True,
        plots=True,
    )

    map50    = results.results_dict.get("metrics/mAP50(B)",    "N/A")
    map5095  = results.results_dict.get("metrics/mAP50-95(B)", "N/A")

    print("\n── v2 Training Complete ───────────────────────────────────────")
    print(f"Best weights: {MODELS_DIR}/fathomnet_v2/weights/best.pt")
    print(f"mAP@50:       {map50}")
    print(f"mAP@50-95:    {map5095}")
    print("\nNext: python evaluate.py --weights models/fathomnet_v2/weights/best.pt")


if __name__ == "__main__":
    main()
