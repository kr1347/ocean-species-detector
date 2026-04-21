"""
Experiment 3: YOLOv8m with expanded dataset and targeted augmentation for low-AP classes.

Modifications from Experiment 2:
    1. YOLOv8m backbone (25M params vs 11M) — deeper C2f blocks and wider feature maps
       provide greater representational capacity for fine-grained species discrimination.
    2. Expanded dataset — 600 images per common species; Ptereleotris and Ophiuroidea
       uncapped (7,686 and 3,061 images respectively) to address their low AP scores
       (0.331 and 0.376 in Experiment 2).
    3. imgsz=896 — higher resolution improves localization of morphologically small
       species with limited pixel coverage at standard resolution.
    4. copy_paste=0.5 — increased copy-paste rate (Ghiasi et al., 2021) to further
       augment rare-class representation in training batches.
    5. degrees=10 — rotational augmentation (±10°) accounts for the orientation-
       invariant nature of deep-sea organism detection in ROV footage.
    6. scale=0.7 — multi-scale training improves robustness to size variation across
       species and depth-dependent perspective distortion.
    7. cls=0.7 — elevated classification loss weight penalizes inter-species confusion
       more strongly, beneficial for morphologically similar species pairs.
    8. cos_lr=True — cosine annealing schedule (Loshchilov & Hutter, 2017) provides
       smoother convergence than step or linear decay over 150 epochs.

Target: mAP@50 > 0.82, mAP@50-95 > 0.58

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

    model = YOLO("yolov8m.pt")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    results = model.train(
        data=str(data_yaml),
        project=str(MODELS_DIR),
        name="fathomnet_v3",
        device="mps",

        epochs=150,
        batch=8,
        imgsz=896,
        patience=40,
        save_period=10,

        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,

        # Loss weights
        box=7.5,
        cls=0.7,
        dfl=1.5,

        # Backbone frozen during warmup to stabilize detection head initialization
        freeze=5,

        # Augmentation: underwater domain-specific + small-object strategies
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.45,
        flipud=0.3,
        fliplr=0.5,
        degrees=10,
        scale=0.7,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.5,
        erasing=0.4,

        verbose=True,
        plots=True,
    )

    map50   = results.results_dict.get("metrics/mAP50(B)",    "N/A")
    map5095 = results.results_dict.get("metrics/mAP50-95(B)", "N/A")

    print("\n── Experiment 3 Complete ──────────────────────────────────────")
    print(f"Best weights: {MODELS_DIR}/fathomnet_v3/weights/best.pt")
    print(f"mAP@50:       {map50}")
    print(f"mAP@50-95:    {map5095}")
    print("\nNext: python evaluate.py --weights models/fathomnet_v3/weights/best.pt")


if __name__ == "__main__":
    main()
