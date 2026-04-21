"""
Experiment 1: Baseline fine-tuning of YOLOv8n on FathomNet deep-sea species data.

Architecture:
    YOLOv8n (nano) — CSPDarknet backbone, C2f neck (FPN + PAN), decoupled detection head.
    Pre-trained on COCO (80 classes, 118k images); fine-tuned on 10 deep-sea species.

Transfer learning rationale:
    The COCO-pretrained backbone encodes low- and mid-level features (edges, textures,
    shapes) transferable to underwater imagery. Fine-tuning adapts the classification
    head and higher-level feature maps to the target domain.

Dataset:
    FathomNet (MBARI) — expert-annotated ROV imagery, 70/15/15 train/val/test split.
    10 species: Lutjanus campechanus, Stenotomus caprinus, Rhomboplites aurorubens,
    Strongylocentrotus fragilis, Ptereleotris, Pagrus pagrus, Chromis,
    Epinephelus morio, Ophiuroidea, Balistes capriscus.

Evaluation:
    mAP@50    — mean Average Precision at IoU threshold 0.50 (PASCAL VOC standard)
    mAP@50-95 — mean AP averaged over IoU thresholds 0.50:0.05:0.95 (COCO standard)

Baseline results: mAP@50 = 0.720, mAP@50-95 = 0.485

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py
    PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --model yolov8s.pt --epochs 100
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import yaml

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on FathomNet species data")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 variant: yolov8n (3M), yolov8s (11M), yolov8m (25M)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def verify_dataset(data_yaml: Path) -> bool:
    if not data_yaml.exists():
        print(f"Dataset config not found: {data_yaml}")
        print("Run:  python data/download_fathomnet.py  first.")
        return False
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg["path"])
    train_imgs = list((base / "images" / "train").glob("*.jpg"))
    val_imgs = list((base / "images" / "val").glob("*.jpg"))
    print(f"Dataset verified — train: {len(train_imgs)}, val: {len(val_imgs)} images")
    print(f"Classes ({cfg['nc']}): {list(cfg['names'].values())}")
    return True


def main():
    args = parse_args()
    data_yaml = DATA_DIR / "dataset.yaml"

    if not verify_dataset(data_yaml):
        return

    model = YOLO(args.model)
    print(f"\nModel: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(MODELS_DIR),
        name="fathomnet_v1",
        resume=args.resume,
        device="mps",

        # Optimizer: AdamW with cosine LR decay
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,

        # Domain-specific augmentation for underwater imagery:
        # HSV shifts simulate depth-dependent color absorption and scattering.
        # Vertical flip accounts for organisms appearing in any orientation.
        # Mosaic augmentation improves detection of rare species by combining scenes.
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        # Early stopping: halt if validation mAP does not improve for 20 epochs
        patience=20,
        save_period=10,

        verbose=True,
        plots=True,
    )

    print("\n── Training Complete ───────────────────────────────────────────")
    print(f"Best weights: {MODELS_DIR}/fathomnet_v1/weights/best.pt")
    print(f"mAP@50:       {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP@50-95:    {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print("\nNext step: python evaluate.py")


if __name__ == "__main__":
    main()
