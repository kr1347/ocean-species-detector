"""
Step 2: Fine-tune YOLOv8 on FathomNet marine species data.

── Why YOLOv8? ──────────────────────────────────────────────────────────────
You could use ResNet (which you know) for classification: "what species is in
this image?" But detection answers a harder question: "where is each species,
and what is it?" This is what real AUV survey systems need — the vehicle needs
bounding boxes to track and count organisms autonomously.

YOLOv8 architecture:
  Backbone  →  feature extraction (like ResNet, but optimized for speed)
  Neck (FPN) →  multi-scale feature fusion using attention
  Head       →  predicts boxes + class probabilities

The connection to your essay grading work:
  Your model attended over memory slots to find similar essays.
  YOLOv8's neck attends over spatial feature maps at different scales
  to find objects at different sizes. Same mechanism, different domain.

── Transfer Learning ─────────────────────────────────────────────────────────
We load yolov8n.pt — pre-trained on COCO (80 classes, 118k images).
Just like you froze GloVe embeddings (trainable=False) and trained the
attention layers, here we fine-tune on our 10 underwater species.
The backbone has already learned edges, textures, shapes. We teach it
the final "what does a Nanomia bijuga look like" step.

── Expected training time ────────────────────────────────────────────────────
  CPU only:  ~3-4 hours for 50 epochs (not recommended)
  M1/M2 Mac: ~30-45 min for 50 epochs (uses MPS backend automatically)
  GPU (T4):  ~15-20 min for 50 epochs (use Google Colab if no GPU)
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import yaml

# Known PyTorch MPS bug: advanced tensor indexing in the task-aligned loss
# crashes with a shape mismatch on Apple Silicon. This env var tells PyTorch
# to fall back those specific ops to CPU instead of crashing.
# Safe to always set — it only affects unsupported MPS ops, not the whole run.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on FathomNet species data")
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="YOLOv8 model size: yolov8n (nano), yolov8s (small), yolov8m (medium). "
             "Start with nano — it trains fast and you can upgrade once the pipeline works."
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs. 50 is a good starting point. "
             "Your essay model used 200 because text models converge slower."
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size. Reduce to 8 if you get memory errors."
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size. 640 is YOLOv8 default. "
             "Underwater images often have important fine detail — don't go below 416."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint."
    )
    return parser.parse_args()


def verify_dataset(data_yaml: Path) -> bool:
    """Check that the dataset exists and has images before training."""
    if not data_yaml.exists():
        print(f"Dataset config not found: {data_yaml}")
        print("Run:  python data/download_fathomnet.py  first.")
        return False

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg["path"])
    train_imgs = list((base / "images" / "train").glob("*.jpg"))
    val_imgs = list((base / "images" / "val").glob("*.jpg"))

    if not train_imgs:
        print("No training images found. Run the download script first.")
        return False

    print(f"Dataset verified — train: {len(train_imgs)}, val: {len(val_imgs)} images")
    print(f"Classes ({cfg['nc']}): {list(cfg['names'].values())[:5]}...")
    return True


def main():
    args = parse_args()
    data_yaml = DATA_DIR / "dataset.yaml"

    if not verify_dataset(data_yaml):
        return

    # Load model
    # If the .pt file isn't local, ultralytics downloads it automatically (~6MB for nano)
    model = YOLO(args.model)
    print(f"\nLoaded: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

    # ── Training ──────────────────────────────────────────────────────────────
    # Ultralytics handles:
    #   - DataLoader with augmentation (mosaic, mixup, flips, color jitter)
    #   - Learning rate scheduling (cosine decay, same idea as your exponential decay)
    #   - Gradient clipping (you did this manually — here it's built in)
    #   - Mixed precision training on GPU
    #   - Checkpoint saving (best.pt = best validation mAP)
    #
    # Key augmentations for underwater imagery:
    #   hsv_h, hsv_s, hsv_v — important because underwater images shift blue/green
    #   flipud, fliplr — organisms can appear in any orientation
    #   mosaic — combines 4 images, helps with rare species

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(MODELS_DIR),
        name="fathomnet_v1",
        resume=args.resume,
        device="mps",       # Apple Silicon GPU — ~3-4x faster than CPU

        # Optimizer settings
        # Adam with lr=0.001 is similar to what your essay model used
        optimizer="AdamW",
        lr0=0.001,          # initial learning rate
        lrf=0.01,           # final lr = lr0 * lrf (cosine decay to 1% of initial)
        weight_decay=0.0005,

        # Augmentation — tuned for underwater imagery
        hsv_h=0.015,        # hue shift (small — preserves species color cues)
        hsv_s=0.5,          # saturation shift (underwater images are often desaturated)
        hsv_v=0.4,          # brightness shift (depth affects lighting drastically)
        flipud=0.3,         # vertical flip (jellyfish, plankton can be upside-down)
        fliplr=0.5,         # horizontal flip
        mosaic=1.0,         # mosaic augmentation (helps with rare species)
        mixup=0.1,          # mixup (mild — helps generalization)

        # Patience: stop early if val mAP doesn't improve for 20 epochs
        # Same concept as your best_kappa_so_far checkpoint saving
        patience=20,
        save_period=10,     # save checkpoint every 10 epochs

        # Logging
        verbose=True,
        plots=True,         # saves training curves, confusion matrix, PR curve
    )

    print("\n── Training Complete ──────────────────────────────────────────")
    print(f"Best model saved to: {MODELS_DIR}/fathomnet_v1/weights/best.pt")
    print(f"Training plots at:   {MODELS_DIR}/fathomnet_v1/")
    print("\nKey metrics from final epoch:")

    # mAP explanation:
    # mAP@50    = mean Average Precision at IoU threshold 0.5
    #             "box overlaps ground truth by at least 50%"
    # mAP@50-95 = averaged over IoU thresholds 0.5 to 0.95 (stricter)
    #             This is the COCO standard — harder to game, more meaningful
    # Analogy to your work: QWK penalized disagreements — mAP@50-95 penalizes
    # imprecise boxes. Both metrics reward being *right* not just *not-wrong*.

    box_map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
    box_map5095 = results.results_dict.get("metrics/mAP50-95(B)", "N/A")
    print(f"  mAP@50:    {box_map50}")
    print(f"  mAP@50-95: {box_map5095}")
    print("\nNext step: run  python evaluate.py")


if __name__ == "__main__":
    main()
