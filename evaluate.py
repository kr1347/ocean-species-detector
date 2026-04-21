"""
Model evaluation on held-out test set.

Metrics reported:
    Precision  — TP / (TP + FP): fraction of detections that are correct
    Recall     — TP / (TP + FN): fraction of ground-truth instances detected
    AP@50      — Area under the Precision-Recall curve at IoU threshold 0.50
                 (PASCAL VOC 2010 standard)
    mAP@50     — Mean AP@50 across all species classes
    mAP@50-95  — Mean AP averaged over IoU thresholds 0.50:0.05:0.95
                 (MS-COCO standard; more sensitive to localization quality)

Note on IoU threshold selection:
    In ecological survey applications, recall is prioritized over precision —
    missed detections (false negatives) carry greater consequence than false
    alarms. The confidence threshold can be tuned on the validation set to
    shift the precision-recall trade-off accordingly.

Usage:
    python evaluate.py
    python evaluate.py --weights models/fathomnet_v3/weights/best.pt
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument(
        "--weights",
        default=str(MODELS_DIR / "fathomnet_v1" / "weights" / "best.pt"),
        help="Path to model weights (.pt)"
    )
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU threshold for NMS (default: 0.5)")
    return parser.parse_args()


def print_per_class_results(results, class_map: dict):
    print("\n── Per-Class Detection Results ────────────────────────────────")
    print(f"{'Species':<35} {'AP@50':>8} {'AP@50-95':>10} {'Recall':>8} {'Precision':>10}")
    print("-" * 75)

    if hasattr(results, 'ap_class_index') and results.ap_class_index is not None:
        for i, cls_idx in enumerate(results.ap_class_index):
            name = class_map.get(str(cls_idx), f"class_{cls_idx}")
            ap50   = results.box.ap50[i] if hasattr(results.box, 'ap50') else 0
            ap5095 = results.box.ap[i]   if hasattr(results.box, 'ap')   else 0
            r      = results.box.r[i]    if hasattr(results.box, 'r')    else 0
            p      = results.box.p[i]    if hasattr(results.box, 'p')    else 0
            print(f"{name:<35} {ap50:>8.3f} {ap5095:>10.3f} {r:>8.3f} {p:>10.3f}")
    else:
        print("  Per-class breakdown unavailable.")

    print("-" * 75)
    map50   = results.box.map50 if hasattr(results.box, 'map50') else 0
    map5095 = results.box.map   if hasattr(results.box, 'map')   else 0
    print(f"{'mAP (all classes)':<35} {map50:>8.3f} {map5095:>10.3f}")


def plot_ap_by_class(results, class_map: dict, save_path: Path):
    """Bar chart of per-class AP@50 for qualitative analysis of class difficulty."""
    if not hasattr(results, 'ap_class_index') or results.ap_class_index is None:
        return

    names    = [class_map.get(str(i), f"class_{i}") for i in results.ap_class_index]
    ap50_vals = results.box.ap50 if hasattr(results.box, 'ap50') else []

    if not len(names) or not len(ap50_vals):
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), ap50_vals, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("AP@50")
    ax.set_title("Per-Class Average Precision (AP@50)\nFathomNet Deep-Sea Species Detection")
    ax.set_ylim(0, 1.05)
    ax.axhline(np.mean(ap50_vals), color="red", linestyle="--",
               label=f"mAP@50 = {np.mean(ap50_vals):.3f}")
    ax.legend()

    for bar, val in zip(bars, ap50_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Per-class AP chart saved: {save_path}")
    plt.close()


def main():
    args = parse_args()
    weights_path = Path(args.weights)

    if not weights_path.exists():
        print(f"Weights not found: {weights_path}")
        return

    class_map_path = DATA_DIR / "class_map.json"
    class_map = {}
    if class_map_path.exists():
        with open(class_map_path) as f:
            class_map = json.load(f)

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    print("\nEvaluating on held-out test set...")
    results = model.val(
        data=str(DATA_DIR / "dataset.yaml"),
        split="test",
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
    )

    print_per_class_results(results, class_map)

    output_dir = weights_path.parent.parent
    plot_ap_by_class(results, class_map, output_dir / "ap_by_class.png")

    map5095 = results.box.map if hasattr(results.box, 'map') else 0
    print(f"\nPrimary metric (COCO standard): mAP@50-95 = {map5095:.3f}")


if __name__ == "__main__":
    main()
