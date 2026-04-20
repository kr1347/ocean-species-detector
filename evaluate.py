"""
Step 3: Evaluate the trained model on the test set.

Produces:
  - Per-class AP (Average Precision) — which species detects well vs. poorly
  - Confusion matrix — where the model confuses species
  - PR curve (Precision-Recall) — same concept as ROC/AUC you used in NLP project
  - F1-confidence curve — what confidence threshold maximizes F1

Understanding the metrics:
  Precision = of all detections, how many were correct?
  Recall    = of all real organisms, how many did we find?
  AP        = area under the precision-recall curve for one class
  mAP       = mean AP across all classes

  In ocean surveys, RECALL is often more important than precision.
  Missing a whale entanglement risk is worse than a false alarm.
  You can tune the confidence threshold to trade precision for recall.
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
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for predictions"
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for NMS (Non-Maximum Suppression)"
    )
    return parser.parse_args()


def print_per_class_results(results, class_map: dict):
    """Print a readable table of per-class AP scores."""
    print("\n── Per-Class Results ──────────────────────────────────────────")
    print(f"{'Class':<35} {'AP@50':>8} {'AP@50-95':>10} {'Recall':>8} {'Precision':>10}")
    print("-" * 75)

    # results.ap_class_index has the class indices
    # results.box.ap gives per-class AP@50-95
    if hasattr(results, 'ap_class_index') and results.ap_class_index is not None:
        for i, cls_idx in enumerate(results.ap_class_index):
            name = class_map.get(str(cls_idx), f"class_{cls_idx}")
            ap50 = results.box.ap50[i] if hasattr(results.box, 'ap50') else 0
            ap5095 = results.box.ap[i] if hasattr(results.box, 'ap') else 0
            r = results.box.r[i] if hasattr(results.box, 'r') else 0
            p = results.box.p[i] if hasattr(results.box, 'p') else 0
            print(f"{name:<35} {ap50:>8.3f} {ap5095:>10.3f} {r:>8.3f} {p:>10.3f}")
    else:
        print("  (Per-class breakdown not available — check ultralytics version)")

    print("-" * 75)
    map50 = results.box.map50 if hasattr(results.box, 'map50') else 0
    map5095 = results.box.map if hasattr(results.box, 'map') else 0
    print(f"{'mAP (all classes)':<35} {map50:>8.3f} {map5095:>10.3f}")


def plot_ap_by_class(results, class_map: dict, save_path: Path):
    """Bar chart of per-class AP@50 — easy to see which species are hardest."""
    if not hasattr(results, 'ap_class_index') or results.ap_class_index is None:
        return

    names = [class_map.get(str(i), f"class_{i}") for i in results.ap_class_index]
    ap50_vals = results.box.ap50 if hasattr(results.box, 'ap50') else []

    if len(names) == 0 or len(ap50_vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), ap50_vals, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("AP@50")
    ax.set_title("Per-Class Average Precision (AP@50)\nFathomNet Marine Species Detection")
    ax.set_ylim(0, 1.05)
    ax.axhline(np.mean(ap50_vals), color="red", linestyle="--", label=f"mAP = {np.mean(ap50_vals):.3f}")
    ax.legend()

    # Annotate bars
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
        print("Run  python train.py  first.")
        return

    # Load class map
    class_map_path = DATA_DIR / "class_map.json"
    class_map = {}
    if class_map_path.exists():
        with open(class_map_path) as f:
            class_map = json.load(f)

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    # Run evaluation on the test split
    print("\nRunning evaluation on test set...")
    results = model.val(
        data=str(DATA_DIR / "dataset.yaml"),
        split="test",
        conf=args.conf,
        iou=args.iou,
        plots=True,  # saves confusion matrix and PR curves automatically
        save_json=True,
    )

    # Print summary
    print_per_class_results(results, class_map)

    # Save per-class AP bar chart
    output_dir = weights_path.parent.parent
    plot_ap_by_class(results, class_map, output_dir / "ap_by_class.png")

    # Key takeaway for your resume:
    # Report the mAP@50-95 number — it's the standard COCO metric
    # and what hiring teams at ocean AI companies will recognize.
    map5095 = results.box.map if hasattr(results.box, 'map') else 0
    print(f"\nResume metric: mAP@50-95 = {map5095:.3f}")
    print("\nNext step: run  python api/serve.py  to deploy the inference API")


if __name__ == "__main__":
    main()
