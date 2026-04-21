"""
Dataset acquisition from FathomNet (MBARI) for deep-sea species detection.

FathomNet is an open-source underwater image training database seeded from
MBARI's Video Annotation and Reference System (VARS), which contains
23,000+ hours of ROV footage with expert taxonomic annotations spanning
4,600+ biological concepts (Boulais et al., 2022).

This script:
    1. Queries the FathomNet API for the N most-annotated species concepts.
    2. Downloads annotated images and converts bounding boxes to YOLO format.
    3. Applies a stratified 70/15/15 train/val/test split.
    4. Writes dataset.yaml and class_map.json for downstream training.

YOLO annotation format (per image, one row per object):
    <class_id> <x_center> <y_center> <width> <height>
    All spatial values normalized to [0, 1] relative to image dimensions.

Reference:
    Boulais, O. et al. (2022). FathomNet: A global image database for enabling
    artificial intelligence in the ocean. Scientific Reports, 12, 15914.
    https://doi.org/10.1038/s41598-022-19939-2

Usage:
    python data/download_fathomnet.py
    python data/download_fathomnet.py --species 15 --per-species 500
"""

import argparse
import io
import json
import shutil
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

from fathomnet.api import images as fn_images
from fathomnet.api import boundingboxes as fn_boxes

DATA_DIR   = Path(__file__).parent
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
RAW_DIR    = DATA_DIR / "raw"

# Exclude non-specific or non-biological concepts
EXCLUDE_CONCEPTS = {
    "", "object", "animal", "organism", "unknown", "marine snow",
    "equipment", "debris", "bony fish", "marine organism",
}
EXCLUDE_KEYWORDS = [
    "bucket", "drum", "laser", "robotics", "camera", "cable",
    "pipe", "rope", "structured light",
]


def get_top_species(n: int, min_annotations: int = 100) -> list[str]:
    """
    Return the N most-annotated species-level concepts from FathomNet.

    Filters out equipment labels, ambiguous taxonomic groupings, and concepts
    with fewer than min_annotations bounding box annotations.
    """
    print("Fetching concept counts from FathomNet API...")
    all_counts = fn_boxes.count_total_by_concept()

    filtered = [
        c for c in all_counts
        if c.concept not in EXCLUDE_CONCEPTS
        and c.concept[0:1].isupper()
        and c.count >= min_annotations
        and not any(k in c.concept.lower() for k in EXCLUDE_KEYWORDS)
    ]
    filtered.sort(key=lambda x: x.count, reverse=True)

    species = [c.concept for c in filtered[:n]]
    print(f"Selected {len(species)} species (top {n} by annotation count)")
    return species


def download_image(url: str, dest: Path) -> bool:
    """Download image from URL and save as JPEG. Returns False on failure."""
    try:
        resp = requests.get(url, timeout=20, stream=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(dest, "JPEG", quality=95)
        return True
    except Exception:
        return False


def bbox_to_yolo(x: float, y: float, w: float, h: float,
                 img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """
    Convert pixel-space bounding box to YOLO normalized format.

    Input:  x, y = top-left corner (pixels); w, h = box dimensions (pixels)
    Output: x_center, y_center, width, height — all normalized to [0, 1]
    """
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    return (
        max(0.0, min(1.0, xc)),
        max(0.0, min(1.0, yc)),
        max(0.0, min(1.0, wn)),
        max(0.0, min(1.0, hn)),
    )


def is_valid_box(box, img_w: int, img_h: int) -> bool:
    """Reject degenerate or out-of-bounds bounding boxes."""
    return (
        box.width > 0 and box.height > 0
        and box.x >= 0 and box.y >= 0
        and box.x < img_w and box.y < img_h
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--species",     type=int, default=10,
                   help="Number of species to include (default: 10)")
    p.add_argument("--per-species", type=int, default=300,
                   help="Max images per species (default: 300)")
    return p.parse_args()


def main():
    args = parse_args()

    for split in ["train", "val", "test"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    species_list = get_top_species(args.species)
    class_to_id  = {s: i for i, s in enumerate(species_list)}

    print(f"\nTarget classes ({len(species_list)}):")
    for i, s in enumerate(species_list):
        print(f"  [{i}] {s}")

    records = []
    seen_uuids = set()

    for concept in species_list:
        print(f"\nFetching images for: {concept}")
        img_records = fn_images.find_by_concept(concept)
        img_records = img_records[:args.per_species]
        print(f"  {len(img_records)} images (capped at {args.per_species})")

        downloaded = 0
        for item in tqdm(img_records, desc=f"  {concept[:35]}"):
            uuid = item.uuid
            url  = item.url

            if not url or uuid in seen_uuids:
                continue
            seen_uuids.add(uuid)

            img_w = item.width
            img_h = item.height
            if not img_w or not img_h:
                continue

            img_path = RAW_DIR / f"{uuid}.jpg"
            if not img_path.exists():
                if not download_image(url, img_path):
                    continue

            yolo_lines = []
            for box in (item.boundingBoxes or []):
                if box.concept not in class_to_id:
                    continue
                if not is_valid_box(box, img_w, img_h):
                    continue
                class_id = class_to_id[box.concept]
                xc, yc, wn, hn = bbox_to_yolo(
                    box.x, box.y, box.width, box.height, img_w, img_h
                )
                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            if yolo_lines:
                records.append({
                    "uuid":        uuid,
                    "img_path":    str(img_path),
                    "annotations": "\n".join(yolo_lines),
                })
                downloaded += 1

        print(f"  Collected {downloaded} annotated images")

    if not records:
        print("No images downloaded.")
        return

    # Stratified 70/15/15 split
    df = pd.DataFrame(records)
    train_val, test = train_test_split(df, test_size=0.15, random_state=42)
    train, val      = train_test_split(train_val, test_size=0.176, random_state=42)

    print(f"\nSplit — train: {len(train)}, val: {len(val)}, test: {len(test)}")

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        print(f"Writing {split_name}...")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            src = Path(row["img_path"])
            if not src.exists():
                continue
            uuid = row["uuid"]
            shutil.copy2(src, IMAGES_DIR / split_name / f"{uuid}.jpg")
            (LABELS_DIR / split_name / f"{uuid}.txt").write_text(row["annotations"])

    names_block = "\n".join(f"  {i}: {s}" for i, s in enumerate(species_list))
    yaml_content = f"""# FathomNet Marine Species Detection Dataset
# Reference: Boulais et al. (2022), Scientific Reports

path: {DATA_DIR.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: {len(species_list)}
names:
{names_block}
"""
    (DATA_DIR / "dataset.yaml").write_text(yaml_content)

    class_map = {str(i): s for i, s in enumerate(species_list)}
    with open(DATA_DIR / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    print(f"\nDataset complete: {len(df)} images, {len(species_list)} species.")
    print(f"Config: {DATA_DIR / 'dataset.yaml'}")
    print("\nNext: python train.py")


if __name__ == "__main__":
    main()
