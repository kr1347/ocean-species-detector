"""
Data download v3 — expanded FathomNet pull for v3 training run.

Changes from v1 (download_fathomnet.py):
  - 600 images per common species (was 300)
  - Unlimited images for weak classes: Ptereleotris, Ophiuroidea
    (these scored 0.331 and 0.376 mAP — need more data)
  - Merges with existing raw/ cache — only downloads what's missing
  - Writes to a separate data_v3/ directory so v1/v2 data is untouched
  - Saves per-class counts so we can verify balance before training

Run:
  python data/download_v3.py
"""

import argparse
import io
import json
import shutil
import requests
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

from fathomnet.api import images as fn_images
from fathomnet.api import boundingboxes as fn_boxes

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
DATA_V3_DIR = SCRIPT_DIR / "data_v3"
IMAGES_DIR  = DATA_V3_DIR / "images"
LABELS_DIR  = DATA_V3_DIR / "labels"
RAW_DIR     = SCRIPT_DIR / "raw"   # shared cache with v1 — reuse downloads

# Our existing 10 classes — keep the same IDs so we can compare apples-to-apples
SPECIES = [
    "Lutjanus campechanus",     # 0 — strong (0.958 mAP)
    "Stenotomus caprinus",      # 1
    "Rhomboplites aurorubens",  # 2 — strong (0.900 mAP)
    "Strongylocentrotus fragilis", # 3
    "Ptereleotris",             # 4 — WEAK (0.331 mAP) — download all available
    "Pagrus pagrus",            # 5
    "Chromis",                  # 6 — strong (0.906 mAP)
    "Epinephelus morio",        # 7 — strong (0.906 mAP)
    "Ophiuroidea",              # 8 — WEAK (0.376 mAP) — download all available
    "Balistes capriscus",       # 9 — strong (0.951 mAP)
]

# Set to resume from a specific species if a previous run crashed mid-way.
# Set to None to start from the beginning.
RESUME_FROM = "Epinephelus morio"

# Download caps: None = no cap (get everything FathomNet has)
PER_SPECIES_CAP = {
    "Lutjanus campechanus":          600,
    "Stenotomus caprinus":           600,
    "Rhomboplites aurorubens":       600,
    "Strongylocentrotus fragilis":   600,
    "Ptereleotris":                  None,  # get all — only 0.331 mAP
    "Pagrus pagrus":                 600,
    "Chromis":                       600,
    "Epinephelus morio":             600,
    "Ophiuroidea":                   None,  # get all — only 0.376 mAP
    "Balistes capriscus":            600,
}

CLASS_TO_ID = {s: i for i, s in enumerate(SPECIES)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_with_retry(fn, *args, retries=3, delay=10, **kwargs):
    """Call a FathomNet SDK function with retries on 504/network errors."""
    import time
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt+1}/{retries} after error: {e}")
                time.sleep(delay)
            else:
                raise


def download_image(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(dest, "JPEG", quality=95)
        return True
    except Exception:
        return False


def bbox_to_yolo(x, y, w, h, img_w, img_h):
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


def is_valid_box(box, img_w, img_h):
    return (
        box.width > 0 and box.height > 0
        and box.x >= 0 and box.y >= 0
        and box.x < img_w and box.y < img_h
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for split in ["train", "val", "test"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    records = []
    seen_uuids = set()
    class_counts = defaultdict(int)

    # If resuming, load existing records from raw dir so UUIDs aren't re-processed
    resume_active = RESUME_FROM is not None
    if resume_active:
        print(f"Resuming from: {RESUME_FROM}")
        for img_path in RAW_DIR.glob("*.jpg"):
            seen_uuids.add(img_path.stem)

    for concept in SPECIES:
        if resume_active:
            if concept != RESUME_FROM:
                print(f"Skipping (already done): {concept}")
                continue
            else:
                resume_active = False  # start processing from here
        cap = PER_SPECIES_CAP[concept]
        cap_str = str(cap) if cap else "ALL"
        print(f"\nFetching: {concept} (cap={cap_str})")

        img_records = fetch_with_retry(fn_images.find_by_concept, concept)
        if cap:
            img_records = img_records[:cap]
        print(f"  Available: {len(img_records)} images")

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
                if box.concept not in CLASS_TO_ID:
                    continue
                if not is_valid_box(box, img_w, img_h):
                    continue
                class_id = CLASS_TO_ID[box.concept]
                xc, yc, wn, hn = bbox_to_yolo(
                    box.x, box.y, box.width, box.height, img_w, img_h
                )
                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
                class_counts[box.concept] += 1

            if yolo_lines:
                records.append({
                    "uuid":        uuid,
                    "img_path":    str(img_path),
                    "annotations": "\n".join(yolo_lines),
                })
                downloaded += 1

        print(f"  Collected: {downloaded} usable images")

    if not records:
        print("No new images downloaded (all already cached or none found).")
        # Still continue to write split if resuming and data_v3 already has files
        existing = list((IMAGES_DIR / "train").glob("*.jpg"))
        if not existing:
            return
        print(f"Using existing {len(existing)} train images from prior run.")
        return

    # ── Merge with any existing split files from prior partial run ─────────────
    existing_records = []
    for split_name in ["train", "val", "test"]:
        for img_file in (IMAGES_DIR / split_name).glob("*.jpg"):
            uuid = img_file.stem
            label_file = LABELS_DIR / split_name / f"{uuid}.txt"
            if label_file.exists():
                existing_records.append({
                    "uuid":        uuid,
                    "img_path":    str(RAW_DIR / f"{uuid}.jpg"),
                    "annotations": label_file.read_text(),
                    "_split":      split_name,
                })

    print(f"New images this run: {len(records)}, existing: {len(existing_records)}")

    # ── Split new records only; existing keep their split ─────────────────────
    df = pd.DataFrame(records)
    train_val, test = train_test_split(df, test_size=0.15, random_state=42)
    train, val      = train_test_split(train_val, test_size=0.176, random_state=42)

    print(f"\nSplit — train: {len(train)}, val: {len(val)}, test: {len(test)}")

    # ── Write files ───────────────────────────────────────────────────────────
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        print(f"Writing {split_name}...")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            src = Path(row["img_path"])
            if not src.exists():
                continue
            uuid = row["uuid"]
            shutil.copy2(src, IMAGES_DIR / split_name / f"{uuid}.jpg")
            (LABELS_DIR / split_name / f"{uuid}.txt").write_text(row["annotations"])

    # ── dataset.yaml ──────────────────────────────────────────────────────────
    names_block = "\n".join(f"  {i}: {s}" for i, s in enumerate(SPECIES))
    yaml_content = f"""# FathomNet v3 — expanded download
path: {DATA_V3_DIR.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: {len(SPECIES)}
names:
{names_block}
"""
    (DATA_V3_DIR / "dataset.yaml").write_text(yaml_content)

    class_map = {str(i): s for i, s in enumerate(SPECIES)}
    with open(DATA_V3_DIR / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    # ── Class balance report ──────────────────────────────────────────────────
    print(f"\n── Annotation counts per class ──")
    for species in SPECIES:
        print(f"  {species:<40} {class_counts[species]:>6} boxes")

    print(f"\nTotal images: {len(df)}")
    print(f"Dataset config: {DATA_V3_DIR / 'dataset.yaml'}")
    print("\nNext: python train_v3.py")


if __name__ == "__main__":
    main()
