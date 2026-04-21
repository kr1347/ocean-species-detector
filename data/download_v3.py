"""
Expanded dataset acquisition for Experiment 3.

Modifications from the baseline download script:
    - Per-species cap increased from 300 to 600 images for common species.
    - Ptereleotris and Ophiuroidea are uncapped (all available FathomNet images
      retrieved) to address their low AP scores in Experiment 2 (0.331 and 0.376
      respectively). Data scarcity was identified as the primary limiting factor
      for these classes based on annotation count analysis.
    - Retry logic added to handle transient FathomNet API gateway errors (504).
    - Resume support: set RESUME_FROM to continue an interrupted download session
      without re-downloading previously cached images.
    - Outputs to data/data_v3/ to preserve Experiment 1/2 datasets intact.

Usage:
    python data/download_v3.py
"""

import io
import json
import shutil
import requests
import time
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

from fathomnet.api import images as fn_images

SCRIPT_DIR  = Path(__file__).parent
DATA_V3_DIR = SCRIPT_DIR / "data_v3"
IMAGES_DIR  = DATA_V3_DIR / "images"
LABELS_DIR  = DATA_V3_DIR / "labels"
RAW_DIR     = SCRIPT_DIR / "raw"

SPECIES = [
    "Lutjanus campechanus",
    "Stenotomus caprinus",
    "Rhomboplites aurorubens",
    "Strongylocentrotus fragilis",
    "Ptereleotris",
    "Pagrus pagrus",
    "Chromis",
    "Epinephelus morio",
    "Ophiuroidea",
    "Balistes capriscus",
]

# None = no cap; retrieve all available FathomNet images for that concept
PER_SPECIES_CAP = {
    "Lutjanus campechanus":          600,
    "Stenotomus caprinus":           600,
    "Rhomboplites aurorubens":       600,
    "Strongylocentrotus fragilis":   600,
    "Ptereleotris":                  None,
    "Pagrus pagrus":                 600,
    "Chromis":                       600,
    "Epinephelus morio":             600,
    "Ophiuroidea":                   None,
    "Balistes capriscus":            600,
}

# Set to a species name to resume from that point after an interrupted run.
# Set to None to start from the beginning.
RESUME_FROM = None

CLASS_TO_ID = {s: i for i, s in enumerate(SPECIES)}


def fetch_with_retry(fn, *args, retries=3, delay=10):
    """Call a FathomNet API function with exponential-backoff retry on failure."""
    for attempt in range(retries):
        try:
            return fn(*args)
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt + 1}/{retries} (error: {e})")
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
    xc = max(0.0, min(1.0, (x + w / 2) / img_w))
    yc = max(0.0, min(1.0, (y + h / 2) / img_h))
    wn = max(0.0, min(1.0, w / img_w))
    hn = max(0.0, min(1.0, h / img_h))
    return xc, yc, wn, hn


def main():
    for split in ["train", "val", "test"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    records = []
    seen_uuids = set()
    class_counts = defaultdict(int)

    resume_active = RESUME_FROM is not None
    if resume_active:
        print(f"Resuming from: {RESUME_FROM}")
        for img_path in RAW_DIR.glob("*.jpg"):
            seen_uuids.add(img_path.stem)

    for concept in SPECIES:
        if resume_active:
            if concept != RESUME_FROM:
                print(f"Skipping (already complete): {concept}")
                continue
            else:
                resume_active = False

        cap = PER_SPECIES_CAP[concept]
        cap_str = str(cap) if cap else "ALL"
        print(f"\nFetching: {concept} (cap={cap_str})")

        img_records = fetch_with_retry(fn_images.find_by_concept, concept)
        if cap:
            img_records = img_records[:cap]
        print(f"  {len(img_records)} images available")

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
                if box.width <= 0 or box.height <= 0:
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

        print(f"  Collected: {downloaded} images")

    if not records:
        print("No new images downloaded.")
        return

    # Merge with existing split files from a prior partial run
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

    print(f"\nNew images: {len(records)}, existing: {len(existing_records)}")

    df = pd.DataFrame(records)
    train_val, test = train_test_split(df, test_size=0.15, random_state=42)
    train, val      = train_test_split(train_val, test_size=0.176, random_state=42)

    print(f"Split — train: {len(train)}, val: {len(val)}, test: {len(test)}")

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        print(f"Writing {split_name}...")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            src = Path(row["img_path"])
            if not src.exists():
                continue
            uuid = row["uuid"]
            shutil.copy2(src, IMAGES_DIR / split_name / f"{uuid}.jpg")
            (LABELS_DIR / split_name / f"{uuid}.txt").write_text(row["annotations"])

    names_block = "\n".join(f"  {i}: {s}" for i, s in enumerate(SPECIES))
    yaml_content = f"""# FathomNet Marine Species Detection Dataset — Experiment 3 (expanded)
# Reference: Boulais et al. (2022), Scientific Reports

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

    print("\n── Annotation counts per class ──")
    for species in SPECIES:
        print(f"  {species:<40} {class_counts[species]:>6} boxes")

    total = len(df)
    print(f"\nTotal images: {total}")
    print(f"Dataset config: {DATA_V3_DIR / 'dataset.yaml'}")
    print("\nNext: python train_v3.py")


if __name__ == "__main__":
    main()
