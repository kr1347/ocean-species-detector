"""
Step 1: Download and prepare FathomNet data for species detection.

─────────────────────────────────────────────────────────────────
WHAT IS FATHOMNET?
  MBARI (Monterey Bay Aquarium Research Institute) has sent ROVs
  into the deep ocean for 30+ years. Marine biologists labeled
  every organism in every frame. FathomNet makes that free to use.
  Think of it as ImageNet, but for the deep sea.

WHAT FORMAT DOES YOLO EXPECT?
  For every image, a .txt file with one line per object:
    <class_id> <x_center> <y_center> <width> <height>
  All values are 0–1 (normalized to image size).

  Example — a jellyfish covering the center-right of the image:
    3 0.75 0.50 0.20 0.30
    ↑      ↑     ↑    ↑    ↑
  class  cx    cy   w    h

  This is different from classification (one label per image).
  Detection: WHERE is each organism, and WHAT is it?

HOW TO RUN:
  python data/download_fathomnet.py
  python data/download_fathomnet.py --species 15 --per-species 500
─────────────────────────────────────────────────────────────────
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

# FathomNet Python SDK — handles auth, retries, and API versioning for us.
# Always prefer an official SDK over hand-crafted HTTP calls. If the API
# changes, the SDK maintainers update it — you don't have to.
from fathomnet.api import images as fn_images
from fathomnet.api import boundingboxes as fn_boxes

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
RAW_DIR    = DATA_DIR / "raw"   # cache — avoid re-downloading on reruns

# Species to exclude — too broad or non-biological
EXCLUDE_CONCEPTS = {
    "", "object", "animal", "organism", "unknown", "marine snow",
    "equipment", "debris", "bony fish", "marine organism",
}
EXCLUDE_KEYWORDS = [
    "bucket", "drum", "laser", "robotics", "camera", "cable",
    "pipe", "rope", "structured light",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_top_species(n: int, min_annotations: int = 100) -> list[str]:
    """
    Return the N most-annotated specific species names from FathomNet.

    HOW THIS WORKS:
      The SDK's count_total_by_concept() returns every labeled concept
      with its total bounding box count. We sort, filter junk, and take
      the top N.

    WHY WE FILTER:
      FathomNet has labels like "bony fish" (21k annotations but useless
      for species-level detection) and equipment labels. We skip those.
      A real species name almost always starts with a capital letter and
      doesn't contain equipment keywords.
    """
    print(f"Fetching concept counts from FathomNet...")
    all_counts = fn_boxes.count_total_by_concept()
    # all_counts is a list of ByConceptCount(concept=str, count=int)

    filtered = [
        c for c in all_counts
        if c.concept not in EXCLUDE_CONCEPTS
        and c.concept[0:1].isupper()       # real names are capitalized
        and c.count >= min_annotations
        and not any(k in c.concept.lower() for k in EXCLUDE_KEYWORDS)
    ]
    filtered.sort(key=lambda x: x.count, reverse=True)

    species = [c.concept for c in filtered[:n]]
    print(f"Selected {len(species)} species (top {n} by annotation count)")
    return species


def download_image(url: str, dest: Path) -> bool:
    """
    Download an image from URL and save as JPEG.

    WHY CONVERT TO JPEG?
      FathomNet hosts both PNG and JPEG images. YOLO expects a consistent
      format. We convert everything to RGB JPEG to avoid surprises.
      PNG is lossless but larger; JPEG is fine for training.

    WHY CACHE IN raw/?
      If the script crashes mid-download, we don't re-download everything.
      This is basic resilience — important when downloading 3000 images.
    """
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
    Convert pixel bounding box → YOLO normalized format.

    INPUT:  x, y = top-left corner (pixels), w, h = box size (pixels)
    OUTPUT: x_center, y_center, width, height — all in [0, 1]

    WHY NORMALIZE?
      YOLO is trained on images of different sizes. Normalizing makes
      annotations resolution-independent. Same idea as normalizing
      input features before sklearn models — consistent scale.

    EXAMPLE:
      Image: 1920×1080
      Box: x=960, y=270, w=192, h=108  (center-right, 10% of image)
      → xc = (960 + 96) / 1920 = 0.55
      → yc = (270 + 54) / 1080 = 0.30
      → wn = 192 / 1920         = 0.10
      → hn = 108 / 1080         = 0.10
    """
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    # Clamp to [0, 1] — annotations occasionally slightly exceed image bounds
    return (
        max(0.0, min(1.0, xc)),
        max(0.0, min(1.0, yc)),
        max(0.0, min(1.0, wn)),
        max(0.0, min(1.0, hn)),
    )


def is_valid_box(box, img_w: int, img_h: int) -> bool:
    """Reject degenerate boxes (zero-size or outside image)."""
    return (
        box.width > 0 and box.height > 0
        and box.x >= 0 and box.y >= 0
        and box.x < img_w and box.y < img_h
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--species",     type=int, default=10,
                   help="Number of species to include (default: 10)")
    p.add_argument("--per-species", type=int, default=300,
                   help="Max images per species (default: 300). "
                        "Caps dominant classes — same class-imbalance fix "
                        "as your ResNet/UTKFace project.")
    return p.parse_args()


def main():
    args = parse_args()
    TOP_N       = args.species
    MAX_PER     = args.per_species

    # Create directory structure
    # YOLO expects: images/train, images/val, images/test
    #               labels/train, labels/val, labels/test
    # Every image has a matching label file with the same stem.
    for split in ["train", "val", "test"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    # ── Step 1: Pick species ───────────────────────────────────────────────────
    species_list = get_top_species(TOP_N)
    class_to_id  = {s: i for i, s in enumerate(species_list)}

    print(f"\nTarget classes ({len(species_list)}):")
    for i, s in enumerate(species_list):
        print(f"  [{i}] {s}")

    # ── Step 2: Download images + build YOLO annotations ──────────────────────
    #
    # For each species we call fn_images.find_by_concept(concept).
    # This returns a list of AImageDTO objects, each with:
    #   .uuid, .url, .width, .height, .boundingBoxes
    #
    # .boundingBoxes is a list of ABoundingBoxDTO, each with:
    #   .concept, .x, .y, .width, .height  (all in pixels)
    #
    # One image can have boxes from MULTIPLE species (multi-label detection).
    # We process all boxes in each image, regardless of which species we
    # queried for — this is more efficient than downloading duplicates.

    records = []
    seen_uuids = set()   # avoid processing the same image twice

    for concept in species_list:
        print(f"\nFetching images for: {concept}")
        img_records = fn_images.find_by_concept(concept)

        # Cap per species to keep the dataset balanced.
        # Dominant class problem: if Lutjanus has 10× more images than
        # Ophiuroidea, the model learns to predict Lutjanus everywhere.
        img_records = img_records[:MAX_PER]
        print(f"  Found {len(img_records)} images (capped at {MAX_PER})")

        downloaded = 0
        for item in tqdm(img_records, desc=f"  {concept[:35]}"):
            uuid = item.uuid
            url  = item.url

            if not url or uuid in seen_uuids:
                continue
            seen_uuids.add(uuid)

            # Use image dimensions from the DTO — no need to open the file
            img_w = item.width
            img_h = item.height
            if not img_w or not img_h:
                continue

            # Download (or use cache)
            img_path = RAW_DIR / f"{uuid}.jpg"
            if not img_path.exists():
                if not download_image(url, img_path):
                    continue

            # Build YOLO label lines for all boxes in this image
            yolo_lines = []
            for box in (item.boundingBoxes or []):
                if box.concept not in class_to_id:
                    continue   # species we don't care about — skip
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

        print(f"  Collected {downloaded} usable annotated images")

    if not records:
        print("\nNo images downloaded. Check internet connection.")
        return

    # ── Step 3: Train / val / test split (70 / 15 / 15) ──────────────────────
    #
    # WHY THIS RATIO?
    #   70% train  — the model learns from this
    #   15% val    — used DURING training to tune hyperparameters & early stop
    #   15% test   — held out COMPLETELY until final evaluation
    #
    #   The test set must never influence any training decision.
    #   Think of it as the "real world" the model hasn't seen.
    #   This is the same concept as holding out a fold in cross-validation.
    #
    # WHY random_state=42?
    #   Reproducibility. Anyone who runs this gets the same split.

    df = pd.DataFrame(records)
    train_val, test = train_test_split(df, test_size=0.15, random_state=42)
    train, val      = train_test_split(train_val, test_size=0.176, random_state=42)
    # 0.176 of 85% ≈ 15% of the full dataset

    print(f"\nSplit — train: {len(train)}, val: {len(val)}, test: {len(test)}")

    # ── Step 4: Write image + label files into split directories ──────────────
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        print(f"Writing {split_name}...")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            src = Path(row["img_path"])
            if not src.exists():
                continue
            uuid = row["uuid"]
            shutil.copy2(src, IMAGES_DIR / split_name / f"{uuid}.jpg")
            (LABELS_DIR / split_name / f"{uuid}.txt").write_text(row["annotations"])

    # ── Step 5: Write dataset.yaml ────────────────────────────────────────────
    #
    # YOLOv8 reads this file to know:
    #   - Where the images and labels are
    #   - How many classes (nc)
    #   - What each class ID maps to
    #
    # The 'path' field is the root — all other paths are relative to it.

    names_block = "\n".join(f"  {i}: {s}" for i, s in enumerate(species_list))
    yaml_content = f"""# FathomNet Marine Species Detection Dataset
# Auto-generated by download_fathomnet.py

path: {DATA_DIR.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: {len(species_list)}
names:
{names_block}
"""
    (DATA_DIR / "dataset.yaml").write_text(yaml_content)

    # Also save a class_map.json for the inference API
    class_map = {str(i): s for i, s in enumerate(species_list)}
    with open(DATA_DIR / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    total = len(df)
    print(f"\nDone. {total} total images across {len(species_list)} species.")
    print(f"Dataset config: {DATA_DIR / 'dataset.yaml'}")
    print("\nNext: python train.py")


if __name__ == "__main__":
    main()
