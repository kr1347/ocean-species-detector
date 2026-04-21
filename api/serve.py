"""



Run with:
  uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload

Test with:
  curl -X POST "http://localhost:8000/detect" \
       -F "file=@path/to/underwater_image.jpg"
"""

import io
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────

WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "fathomnet_v1" / "weights" / "best.pt"
CLASS_MAP_PATH = Path(__file__).parent.parent / "data" / "class_map.json"

DEFAULT_CONF = 0.25   # minimum confidence to report a detection
DEFAULT_IOU = 0.45    # IoU threshold for NMS

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Marine Species Detector",
    description=(
        "Detects deep-sea marine species in underwater imagery. "
        "Trained on FathomNet data from MBARI. "
        "Returns bounding boxes, species names, and confidence scores."
    ),
    version="1.0.0",
)

# Load model once at startup — not on every request
# This is the same pattern as loading a heavy NLP model (BERT, etc.) once
_model: Optional[YOLO] = None
_class_map: dict = {}


@app.on_event("startup")
async def load_model():
    global _model, _class_map

    if not WEIGHTS_PATH.exists():
        raise RuntimeError(
            f"Model weights not found at {WEIGHTS_PATH}. "
            "Run  python train.py  first."
        )

    print(f"Loading model from {WEIGHTS_PATH}...")
    _model = YOLO(str(WEIGHTS_PATH))

    if CLASS_MAP_PATH.exists():
        with open(CLASS_MAP_PATH) as f:
            _class_map = json.load(f)

    print("Model ready.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Quick liveness check — same pattern you used at TCS."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "n_classes": len(_class_map),
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
):
    """
    Run species detection on an uploaded image.

    Returns a list of detections, each with:
      - species: the predicted species name
      - confidence: model confidence (0-1)
      - bbox: [x1, y1, x2, y2] in pixel coordinates (top-left, bottom-right)
      - bbox_normalized: same but normalized to [0,1] relative to image size
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read and decode image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    img_w, img_h = image.size

    # Run inference
    t0 = time.perf_counter()
    results = _model.predict(
        source=np.array(image),
        conf=conf,
        iou=iou,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # Parse detections
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "species": _class_map.get(str(cls_id), f"class_{cls_id}"),
                "class_id": cls_id,
                "confidence": round(conf_score, 4),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "bbox_normalized": {
                    "x1": round(x1 / img_w, 4),
                    "y1": round(y1 / img_h, 4),
                    "x2": round(x2 / img_w, 4),
                    "y2": round(y2 / img_h, 4),
                },
            })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return JSONResponse({
        "image_size": {"width": img_w, "height": img_h},
        "n_detections": len(detections),
        "latency_ms": round(latency_ms, 2),
        "detections": detections,
    })


@app.get("/classes")
async def list_classes():
    """Return the list of detectable species."""
    return {"classes": _class_map}
