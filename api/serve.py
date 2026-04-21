"""
FastAPI inference endpoint for deep-sea marine species detection.

Exposes a REST API for real-time species detection in underwater imagery.
Designed for deployment on vessel edge hardware or cloud inference services.

Endpoints:
    GET  /health   — liveness check; confirms model is loaded
    GET  /classes  — returns the full species class map
    POST /detect   — runs detection on an uploaded image; returns bounding boxes,
                     species labels, confidence scores, and inference latency

Model:
    YOLOv8 fine-tuned on FathomNet (MBARI) deep-sea ROV imagery.
    10 species: Lutjanus campechanus, Stenotomus caprinus, Rhomboplites aurorubens,
    Strongylocentrotus fragilis, Ptereleotris, Pagrus pagrus, Chromis,
    Epinephelus morio, Ophiuroidea, Balistes capriscus.

Run:
    uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload

Test:
    curl -X POST "http://localhost:8000/detect" -F "file=@image.jpg"
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

WEIGHTS_PATH   = Path(__file__).parent.parent / "models" / "fathomnet_v2" / "weights" / "best.pt"
CLASS_MAP_PATH = Path(__file__).parent.parent / "data" / "class_map.json"

DEFAULT_CONF = 0.25
DEFAULT_IOU  = 0.45

app = FastAPI(
    title="Marine Species Detector",
    description=(
        "Real-time detection of deep-sea marine species in underwater imagery. "
        "Trained on FathomNet data (MBARI). "
        "Returns bounding boxes, species classifications, and confidence scores."
    ),
    version="2.0.0",
)

_model: Optional[YOLO] = None
_class_map: dict = {}


@app.on_event("startup")
async def load_model():
    global _model, _class_map

    if not WEIGHTS_PATH.exists():
        raise RuntimeError(
            f"Model weights not found at {WEIGHTS_PATH}. "
            "Run training script first."
        )

    print(f"Loading model from {WEIGHTS_PATH}...")
    _model = YOLO(str(WEIGHTS_PATH))

    if CLASS_MAP_PATH.exists():
        with open(CLASS_MAP_PATH) as f:
            _class_map = json.load(f)

    print(f"Model ready. {len(_class_map)} species classes loaded.")


@app.get("/health")
async def health():
    """Liveness check."""
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

    Returns:
        image_size: width and height in pixels
        n_detections: number of detected organisms
        latency_ms: model inference time in milliseconds
        detections: list of detections, each containing:
            species: predicted species name
            class_id: integer class index
            confidence: model confidence score in [0, 1]
            bbox: [x1, y1, x2, y2] in pixel coordinates
            bbox_normalized: bbox normalized to [0, 1] relative to image dimensions
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    img_w, img_h = image.size

    t0 = time.perf_counter()
    results = _model.predict(
        source=np.array(image),
        conf=conf,
        iou=iou,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id     = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "species":    _class_map.get(str(cls_id), f"class_{cls_id}"),
                "class_id":   cls_id,
                "confidence": round(conf_score, 4),
                "bbox":       [round(x1), round(y1), round(x2), round(y2)],
                "bbox_normalized": {
                    "x1": round(x1 / img_w, 4),
                    "y1": round(y1 / img_h, 4),
                    "x2": round(x2 / img_w, 4),
                    "y2": round(y2 / img_h, 4),
                },
            })

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return JSONResponse({
        "image_size":   {"width": img_w, "height": img_h},
        "n_detections": len(detections),
        "latency_ms":   round(latency_ms, 2),
        "detections":   detections,
    })


@app.get("/classes")
async def list_classes():
    """Return the full species class map."""
    return {"classes": _class_map}
