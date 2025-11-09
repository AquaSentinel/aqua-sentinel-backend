# inference/ship_optimized.py - Optimized using shared utilities
import os
from PIL import Image
from utils.inference_utils import (
    preprocess_ship_image,
    postprocess_ship_detections,
    draw_detections,
    ModelSession,
)

# --- Config ---
TARGET_SIZE = 640
CONF_THRESHOLD = 0.6
NMS_THRESHOLD = 0.45
DEFAULT_MODEL = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "ship_detection.onnx")
)

# --- Global model session ---
_model_session = None


def _get_model_session(model_path: str = None):
    """Get or create model session."""
    global _model_session
    path = model_path or DEFAULT_MODEL

    if _model_session is None or _model_session.model_path != path:
        _model_session = ModelSession(path)

    return _model_session


def detect_ships(
    pil_img: Image.Image,
    model_path: str = None,
    conf_thresh: float = CONF_THRESHOLD,
    nms_thresh: float = NMS_THRESHOLD,
) -> tuple:
    """
    Detect ships in image and return bounding boxes and annotated image.

    Returns:
        tuple: (boxes, annotated_image)
            - boxes: List of (x1, y1, x2, y2, conf) tuples
            - annotated_image: PIL Image with drawn boxes
    """
    # Get model session
    session = _get_model_session(model_path)

    # Preprocess
    input_tensor, ratio, pad, orig_size = preprocess_ship_image(pil_img, TARGET_SIZE)

    # Inference
    outputs = session.predict(input_tensor)

    # Postprocess
    boxes = postprocess_ship_detections(
        outputs[0], orig_size, ratio, pad, conf_thresh, nms_thresh
    )

    # Extract scores for drawing (but don't return them)
    scores = [box[4] for box in boxes] if boxes else []

    # Draw annotations
    annotated_img = draw_detections(pil_img, boxes, scores, "ship", "red")

    return boxes, annotated_img
