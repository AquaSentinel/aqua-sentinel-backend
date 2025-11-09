# inference/debris_optimized.py - Optimized using shared utilities
import os
from PIL import Image
from utils.inference_utils import (
    preprocess_debris_image,
    postprocess_debris_detections,
    draw_detections,
    ModelSession,
)

# --- Config ---
IMAGE_SIZE = (256, 256)  # (W, H)
CONF_THRESHOLD = 0.55
NMS_THRESHOLD = 0.2
DEFAULT_MODEL = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "models", "marine_debris_detector.onnx"
    )
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


def detect_debris(
    pil_img: Image.Image,
    model_path: str = None,
    conf_thresh: float = CONF_THRESHOLD,
    nms_thresh: float = NMS_THRESHOLD,
) -> tuple:
    """
    Detect debris in image and return bounding boxes and annotated image.

    Returns:
        tuple: (boxes, annotated_image)
            - boxes: numpy array of (x1, y1, x2, y2) coordinates
            - annotated_image: PIL Image with drawn boxes
    """
    # Get model session
    session = _get_model_session(model_path)

    # Preprocess
    input_tensor, orig_size = preprocess_debris_image(pil_img, IMAGE_SIZE)

    # Inference - handle different input shapes the model might expect
    try:
        # Try 4D input first (batch dimension)
        outputs = session.predict(input_tensor)
    except Exception:
        try:
            # Try 3D input (no batch dimension)
            input_3d = input_tensor.squeeze(0)  # Remove batch dim
            outputs = session.predict(input_3d)
        except Exception:
            # Try HWC format
            input_hwc = input_tensor.squeeze(0).transpose(1, 2, 0)  # CHW -> HWC
            outputs = session.predict(input_hwc)

    # Postprocess
    boxes, scores = postprocess_debris_detections(
        outputs, orig_size, IMAGE_SIZE, conf_thresh, nms_thresh
    )

    # Draw annotations (scores used for drawing but not returned)
    annotated_img = draw_detections(pil_img, boxes, scores, "debris", "red")

    return boxes, annotated_img
