import os
import io
import numpy as np
import cv2
import onnxruntime as ort
from typing import Tuple
from PIL import Image, ImageDraw

# --- Config ---
TARGET_SIZE = 640
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45
DEFAULT_MODEL = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "ship_detection.onnx")
)

# --- Globals ---
_session = None
_input_name = None
_output_name = None


def _load_model(model_path: str = None):
    """Lazy load ONNX model."""
    global _session, _input_name, _output_name
    if _session is not None:
        return
    model_path = model_path or DEFAULT_MODEL
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Ship ONNX model not found: {model_path}")
    _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _input_name = _session.get_inputs()[0].name
    _output_name = _session.get_outputs()[0].name


def _letterbox(im, new_shape=640, color=(114, 114, 114)):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def _preprocess_pil(pil_img: Image.Image, target_size: int = TARGET_SIZE):
    """Prepare image for ONNX inference."""
    img = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # RGB->BGR
    img, r, (dw, dh) = _letterbox(img, target_size)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    return img, r, (dw, dh), pil_img.size


def _non_max_suppression(prediction, conf_thres=0.4, iou_thres=0.45):
    """Simple NMS for YOLO output."""
    boxes = prediction[:, :4]
    scores = prediction[:, 4]

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return prediction[keep]


def _postprocess(output, orig_size, ratio, pad, conf_thres=0.4, iou_thres=0.45):
    """Convert model outputs into image-space boxes."""
    pred = np.squeeze(output)
    if pred.shape[0] == 5:
        pred = pred.T  # (5, N) -> (N, 5)

    pred = pred.T if pred.shape[0] == 5 else pred
    pred = pred.T if pred.shape[0] == 5 else pred
    pred = pred.T if pred.shape[0] == 5 else pred  # safety (some exports nest deeply)

    if pred.shape[1] < 5:
        raise ValueError(f"Unexpected model output shape: {pred.shape}")

    pred = pred[pred[:, 4] > conf_thres]
    if len(pred) == 0:
        return []

    nms_boxes = _non_max_suppression(pred, conf_thres, iou_thres)

    # Undo padding and scale
    boxes_out = []
    for (cx, cy, w, h, conf) in nms_boxes:
        x1 = (cx - w / 2 - pad[0]) / ratio
        y1 = (cy - h / 2 - pad[1]) / ratio
        x2 = (cx + w / 2 - pad[0]) / ratio
        y2 = (cy + h / 2 - pad[1]) / ratio
        x1 = np.clip(x1, 0, orig_size[0] - 1)
        y1 = np.clip(y1, 0, orig_size[1] - 1)
        x2 = np.clip(x2, 0, orig_size[0] - 1)
        y2 = np.clip(y2, 0, orig_size[1] - 1)
        boxes_out.append((x1, y1, x2, y2, conf))
    return boxes_out


def run_ship(image_bytes: bytes, model_path: str = None) -> bytes:
    """Main callable from Flask app."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    _load_model(model_path)
    img, r, pad, orig_size = _preprocess_pil(pil_img)

    output = _session.run(None, { _input_name: img })[0]
    boxes = _postprocess(output, orig_size, r, pad, CONF_THRESHOLD, NMS_THRESHOLD)

    out_pil = pil_img.copy()
    draw = ImageDraw.Draw(out_pil)
    for (x1, y1, x2, y2, conf) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 2, y1 - 12), f"{conf:.2f}", fill="red")

    # Diagnostic logging: how many detections?
    # try:
    #     det_count = len(boxes) if boxes is not None else 0
    # except Exception:
    #     det_count = 0
    # print(f"[ship_infer] detections={det_count}")

    # # If debug env var is set, save a copy to cwd for inspection
    # try:
    #     if os.environ.get("AQS_DEBUG_DETECTIONS") == "1":
    #         dbg_path = os.path.join(os.getcwd(), "ship_out_debug.png")
    #         out_pil.save(dbg_path)
    #         print(f"[ship_infer] saved debug image: {dbg_path}")
    # except Exception as e:
    #     print("[ship_infer] failed to write debug image:", e)

    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    return buf.getvalue()
