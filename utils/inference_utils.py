# utils/inference_utils.py - Shared efficient inference utilities
import os
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Union
import onnxruntime as ort


def letterbox_resize(image: np.ndarray, target_size: int, color=(114, 114, 114)):
    """Efficient letterbox resize for YOLO models."""
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    # Calculate ratio (new / old)
    r = min(target_size[0] / shape[0], target_size[1] / shape[1])

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = target_size[1] - new_unpad[0], target_size[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # if not same shape
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return image, r, (dw, dh)


def preprocess_ship_image(
    pil_img: Image.Image, target_size: int = 640
) -> Tuple[np.ndarray, float, Tuple[float, float], Tuple[int, int]]:
    """Preprocess image for ship detection model."""
    # Convert PIL to numpy (RGB -> BGR for consistency)
    img = np.array(pil_img.convert("RGB"))[:, :, ::-1]
    orig_size = pil_img.size

    # Letterbox resize
    img, ratio, pad = letterbox_resize(img, target_size)

    # Convert to model input format: (1, 3, H, W), normalized
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, 0)  # Add batch dimension
    img = np.ascontiguousarray(img, dtype=np.float16) / 255.0

    return img, ratio, pad, orig_size


def preprocess_debris_image(
    pil_img: Image.Image, target_size: Tuple[int, int] = (256, 256)
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Preprocess image for debris detection model."""
    orig_size = pil_img.size  # (width, height)

    # Resize image
    resized = pil_img.resize(target_size)

    # Convert to model input format: (1, 3, H, W), normalized
    img = np.array(resized, dtype=np.float16) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, 0)  # Add batch dimension

    return img, orig_size


def efficient_nms(
    prediction: np.ndarray, conf_thresh: float = 0.6, iou_thresh: float = 0.45
) -> List[Tuple[float, float, float, float, float]]:
    """Simple NMS for YOLO output - matches original ship.py implementation."""
    boxes = prediction[:, :4]  # cx, cy, w, h
    scores = prediction[:, 4]

    # Convert to x1, y1, x2, y2 for area calculation
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
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    # Return the kept predictions with original format (cx, cy, w, h, conf)
    return [
        (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i]) for i in keep
    ]


def postprocess_ship_detections(
    output: np.ndarray,
    orig_size: Tuple[int, int],
    ratio: float,
    pad: Tuple[float, float],
    conf_thresh: float = 0.6,
    iou_thresh: float = 0.45,
) -> List[Tuple[float, float, float, float, float]]:
    """Postprocess ship model outputs to get final bounding boxes."""
    # Handle different output shapes - match original ship.py exactly
    pred = np.squeeze(output)
    if pred.shape[0] == 5:
        pred = pred.T  # (5, N) -> (N, 5)

    pred = pred.T if pred.shape[0] == 5 else pred
    pred = pred.T if pred.shape[0] == 5 else pred
    pred = pred.T if pred.shape[0] == 5 else pred  # safety (some exports nest deeply)

    if pred.shape[1] < 5:
        raise ValueError(f"Unexpected model output shape: {pred.shape}")

    # Filter by confidence first
    pred = pred[pred[:, 4] > conf_thresh]
    if len(pred) == 0:
        return []

    # Apply NMS - this returns list of (cx, cy, w, h, conf) tuples
    nms_boxes = efficient_nms(pred, conf_thresh, iou_thresh)

    # Convert to image coordinates and format (x1, y1, x2, y2, conf)
    boxes_out = []
    for cx, cy, w, h, conf in nms_boxes:
        # Undo padding and scale - match original exactly
        x1 = (cx - w / 2 - pad[0]) / ratio
        y1 = (cy - h / 2 - pad[1]) / ratio
        x2 = (cx + w / 2 - pad[0]) / ratio
        y2 = (cy + h / 2 - pad[1]) / ratio
        x1 = np.clip(x1, 0, orig_size[0] - 1)  # orig_size is (width, height)
        y1 = np.clip(y1, 0, orig_size[1] - 1)
        x2 = np.clip(x2, 0, orig_size[0] - 1)
        y2 = np.clip(y2, 0, orig_size[1] - 1)
        boxes_out.append((x1, y1, x2, y2, conf))

    return boxes_out


def postprocess_debris_detections(
    outputs: Union[List, Tuple, np.ndarray],
    orig_size: Tuple[int, int],
    model_size: Tuple[int, int] = (256, 256),
    conf_thresh: float = 0.55,
    iou_thresh: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Postprocess debris model outputs to get final bounding boxes."""
    # Handle different output formats
    if isinstance(outputs, (list, tuple)):
        if len(outputs) >= 3:
            boxes = np.array(outputs[0])
            scores = np.array(outputs[2])
        elif len(outputs) == 2:
            boxes = np.array(outputs[0])
            scores = np.array(outputs[1])
        else:
            boxes = np.array(outputs[0])
            scores = np.array([])
    else:
        boxes = np.array(outputs)
        scores = np.array([])

    if boxes.size == 0:
        return np.array([]), np.array([])

    # Filter by confidence
    if scores.size > 0:
        scores = scores.flatten()
        mask = scores >= conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]
    else:
        scores = np.ones(len(boxes))  # Default confidence if not provided

    if len(boxes) == 0:
        return np.array([]), np.array([])

    # Convert boxes to format for NMS (x, y, w, h)
    boxes_nms = boxes.copy()
    if boxes.shape[1] >= 4:
        boxes_nms[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
        boxes_nms[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1

    # Apply NMS
    try:
        indices = cv2.dnn.NMSBoxes(
            boxes_nms.tolist(), scores.tolist(), conf_thresh, iou_thresh
        )
        if len(indices) == 0:
            return np.array([]), np.array([])
        indices = indices.flatten()
    except Exception:
        indices = np.arange(len(boxes))  # Fallback: keep all boxes

    final_boxes = boxes[indices]
    final_scores = scores[indices]

    # Scale boxes from model size to original size
    orig_w, orig_h = orig_size
    model_w, model_h = model_size
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h

    final_boxes[:, [0, 2]] *= scale_x  # Scale x coordinates
    final_boxes[:, [1, 3]] *= scale_y  # Scale y coordinates

    # Clip to image bounds
    final_boxes = np.clip(final_boxes, [0, 0, 0, 0], [orig_w, orig_h, orig_w, orig_h])

    return final_boxes.astype(int), final_scores


def draw_detections(
    pil_img: Image.Image,
    boxes: Union[List, np.ndarray],
    scores: Union[List, np.ndarray],
    class_name: str = "detection",
    color: str = "red",
) -> Image.Image:
    """Draw bounding boxes on image efficiently."""
    from PIL import ImageDraw

    img_copy = pil_img.copy()
    draw = ImageDraw.Draw(img_copy)

    for i, box in enumerate(boxes):
        if len(box) >= 4:
            x1, y1, x2, y2 = box[:4]
        else:
            continue

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    return img_copy


class ModelSession:
    """Efficient model session manager with lazy loading."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._session = None
        self._input_name = None
        self._output_names = None

    def _load(self):
        """Lazy load the model."""
        if self._session is not None:
            return

        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]

    def predict(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Run inference."""
        self._load()
        return self._session.run(self._output_names, {self._input_name: input_tensor})

    @property
    def input_shape(self):
        """Get expected input shape."""
        self._load()
        return self._session.get_inputs()[0].shape
