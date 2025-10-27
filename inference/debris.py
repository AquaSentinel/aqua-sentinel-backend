# inference/debris.py
"""
Debris detection module (robust to models that expect rank-3 or rank-4 inputs).
Exposes detect_image(pil_image, model_path=None) -> PIL.Image with boxes drawn.
"""
import os
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import onnxruntime as ort

# module-level session (lazy)
_session = None
_input_meta = None

# defaults
IMAGE_SIZE = (256, 256)  # (W, H)
CONF_THRESHOLD = 0.55
NMS_IOU = 0.4
CLASS_NAME = "debris"


def _load_model(model_path: str):
    global _session, _input_meta
    if _session is not None:
        return
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Debris ONNX model not found: {model_path}")
    _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _input_meta = _session.get_inputs()[0]


def _preprocess(pil_img: Image.Image, target_size: Tuple[int, int] = IMAGE_SIZE):
    orig_w, orig_h = pil_img.size
    resized = pil_img.resize(target_size)
    # produce channels-first (C, H, W) and add batch dim -> (1, C, H, W)
    image_np = np.array(resized).astype(np.float32) / 255.0
    # If image_np shape is (H, W, C) -> transpose to C,H,W
    if image_np.ndim == 3:
        image_chw = image_np.transpose(2, 0, 1)
    else:
        raise ValueError("Unexpected image array shape during preprocess.")
    image_np_batched = np.expand_dims(image_chw, axis=0).astype(np.float32)
    return image_np_batched, (orig_h, orig_w)


def _postprocess(outputs, orig_shape: Tuple[int, int], resize_size: Tuple[int, int] = IMAGE_SIZE,
                 conf_thresh: float = CONF_THRESHOLD, nms_iou: float = NMS_IOU):
    # outputs may be (boxes, labels, scores) or (boxes, scores) or just a single array
    boxes, scores = None, None
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 1:
            boxes = np.array(outputs[0])
            scores = np.array([])
        elif len(outputs) == 2:
            boxes = np.array(outputs[0])
            scores = np.array(outputs[1])
        else:
            boxes = np.array(outputs[0])
            scores = np.array(outputs[-1])
    else:
        arr = np.array(outputs)
        return np.array([]), np.array([])

    if boxes.size == 0:
        return np.array([]), np.array([])

    scores = scores.flatten() if scores.ndim > 0 else scores
    mask = scores >= conf_thresh if scores.size > 0 else np.array([True] * len(boxes))
    boxes = boxes[mask]
    scores = scores[mask]
    if boxes.size == 0:
        return np.array([]), np.array([])

    # convert to x,y,w,h for NMS
    boxes_for_nms = boxes.copy()
    boxes_for_nms[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes_for_nms[:, 3] = boxes[:, 3] - boxes[:, 1]
    boxes_list = boxes_for_nms.tolist()
    scores_list = scores.tolist()
    try:
        indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, conf_thresh, nms_iou)
    except Exception:
        indices = []
    if len(indices) == 0:
        return np.array([]), np.array([])
    if isinstance(indices, (list, tuple)):
        indices = np.array(indices).flatten()
    else:
        indices = indices.flatten()
    boxes = boxes[indices]
    scores = scores[indices]

    # scale boxes from resized->orig
    orig_h, orig_w = orig_shape
    resize_w, resize_h = resize_size
    scale_x = orig_w / resize_w
    scale_y = orig_h / resize_h
    boxes[:, 0] = (boxes[:, 0] * scale_x).astype(int)
    boxes[:, 1] = (boxes[:, 1] * scale_y).astype(int)
    boxes[:, 2] = (boxes[:, 2] * scale_x).astype(int)
    boxes[:, 3] = (boxes[:, 3] * scale_y).astype(int)
    boxes = np.clip(boxes, [0, 0, 0, 0], [orig_w, orig_h, orig_w, orig_h])
    return boxes.astype(int), scores


def _draw_boxes(pil_img: Image.Image, boxes, scores):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except Exception:
        from PIL import ImageFont as _IF
        font = _IF.load_default()
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = [int(x) for x in box]
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        score_text = f"{scores[i]:.2f}" if (scores is not None and i < len(scores)) else ""
        text = f"{CLASS_NAME} {score_text}".strip()
        if text:
            text_pos = (xmin, max(0, ymin - 14))
            draw.rectangle([text_pos[0], text_pos[1], text_pos[0] + 100, text_pos[1] + 14], fill="red")
            draw.text(text_pos, text, fill="white", font=font)
    return pil_img


def detect_image(pil_image: Image.Image, model_path: str = None) -> Image.Image:
    """
    High-level API used by app.py.
    Handles models that expect rank-3 or rank-4 inputs by trying sensible variants.
    """
    _load_model(model_path)
    input_tensor_batched, orig_shape = _preprocess(pil_image)  # shape: (1, C, H, W)

    # Determine expected input shape/rank from model metadata
    expected_shape = None
    try:
        expected_shape = _input_meta.shape  # e.g. [None, 3, 256, 256] or [3,256,256]
    except Exception:
        expected_shape = None

    # We'll attempt to feed a tensor that matches model expectations:
    #  - if model expects 4 dims -> feed batched (1,C,H,W)
    #  - if model expects 3 dims -> try (C,H,W) first, then (H,W,C)
    tried_shapes = []
    last_exc = None

    # Helper to run safely and capture exception
    def try_run(feed):
        nonlocal last_exc
        try:
            outputs = _session.run(None, { _input_meta.name: feed })
            return outputs
        except Exception as e:
            last_exc = e
            return None

    # Decide based on expected_shape if available
    if expected_shape is not None and len(expected_shape) == 4:
        # Use batched tensor directly
        tried_shapes.append(("batched", input_tensor_batched.shape))
        outputs = try_run(input_tensor_batched)
        if outputs is None:
            raise RuntimeError(f"Debris model rejected a batched input; last error: {last_exc}")
    else:
        # Model expects rank-3 (or unknown). Try (C,H,W) first (most likely),
        # then try (H,W,C) to be robust.
        # 1) try channels-first (C,H,W)
        feed1 = input_tensor_batched[0]  # (C,H,W)
        tried_shapes.append(("C,H,W", feed1.shape))
        outputs = try_run(feed1)
        if outputs is None:
            # 2) try channels-last (H,W,C)
            feed2 = feed1.transpose(1, 2, 0)  # (H,W,C)
            tried_shapes.append(("H,W,C", feed2.shape))
            outputs = try_run(feed2)
            if outputs is None:
                # 3) as a last-resort try the batched input shape (1,C,H,W)
                tried_shapes.append(("batched", input_tensor_batched.shape))
                outputs = try_run(input_tensor_batched)

        if outputs is None:
            # none of the attempts succeeded; raise informative error
            raise RuntimeError(
                "Debris model input mismatch. Tried these feed shapes: "
                f"{tried_shapes}. Last ONNX error: {last_exc}"
            )

    # Postprocess outputs normally
    boxes, scores = _postprocess(outputs, orig_shape, resize_size=IMAGE_SIZE)

    # Diagnostic logging + optional debug save
    # try:
    #     det_count = 0 if boxes is None or boxes.size == 0 else len(boxes)
    # except Exception:
    #     det_count = 0
    # print(f"[debris_infer] detections={det_count}")

    # if os.environ.get("AQS_DEBUG_DETECTIONS") == "1":
    #     try:
    #         out_img = pil_image.copy()
    #         if det_count > 0:
    #             out_img = _draw_boxes(out_img, boxes, scores)
    #         dbg_path = os.path.join(os.getcwd(), "debris_out_debug.png")
    #         out_img.save(dbg_path)
    #         print(f"[debris_infer] saved debug image: {dbg_path}")
    #     except Exception as e:
    #         print("[debris_infer] failed to write debug image:", e)

    if boxes.size == 0:
        return pil_image
    return _draw_boxes(pil_image.copy(), boxes, scores)
