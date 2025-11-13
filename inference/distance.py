# inference/distance.py
"""
Distance computation between detected ships and debris.
Reuses ship_2.py (blue boxes) and debris_2.py (red boxes).
"""

import math
import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import io

from inference_ipynb.ship_2 import run_ship
from inference_ipynb.debris_2 import detect_image


# --- Constants ---
BLUE = "blue"
RED = "red"
GREEN = (0, 255, 0)


def _detect_color_boxes(img_bgr: np.ndarray, color: str, min_area: int = 120) -> List[Tuple[int, int, int, int]]:
    """
    Detects rectangles drawn in BLUE (ships) or RED (debris).
    Uses HSV masking and morphological ops to solidify outlines.
    Returns a list of (x1, y1, x2, y2) bounding boxes.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if color == "blue":
        lower, upper = (90, 40, 40), (135, 255, 255)
        mask = cv2.inRange(hsv, lower, upper)
    else:
        lower1, upper1 = (0, 70, 70), (12, 255, 255)
        lower2, upper2 = (170, 70, 70), (180, 255, 255)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append((x, y, x + w, y + h))
    return boxes


def _centers(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    """Return centers (cx, cy) for list of boxes."""
    return [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in boxes]


def detect_distance(
    ship_pil: Image.Image,
    debris_pil: Image.Image,
    ship_model_path: Optional[str] = None,
    debris_model_path: Optional[str] = None,
) -> Tuple[float, Image.Image]:
    """
    Runs ship + debris detectors, computes pairwise distances,
    and returns (min_distance_in_pixels, annotated_PIL_image).
    """

    try:
        # --- Step 1: Run inference on both images ---
        ship_buf = run_ship(
            image_bytes=_pil_to_bytes(ship_pil),
            model_path=ship_model_path,
        )
        ship_det_pil = Image.open(io.BytesIO(ship_buf)).convert("RGB")

        debris_det_pil = detect_image(debris_pil, model_path=debris_model_path)

        ship_cv = cv2.cvtColor(np.array(ship_det_pil), cv2.COLOR_RGB2BGR)
        debris_cv = cv2.cvtColor(np.array(debris_det_pil), cv2.COLOR_RGB2BGR)

        # --- Step 2: Match sizes ---
        sh_h, sh_w = ship_cv.shape[:2]
        db_h, db_w = debris_cv.shape[:2]
        debris_resized = cv2.resize(debris_cv, (sh_w, sh_h), interpolation=cv2.INTER_LINEAR)

        # --- Step 3: Detect colored rectangles ---
        ship_boxes = _detect_color_boxes(ship_cv, "blue")
        debris_boxes = _detect_color_boxes(debris_resized, "red")

        # if not ship_boxes or not debris_boxes:
        #     print("⚠️ Warning: No boxes detected — check model outputs or HSV thresholds.")

        # --- Step 4: Compute pairwise distances ---
        ship_centers = _centers(ship_boxes)
        debris_centers = _centers(debris_boxes)

        pairs = []
        for i, s in enumerate(ship_centers):
            for j, d in enumerate(debris_centers):
                pairs.append(((i, j), math.dist(s, d)))

        min_dist = float("inf")
        if pairs:
            _, min_dist = min(pairs, key=lambda x: x[1])

        # --- Step 5: Visualization ---
        combined = cv2.addWeighted(ship_cv, 0.6, debris_resized, 0.6, 0)

        # Draw ship boxes in yellow
        YELLOW = (0, 255, 255)
        for box in ship_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(combined, (x1, y1), (x2, y2), YELLOW, 3)

        # Draw all lines in gray, min distance in green
        GRAY = (180, 180, 180)
        min_pair = None
        if pairs:
            min_pair = min(pairs, key=lambda x: x[1])[0]
        for (i_j, dist) in pairs:
            i, j = i_j
            c1, c2 = ship_centers[i], debris_centers[j]
            color = GREEN if i_j == min_pair else GRAY
            cv2.line(combined, c1, c2, color, 2)
            mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
            if i_j == min_pair:
                cv2.putText(combined, f"{dist:.1f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2, cv2.LINE_AA)

        annotated = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        return min_dist, annotated

    except Exception as e:
        print(f"❌ distance.py error: {e}")
        return float("inf"), ship_pil.convert("RGB")


def _pil_to_bytes(pil_img: Image.Image) -> bytes:
    """Utility: convert PIL image to PNG bytes."""
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()
