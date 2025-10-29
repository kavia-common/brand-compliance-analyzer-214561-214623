from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class Detection:
    """Simple detection box in image pixel coordinates."""
    x: int
    y: int
    w: int
    h: int
    score: float
    template_id: int


def pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR numpy array."""
    arr = np.array(img.convert("RGBA"))
    # Handle alpha on white
    if arr.shape[2] == 4:
        # premultiply alpha on white
        alpha = arr[:, :, 3:4] / 255.0
        rgb = arr[:, :, :3]
        white = np.ones_like(rgb) * 255
        rgb = rgb * alpha + white * (1 - alpha)
        arr = rgb.astype(np.uint8)
    else:
        arr = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def cv_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    """Encode OpenCV BGR image to PNG bytes."""
    ok, buff = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG.")
    return buff.tobytes()


def clamp_rect(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """Clamp rectangle to image boundaries."""
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


def nms(detections: List[Detection], iou_threshold: float = 0.3) -> List[Detection]:
    """Standard NMS on detections using IoU."""
    if not detections:
        return []

    scores = np.array([d.score for d in detections], dtype=np.float32)
    idxs = cv2.dnn.NMSBoxes(
        bboxes=[(int(d.x), int(d.y), int(d.w), int(d.h)) for d in detections],
        scores=scores.tolist(),
        score_threshold=0.0,  # already filtered elsewhere
        nms_threshold=iou_threshold,
    )
    if len(idxs) == 0:
        return []
    keep = []
    idxs = [int(i) for i in np.array(idxs).reshape(-1)]
    for i in idxs:
        keep.append(detections[i])
    return keep


def multi_scale_template_match(
    image_bgr: np.ndarray,
    templates_bgr: List[np.ndarray],
    match_threshold: float = 0.8,
    scales: List[float] | None = None,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> List[Detection]:
    """
    Run multi-scale template matching for multiple templates.
    Returns a list of Detection in pixel coordinates of image_bgr.
    """
    if scales is None:
        scales = [1.0, 0.9, 1.1, 0.8, 1.2]

    H, W = image_bgr.shape[:2]
    detections: List[Detection] = []

    for t_idx, tpl in enumerate(templates_bgr):
        th, tw = tpl.shape[:2]
        for s in scales:
            new_w = max(5, int(tw * s))
            new_h = max(5, int(th * s))
            tpl_resized = cv2.resize(tpl, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if new_w > W or new_h > H:
                continue

            res = cv2.matchTemplate(image_bgr, tpl_resized, method)
            # Locations above threshold
            loc = np.where(res >= match_threshold)
            for (y, x) in zip(*loc):
                score = float(res[y, x])
                detections.append(Detection(x=x, y=y, w=new_w, h=new_h, score=score, template_id=t_idx))

    # Apply NMS to prune duplicates
    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    # cull to reasonable size pre-nms
    detections = detections[:5000]
    pruned = nms(detections, iou_threshold=0.3)
    return pruned


def draw_boxes(img_bgr: np.ndarray, boxes: List[Detection], color=(0, 0, 255), thickness: int = 2) -> np.ndarray:
    """Draw detection boxes on a copy of the image."""
    out = img_bgr.copy()
    for d in boxes:
        cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), color, thickness)
    return out
