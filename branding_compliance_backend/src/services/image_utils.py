from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

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
    """Convert PIL Image to OpenCV BGR numpy array, normalizing alpha to white."""
    arr = np.array(img.convert("RGBA"))
    # Handle alpha on white
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3:4] / 255.0
        rgb = arr[:, :, :3].astype(np.float32)
        white = np.ones_like(rgb, dtype=np.float32) * 255.0
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


def nms(detections: List[Detection], iou_threshold: float = 0.35, min_dist: int = 6) -> List[Detection]:
    """
    Non-maximum suppression with tunable IoU and a minimum center distance to avoid
    over-suppression of nearby small matches.
    """
    if not detections:
        return []

    # Sort by score desc
    dets = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for d in dets:
        keep = True
        cx = d.x + d.w / 2.0
        cy = d.y + d.h / 2.0
        for k in kept:
            # IoU check
            xx1 = max(d.x, k.x)
            yy1 = max(d.y, k.y)
            xx2 = min(d.x + d.w, k.x + k.w)
            yy2 = min(d.y + d.h, k.y + k.h)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            iou = inter / (d.w * d.h + k.w * k.h - inter + 1e-6)
            if iou > iou_threshold:
                keep = False
                break
            # Min distance check
            kcx = k.x + k.w / 2.0
            kcy = k.y + k.h / 2.0
            if (cx - kcx) ** 2 + (cy - kcy) ** 2 < (min_dist ** 2):
                keep = False
                break
        if keep:
            kept.append(d)
    return kept


def _to_gray_preproc(img_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for robust template and feature matching:
    - convert to gray
    - histogram equalization (CLAHE) to normalize contrast
    - light gaussian blur to reduce noise
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def _normalize_template(template_bgr: np.ndarray) -> np.ndarray:
    """
    Normalize a template image:
    - convert RGBA to BGR over white
    - convert to grayscale with CLAHE
    """
    g = _to_gray_preproc(template_bgr)
    return g


def multi_scale_template_match(
    image_bgr: np.ndarray,
    templates_bgr: List[np.ndarray],
    match_threshold: float = 0.8,
    scales: Optional[List[float]] = None,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> List[Detection]:
    """
    Run multi-scale template matching for multiple templates with robust preprocessing.
    Returns a list of Detection in pixel coordinates of image_bgr.
    """
    if scales is None:
        # wider range of scales to handle different rendering sizes and DPI
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.45, 1.6]

    page_gray = _to_gray_preproc(image_bgr)
    H, W = page_gray.shape[:2]
    detections: List[Detection] = []

    for t_idx, tpl_bgr in enumerate(templates_bgr):
        tpl_gray = _normalize_template(tpl_bgr)
        th, tw = tpl_gray.shape[:2]
        if th < 5 or tw < 5:
            continue
        for s in scales:
            new_w = max(5, int(tw * s))
            new_h = max(5, int(th * s))
            if new_w > W or new_h > H:
                continue
            tpl_resized = cv2.resize(tpl_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

            res = cv2.matchTemplate(page_gray, tpl_resized, method)

            # Adaptive threshold: relax slightly for tiny templates
            adaptive_thresh = match_threshold
            if min(new_w, new_h) < 28:
                adaptive_thresh = max(0.58, match_threshold - 0.12)

            ys, xs = np.where(res >= adaptive_thresh)
            for (y, x) in zip(ys, xs):
                score = float(res[y, x])
                detections.append(Detection(x=x, y=y, w=new_w, h=new_h, score=score, template_id=t_idx))

    # Keep top-k before NMS
    detections = sorted(detections, key=lambda d: d.score, reverse=True)[:8000]
    pruned = nms(detections, iou_threshold=0.4, min_dist=6)
    return pruned


def _feature_detector() -> Optional[cv2.Feature2D]:
    """Create an ORB or AKAZE feature detector as available."""
    try:
        return cv2.AKAZE_create()
    except Exception:
        try:
            return cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, edgeThreshold=15)
        except Exception:
            return None


def feature_match_fallback(
    image_bgr: np.ndarray,
    templates_bgr: List[np.ndarray],
    rotations_deg: Optional[List[float]] = None,
    min_inliers: int = 8,
    ransac_reproj_thresh: float = 3.0,
) -> List[Detection]:
    """
    Fallback detection using feature matching with RANSAC homography.
    - Supports small rotations (-10..+10 deg by default).
    - Returns estimated bounding boxes for located templates.
    """
    if rotations_deg is None:
        rotations_deg = [0, -10, -7, -5, -3, 3, 5, 7, 10]

    gray_page = _to_gray_preproc(image_bgr)
    H, W = gray_page.shape[:2]

    det = _feature_detector()
    if det is None:
        return []

    detections: List[Detection] = []
    kp2, des2 = det.detectAndCompute(gray_page, None)
    if des2 is None or len(kp2) < 2:
        return []

    for t_idx, tpl_bgr in enumerate(templates_bgr):
        tpl_gray = _normalize_template(tpl_bgr)
        th, tw = tpl_gray.shape[:2]
        if th < 5 or tw < 5:
            continue

        for angle in rotations_deg:
            # Rotate template around its center
            center = (tw / 2.0, th / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            nW = int((th * sin) + (tw * cos))
            nH = int((th * cos) + (tw * sin))
            M[0, 2] += (nW / 2) - center[0]
            M[1, 2] += (nH / 2) - center[1]
            tpl_rot = cv2.warpAffine(tpl_gray, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            kp1, des1 = det.detectAndCompute(tpl_rot, None)
            if des1 is None or len(kp1) < 2:
                continue

            # Matcher
            try:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)
            except Exception:
                continue

            good = []
            for m_n in matches:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) >= min_inliers:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                Hm, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh)
                if Hm is not None and mask is not None and mask.sum() >= min_inliers:
                    # Map template corners
                    h2, w2 = tpl_rot.shape[:2]
                    corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
                    proj = cv2.perspectiveTransform(corners, Hm)
                    xs = proj[:, 0, 0]
                    ys = proj[:, 0, 1]
                    x0 = max(0, int(np.min(xs)))
                    y0 = max(0, int(np.min(ys)))
                    x1 = min(W - 1, int(np.max(xs)))
                    y1 = min(H - 1, int(np.max(ys)))
                    if x1 > x0 and y1 > y0:
                        w_box = x1 - x0
                        h_box = y1 - y0
                        # Confidence proxy: ratio of inliers to total good matches, clipped to [0,1]
                        score = float(np.clip(mask.mean(), 0.0, 1.0))
                        detections.append(Detection(x=x0, y=y0, w=w_box, h=h_box, score=score, template_id=t_idx))

    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    detections = nms(detections, iou_threshold=0.45, min_dist=8)
    return detections


def draw_boxes(img_bgr: np.ndarray, boxes: List[Detection], color=(0, 0, 255), thickness: int = 2) -> np.ndarray:
    """Draw detection boxes on a copy of the image."""
    out = img_bgr.copy()
    for d in boxes:
        cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), color, thickness)
    return out
