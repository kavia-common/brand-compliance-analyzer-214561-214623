from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image

from src.services.detect_config import DetectionConfig, default_detection_config


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


def _nms_dynamic(detections: List[Detection], iou_threshold: float, min_dist: int) -> List[Detection]:
    """NMS with IoU and center distance. Thresholds provided by caller (can be tuned by template size)."""
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for d in dets:
        keep = True
        cx = d.x + d.w / 2.0
        cy = d.y + d.h / 2.0
        for k in kept:
            xx1 = max(d.x, k.x)
            yy1 = max(d.y, k.y)
            xx2 = min(d.x + d.w, k.x + k.w)
            yy2 = min(d.y + d.h, k.y + k.h)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            iou = inter / (d.w * d.h + k.w * k.h - inter + 1e-6)
            if iou > iou_threshold:
                keep = False
                break
            kcx = k.x + k.w / 2.0
            kcy = k.y + k.h / 2.0
            if (cx - kcx) ** 2 + (cy - kcy) ** 2 < (min_dist ** 2):
                keep = False
                break
        if keep:
            kept.append(d)
    return kept


def _apply_tophat(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply morphological white top-hat to suppress background shading and emphasize bright logo edges."""
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    return tophat


def _to_gray_preproc(img_bgr: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """
    Preprocess an image for robust template and feature matching:
    - convert to gray
    - histogram equalization (CLAHE) with tunable clip limit and tile size
    - optional morphological top-hat to suppress background variation
    - light gaussian blur to reduce noise
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip_limit),
                            tileGridSize=(int(cfg.clahe_tile_grid), int(cfg.clahe_tile_grid)))
    gray = clahe.apply(gray)

    if cfg.use_tophat:
        gray = _apply_tophat(gray, cfg.tophat_kernel)

    k = max(1, int(cfg.gaussian_blur))
    if k % 2 == 0:
        k += 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def _normalize_template(template_bgr: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """
    Normalize a template image using grayscale + CLAHE + (optional) tophat per config.
    """
    g = _to_gray_preproc(template_bgr, cfg)
    return g


def _estimate_scale_prior(page_shape: Tuple[int, int], template_shapes: List[Tuple[int, int]]) -> float:
    """
    Estimate a scale prior from page and template sizes.
    Returns a multiplier that nudges scales toward expected size on page.
    Simple heuristic: favor scales that keep template width 2%-25% of page width.
    """
    H, W = page_shape
    if W <= 0:
        return 1.0
    target_min = 0.02 * W
    target_max = 0.25 * W
    # Use average template width for heuristic
    if not template_shapes:
        return 1.0
    avg_tw = float(np.mean([tw for (_, tw) in template_shapes]))
    if avg_tw <= 0:
        return 1.0
    # If template appears too small/large, nudge
    if avg_tw < target_min:
        return min(1.6, max(1.1, target_min / max(1.0, avg_tw)))
    if avg_tw > target_max:
        return max(0.4, min(0.9, target_max / avg_tw))
    return 1.0


def multi_scale_template_match(
    image_bgr: np.ndarray,
    templates_bgr: List[np.ndarray],
    match_threshold: float = 0.8,
    scales: Optional[List[float]] = None,
    method: int = cv2.TM_CCOEFF_NORMED,
    cfg: Optional[DetectionConfig] = None,
    page_diag: Optional[Dict[str, Any]] = None,
) -> List[Detection]:
    """
    Run multi-scale template matching for multiple templates with robust preprocessing.
    - Grayscale + CLAHE (tunable) + optional top-hat
    - Small-angle rotation search (-15..+15 deg) on templates
    - Scale prior: adjust scales list with a multiplicative nudge derived from template/page sizes
    - Use normalized cross-correlation (TM_CCOEFF_NORMED) on grayscale
    Returns a list of Detection in pixel coordinates of image_bgr.
    """
    cfg = cfg or default_detection_config()
    cfg.ensure_defaults()

    page_gray = _to_gray_preproc(image_bgr, cfg)
    H, W = page_gray.shape[:2]
    detections: List[Detection] = []

    # Prepare template shapes and scale prior
    tpl_shapes = []
    for tpl_bgr in templates_bgr:
        g = _normalize_template(tpl_bgr, cfg)
        tpl_shapes.append(g.shape[:2])

    nudge = _estimate_scale_prior(page_gray.shape[:2], tpl_shapes)
    base_scales = scales if scales is not None else cfg.scales
    scales_eff = [max(0.2, min(3.0, s * nudge)) for s in base_scales]

    # Diagnostics container
    if page_diag is not None:
        page_diag.setdefault("scale_prior", nudge)
        page_diag.setdefault("used_scales", scales_eff)
        page_diag.setdefault("rotations", cfg.rotation_degrees)
        page_diag.setdefault("reasons", [])

    for t_idx, tpl_bgr in enumerate(templates_bgr):
        tpl_gray_base = _normalize_template(tpl_bgr, cfg)
        th0, tw0 = tpl_gray_base.shape[:2]
        if th0 < 5 or tw0 < 5:
            if page_diag is not None:
                page_diag["reasons"].append(f"template_{t_idx}_too_small")
            continue

        # Rotation search
        for angle in cfg.rotation_degrees:
            # Rotate template around its center
            center = (tw0 / 2.0, th0 / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            nW = int((th0 * sin) + (tw0 * cos))
            nH = int((th0 * cos) + (tw0 * sin))
            M[0, 2] += (nW / 2) - center[0]
            M[1, 2] += (nH / 2) - center[1]
            tpl_gray = cv2.warpAffine(tpl_gray_base, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            th, tw = tpl_gray.shape[:2]
            if th < 5 or tw < 5:
                if page_diag is not None:
                    page_diag["reasons"].append(f"template_{t_idx}_rotated_too_small")
                continue

            for s in scales_eff:
                new_w = max(5, int(tw * s))
                new_h = max(5, int(th * s))
                if new_w > W or new_h > H:
                    continue
                tpl_resized = cv2.resize(tpl_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

                res = cv2.matchTemplate(page_gray, tpl_resized, method)

                # Adaptive threshold: relax slightly for tiny templates
                adaptive_thresh = match_threshold
                if min(new_w, new_h) < cfg.tiny_template_min_side:
                    adaptive_thresh = max(cfg.tiny_threshold_floor, match_threshold - cfg.tiny_template_relax)

                ys, xs = np.where(res >= adaptive_thresh)
                for (y, x) in zip(ys, xs):
                    score = float(res[y, x])
                    detections.append(Detection(x=x, y=y, w=new_w, h=new_h, score=score, template_id=t_idx))

    # Keep top-k before NMS
    detections = sorted(detections, key=lambda d: d.score, reverse=True)[: cfg.pre_nms_topk]

    # Size-aware NMS tuning: enlarge min_dist for small templates to avoid duplicates
    if detections:
        median_w = int(np.median([d.w for d in detections]))
        min_dist = max(cfg.nms_min_dist, int(max(6, median_w * 0.15)))
    else:
        min_dist = cfg.nms_min_dist
    pruned = _nms_dynamic(detections, iou_threshold=cfg.nms_iou, min_dist=min_dist)

    if page_diag is not None and not pruned:
        # Provide likely reasons if nothing survived
        min_side = min(H, W)
        if min_side < 400:
            page_diag["reasons"].append("page_small_maybe_increase_dpi")
        if cfg.use_tophat and cfg.tophat_kernel < 7:
            page_diag["reasons"].append("low_contrast_try_higher_clahe_or_tophat_kernel")
        page_diag["reasons"].append("no_template_match_above_threshold")

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
    cfg: Optional[DetectionConfig] = None,
    page_diag: Optional[Dict[str, Any]] = None,
) -> List[Detection]:
    """
    Fallback detection using feature matching with RANSAC homography.
    - Supports small rotations (-~12..+~12 deg by default).
    - Returns estimated bounding boxes for located templates.
    """
    cfg = cfg or default_detection_config()
    cfg.ensure_defaults()

    if rotations_deg is None:
        rotations_deg = cfg.feature_rotations

    gray_page = _to_gray_preproc(image_bgr, cfg)
    H, W = gray_page.shape[:2]

    det = _feature_detector()
    if det is None:
        if page_diag is not None:
            page_diag.setdefault("reasons", []).append("feature_detector_unavailable")
        return []

    detections: List[Detection] = []
    kp2, des2 = det.detectAndCompute(gray_page, None)
    if des2 is None or len(kp2) < 2:
        if page_diag is not None:
            page_diag.setdefault("reasons", []).append("insufficient_keypoints_page")
        return []

    for t_idx, tpl_bgr in enumerate(templates_bgr):
        tpl_gray = _normalize_template(tpl_bgr, cfg)
        th, tw = tpl_gray.shape[:2]
        if th < 5 or tw < 5:
            if page_diag is not None:
                page_diag.setdefault("reasons", []).append(f"template_{t_idx}_too_small_for_feature_match")
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
                        score = float(np.clip(mask.mean(), 0.0, 1.0))
                        detections.append(Detection(x=x0, y=y0, w=w_box, h=h_box, score=score, template_id=t_idx))

    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    detections = _nms_dynamic(detections, iou_threshold=0.45, min_dist=8)
    if page_diag is not None and not detections:
        page_diag.setdefault("reasons", []).append("feature_match_found_none")
    return detections


def draw_boxes(img_bgr: np.ndarray, boxes: List[Detection], color=(0, 0, 255), thickness: int = 2) -> np.ndarray:
    """Draw detection boxes on a copy of the image."""
    out = img_bgr.copy()
    for d in boxes:
        cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), color, thickness)
    return out
