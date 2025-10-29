from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from PIL import Image
import numpy as np

# Optional import handling for OpenCV; we guard usages where necessary.
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 might not be installed in test env
    cv2 = None  # type: ignore


@dataclass
class Detection:
    """Represents one detected old-logo location in an image."""
    x: float
    y: float
    width: float
    height: float
    score: float
    method: str  # 'template' or 'orb'
    points: Optional[List[Tuple[float, float]]] = None  # corners for perspective

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.points is None:
            d["points"] = []
        return d


class DetectorConfig:
    """Lightweight config pulled from env with defaults.

    DETECTOR env:
      - template: multi-scale normalized cross correlation
      - orb: feature-based matching robust to rotation/scale
      - hybrid: both (default)

    QUALITY env adjusts scale range and feature thresholds:
      - fast: fewer scales, lower nfeatures
      - balanced: sensible defaults
      - best: more scales and features for robustness
    """
    def __init__(self) -> None:
        # DETECTOR: template|orb|hybrid
        self.detector = os.getenv("DETECTOR", "hybrid").lower()
        # QUALITY: fast|balanced|best => controls scale steps and feature thresholds
        self.quality = os.getenv("QUALITY", "balanced").lower()

        # Tunables
        if self.quality == "fast":
            self.scales = [1.0, 0.75]  # fewer scales
            self.min_match_count = 8
            self.orb_nfeatures = 300
        elif self.quality == "best":
            # wider range for robustness as per acceptance: search scales 0.5–1.5 (subset covered via shape limits)
            self.scales = [1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
            self.min_match_count = 18
            self.orb_nfeatures = 1500
        else:  # balanced
            self.scales = [1.2, 1.0, 0.9, 0.8, 0.7]
            self.min_match_count = 12
            self.orb_nfeatures = 1000


class VisionUtils:
    """Static helpers for detecting and replacing logos using OpenCV + Pillow."""

    @staticmethod
    def _imread(path: Path) -> Optional[np.ndarray]:
        """Read image as BGR using OpenCV if available, else Pillow -> numpy."""
        if not path.exists():
            return None
        if cv2 is not None:
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            return img
        # Fallback to Pillow
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                arr = np.array(im)  # RGB
                # convert to BGR to keep further processing consistent
                return arr[:, :, ::-1].copy()
        except Exception:
            return None

    @staticmethod
    def _ensure_cv2() -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for detection and is not available")

    # PUBLIC_INTERFACE
    @staticmethod
    def detect_old_logo(image_path: Path, template_path: Path, config: Optional[DetectorConfig] = None) -> List[Detection]:
        """Detect instances of the old logo in an image.

        Uses:
          - Multi-scale normalized template matching (scale-invariant within configured scales)
          - ORB feature matching with homography (rotation and perspective tolerant)

        Tunables via env:
          - DETECTOR=template|orb|hybrid (default hybrid)
          - QUALITY=fast|balanced|best (affects scale list, ORB nfeatures, match thresholds)

        Returns list of Detection entries with bounding boxes, method, score, and optional corner points.
        """
        VisionUtils._ensure_cv2()
        cfg = config or DetectorConfig()

        img = VisionUtils._imread(image_path)
        templ = VisionUtils._imread(template_path)
        if img is None or templ is None:
            return []

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

        # Normalize lighting/contrast slightly to improve robustness
        img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
        templ_gray = cv2.normalize(templ_gray, None, 0, 255, cv2.NORM_MINMAX)

        detections: List[Detection] = []
        log = logging.getLogger("vision.detect")
        log.info("detect:start image=%s template=%s detector=%s quality=%s", image_path, template_path, cfg.detector, cfg.quality)

        def _template_multi_scale() -> List[Detection]:
            dets: List[Detection] = []
            # Support scale range 0.5–2.0 under best quality to increase robustness
            scales = cfg.scales
            if cfg.quality == "best":
                extra = [1.8, 1.6, 1.4, 1.3, 1.1, 0.95, 0.85, 0.65, 0.55]
                # ensure uniqueness while preserving order
                seen = set()
                merged = []
                for s in list(scales) + extra:
                    if s not in seen:
                        seen.add(s)
                        merged.append(s)
                scales = merged
            for scale in scales:
                # Resize template for this scale
                if abs(scale - 1.0) < 1e-3:
                    t_scaled = templ_gray
                else:
                    new_w = max(8, int(templ_gray.shape[1] * scale))
                    new_h = max(8, int(templ_gray.shape[0] * scale))
                    t_scaled = cv2.resize(templ_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if t_scaled.shape[0] >= img_gray.shape[0] or t_scaled.shape[1] >= img_gray.shape[1]:
                    continue
                # Try multiple rotations for tolerance (e.g., -15, 0, +15)
                for angle in (-15, 0, 15):
                    if angle != 0:
                        center = (t_scaled.shape[1] // 2, t_scaled.shape[0] // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        t_rot = cv2.warpAffine(t_scaled, M, (t_scaled.shape[1], t_scaled.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    else:
                        t_rot = t_scaled
                    if t_rot.shape[0] >= img_gray.shape[0] or t_rot.shape[1] >= img_gray.shape[1]:
                        continue
                    res = cv2.matchTemplate(img_gray, t_rot, cv2.TM_CCOEFF_NORMED)
                    # Threshold selection based on quality
                    if cfg.quality == "best":
                        thr = 0.68
                    elif cfg.quality == "fast":
                        thr = 0.82
                    else:
                        thr = 0.75
                    loc = np.where(res >= thr)
                    w, h = t_rot.shape[1], t_rot.shape[0]
                    # Collect raw matches
                    for pt in zip(*loc[::-1]):
                        score = float(res[pt[1], pt[0]])
                        dets.append(Detection(
                            x=float(pt[0]), y=float(pt[1]),
                            width=float(w), height=float(h),
                            score=score, method="template", points=None
                        ))
            log.info("detect:template_candidates=%d", len(dets))
            # Non-maximum suppression (greedy)
            dets_sorted = sorted(dets, key=lambda d: d.score, reverse=True)
            kept: List[Detection] = []
            def iou(a: Detection, b: Detection) -> float:
                ax1, ay1 = a.x, a.y
                ax2, ay2 = a.x + a.width, a.y + a.height
                bx1, by1 = b.x, b.y
                bx2, by2 = b.x + b.width, b.y + b.height
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                a_area = (ax2 - ax1) * (ay2 - ay1)
                b_area = (bx2 - bx1) * (by2 - by1)
                union = a_area + b_area - inter_area + 1e-6
                return inter_area / union
            for d in dets_sorted:
                if all(iou(d, k) < 0.3 for k in kept):
                    kept.append(d)
            return kept

        def _orb_feature_match() -> List[Detection]:
            dets: List[Detection] = []
            try:
                orb = cv2.ORB_create(nfeatures=cfg.orb_nfeatures)
                kp1, des1 = orb.detectAndCompute(templ_gray, None)
                kp2, des2 = orb.detectAndCompute(img_gray, None)
                if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
                    log.info("detect:orb_no_descriptors")
                    return dets
                # BFMatcher with Hamming for ORB
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)
                good = []
                # Lowe's ratio test
                for m, n in matches:
                    if m.distance < 0.74 * n.distance:
                        good.append(m)
                log.info("detect:orb_matches total=%d good=%d", len(matches), len(good))
                if len(good) >= cfg.min_match_count:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        h, w = templ_gray.shape
                        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
                        xs = dst[:, 0]
                        ys = dst[:, 1]
                        x_min, y_min = float(xs.min()), float(ys.min())
                        x_max, y_max = float(xs.max()), float(ys.max())
                        score = min(1.0, float(len(good)) / float(cfg.min_match_count + 1))
                        dets.append(Detection(
                            x=x_min,
                            y=y_min,
                            width=(x_max - x_min),
                            height=(y_max - y_min),
                            score=score,
                            method="orb",
                            points=[(float(x), float(y)) for x, y in dst.tolist()],
                        ))
                # As a fallback, also try FLANN if available for ORB (LSH index)
                try:
                    FLANN_INDEX_LSH = 6
                    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
                    search_params = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2)
                    good_flann = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good_flann.append(m)
                    log.info("detect:flann_matches total=%d good=%d", len(matches), len(good_flann))
                    if len(good_flann) >= cfg.min_match_count and len(good_flann) > len(good):
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_flann]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_flann]).reshape(-1, 1, 2)
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if M is not None:
                            h, w = templ_gray.shape
                            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
                            xs = dst[:, 0]
                            ys = dst[:, 1]
                            x_min, y_min = float(xs.min()), float(ys.min())
                            x_max, y_max = float(xs.max()), float(ys.max())
                            score = min(1.0, float(len(good_flann)) / float(cfg.min_match_count + 1))
                            dets.append(Detection(
                                x=x_min,
                                y=y_min,
                                width=(x_max - x_min),
                                height=(y_max - y_min),
                                score=score,
                                method="orb",
                                points=[(float(x), float(y)) for x, y in dst.tolist()],
                            ))
                except Exception:
                    pass
            except Exception:
                # If cv2 ORB pipeline fails, just return empty
                return dets
            return dets

        if cfg.detector in ("template", "hybrid"):
            detections.extend(_template_multi_scale())
        if cfg.detector in ("orb", "hybrid"):
            detections.extend(_orb_feature_match())

        # Keep top-N (avoid too many boxes)
        detections = sorted(detections, key=lambda d: d.score, reverse=True)
        return detections[:10]

    # PUBLIC_INTERFACE
    @staticmethod
    def save_detections_json(detections: Dict[str, List[Detection]], out_path: Path) -> None:
        """Persist detections to JSON file: { image_rel_path: [Detection...] }."""
        data = {k: [d.to_json() for d in v] for k, v in detections.items()}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _pil_read(path: Path) -> Image.Image:
        with Image.open(path) as im:
            return im.convert("RGBA")

    @staticmethod
    def _pil_save_keep_format(image: Image.Image, dst: Path, quality_mode: str = "balanced") -> None:
        """Save keeping original extension semantics. Use sensible quality defaults."""
        fmt = dst.suffix.lower()
        params: Dict[str, Any] = {}
        if fmt in (".jpg", ".jpeg"):
            # Adjust quality based on QUALITY
            quality = 85 if quality_mode == "balanced" else (75 if quality_mode == "fast" else 92)
            image = image.convert("RGB")
            params.update({"quality": quality, "optimize": True, "progressive": True})
            image.save(dst, format="JPEG", **params)
        elif fmt == ".png":
            # Pillow uses 0 (no compress) - 9 (max)
            compress = 6 if quality_mode == "balanced" else (3 if quality_mode == "fast" else 7)
            params.update({"compress_level": compress})
            image.save(dst, format="PNG", **params)
        elif fmt == ".webp":
            quality = 80 if quality_mode == "balanced" else (70 if quality_mode == "fast" else 90)
            image.save(dst, format="WEBP", quality=quality, method=4)
        else:
            # default fall-back
            image.save(dst)

    @staticmethod
    def _build_padded_patch(logo_rgba: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Create a white RGBA patch of (target_w,target_h) and center the logo preserving aspect ratio."""
        target_w = max(1, int(round(target_w)))
        target_h = max(1, int(round(target_h)))
        patch = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
        lw, lh = logo_rgba.size
        # compute scale to fit inside target while preserving aspect ratio
        scale = min(target_w / lw, target_h / lh) if lw > 0 and lh > 0 else 1.0
        new_w = max(1, int(round(lw * scale)))
        new_h = max(1, int(round(lh * scale)))
        resized = logo_rgba.resize((new_w, new_h), Image.LANCZOS)
        # center the resized logo
        left = (target_w - new_w) // 2
        top = (target_h - new_h) // 2
        patch.alpha_composite(resized, dest=(left, top))
        return patch

    # PUBLIC_INTERFACE
    @staticmethod
    def replace_logo(
        image_path: Path,
        new_logo_png_path: Path,
        detections: List[Detection],
        out_path: Path,
        feather: int = 6,
        quality_mode: str = "balanced",
    ) -> Path:
        """Apply new logo on top of detected regions with aspect-ratio preservation and white padding.

        Behavior:
          - Detections below ~0.75 confidence are skipped.
          - For axis-aligned boxes: create a white patch the size of bbox, center the logo keeping aspect ratio.
          - For rotated/perspective detections: build a white patch for the bounding rect then warp it to the quad.
          - White padding fills any empty space to completely cover the detected bbox.
        """
        # Filter by confidence threshold ~0.75
        conf_thr = float(os.getenv("DETECTION_CONFIDENCE_THRESHOLD", "0.75"))
        good_dets = [d for d in (detections or []) if (d.score is None or d.score >= conf_thr)]
        if not good_dets:
            # If nothing to replace, copy or save original
            base = VisionUtils._pil_read(image_path)
            VisionUtils._pil_save_keep_format(base, out_path, quality_mode)
            return out_path

        base_im = VisionUtils._pil_read(image_path)  # RGBA
        overlay_logo = VisionUtils._pil_read(new_logo_png_path)  # RGBA

        # Working in numpy for geometric warp; convert to OpenCV space when needed
        base_bgra = cv2.cvtColor(np.array(base_im), cv2.COLOR_RGBA2BGRA) if cv2 is not None else np.array(base_im)

        for det in good_dets:
            try:
                # If we have 4-point quad and cv2, use perspective warp of a prepared white patch
                if det.points and len(det.points) == 4 and cv2 is not None:
                    dst_pts = np.float32(det.points)
                    # bounding rect dimensions for patch
                    xs = dst_pts[:, 0]
                    ys = dst_pts[:, 1]
                    x_min, y_min = float(xs.min()), float(ys.min())
                    x_max, y_max = float(xs.max()), float(ys.max())
                    rect_w = max(1, int(round(x_max - x_min)))
                    rect_h = max(1, int(round(y_max - y_min)))
                    # Build padded patch with white background
                    patch = VisionUtils._build_padded_patch(overlay_logo, rect_w, rect_h)
                    patch_bgra = cv2.cvtColor(np.array(patch), cv2.COLOR_RGBA2BGRA)

                    src_pts = np.float32([[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]])
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    warped = cv2.warpPerspective(patch_bgra, M, (base_bgra.shape[1], base_bgra.shape[0]))
                    # Alpha blend with optional feather
                    alpha = warped[:, :, 3] / 255.0
                    if feather and feather > 0:
                        k = feather | 1
                        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
                    for c in range(3):
                        base_bgra[:, :, c] = (1.0 - alpha) * base_bgra[:, :, c] + alpha * warped[:, :, c]
                    base_bgra[:, :, 3] = 255
                else:
                    # Axis-aligned bbox path (works without cv2 as well)
                    x1 = int(max(0, round(det.x)))
                    y1 = int(max(0, round(det.y)))
                    bw = int(round(det.width))
                    bh = int(round(det.height))
                    if bw <= 0 or bh <= 0:
                        continue
                    # Patch with white padding and aspect preservation
                    patch = VisionUtils._build_padded_patch(overlay_logo, bw, bh)
                    if cv2 is not None:
                        patch_bgra = cv2.cvtColor(np.array(patch), cv2.COLOR_RGBA2BGRA)
                        x2 = min(base_bgra.shape[1], x1 + bw)
                        y2 = min(base_bgra.shape[0], y1 + bh)
                        # Adjust patch to clipped region if bbox extends outside image
                        pw = x2 - x1
                        ph = y2 - y1
                        if pw <= 0 or ph <= 0:
                            continue
                        patch_roi = patch_bgra[0:ph, 0:pw, :]
                        alpha = patch_roi[:, :, 3] / 255.0
                        if feather and feather > 0:
                            k = feather | 1
                            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
                        for c in range(3):
                            base_bgra[y1:y2, x1:x2, c] = (1.0 - alpha) * base_bgra[y1:y2, x1:x2, c] + alpha * patch_roi[:, :, c]
                        base_bgra[y1:y2, x1:x2, 3] = 255
                    else:
                        # Pillow fallback
                        base_im.alpha_composite(patch, dest=(x1, y1))
            except Exception as e:
                logging.getLogger("vision.replace").warning("Failed to apply replacement: %s", e)

        # Save back using Pillow to preserve original format
        if cv2 is not None:
            out_rgba = cv2.cvtColor(base_bgra, cv2.COLOR_BGRA2RGBA)
            pil_out = Image.fromarray(out_rgba)
        else:
            pil_out = base_im

        # Ensure parent exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        VisionUtils._pil_save_keep_format(pil_out, out_path, quality_mode)
        return out_path
