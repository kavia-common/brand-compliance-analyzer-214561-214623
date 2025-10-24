from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from PIL import Image, ImageFilter
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
    """Lightweight config pulled from env with defaults."""
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
            self.scales = [1.2, 1.0, 0.9, 0.8, 0.7, 0.6]
            self.min_match_count = 18
            self.orb_nfeatures = 1500
        else:  # balanced
            self.scales = [1.0, 0.9, 0.8, 0.7]
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
        """Detect instances of the old logo in an image using template matching and/or ORB.

        Returns list of Detection entries with bounding boxes and scores.
        """
        VisionUtils._ensure_cv2()
        cfg = config or DetectorConfig()

        img = VisionUtils._imread(image_path)
        templ = VisionUtils._imread(template_path)
        if img is None or templ is None:
            return []

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

        detections: List[Detection] = []

        def _template_multi_scale() -> List[Detection]:
            dets: List[Detection] = []
            for scale in cfg.scales:
                # Resize template for this scale
                if abs(scale - 1.0) < 1e-3:
                    t_scaled = templ_gray
                else:
                    new_w = max(8, int(templ_gray.shape[1] * scale))
                    new_h = max(8, int(templ_gray.shape[0] * scale))
                    t_scaled = cv2.resize(templ_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if t_scaled.shape[0] >= img_gray.shape[0] or t_scaled.shape[1] >= img_gray.shape[1]:
                    continue
                res = cv2.matchTemplate(img_gray, t_scaled, cv2.TM_CCOEFF_NORMED)
                # Threshold selection based on quality
                if cfg.quality == "best":
                    thr = 0.7
                elif cfg.quality == "fast":
                    thr = 0.8
                else:
                    thr = 0.75
                loc = np.where(res >= thr)
                w, h = t_scaled.shape[1], t_scaled.shape[0]
                # Collect raw matches
                for pt in zip(*loc[::-1]):
                    score = float(res[pt[1], pt[0]])
                    dets.append(Detection(
                        x=float(pt[0]), y=float(pt[1]),
                        width=float(w), height=float(h),
                        score=score, method="template", points=None
                    ))
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
                    return dets
                # BFMatcher with Hamming for ORB
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des1, des2, k=2)
                good = []
                # Lowe's ratio test
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
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
        """Apply new logo on top of detected regions, with affine/perspective transformations and soft edges."""
        if not detections:
            # If nothing to replace, copy or save original
            # We use Pillow to load and re-save to keep consistent pipeline.
            base = VisionUtils._pil_read(image_path)
            VisionUtils._pil_save_keep_format(base, out_path, quality_mode)
            return out_path

        base_im = VisionUtils._pil_read(image_path)  # RGBA
        overlay_logo = VisionUtils._pil_read(new_logo_png_path)  # assumed RGBA with transparency

        # Working in numpy for geometric warp; convert to OpenCV space when needed
        base_bgra = cv2.cvtColor(np.array(base_im), cv2.COLOR_RGBA2BGRA) if cv2 is not None else np.array(base_im)

        for det in detections:
            try:
                # Determine destination quadrilateral
                if det.points and len(det.points) == 4 and cv2 is not None:
                    dst_pts = np.float32(det.points)
                else:
                    # Use axis-aligned box
                    x1, y1 = det.x, det.y
                    x2, y2 = det.x + det.width, det.y + det.height
                    dst_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

                # Source quad: full logo image
                h, w = overlay_logo.size[1], overlay_logo.size[0]
                src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

                if cv2 is not None:
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    logo_bgra = cv2.cvtColor(np.array(overlay_logo), cv2.COLOR_RGBA2BGRA)
                    warped = cv2.warpPerspective(logo_bgra, M, (base_bgra.shape[1], base_bgra.shape[0]))
                    # Feathered alpha blending
                    alpha = warped[:, :, 3] / 255.0
                    if feather > 0:
                        alpha = cv2.GaussianBlur(alpha, (feather | 1, feather | 1), 0)
                    for c in range(3):
                        base_bgra[:, :, c] = (1.0 - alpha) * base_bgra[:, :, c] + alpha * warped[:, :, c]
                    # update alpha to fully opaque in result
                    base_bgra[:, :, 3] = 255
                else:
                    # Pillow-only fallback: approximate using resize + paste
                    x1, y1 = int(dst_pts[:, 0].min()), int(dst_pts[:, 1].min())
                    x2, y2 = int(dst_pts[:, 0].max()), int(dst_pts[:, 1].max())
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        continue
                    resized = overlay_logo.resize((x2 - x1, y2 - y1), Image.LANCZOS)
                    if feather > 0:
                        # soften the edges of the alpha channel
                        r, g, b, a = resized.split()
                        a = a.filter(ImageFilter.GaussianBlur(radius=max(1, feather // 2)))
                        resized = Image.merge("RGBA", (r, g, b, a))
                    base_im.alpha_composite(resized, dest=(x1, y1))
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
