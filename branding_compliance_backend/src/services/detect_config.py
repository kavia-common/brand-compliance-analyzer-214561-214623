from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class DetectionConfig:
    """Centralized configuration for detection pipeline parameters.

    All parameters have sane defaults and can be overridden per job if needed.
    """
    # PDF rasterization
    dpi_min: int = 200
    dpi_max: int = 600

    # Template matching
    match_threshold: float = 0.8
    # Base scales; scale prior will adjust these per-document dynamically
    scales: List[float] = None  # type: ignore
    # Rotation search for template matching (small-angle)
    rotation_degrees: List[int] = None  # type: ignore

    # CLAHE tuning (for low-contrast pages)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    gaussian_blur: int = 3  # kernel size

    # Background suppression (morphological top-hat)
    use_tophat: bool = True
    tophat_kernel: int = 9  # odd size

    # Fallback feature-based matching
    feature_rotations: List[int] = None  # type: ignore
    feature_min_inliers: int = 8
    feature_ransac_reproj_thresh: float = 3.0

    # NMS parameters (tuned by template size at runtime)
    nms_iou: float = 0.4
    nms_min_dist: int = 6

    # Adaptive threshold relaxation for tiny templates
    tiny_template_relax: float = 0.12
    tiny_template_min_side: int = 28
    tiny_threshold_floor: float = 0.58

    # Limit candidate matches before NMS for performance
    pre_nms_topk: int = 8000

    # Color-invariant: always convert to grayscale and NCC (TM_CCOEFF_NORMED is NCC-like after normalization)
    use_grayscale_ncc: bool = True

    # Enable per-page diagnostics and failure reasons
    collect_page_diagnostics: bool = True

    def ensure_defaults(self) -> None:
        if self.scales is None:
            self.scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.45, 1.6]
        if self.rotation_degrees is None:
            # small-angle rotation search for template matching
            self.rotation_degrees = list(range(-15, 16, 3))
        if self.feature_rotations is None:
            self.feature_rotations = [0, -12, -9, -6, -3, 3, 6, 9, 12]

    # PUBLIC_INTERFACE
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for metadata and API responses."""
        self.ensure_defaults()
        return asdict(self)


# PUBLIC_INTERFACE
def default_detection_config() -> DetectionConfig:
    """Return a DetectionConfig with sane defaults."""
    cfg = DetectionConfig()
    cfg.ensure_defaults()
    return cfg
