from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

logger = logging.getLogger("json_utils")
if not logger.handlers:
    # Minimal console handler; in production, app-level logging config may override this.
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[json_utils] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# PUBLIC_INTERFACE
def to_native_jsonable(value: Any, _depth: int = 0, _max_depth: int = 50) -> Any:
    """Recursively convert a Python object graph to built-in JSON-serializable types.

    This function:
    - Converts numpy scalar types (np.integer, np.floating, np.bool_) to int/float/bool.
    - Converts numpy arrays to lists via .tolist().
    - Converts sets and tuples to lists.
    - Converts dataclasses and pydantic models to dicts and recurses.
    - Leaves bytes as base64-encoded str? For safety we leave bytes unchanged here since responses
      shouldn't include raw bytes; metadata should avoid bytes entirely. If encountered, str() cast.
    - Recurses into dicts and lists.

    Args:
        value: Any input value (possibly containing numpy or model instances)
        _depth: Internal recursion depth guard
        _max_depth: Maximum recursion depth to prevent runaway recursion

    Returns:
        A value consisting only of native Python types (dict, list, str, int, float, bool, None).
    """
    if _depth > _max_depth:
        logger.warning("to_native_jsonable: maximum recursion depth exceeded; returning string repr")
        try:
            return str(value)
        except Exception:
            return None

    # Fast path for already-native types
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Numpy handling
    if np is not None:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.ndarray,)):
            try:
                return value.tolist()
            except Exception:
                # Fallback: iterate
                return [to_native_jsonable(v, _depth=_depth + 1, _max_depth=_max_depth) for v in value]

    # Pydantic BaseModel support without importing pydantic directly
    # We duck-type by presence of model_dump or dict attributes
    try:
        if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
            try:
                value = value.model_dump()
            except Exception:
                value = dict(value)  # type: ignore
        elif hasattr(value, "dict") and callable(getattr(value, "dict")):
            try:
                value = value.dict()
            except Exception:
                value = dict(value)  # type: ignore
    except Exception:
        pass

    # Dataclass handling
    if is_dataclass(value):
        try:
            value = asdict(value)
        except Exception:
            # Fallback turning to string
            return str(value)

    # Mapping (dict-like)
    if isinstance(value, dict):
        return {
            to_native_jsonable(k, _depth=_depth + 1, _max_depth=_max_depth)
            if not isinstance(k, (str, int, float, bool))
            else k: to_native_jsonable(v, _depth=_depth + 1, _max_depth=_max_depth)
            for k, v in value.items()
        }

    # Sequence types
    if isinstance(value, (list, tuple, set)):
        return [to_native_jsonable(v, _depth=_depth + 1, _max_depth=_max_depth) for v in value]

    # Bytes: avoid in metadata; convert to string as last resort
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return value.decode("utf-8", errors="replace")  # best-effort
        except Exception:
            return str(value)

    # Objects with __dict__
    if hasattr(value, "__dict__"):
        try:
            return to_native_jsonable(vars(value), _depth=_depth + 1, _max_depth=_max_depth)
        except Exception:
            pass

    # Fallback: string representation
    try:
        return str(value)
    except Exception:
        return None


# PUBLIC_INTERFACE
def normalize_for_json_inplace(obj: Any) -> Any:
    """Convenience wrapper that logs a short message and returns normalized object.

    This is primarily intended for use before saving metadata or returning API responses.
    """
    out = to_native_jsonable(obj)
    logger.info("Applied JSON normalization.")
    return out
