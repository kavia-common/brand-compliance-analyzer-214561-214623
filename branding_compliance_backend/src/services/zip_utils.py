from __future__ import annotations

import io
import os
import zipfile
from typing import Iterable, Tuple


# PUBLIC_INTERFACE
def create_zip_from_files(file_iter: Iterable[Tuple[str, str]]) -> bytes:
    """Create a ZIP archive from an iterable of (arcname, file_path) items and return its bytes.

    This reads input files from disk and writes them into an in-memory zip with the given arcnames.

    Args:
        file_iter: Iterable yielding tuples of (arcname, file_path). arcname is the name within the zip.

    Returns:
        Bytes of the created zip file.

    Raises:
        FileNotFoundError: If any provided file_path does not exist.
        OSError: For general I/O issues.
    """
    # Use BytesIO to keep in memory; for very large payloads consider streaming.
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, file_path in file_iter:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing file for zip: {file_path}")
            # Ensure arcname has no leading separators
            arcname = arcname.lstrip("/\\")
            zf.write(file_path, arcname)
    bio.seek(0)
    return bio.getvalue()


# PUBLIC_INTERFACE
def create_zip_from_memory(files: Iterable[Tuple[str, bytes]]) -> bytes:
    """Create a zip from in-memory (filename, content_bytes) pairs.

    Args:
        files: Iterable of (arcname, data_bytes)

    Returns:
        Zip bytes
    """
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, data in files:
            arcname = arcname.lstrip("/\\")
            zf.writestr(arcname, data)
    bio.seek(0)
    return bio.getvalue()
