"""Image encoding with an OpenCV fast path and a dependency-free PNG fallback."""
from __future__ import annotations

import struct
import zlib

import numpy as np

try:  # pragma: no cover - depends on optional install
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def encode(image: np.ndarray, quality: int = 80) -> tuple[bytes, str]:
    """Returns (bytes, extension)."""
    if cv2 is not None:
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            return buf.tobytes(), "jpg"
    return _png(image), "png"


def _png(image: np.ndarray) -> bytes:
    h, w = image.shape[:2]
    if image.ndim == 2:
        rgb, ctype = image, 0
    else:
        rgb, ctype = image[:, :, ::-1], 2      # BGR -> RGB
    raw = b"".join(b"\x00" + np.ascontiguousarray(rgb[y]).tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
