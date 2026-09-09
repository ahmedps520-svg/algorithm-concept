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


def decode(data: bytes) -> np.ndarray:
    """Inverse of ``encode``: OpenCV when available, else our own filter-0 PNG layout."""
    if cv2 is not None:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("unsupported image format without OpenCV")
    pos, idat, w = 8, b"", 0
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4]); tag = data[pos + 4:pos + 8]; body = data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", body[:10])
            if depth != 8 or ctype not in (0, 2):
                raise ValueError("only 8-bit grey/RGB PNG supported")
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    ch = 3 if ctype == 2 else 1
    stride = w * ch + 1
    rows = np.frombuffer(raw, np.uint8).reshape(h, stride)
    if rows[:, 0].any():
        raise ValueError("PNG uses row filters; OpenCV is required to decode it")
    img = rows[:, 1:].reshape(h, w, ch)
    return np.ascontiguousarray(img[:, :, ::-1]) if ch == 3 else np.ascontiguousarray(img)


def resize_nearest(img: np.ndarray, width: int) -> np.ndarray:
    """Dependency-free resize (nearest neighbour), width-driven, aspect preserved."""
    h, w = img.shape[:2]
    height = max(1, round(h * width / w))
    ys = (np.arange(height) * h / height).astype(int)
    xs = (np.arange(width) * w / width).astype(int)
    return img[ys][:, xs]
