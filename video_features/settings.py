from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


def _get(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


def _get_int(name: str, default: int) -> int:
    try:
        return int(_get(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(_get(name, str(default)))
    except ValueError:
        return default


def _get_tuple_ints(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    raw = _get(name, ",".join(str(x) for x in default)).split(",")
    try:
        vals = tuple(int(x.strip()) for x in raw if x.strip())
        return vals if vals else default
    except ValueError:
        return default


@dataclass(frozen=True)
class AppSettings:
    app_title: str
    static_dir: Path
    allowed_extensions: Tuple[str, ...]
    allowed_mime_prefix: str
    max_upload_bytes: int
    upload_chunk_bytes: int
    temp_volume_dir: Path
    volume_quota_bytes: int
    volume_min_free_bytes: int

    @staticmethod
    def load() -> "AppSettings":
        return AppSettings(
            app_title=_get("APP_TITLE", "VisionSense API"),
            static_dir=Path(_get("STATIC_DIR", "static")),
            allowed_extensions=tuple(ext if ext.startswith(".") else f".{ext}" for ext in _get("ALLOWED_EXTENSIONS", ".mp4,.mov,.mkv,.avi,.webm").split(",")),
            allowed_mime_prefix=_get("ALLOWED_MIME_PREFIX", "video/"),
            max_upload_bytes=_get_int("MAX_UPLOAD_BYTES", 500 * 1024 * 1024),
            upload_chunk_bytes=_get_int("UPLOAD_CHUNK_BYTES", 1 * 1024 * 1024),
            temp_volume_dir=Path(_get("TEMP_VOLUME_DIR", "/tmp/video-tmp")),
            volume_quota_bytes=_get_int("VOLUME_QUOTA_BYTES", 10 * 1024 * 1024 * 1024),
            volume_min_free_bytes=_get_int("VOLUME_MIN_FREE_BYTES", 2 * 1024 * 1024 * 1024),
        )


@dataclass(frozen=True)
class ExtractorSettings:
    frame_stride: int
    resize_width: int
    shot_threshold: float
    text_sample_stride: int
    text_min_chars: int
    yolo_model: str
    yolo_frame_stride: int
    farneback_pyr_scale: float
    farneback_levels: int
    farneback_winsize: int
    farneback_iterations: int
    farneback_poly_n: int
    farneback_poly_sigma: float
    farneback_flags: int
    histogram_bins: Tuple[int, int, int]
    histogram_ranges: Tuple[int, int, int, int, int, int]

    @staticmethod
    def load() -> "ExtractorSettings":
        return ExtractorSettings(
            frame_stride=_get_int("FRAME_STRIDE", 5),
            resize_width=_get_int("RESIZE_WIDTH", 640),
            shot_threshold=_get_float("SHOT_THRESHOLD", 0.45),
            text_sample_stride=_get_int("TEXT_SAMPLE_STRIDE", 10),
            text_min_chars=_get_int("TEXT_MIN_CHARS", 8),
            yolo_model=_get("YOLO_MODEL", "yolov8n.pt"),
            yolo_frame_stride=_get_int("YOLO_FRAME_STRIDE", 15),
            farneback_pyr_scale=_get_float("FARNEBACK_PYR_SCALE", 0.5),
            farneback_levels=_get_int("FARNEBACK_LEVELS", 3),
            farneback_winsize=_get_int("FARNEBACK_WINSIZE", 15),
            farneback_iterations=_get_int("FARNEBACK_ITERATIONS", 3),
            farneback_poly_n=_get_int("FARNEBACK_POLY_N", 5),
            farneback_poly_sigma=_get_float("FARNEBACK_POLY_SIGMA", 1.2),
            farneback_flags=_get_int("FARNEBACK_FLAGS", 0),
            histogram_bins=tuple(_get_tuple_ints("HISTOGRAM_BINS", (8, 8, 8)))[:3],  # type: ignore
            histogram_ranges=tuple(_get_tuple_ints("HISTOGRAM_RANGES", (0, 180, 0, 256, 0, 256)))[:6],  # type: ignore
        )
