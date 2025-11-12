from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO


@dataclass(frozen=True)
class FarnebackParams:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


@dataclass(frozen=True)
class HistogramParams:
    bins: Tuple[int, int, int] = (8, 8, 8)
    ranges: Tuple[int, int, int, int, int, int] = (0, 180, 0, 256, 0, 256)


@dataclass(frozen=True)
class FeatureExtractionConfig:
    frame_stride: int = 5
    resize_width: int = 640
    shot_threshold: float = 0.45
    text_sample_stride: int = 10
    text_min_chars: int = 8
    yolo_model: str = "yolov8n.pt"
    yolo_frame_stride: int = 15
    farneback: FarnebackParams = field(default_factory=FarnebackParams)
    hist: HistogramParams = field(default_factory=HistogramParams)


class VideoFeatureExtractor:
    def __init__(self, config: Optional[FeatureExtractionConfig] = None) -> None:
        self.config = config or FeatureExtractionConfig()
        self.detector = YOLO(self.config.yolo_model)

    def extract_features(self, video_path: str | Path) -> Dict[str, Any]:
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video not found: {video_file}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_file}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = (total_frames / fps) if fps > 0 else None

        absolute_frame_index = -1
        processed_frames = 0
        hard_cuts = 0
        text_samples = 0
        text_positives = 0
        motion_samples = 0
        motion_magnitude_sum = 0.0
        person_total = 0
        object_total = 0

        prev_hist: Optional[np.ndarray] = None
        prev_gray: Optional[np.ndarray] = None

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                absolute_frame_index += 1
                if (absolute_frame_index % self.config.frame_stride) != 0:
                    continue

                processed_frames += 1
                frame_small_bgr = self._resize_to_width(frame_bgr, self.config.resize_width)

                curr_hist = self._hsv_histogram(frame_small_bgr)
                if prev_hist is not None:
                    color_distance = self._bhattacharyya(prev_hist, curr_hist)
                    if color_distance > self.config.shot_threshold:
                        hard_cuts += 1
                prev_hist = curr_hist

                curr_gray = cv2.cvtColor(frame_small_bgr, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    avg_mag = self._avg_optical_flow_magnitude(prev_gray, curr_gray)
                    motion_magnitude_sum += avg_mag
                    motion_samples += 1
                prev_gray = curr_gray

                if (processed_frames % self.config.text_sample_stride) == 0:
                    text_samples += 1
                    if self._contains_text(frame_small_bgr, min_chars=self.config.text_min_chars):
                        text_positives += 1

                if (processed_frames % self.config.yolo_frame_stride) == 0:
                    persons, objects = self._count_people_vs_objects(frame_small_bgr)
                    person_total += persons
                    object_total += objects
        finally:
            cap.release()

        avg_motion = (motion_magnitude_sum / motion_samples) if motion_samples else 0.0
        text_ratio = (text_positives / text_samples) if text_samples else 0.0
        person_object_ratio = (person_total / object_total) if object_total else None

        return {
            "video_path": str(video_file.resolve()),
            "duration_seconds": duration_seconds,
            "frames_total": total_frames,
            "frames_processed": processed_frames,
            "hard_cuts": hard_cuts,
            "avg_motion_magnitude": avg_motion,
            "text_present_ratio": text_ratio,
            "people_detections": person_total,
            "object_detections": object_total,
            "person_to_object_ratio": person_object_ratio,
        }

    def _resize_to_width(self, frame_bgr: np.ndarray, target_w: int) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if w <= target_w:
            return frame_bgr
        scale = target_w / float(w)
        new_size = (target_w, int(h * scale))
        return cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)

    def _hsv_histogram(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            images=[hsv],
            channels=[0, 1, 2],
            mask=None,
            histSize=list(self.config.hist.bins),
            ranges=list(self.config.hist.ranges),
        )
        cv2.normalize(src=hist, dst=hist)
        return hist

    @staticmethod
    def _bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))

    def _avg_optical_flow_magnitude(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        p = self.config.farneback
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_gray,
            next=curr_gray,
            flow=None,
            pyr_scale=p.pyr_scale,
            levels=p.levels,
            winsize=p.winsize,
            iterations=p.iterations,
            poly_n=p.poly_n,
            poly_sigma=p.poly_sigma,
            flags=p.flags,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag))

    def _contains_text(self, frame_bgr: np.ndarray, min_chars: int) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            src=blur,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2,
        )
        data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
        text = "".join(data.get("text", [])).strip()
        return len(text) >= min_chars

    def _count_people_vs_objects(self, frame_bgr: np.ndarray) -> Tuple[int, int]:
        result = self.detector(frame_bgr, verbose=False)[0]
        names = getattr(result, "names", getattr(self.detector, "names", {})) or {}
        people, others = 0, 0
        if not hasattr(result, "boxes") or result.boxes is None:
            return people, others
        classes = result.boxes.cls
        if classes is None:
            return people, others
        for cls_id in classes.int().tolist():
            label = names.get(int(cls_id), "")
            if label == "person":
                people += 1
            else:
                others += 1
        return people, others
