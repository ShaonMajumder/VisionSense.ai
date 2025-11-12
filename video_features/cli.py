import argparse
import json
from pathlib import Path

from video_features.extractor import FeatureExtractionConfig, VideoFeatureExtractor, FarnebackParams, HistogramParams
from video_features.settings import ExtractorSettings


def _env_cfg() -> FeatureExtractionConfig:
    e = ExtractorSettings.load()
    return FeatureExtractionConfig(
        frame_stride=e.frame_stride,
        resize_width=e.resize_width,
        shot_threshold=e.shot_threshold,
        text_sample_stride=e.text_sample_stride,
        text_min_chars=e.text_min_chars,
        yolo_model=e.yolo_model,
        yolo_frame_stride=e.yolo_frame_stride,
        farneback=FarnebackParams(
            pyr_scale=e.farneback_pyr_scale,
            levels=e.farneback_levels,
            winsize=e.farneback_winsize,
            iterations=e.farneback_iterations,
            poly_n=e.farneback_poly_n,
            poly_sigma=e.farneback_poly_sigma,
            flags=e.farneback_flags,
        ),
        hist=HistogramParams(bins=e.histogram_bins, ranges=e.histogram_ranges),
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract structural, temporal, and semantic video features.")
    p.add_argument("--video", required=True)
    p.add_argument("--output")
    p.add_argument("--frame-stride", type=int)
    p.add_argument("--resize-width", type=int)
    p.add_argument("--shot-threshold", type=float)
    p.add_argument("--text-sample-stride", type=int)
    p.add_argument("--text-min-chars", type=int)
    p.add_argument("--yolo-frame-stride", type=int)
    p.add_argument("--yolo-model")
    return p


def _merge(base: FeatureExtractionConfig, args: argparse.Namespace) -> FeatureExtractionConfig:
    return FeatureExtractionConfig(
        frame_stride=args.frame_stride if args.frame_stride is not None else base.frame_stride,
        resize_width=args.resize_width if args.resize_width is not None else base.resize_width,
        shot_threshold=args.shot_threshold if args.shot_threshold is not None else base.shot_threshold,
        text_sample_stride=args.text_sample_stride if args.text_sample_stride is not None else base.text_sample_stride,
        text_min_chars=args.text_min_chars if args.text_min_chars is not None else base.text_min_chars,
        yolo_frame_stride=args.yolo_frame_stride if args.yolo_frame_stride is not None else base.yolo_frame_stride,
        yolo_model=args.yolo_model if args.yolo_model is not None else base.yolo_model,
        farneback=base.farneback,
        hist=base.hist,
    )


def main() -> None:
    args = _parser().parse_args()
    cfg = _merge(_env_cfg(), args)
    extractor = VideoFeatureExtractor(cfg)
    features = extractor.extract_features(args.video)
    print(json.dumps(features, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(features, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
