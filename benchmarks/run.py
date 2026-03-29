"""Benchmark harness: orchestrates dataset loading, detection, and reporting."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from benchmarks.config import CONFIDENCE_THRESHOLD, IOU_THRESHOLDS, MICRO_SIZE, RESULTS_DIR
from benchmarks.datasets.base import ImageSample
from benchmarks.metrics.detection import evaluate_dataset
from benchmarks.metrics.performance import MemoryTracker, measure_fps
from benchmarks.metrics.video import compute_flicker_rate, compute_temporal_consistency
from benchmarks.report import generate_report


def get_all_detectors(include_competitors: bool = True) -> list:
    """Load all available detectors, skipping those with missing deps."""
    detectors = []
    errors = {}

    # Always include blur-daddy detectors
    from benchmarks.detectors.blur_daddy_mtcnn import BlurDaddyMTCNN
    from benchmarks.detectors.blur_daddy_yolo import BlurDaddyYOLO

    detectors.append(BlurDaddyYOLO())
    detectors.append(BlurDaddyMTCNN())

    if not include_competitors:
        return detectors, errors

    competitor_classes = [
        ("benchmarks.detectors.deface_centerface", "DefaceCenterFace"),
        ("benchmarks.detectors.insightface_det", "RetinaFaceDetector"),
        ("benchmarks.detectors.insightface_det", "SCRFDDetector"),
        ("benchmarks.detectors.mediapipe_face", "MediaPipeFaceDetector"),
    ]

    for module_path, class_name in competitor_classes:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            det = cls()
            # Test that the underlying library is importable
            det.warmup()
            detectors.append(det)
        except Exception as e:
            errors[class_name] = str(e)

    return detectors, errors


def run_image_benchmark(
    detectors: list,
    samples: list[ImageSample],
    iou_thresholds: list[float],
) -> tuple[dict, dict]:
    """Run detection + performance benchmarks on image samples.

    Returns (detection_results, performance_results).
    """
    # Load all images once
    images = []
    ground_truths = []
    for sample in samples:
        img = cv2.imread(str(sample.path))
        if img is None:
            continue
        images.append(img)
        ground_truths.append(sample.ground_truth_boxes)

    print(f"  Loaded {len(images)} images")

    detection_results = {}
    performance_results = {}

    for det in detectors:
        print(f"  Running {det.name}...")

        # Warmup
        try:
            det.warmup()
        except Exception as e:
            print(f"    Warmup failed: {e}")
            continue

        # Run detections
        all_detections = []
        mem_tracker = MemoryTracker()
        mem_tracker.start()

        for img in images:
            try:
                dets = det.detect(img)
                # Filter by confidence
                dets = [d for d in dets if d.confidence >= CONFIDENCE_THRESHOLD]
                all_detections.append([(d.box, d.confidence) for d in dets])
            except Exception as e:
                print(f"    Detection failed on an image: {e}")
                all_detections.append([])

        peak_mem = mem_tracker.stop()

        # Detection quality at each IoU threshold
        det_iou_results = {}
        for iou_t in iou_thresholds:
            result = evaluate_dataset(all_detections, ground_truths, iou_t)
            det_iou_results[f"IoU={iou_t}"] = result

        detection_results[det.name] = det_iou_results

        # Performance
        perf = measure_fps(det, images, warmup_runs=min(3, len(images)))
        perf["peak_memory_mb"] = peak_mem
        performance_results[det.name] = perf

    return detection_results, performance_results


def run_video_benchmark(
    detectors: list, video_paths: list[Path], max_frames: int = 150
) -> dict:
    """Run video-specific metrics on video files.

    Args:
        max_frames: Max frames to sample per video. Samples evenly across the video.
    """
    video_results = {}

    for det in detectors:
        print(f"    {det.name}...")
        all_temporal = []
        all_flicker = []

        for vpath in video_paths:
            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                print(f"      Cannot open video: {vpath}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Sample evenly if video is too long
            if total_frames > max_frames:
                step = total_frames / max_frames
                sample_indices = {int(i * step) for i in range(max_frames)}
            else:
                sample_indices = None  # process all

            frame_detections = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if sample_indices is not None and frame_idx not in sample_indices:
                    frame_idx += 1
                    continue
                try:
                    dets = det.detect(frame)
                    frame_detections.append(dets)
                except Exception:
                    frame_detections.append([])
                frame_idx += 1
            cap.release()

            print(f"      {vpath.name}: {len(frame_detections)}/{total_frames} frames sampled")

            if frame_detections:
                all_temporal.append(compute_temporal_consistency(frame_detections))
                all_flicker.append(compute_flicker_rate(frame_detections))

        # Average across videos
        if all_temporal:
            avg_tc = {
                "mean_iou": np.mean([t["mean_iou"] for t in all_temporal]),
                "match_rate": np.mean([t["match_rate"] for t in all_temporal]),
            }
            avg_fl = {
                "flicker_rate": np.mean([f["flicker_rate"] for f in all_flicker]),
                "total_flickers": sum(f["total_flickers"] for f in all_flicker),
            }
            video_results[det.name] = {
                "temporal_consistency": avg_tc,
                "flicker": avg_fl,
                "num_videos": len(video_paths),
            }

    return video_results


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Blur Daddy Benchmark Suite")
    parser.add_argument(
        "--tier",
        choices=["micro", "full"],
        default="micro",
        help="Benchmark tier: micro (~50 images, fast) or full (~1000 images)",
    )
    parser.add_argument(
        "--no-competitors",
        action="store_true",
        help="Only benchmark blur-daddy detectors, skip competitors",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video benchmarks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for benchmark results",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        help="Specific detectors to run (by name)",
    )
    opts = parser.parse_args(args)

    print("=" * 60)
    print("Blur Daddy Benchmark Suite")
    print("=" * 60)

    # Load detectors
    print("\nLoading detectors...")
    detectors, detector_errors = get_all_detectors(
        include_competitors=not opts.no_competitors
    )

    if opts.detectors:
        detectors = [d for d in detectors if d.name in opts.detectors]

    print(f"  Active: {[d.name for d in detectors]}")
    if detector_errors:
        print(f"  Skipped: {detector_errors}")

    # Load dataset
    print("\nLoading dataset...")
    from benchmarks.datasets.open_images import OpenImagesDataset

    dataset = OpenImagesDataset()
    if not dataset.is_ready():
        print("  Open Images V7 not found. Downloading...")
        dataset.setup()

    if opts.tier == "micro":
        samples = dataset.get_micro_samples(MICRO_SIZE)
    else:
        samples = dataset.get_samples(limit=1000)

    print(f"  {len(samples)} images loaded ({opts.tier} tier)")

    # Run image benchmarks
    print("\nRunning image benchmarks...")
    detection_results, performance_results = run_image_benchmark(
        detectors, samples, IOU_THRESHOLDS
    )

    # Run video benchmarks
    video_results = {}
    if not opts.no_video:
        print("\nChecking for video datasets...")
        from benchmarks.datasets.curated_clips import CuratedClipsDataset

        curated = CuratedClipsDataset()
        video_paths = []

        if curated.is_ready():
            for vs in curated.get_samples():
                video_paths.append(vs.path)

        # Also check sample_videos in project root
        sample_vids = Path(__file__).parent.parent / "sample_videos"
        if sample_vids.is_dir():
            for vf in sorted(sample_vids.glob("*.mp4")):
                video_paths.append(vf)

        if video_paths:
            print(f"  Running video benchmarks on {len(video_paths)} videos...")
            video_results = run_video_benchmark(detectors, video_paths)
        else:
            print("  No video files found, skipping video benchmarks")

    # Assemble results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier": opts.tier,
        "dataset": dataset.name,
        "num_images": len(samples),
        "iou_thresholds": IOU_THRESHOLDS,
        "detection": detection_results,
        "performance": performance_results,
        "video": video_results,
        "errors": detector_errors,
    }

    # Generate report
    print("\nGenerating report...")
    json_path, md_path = generate_report(results, opts.output_dir)
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    _print_summary(detection_results, performance_results)


def _print_summary(detection_results: dict, performance_results: dict) -> None:
    """Print a quick summary table to stdout."""
    print(f"\n{'Detector':<25} {'AP@0.5':>8} {'AP@0.75':>8} {'FPS':>8} {'ms/img':>8} {'Mem MB':>8}")
    print("-" * 75)
    for name in detection_results:
        det = detection_results[name]
        perf = performance_results.get(name, {})
        ap50 = det.get("IoU=0.5", {}).get("ap", 0)
        ap75 = det.get("IoU=0.75", {}).get("ap", 0)
        fps = perf.get("fps", 0)
        ms = perf.get("avg_ms_per_image", 0)
        mem = perf.get("peak_memory_mb", "N/A")
        print(f"{name:<25} {ap50:>8.3f} {ap75:>8.3f} {fps:>8.1f} {ms:>8.1f} {mem:>8}")


if __name__ == "__main__":
    main()
