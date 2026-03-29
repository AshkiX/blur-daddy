"""Video-specific metrics: temporal consistency, flicker rate."""

from __future__ import annotations

from benchmarks.detectors.base import Detection
from benchmarks.metrics.detection import compute_iou


def compute_temporal_consistency(
    frame_detections: list[list[Detection]],
    iou_threshold: float = 0.5,
) -> dict:
    """Measure how consistently faces are detected across consecutive frames.

    For each consecutive frame pair, match detections by IoU and measure
    how stable detection is.

    Returns:
        dict with mean_iou (of matched pairs), match_rate (fraction of
        detections that found a match in the next frame).
    """
    if len(frame_detections) < 2:
        return {"mean_iou": 0.0, "match_rate": 0.0, "num_transitions": 0}

    total_iou = 0.0
    total_matched = 0
    total_dets = 0

    for i in range(len(frame_detections) - 1):
        curr = frame_detections[i]
        next_ = frame_detections[i + 1]
        total_dets += len(curr)

        matched_next = set()
        for det in curr:
            best_iou = 0.0
            best_idx = -1
            for j, ndet in enumerate(next_):
                if j in matched_next:
                    continue
                iou = compute_iou(det.box, ndet.box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= iou_threshold and best_idx >= 0:
                total_iou += best_iou
                total_matched += 1
                matched_next.add(best_idx)

    return {
        "mean_iou": round(total_iou / total_matched, 4) if total_matched > 0 else 0.0,
        "match_rate": round(total_matched / total_dets, 4) if total_dets > 0 else 0.0,
        "num_transitions": len(frame_detections) - 1,
    }


def compute_flicker_rate(
    frame_detections: list[list[Detection]],
    iou_threshold: float = 0.5,
) -> dict:
    """Measure flicker: faces that appear/disappear for only 1-2 frames.

    A "flicker" is when detection count changes for 1-2 frames then reverts.

    Returns:
        dict with flicker_rate (fraction of frames that are flickers),
        total_flickers, total_frames.
    """
    if len(frame_detections) < 3:
        return {"flicker_rate": 0.0, "total_flickers": 0, "total_frames": len(frame_detections)}

    counts = [len(dets) for dets in frame_detections]
    flickers = 0

    for i in range(1, len(counts) - 1):
        # A flicker: count changed from prev, then reverts to same as prev
        if counts[i] != counts[i - 1] and counts[i + 1] == counts[i - 1]:
            flickers += 1

    return {
        "flicker_rate": round(flickers / (len(counts) - 2), 4) if len(counts) > 2 else 0.0,
        "total_flickers": flickers,
        "total_frames": len(counts),
    }
