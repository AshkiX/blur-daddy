"""Detection quality metrics: IoU, precision, recall, mAP."""

from __future__ import annotations


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _match_detections_scored(
    detections: list[tuple[tuple, float]],
    ground_truths: list[tuple],
    iou_threshold: float,
) -> list[tuple[float, bool]]:
    """Match detections to ground truths greedily by confidence.

    Returns list of (confidence, is_true_positive) sorted by confidence descending.
    """
    dets_sorted = sorted(detections, key=lambda d: d[1], reverse=True)
    matched_gt = set()
    results = []

    for det_box, conf in dets_sorted:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            results.append((conf, True))
            matched_gt.add(best_gt_idx)
        else:
            results.append((conf, False))

    return results


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """Compute Average Precision using all-point interpolation (PASCAL VOC 2010+)."""
    if not precisions:
        return 0.0

    # Add sentinel values
    precisions = [0.0] + precisions + [0.0]
    recalls = [0.0] + recalls + [recalls[-1] if recalls else 0.0]

    # VOC 2010+ requires monotonic precision for interpolation
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Find points where recall changes
    ap = 0.0
    for i in range(1, len(recalls)):
        if recalls[i] != recalls[i - 1]:
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap


def evaluate_image(
    detections: list[tuple[tuple, float]],
    ground_truths: list[tuple],
    iou_threshold: float,
) -> dict:
    """Evaluate detections for a single image."""
    scored = _match_detections_scored(detections, ground_truths, iou_threshold)
    tp = sum(1 for _, is_tp in scored if is_tp)
    fp = sum(1 for _, is_tp in scored if not is_tp)
    fn = len(ground_truths) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_dataset(
    all_detections: list[list[tuple[tuple, float]]],
    all_ground_truths: list[list[tuple]],
    iou_threshold: float,
) -> dict:
    """Evaluate detections across a full dataset. Computes mAP, precision, recall."""
    all_scored = []
    total_gt = 0

    for dets, gts in zip(all_detections, all_ground_truths):
        total_gt += len(gts)
        all_scored.extend(_match_detections_scored(dets, gts, iou_threshold))

    # Sort all detections by confidence for AP curve
    all_scored.sort(key=lambda x: x[0], reverse=True)

    # Compute precision-recall curve
    precisions = []
    recalls = []
    tp_cumsum = 0
    fp_cumsum = 0

    for _, is_tp in all_scored:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
        recalls.append(tp_cumsum / total_gt if total_gt > 0 else 0.0)

    ap = compute_ap(precisions, recalls)
    total_tp = tp_cumsum
    total_fp = fp_cumsum
    total_fn = total_gt - total_tp

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_gt": total_gt,
        "total_detections": len(all_scored),
    }
