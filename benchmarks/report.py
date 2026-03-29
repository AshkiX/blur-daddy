"""Benchmark report generation: JSON + Markdown."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def generate_report(results: dict, output_dir: Path) -> tuple[Path, Path]:
    """Write benchmark results as JSON and Markdown.

    Args:
        results: Full benchmark results dict.
        output_dir: Directory to write reports to.

    Returns:
        (json_path, md_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # JSON report
    json_path = output_dir / f"benchmark_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Markdown report
    md_path = output_dir / f"benchmark_{timestamp}.md"
    md = _render_markdown(results)
    with open(md_path, "w") as f:
        f.write(md)

    return json_path, md_path


def _render_markdown(results: dict) -> str:
    lines = [
        "# Benchmark Report",
        "",
        f"**Date:** {results.get('timestamp', 'N/A')}  ",
        f"**Tier:** {results.get('tier', 'N/A')}  ",
        f"**Dataset:** {results.get('dataset', 'N/A')}  ",
        f"**Images:** {results.get('num_images', 'N/A')}  ",
        "",
    ]

    # Detection quality table
    det_results = results.get("detection", {})
    if det_results:
        lines.append("## Detection Quality")
        lines.append("")
        lines.append("| Detector | IoU | AP | Precision | Recall | F1 |")
        lines.append("|----------|-----|-----|-----------|--------|----|")
        for detector_name, iou_results in det_results.items():
            for iou_str, metrics in iou_results.items():
                lines.append(
                    f"| {detector_name} | {iou_str} | "
                    f"{metrics.get('ap', 0):.3f} | "
                    f"{metrics.get('precision', 0):.3f} | "
                    f"{metrics.get('recall', 0):.3f} | "
                    f"{metrics.get('f1', 0):.3f} |"
                )
        lines.append("")

    # Performance table
    perf_results = results.get("performance", {})
    if perf_results:
        lines.append("## Performance")
        lines.append("")
        lines.append("| Detector | FPS | Avg ms/img | Peak Memory (MB) |")
        lines.append("|----------|-----|------------|------------------|")
        for detector_name, metrics in perf_results.items():
            lines.append(
                f"| {detector_name} | "
                f"{metrics.get('fps', 0):.1f} | "
                f"{metrics.get('avg_ms_per_image', 0):.1f} | "
                f"{metrics.get('peak_memory_mb', 'N/A')} |"
            )
        lines.append("")

    # Video metrics
    video_results = results.get("video", {})
    if video_results:
        lines.append("## Video Metrics")
        lines.append("")
        lines.append("| Detector | Temporal Consistency | Match Rate | Flicker Rate |")
        lines.append("|----------|---------------------|------------|--------------|")
        for detector_name, metrics in video_results.items():
            tc = metrics.get("temporal_consistency", {})
            fl = metrics.get("flicker", {})
            lines.append(
                f"| {detector_name} | "
                f"{tc.get('mean_iou', 0):.3f} | "
                f"{tc.get('match_rate', 0):.3f} | "
                f"{fl.get('flicker_rate', 0):.3f} |"
            )
        lines.append("")

    # Errors/skipped
    errors = results.get("errors", {})
    if errors:
        lines.append("## Skipped / Errors")
        lines.append("")
        for name, reason in errors.items():
            lines.append(f"- **{name}**: {reason}")
        lines.append("")

    return "\n".join(lines)
