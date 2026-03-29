"""Performance metrics: FPS and peak memory tracking."""

from __future__ import annotations

import threading
import time

import psutil


class MemoryTracker:
    """Track peak RSS memory usage in a background thread."""

    def __init__(self, poll_interval: float = 0.05):
        self._poll_interval = poll_interval
        self._peak_mb = 0.0
        self._baseline_mb = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        process = psutil.Process()
        self._baseline_mb = process.memory_info().rss / 1024 / 1024
        self._peak_mb = self._baseline_mb
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        process = psutil.Process()
        while self._running:
            mem_mb = process.memory_info().rss / 1024 / 1024
            if mem_mb > self._peak_mb:
                self._peak_mb = mem_mb
            time.sleep(self._poll_interval)

    def stop(self) -> float:
        """Stop tracking and return peak memory delta in MB."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return round(self._peak_mb - self._baseline_mb, 2)

    @property
    def peak_mb(self) -> float:
        return round(self._peak_mb, 2)


def measure_fps(detector, images: list, warmup_runs: int = 3) -> dict:
    """Measure detector FPS on a list of images.

    Returns dict with fps, total_time, num_images, avg_ms_per_image.
    """
    # Warmup
    for img in images[:warmup_runs]:
        detector.detect(img)

    start = time.perf_counter()
    for img in images:
        detector.detect(img)
    elapsed = time.perf_counter() - start

    num = len(images)
    return {
        "fps": num / elapsed if elapsed > 0 else 0.0,
        "total_time_s": round(elapsed, 3),
        "num_images": num,
        "avg_ms_per_image": round((elapsed / num) * 1000, 1) if num > 0 else 0.0,
    }
