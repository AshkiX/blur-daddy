"""Pytest plugin that generates visual milestone reports as Markdown.

Usage:
    uv run pytest --milestone-report M0
    # produces reports/M0.md + reports/M0/*.png
"""

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pytest

REPORTS_DIR = Path(__file__).parent.parent / "reports"
HISTORY_FILE = REPORTS_DIR / "history.json"
SAMPLE_IMAGE = Path(__file__).parent.parent / "sample_images" / "sample1.jpg"


def pytest_addoption(parser):
    parser.addoption(
        "--milestone-report",
        default=None,
        metavar="NAME",
        help="Generate a milestone report (e.g. --milestone-report M0)",
    )


def pytest_configure(config):
    config._ms_name = config.getoption("--milestone-report", default=None)
    config._ms_results = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        module = item.nodeid.split("::")[0].replace("tests/", "").replace(".py", "")
        item.config._ms_results.append(
            {
                "nodeid": item.nodeid,
                "module": module,
                "passed": report.passed,
                "skipped": report.skipped,
                "duration": round(report.duration, 4),
            }
        )


def pytest_sessionfinish(session, exitstatus):
    name = session.config._ms_name
    if not name:
        return
    results = session.config._ms_results

    REPORTS_DIR.mkdir(exist_ok=True)
    images_dir = REPORTS_DIR / name
    images_dir.mkdir(exist_ok=True)

    # --- Aggregate test results ---
    modules = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0, "tests": []})
    total_passed = total_failed = total_skipped = 0
    for r in results:
        mod = modules[r["module"]]
        mod["tests"].append(r)
        if r["skipped"]:
            mod["skipped"] += 1
            total_skipped += 1
        elif r["passed"]:
            mod["passed"] += 1
            total_passed += 1
        else:
            mod["failed"] += 1
            total_failed += 1

    total = total_passed + total_failed + total_skipped

    # --- Performance benchmarks ---
    perf = _run_benchmarks()

    # --- Visual samples ---
    sample_files = _generate_visual_samples(images_dir)

    # --- Build milestone data ---
    milestone_data = {
        "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "passed": total_passed,
        "failed": total_failed,
        "skipped": total_skipped,
        "perf": perf,
    }

    # --- Update history ---
    history = _load_history()
    history = [h for h in history if h["name"] != name]
    history.append(milestone_data)
    history.sort(key=lambda h: h["name"])
    _save_history(history)

    # --- Render Markdown ---
    md = _render_markdown(name, milestone_data, modules, perf, sample_files, history)
    report_path = REPORTS_DIR / f"{name}.md"
    report_path.write_text(md)

    print(f"\n{'='*60}")
    print(f"  Milestone report: {report_path}")
    print(f"  {total_passed} passed / {total_failed} failed / {total_skipped} skipped")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def _run_benchmarks():
    perf = {}
    try:
        from blur_daddy import BlurDaddy

        if not SAMPLE_IMAGE.exists():
            return perf

        path = str(SAMPLE_IMAGE)

        # YOLO detect
        bd_yolo = BlurDaddy(model="yolov8n-face")
        t0 = time.perf_counter()
        bd_yolo.detect(path)
        perf["detect(model='yolov8n-face')"] = round(time.perf_counter() - t0, 3)

        # MTCNN detect
        bd_mtcnn = BlurDaddy(model="mtcnn")
        t0 = time.perf_counter()
        bd_mtcnn.detect(path)
        perf["detect(model='mtcnn')"] = round(time.perf_counter() - t0, 3)

        # Blur methods
        for method in ("gaussian", "pixelation", "elliptical"):
            bd = BlurDaddy(method=method)
            t0 = time.perf_counter()
            bd.blur(path)
            perf[f"blur(method='{method}')"] = round(time.perf_counter() - t0, 3)

        # keep= (detect + blur with protection)
        preview = bd_yolo.detect(path)
        if len(preview.faces) >= 2:
            t0 = time.perf_counter()
            bd_yolo.blur(path, keep=[preview.faces[0]])
            perf["blur(keep=[face-0])"] = round(time.perf_counter() - t0, 3)

    except Exception as exc:
        perf["error"] = str(exc)

    return perf


# ---------------------------------------------------------------------------
# Visual samples — saves PNGs to images_dir
# ---------------------------------------------------------------------------

def _generate_visual_samples(images_dir):
    """Generate sample images and save as PNGs. Returns list of (label, filename)."""
    samples = []
    try:
        from blur_daddy import BlurDaddy

        if not SAMPLE_IMAGE.exists():
            return samples

        img_bgr = cv2.imread(str(SAMPLE_IMAGE))
        scale = min(1.0, 600.0 / img_bgr.shape[1])

        def _resize(img):
            if scale < 1.0:
                return cv2.resize(img, None, fx=scale, fy=scale)
            return img

        def _save_rgb(rgb_img, filename):
            bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(images_dir / filename), _resize(bgr))

        def _save_bgr(bgr_img, filename):
            cv2.imwrite(str(images_dir / filename), _resize(bgr_img))

        # 1. Original
        _save_bgr(img_bgr, "original.png")
        samples.append(("Original", "original.png"))

        # 2. Detection preview
        bd = BlurDaddy()
        preview = bd.detect(str(SAMPLE_IMAGE))
        if not preview.faces:
            return samples

        annotated = preview.image.copy()
        for face in preview.faces:
            x1, y1, x2, y2 = face.box_int
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, face.id, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        _save_rgb(annotated, "detect.png")
        samples.append(("detect()", "detect.png"))

        # 3. Three blur methods
        for method in ("gaussian", "pixelation", "elliptical"):
            bd_m = BlurDaddy(method=method)
            result = bd_m.blur(str(SAMPLE_IMAGE))
            filename = f"blur_{method}.png"
            _save_rgb(result.image, filename)
            samples.append((f"blur(method='{method}')", filename))

        # 4. keep= demo
        if len(preview.faces) >= 2:
            result_keep = bd.blur(str(SAMPLE_IMAGE), keep=[preview.faces[0]])
            _save_rgb(result_keep.image, "blur_keep.png")
            samples.append(("blur(keep=[face-0])", "blur_keep.png"))

    except Exception:
        pass

    return samples


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def _load_history():
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


def _save_history(history):
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _render_markdown(name, data, modules, perf, sample_files, history):
    passed, failed, skipped, total = data["passed"], data["failed"], data["skipped"], data["total"]
    status = "ALL PASSING" if failed == 0 else f"{failed} FAILING"
    timestamp = datetime.fromisoformat(data["timestamp"]).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# {name} — Milestone Report",
        "",
        f"**{timestamp}** | **{status}** | {total} total | {passed} passed | {failed} failed | {skipped} skipped",
        "",
    ]

    # --- Tests by module ---
    lines.append("## Tests by Module")
    lines.append("")
    lines.append("| Module | Pass | Fail | Skip | Status |")
    lines.append("|--------|------|------|------|--------|")
    for mod_name, mod in sorted(modules.items()):
        p, f, s = mod["passed"], mod["failed"], mod["skipped"]
        icon = "pass" if f == 0 else "FAIL"
        lines.append(f"| {mod_name} | {p} | {f} | {s} | {icon} |")
    lines.append("")

    # --- Failed tests ---
    if failed > 0:
        failed_tests = []
        for r in data.get("_results", []):
            if not r.get("passed") and not r.get("skipped"):
                failed_tests.append(r)
        if failed_tests:
            lines.append("## Failed Tests")
            lines.append("")
            for t in failed_tests:
                lines.append(f"- `{t['nodeid']}`")
            lines.append("")

    # --- Performance ---
    if perf and "error" not in perf:
        lines.append("## Performance")
        lines.append("")
        lines.append("| Operation | Time |")
        lines.append("|-----------|------|")
        for op, secs in perf.items():
            lines.append(f"| `{op}` | {secs:.3f}s |")
        lines.append("")

    # --- Visual samples ---
    if sample_files:
        lines.append("## Visual Samples")
        lines.append("")
        # Row of images as a table
        headers = " | ".join(label for label, _ in sample_files)
        separator = " | ".join("---" for _ in sample_files)
        images = " | ".join(f"![{label}]({name}/{filename})" for label, filename in sample_files)
        lines.append(f"| {headers} |")
        lines.append(f"| {separator} |")
        lines.append(f"| {images} |")
        lines.append("")

    # --- Milestone trend ---
    lines.append("## Milestone Trend")
    lines.append("")
    lines.append("| Milestone | Total | Passed | Failed |")
    lines.append("|-----------|-------|--------|--------|")
    for h in history:
        marker = " **<- current**" if h["name"] == name else ""
        lines.append(f"| {h['name']}{marker} | {h['total']} | {h['passed']} | {h['failed']} |")
    lines.append("")

    lines.append(f"---\n*Generated by `pytest --milestone-report {name}`*")
    return "\n".join(lines)
