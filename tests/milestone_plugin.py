"""Pytest plugin that generates visual milestone reports as HTML.

Usage:
    uv run pytest --milestone-report M0
    # produces reports/M0.html (self-contained)
"""

import base64
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
    sample_images = _generate_visual_samples()

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

    # --- Render HTML ---
    html = _render_html(name, milestone_data, modules, perf, sample_images, history)
    report_path = REPORTS_DIR / f"{name}.html"
    report_path.write_text(html)

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
# Visual samples — returns base64-encoded PNGs
# ---------------------------------------------------------------------------

def _generate_visual_samples():
    """Generate sample images and return as base64 data URIs.

    Returns list of (label, base64_data_uri).
    """
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

        def _to_b64(bgr_img):
            resized = _resize(bgr_img)
            _, buf = cv2.imencode(".png", resized)
            return "data:image/png;base64," + base64.b64encode(buf).decode()

        def _rgb_to_b64(rgb_img):
            return _to_b64(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # 1. Original
        samples.append(("Original", _to_b64(img_bgr)))

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
        samples.append(("detect()", _rgb_to_b64(annotated)))

        # 3. Three blur methods
        for method in ("gaussian", "pixelation", "elliptical"):
            bd_m = BlurDaddy(method=method)
            result = bd_m.blur(str(SAMPLE_IMAGE))
            samples.append((f"blur(method='{method}')", _rgb_to_b64(result.image)))

        # 4. keep= demo
        if len(preview.faces) >= 2:
            result_keep = bd.blur(str(SAMPLE_IMAGE), keep=[preview.faces[0]])
            samples.append(("blur(keep=[face-0])", _rgb_to_b64(result_keep.image)))

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
# HTML rendering
# ---------------------------------------------------------------------------

def _render_html(name, data, modules, perf, sample_images, history):
    passed, failed, skipped, total = data["passed"], data["failed"], data["skipped"], data["total"]
    status = "ALL PASSING" if failed == 0 else f"{failed} FAILING"
    status_class = "pass" if failed == 0 else "fail"
    timestamp = datetime.fromisoformat(data["timestamp"]).strftime("%Y-%m-%d %H:%M UTC")

    # --- Tests by module rows ---
    module_rows = ""
    for mod_name, mod in sorted(modules.items()):
        p, f, s = mod["passed"], mod["failed"], mod["skipped"]
        row_class = "pass" if f == 0 else "fail"
        icon = "PASS" if f == 0 else "FAIL"
        module_rows += (
            f"<tr class='{row_class}'><td>{mod_name}</td>"
            f"<td>{p}</td><td>{f}</td><td>{s}</td><td>{icon}</td></tr>\n"
        )

    # --- Failed tests ---
    failed_section = ""
    if failed > 0:
        failed_tests = [r for r in data.get("_results", []) if not r.get("passed") and not r.get("skipped")]
        if failed_tests:
            items = "".join(f"<li><code>{t['nodeid']}</code></li>" for t in failed_tests)
            failed_section = f"<h2>Failed Tests</h2><ul>{items}</ul>"

    # --- Performance rows ---
    perf_section = ""
    if perf and "error" not in perf:
        perf_rows = ""
        for op, secs in perf.items():
            perf_rows += f"<tr><td><code>{op}</code></td><td>{secs:.3f}s</td></tr>\n"
        perf_section = f"""<h2>Performance</h2>
<table><thead><tr><th>Operation</th><th>Time</th></tr></thead>
<tbody>{perf_rows}</tbody></table>"""

    # --- Visual samples ---
    samples_section = ""
    if sample_images:
        cards = ""
        for label, data_uri in sample_images:
            cards += f"""<div class="sample-card">
<img src="{data_uri}" alt="{label}">
<div class="sample-label">{label}</div>
</div>\n"""
        samples_section = f'<h2>Visual Samples</h2><div class="samples-grid">{cards}</div>'

    # --- Milestone trend rows ---
    trend_rows = ""
    for h in history:
        marker = " (current)" if h["name"] == name else ""
        row_class = "current" if h["name"] == name else ""
        trend_rows += (
            f"<tr class='{row_class}'><td>{h['name']}{marker}</td>"
            f"<td>{h['total']}</td><td>{h['passed']}</td><td>{h['failed']}</td></tr>\n"
        )

    css = (
        "body { font-family: -apple-system, BlinkMacSystemFont, "
        "'Segoe UI', Roboto, sans-serif; max-width: 960px; "
        "margin: 0 auto; padding: 2rem; color: #1a1a1a; background: #fafafa; }\n"
        "h1 { margin-bottom: 0.25rem; }\n"
        ".summary { color: #555; margin-bottom: 2rem; font-size: 0.95rem; }\n"
        ".summary .status { font-weight: bold; }\n"
        ".summary .status.pass { color: #16a34a; }\n"
        ".summary .status.fail { color: #dc2626; }\n"
        "table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }\n"
        "th, td { text-align: left; padding: 0.5rem 0.75rem; "
        "border-bottom: 1px solid #e5e5e5; }\n"
        "th { background: #f5f5f5; font-weight: 600; }\n"
        "tr.pass td:last-child { color: #16a34a; font-weight: 600; }\n"
        "tr.fail td:last-child { color: #dc2626; font-weight: 600; }\n"
        "tr.current { background: #eff6ff; }\n"
        "code { background: #f0f0f0; padding: 0.15em 0.35em; "
        "border-radius: 3px; font-size: 0.9em; }\n"
        ".samples-grid { display: flex; flex-wrap: wrap; gap: 1rem; "
        "margin-bottom: 2rem; }\n"
        ".sample-card { flex: 1 1 180px; max-width: 280px; text-align: center; }\n"
        ".sample-card img { width: 100%; border: 1px solid #ddd; "
        "border-radius: 4px; }\n"
        ".sample-label { font-size: 0.8rem; color: #555; margin-top: 0.25rem; }\n"
        ".footer { color: #999; font-size: 0.8rem; "
        "border-top: 1px solid #e5e5e5; padding-top: 1rem; margin-top: 2rem; }"
    )

    summary_line = (
        f"{timestamp} | <span class=\"status {status_class}\">{status}</span>"
        f" | {total} total | {passed} passed"
        f" | {failed} failed | {skipped} skipped"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{name} — Milestone Report</title>
<style>{css}</style>
</head>
<body>
<h1>{name} — Milestone Report</h1>
<p class="summary">{summary_line}</p>

<h2>Tests by Module</h2>
<table>
<thead><tr><th>Module</th><th>Pass</th><th>Fail</th><th>Skip</th><th>Status</th></tr></thead>
<tbody>{module_rows}</tbody>
</table>

{failed_section}
{perf_section}
{samples_section}

<h2>Milestone Trend</h2>
<table>
<thead><tr><th>Milestone</th><th>Total</th><th>Passed</th><th>Failed</th></tr></thead>
<tbody>{trend_rows}</tbody>
</table>

<div class="footer">Generated by <code>pytest --milestone-report {name}</code></div>
</body>
</html>"""
