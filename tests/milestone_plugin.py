"""Pytest plugin that generates visual milestone reports.

Usage:
    uv run pytest --milestone-report M0
    open reports/M0.html
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

    # --- Visual samples (before/after blurred images) ---
    samples = _generate_visual_samples()

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
    # Replace existing entry for this milestone, or append
    history = [h for h in history if h["name"] != name]
    history.append(milestone_data)
    history.sort(key=lambda h: h["name"])
    _save_history(history)

    # --- Render HTML ---
    html = _render_html(name, milestone_data, modules, perf, samples, history)
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
        from blur_daddy.blur import apply_elliptical_gaussian_blur, apply_rect_gaussian_blur, apply_rect_pixelation
        from blur_daddy.detection import detect_faces_mtcnn, detect_faces_yolo

        if not SAMPLE_IMAGE.exists():
            return perf

        img_bgr = cv2.imread(str(SAMPLE_IMAGE))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # YOLO detection
        t0 = time.perf_counter()
        boxes_y, confs_y, _ = detect_faces_yolo(img_rgb)
        perf["YOLO detect"] = round(time.perf_counter() - t0, 3)

        # MTCNN detection
        t0 = time.perf_counter()
        boxes_m, _, landmarks_m = detect_faces_mtcnn(img_rgb)
        perf["MTCNN detect"] = round(time.perf_counter() - t0, 3)

        # Use YOLO boxes for blur benchmarks (fall back to MTCNN)
        boxes = boxes_y if boxes_y else (boxes_m.tolist() if boxes_m is not None else [[50, 50, 200, 200]])

        t0 = time.perf_counter()
        apply_rect_gaussian_blur(img_bgr.copy(), boxes)
        perf["Gaussian blur"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        apply_rect_pixelation(img_bgr.copy(), boxes)
        perf["Pixelation"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        apply_elliptical_gaussian_blur(img_bgr.copy(), boxes, landmarks_m)
        perf["Elliptical blur"] = round(time.perf_counter() - t0, 3)

    except Exception as exc:
        perf["error"] = str(exc)

    return perf


# ---------------------------------------------------------------------------
# Visual samples
# ---------------------------------------------------------------------------

def _generate_visual_samples():
    """Run the blur pipeline on the sample image and return base64-encoded PNGs."""
    samples = {}
    try:
        from blur_daddy.blur import apply_elliptical_gaussian_blur, apply_rect_gaussian_blur, apply_rect_pixelation
        from blur_daddy.detection import detect_faces_mtcnn, detect_faces_yolo

        if not SAMPLE_IMAGE.exists():
            return samples

        img_bgr = cv2.imread(str(SAMPLE_IMAGE))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes_y, _, _ = detect_faces_yolo(img_rgb)
        boxes_m, _, landmarks_m = detect_faces_mtcnn(img_rgb)
        boxes = boxes_y if boxes_y else (boxes_m.tolist() if boxes_m is not None else None)

        if boxes is None:
            return samples

        # Resize for report (max 600px wide)
        scale = min(1.0, 600.0 / img_bgr.shape[1])
        def _resize(img):
            if scale < 1.0:
                return cv2.resize(img, None, fx=scale, fy=scale)
            return img

        samples["Original"] = _img_to_b64(_resize(img_bgr))

        # Draw detection boxes on original
        annotated = img_bgr.copy()
        for box in boxes:
            x1, y1, x2, y2 = [int(c) for c in box]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        samples["Detected faces"] = _img_to_b64(_resize(annotated))

        # Gaussian
        samples["Gaussian blur"] = _img_to_b64(_resize(apply_rect_gaussian_blur(img_bgr.copy(), boxes)))

        # Pixelation
        samples["Pixelation"] = _img_to_b64(_resize(apply_rect_pixelation(img_bgr.copy(), boxes)))

        # Elliptical
        samples["Elliptical blur"] = _img_to_b64(
            _resize(apply_elliptical_gaussian_blur(img_bgr.copy(), boxes, landmarks_m))
        )

    except Exception:
        pass

    return samples


def _img_to_b64(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("ascii")


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
# HTML rendering (self-contained, no Jinja2 dependency)
# ---------------------------------------------------------------------------

def _render_html(name, data, modules, perf, samples, history):
    passed, failed, skipped, total = data["passed"], data["failed"], data["skipped"], data["total"]
    status_color = "#22c55e" if failed == 0 else "#ef4444"
    status_text = "ALL PASSING" if failed == 0 else f"{failed} FAILING"

    # --- Module rows ---
    module_rows = ""
    for mod_name, mod in sorted(modules.items()):
        p, f, s = mod["passed"], mod["failed"], mod["skipped"]
        bar_w = 100
        pw = int(p / max(p + f + s, 1) * bar_w)
        fw = int(f / max(p + f + s, 1) * bar_w)
        sw = bar_w - pw - fw
        icon = "\u2705" if f == 0 else "\u274c"
        module_rows += f"""
        <tr>
            <td>{icon} {mod_name}</td>
            <td>{p}</td><td>{f}</td><td>{s}</td>
            <td>
                <div class="bar">
                    <div class="bar-pass" style="width:{pw}%"></div>
                    <div class="bar-fail" style="width:{fw}%"></div>
                    <div class="bar-skip" style="width:{sw}%"></div>
                </div>
            </td>
        </tr>"""

    # --- Failed test details ---
    failed_section = ""
    if failed > 0:
        failed_tests = [r for r in data.get("_results", []) if not r["passed"] and not r["skipped"]]
        if failed_tests:
            rows = "".join(f"<li><code>{t['nodeid']}</code></li>" for t in failed_tests)
            failed_section = f'<div class="card"><h2>Failed Tests</h2><ul>{rows}</ul></div>'

    # --- Perf table ---
    perf_rows = ""
    for op, secs in perf.items():
        if op == "error":
            continue
        bar_max = max(perf.values()) if perf else 1
        pct = secs / bar_max * 100 if bar_max else 0
        perf_rows += f"""
        <tr>
            <td>{op}</td>
            <td>{secs:.3f}s</td>
            <td><div class="perf-bar" style="width:{pct}%"></div></td>
        </tr>"""

    # --- Visual samples ---
    sample_cards = ""
    for label, b64 in samples.items():
        sample_cards += f"""
        <div class="sample-card">
            <img src="data:image/png;base64,{b64}" alt="{label}">
            <div class="sample-label">{label}</div>
        </div>"""

    # --- History trend (simple ASCII-style bar chart in HTML) ---
    trend_rows = ""
    max_tests = max((h["total"] for h in history), default=1)
    arrow = " \u2190 current"
    for h in history:
        is_current = h["name"] == name
        pct = h["passed"] / max(max_tests, 1) * 100
        fpct = h["failed"] / max(max_tests, 1) * 100
        bold = "font-weight:700;" if is_current else ""
        current_marker = arrow if is_current else ""
        trend_rows += f"""
        <tr style="{bold}">
            <td>{h['name']}{current_marker}</td>
            <td>{h['total']}</td>
            <td>{h['passed']}</td>
            <td>{h['failed']}</td>
            <td>
                <div class="bar">
                    <div class="bar-pass" style="width:{pct}%"></div>
                    <div class="bar-fail" style="width:{fpct}%"></div>
                </div>
            </td>
        </tr>"""

    timestamp = datetime.fromisoformat(data["timestamp"]).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blur Daddy — {name} Report</title>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8;
    --green: #22c55e; --red: #ef4444; --yellow: #eab308; --blue: #3b82f6;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
         background: var(--bg); color: var(--text); padding: 2rem; line-height: 1.6; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.1rem; margin-bottom: 1rem; color: var(--muted); font-weight: 500; }}
  .header {{ text-align: center; margin-bottom: 2rem; }}
  .header .status {{ display: inline-block; padding: 0.25rem 1rem; border-radius: 999px;
                     font-size: 0.85rem; font-weight: 700; letter-spacing: 0.05em;
                     background: {status_color}22; color: {status_color}; border: 1px solid {status_color}44; }}
  .header .meta {{ color: var(--muted); font-size: 0.8rem; margin-top: 0.5rem; }}
  .stats {{ display: flex; gap: 1rem; justify-content: center; margin: 1.5rem 0; flex-wrap: wrap; }}
  .stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
           padding: 1rem 1.5rem; text-align: center; min-width: 120px; }}
  .stat .num {{ font-size: 2rem; font-weight: 700; }}
  .stat .label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
           padding: 1.5rem; margin-bottom: 1.5rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ text-align: left; color: var(--muted); font-weight: 500; padding: 0.5rem 0.75rem;
       border-bottom: 1px solid var(--border); font-size: 0.75rem; text-transform: uppercase;
       letter-spacing: 0.05em; }}
  td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border)22; }}
  .bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--border); }}
  .bar-pass {{ background: var(--green); }}
  .bar-fail {{ background: var(--red); }}
  .bar-skip {{ background: var(--yellow); }}
  .perf-bar {{ height: 8px; border-radius: 4px; background: var(--blue); min-width: 2px; }}
  .samples {{ display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center; }}
  .sample-card {{ background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
                  overflow: hidden; max-width: 320px; }}
  .sample-card img {{ width: 100%; display: block; }}
  .sample-label {{ padding: 0.5rem; text-align: center; font-size: 0.8rem; color: var(--muted); }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
  @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  a {{ color: var(--blue); }}
</style>
</head>
<body>

<div class="header">
    <h1>blur-daddy / {name}</h1>
    <div class="meta">{timestamp}</div>
    <div style="margin-top:0.75rem"><span class="status">{status_text}</span></div>
</div>

<div class="stats">
    <div class="stat"><div class="num" style="color:var(--text)">{total}</div><div class="label">Total</div></div>
    <div class="stat"><div class="num" style="color:var(--green)">{passed}</div><div class="label">Passed</div></div>
    <div class="stat"><div class="num" style="color:var(--red)">{failed}</div><div class="label">Failed</div></div>
    <div class="stat"><div class="num" style="color:var(--yellow)">{skipped}</div><div class="label">Skipped</div></div>
</div>

{failed_section}

<div class="grid">
<div class="card">
    <h2>Tests by Module</h2>
    <table>
        <tr><th>Module</th><th>Pass</th><th>Fail</th><th>Skip</th><th>Distribution</th></tr>
        {module_rows}
    </table>
</div>

<div class="card">
    <h2>Performance (seconds)</h2>
    <table>
        <tr><th>Operation</th><th>Time</th><th></th></tr>
        {perf_rows}
    </table>
</div>
</div>

<div class="card">
    <h2>Visual Output Samples</h2>
    <div class="samples">
        {sample_cards if sample_cards else '<p style="color:var(--muted)">No sample images available.</p>'}
    </div>
</div>

<div class="card">
    <h2>Milestone Trend</h2>
    <table>
        <tr><th>Milestone</th><th>Total</th><th>Pass</th><th>Fail</th><th>Trend</th></tr>
        {trend_rows}
    </table>
</div>

<div style="text-align:center; margin-top:2rem; color:var(--muted); font-size:0.75rem;">
    Generated by <code>pytest --milestone-report {name}</code>
</div>

</body>
</html>"""
