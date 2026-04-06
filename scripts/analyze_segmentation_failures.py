#!/usr/bin/env python3
"""
Analyze why 21 slides failed segmentation with grandqc/hest/otsu.

Modes:
1. --scan-logs: Quick scan of existing TRIDENT logs for error snippets (no TRIDENT run).
2. Full run: For each slide, runs TRIDENT --task seg for each segmenter, records results.

Run on cluster (interactive GPU session or PBS):
  cd /storage/brno2/home/filipsec/MIL
  python scripts/analyze_segmentation_failures.py \\
    --missing-csv data/lists/missing_21_slides.csv --scan-logs  # quick, no GPU
  python scripts/analyze_segmentation_failures.py \\             # full diagnostic
    --missing-csv data/lists/missing_21_slides.csv \\
    --output data/analysis/segmentation_diagnosis.csv

Requires: conda env 'trident' with TRIDENT deps; GPU for grandqc/hest (otsu is CPU).
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


SEGMENTERS = ["grandqc", "hest", "otsu"]


def norm_id(raw: str) -> str:
    s = (raw or "").strip()
    for suf in (".tiff", ".tif"):
        if s.lower().endswith(suf):
            return s[: -len(suf)]
    return s


def check_wsi(wsi_path: Path) -> tuple[bool, str]:
    """Check if WSI exists and is readable (basic)."""
    if not wsi_path.exists():
        return False, "file_missing"
    if wsi_path.stat().st_size == 0:
        return False, "file_empty"
    return True, "ok"


def run_segmentation(
    trident_repo: Path,
    wsi_dir: Path,
    job_dir: Path,
    slide_id: str,
    wsi_name: str,
    segmenter: str,
    python_path: str | None = None,
) -> tuple[bool, str]:
    """Run TRIDENT --task seg for one slide + segmenter. Returns (success, message)."""
    job_dir.mkdir(parents=True, exist_ok=True)

    # Single-slide CSV for custom_list_of_wsis
    csv_path = job_dir / "single_slide.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wsi"])
        w.writerow([wsi_name])

    cmd = [
        sys.executable if python_path is None else python_path,
        str(trident_repo / "run_batch_of_slides.py"),
        "--task",
        "seg",
        "--wsi_dir",
        str(wsi_dir),
        "--job_dir",
        str(job_dir),
        "--custom_list_of_wsis",
        str(csv_path),
        "--wsi_ext",
        ".tiff",
        "--segmenter",
        segmenter,
        "--gpu",
        "0",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(trident_repo),
            capture_output=True,
            text=True,
            timeout=600,
        )
        out = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0:
            return True, "ok"
        # Extract last meaningful error (avoid full traceback)
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        err_snippet = "\n".join(lines[-5:]) if lines else "unknown"
        return False, err_snippet
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def scan_logs(logs_dir: Path, slide_ids: list[str]) -> list[dict]:
    """Scan TRIDENT logs for mentions of slide IDs and error patterns."""
    error_patterns = [
        r"no contour|No contour|NO CONTOUR",
        r"failed|Failure|Error|error",
        r"skip|Skip|SKIP",
        r"exception|Exception",
    ]
    results: list[dict] = []
    for sid in slide_ids:
        results.append({"slide_id": sid, "found_in_logs": "", "log_snippet": ""})

    if not logs_dir.exists():
        return results

    log_files = list(logs_dir.glob("trident_*.o")) + list(logs_dir.glob("trident_*.e"))
    for lf in sorted(log_files, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            text = lf.read_text(errors="replace")
        except Exception:
            continue
        for i, sid in enumerate(slide_ids):
            if sid not in text:
                continue
            # Extract context around first mention
            idx = text.find(sid)
            start = max(0, idx - 80)
            end = min(len(text), idx + len(sid) + 120)
            ctx = text[start:end].replace("\n", " ")
            snippet = re.sub(r"\s+", " ", ctx).strip()[:300]
            for pat in error_patterns:
                if re.search(pat, ctx, re.I):
                    results[i]["found_in_logs"] = lf.name
                    results[i]["log_snippet"] = snippet
                    break
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnose segmentation failures for missing slides."
    )
    ap.add_argument("--missing-csv", type=Path, required=True)
    ap.add_argument(
        "--scan-logs",
        action="store_true",
        help="Only scan existing TRIDENT logs for errors (no TRIDENT run)",
    )
    ap.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("/storage/brno2/home/filipsec/MIL/logs"),
        help="Dir with trident_*.o / trident_*.e logs",
    )
    ap.add_argument(
        "--wsi-dir",
        type=Path,
        default=Path("/storage/brno2/home/filipsec/MIL/data/raw/train_images"),
    )
    ap.add_argument(
        "--trident-repo",
        type=Path,
        default=Path("/storage/brno2/home/filipsec/repos/trident"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/analysis/segmentation_diagnosis.csv"),
    )
    ap.add_argument("--python", type=str, default=None, help="Path to trident env python")
    ap.add_argument(
        "--segmenters",
        nargs="+",
        default=SEGMENTERS,
        help=f"Segmenters to test (default: {SEGMENTERS})",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N slides (for quick test)",
    )
    args = ap.parse_args()

    if not args.missing_csv.exists():
        raise SystemExit(f"Missing CSV not found: {args.missing_csv}")

    ids_from_csv: list[str] = []
    with args.missing_csv.open(newline="") as f:
        r = csv.DictReader(f)
        col = "wsi" if r.fieldnames and "wsi" in (r.fieldnames or []) else (r.fieldnames[0] if r.fieldnames else None)
        if not col:
            raise SystemExit("No column found in CSV")
        for row in r:
            val = row.get(col, "").strip()
            if val:
                ids_from_csv.append(norm_id(val))

    if args.scan_logs:
        results = scan_logs(args.logs_dir, ids_from_csv)
        out_path = args.output.parent / "segmentation_log_scan.csv"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["slide_id", "found_in_logs", "log_snippet"])
            w.writeheader()
            w.writerows(results)
        print(f"Scan complete. Results: {out_path}")
        for r in results:
            if r["found_in_logs"]:
                print(f"  {r['slide_id']}: {r['found_in_logs']} -> {r['log_snippet'][:100]}...")
        return

    if not args.trident_repo.exists():
        raise SystemExit(f"TRIDENT repo not found: {args.trident_repo}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resolve Python
    py = args.python
    if not py:
        for d in [
            Path.home() / ".conda/envs/trident/bin/python",
            Path("/storage/brno2/home/filipsec/.conda/envs/trident/bin/python"),
        ]:
            if d.exists():
                py = str(d)
                break
    if not py:
        py = sys.executable

    if args.limit:
        ids_from_csv = ids_from_csv[: args.limit]

    segmenters = args.segmenters
    work_dir = args.output.parent / "seg_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, slide_id in enumerate(ids_from_csv):
        wsi_name = f"{slide_id}.tiff"
        wsi_path = args.wsi_dir / wsi_name
        wsi_ok, wsi_msg = check_wsi(wsi_path)

        row: dict = {
            "slide_id": slide_id,
            "wsi_exists": "yes" if wsi_ok else "no",
            "wsi_note": wsi_msg,
        }
        for seg in segmenters:
            row[f"{seg}_ok"] = ""
            row[f"{seg}_note"] = ""

        if not wsi_ok:
            for seg in segmenters:
                row[f"{seg}_ok"] = "skip"
                row[f"{seg}_note"] = f"wsi_{wsi_msg}"
            rows.append(row)
            print(f"[{i+1}/{len(ids_from_csv)}] {slide_id}: WSI {wsi_msg}, skipping segmenters")
            continue

        for seg in segmenters:
            seg_job = work_dir / f"{slide_id}_{seg}"
            ok, msg = run_segmentation(
                args.trident_repo,
                args.wsi_dir,
                seg_job,
                slide_id,
                wsi_name,
                seg,
                python_path=py,
            )
            row[f"{seg}_ok"] = "yes" if ok else "no"
            row[f"{seg}_note"] = msg[:200] if msg != "ok" else ""
            status = "OK" if ok else "FAIL"
            print(f"[{i+1}/{len(ids_from_csv)}] {slide_id} | {seg}: {status} {msg[:80] if not ok else ''}")
        rows.append(row)

    # Write report
    fieldnames = ["slide_id", "wsi_exists", "wsi_note"] + [
        f for seg in segmenters for f in (f"{seg}_ok", f"{seg}_note")
    ]
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nReport written to: {args.output}")
    # Summary
    any_ok = sum(1 for r in rows if any(r.get(f"{s}_ok") == "yes" for s in segmenters))
    all_fail = sum(1 for r in rows if all(r.get(f"{s}_ok") != "yes" for s in segmenters))
    print(f"Slides with at least one segmenter OK: {any_ok}")
    print(f"Slides where all segmenters failed: {all_fail}")


if __name__ == "__main__":
    main()
