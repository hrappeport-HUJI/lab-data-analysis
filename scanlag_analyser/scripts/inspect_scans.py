"""Report what a ScanLag scan folder holds, before anything is processed.

    python scripts/inspect_scans.py <scan folder>

Per scanner (as the pipeline groups them): the file name prefix, number of scans, where time zero
comes from (the robot log's "Starting." line, or the first scan), the first and last scan, the
duration, the usual interval, and any gap longer than 1.5 intervals (scans missing). A pause right
after the first scan is the usual setup (a background scan of the fresh plates, then the regular
scans once colonies can appear) and is reported as such, not as a gap. Also lists the robot logs,
and files that look like scans but don't follow the naming rule.
Nothing is written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from scanlag.cli import split_prompt_args
from scanlag.pipeline.scans import SCAN_NAME, SKIP_DIRS, find_scanners


def main() -> None:
    args = split_prompt_args(sys.argv[1:])
    if len(args) != 1:
        sys.exit(__doc__)
    root = Path(args[0]).resolve()
    if not root.is_dir():
        sys.exit(f"{root} is not a folder")
    try:
        scanners = find_scanners(root)
    except FileNotFoundError as e:
        sys.exit(str(e))
    print(f"{root}: {len(scanners)} scanner(s)")
    for s in scanners:
        minutes = np.asarray(s.times_min)
        steps = np.diff(minutes)
        step = float(np.median(steps)) if len(steps) else float("nan")
        step = float(np.median(steps[1:])) if len(steps) > 1 else step  # the background pause doesn't count
        gaps = [(i, s.scans[i].name, s.scans[i + 1].name, d) for i, d in enumerate(steps) if d > 1.5 * step]
        print(f"\n{s.name}: prefix {s.prefix!r}, {len(s.scans)} scans in {s.directory.relative_to(root.parent)}")
        print(f"  time zero: {s.start:%Y-%m-%d %H:%M} (from {s.time_zero})")
        when = "at time zero" if abs(minutes[0]) < 1.5 else f"{minutes[0]:.0f} min after time zero"
        print(f"  first scan (background): {s.scans[0].name}, {when}")
        print(f"  last scan: {s.scans[-1].name}, {minutes[-1] / 60:.1f} h after time zero")
        print(f"  usual interval: {step:.0f} min")
        if s.empty:
            print(f"  WARNING: {len(s.empty)} empty (0-byte) scan(s) left out: " + ", ".join(p.name for p in s.empty))
        for i, a, b, d in gaps:
            if i == 0:
                print(f"  background pause: regular scans start {d:.0f} min after the background scan, with {b}")
            else:
                print(f"  GAP: {d:.0f} min between {a} and {b} (about {round(d / step) - 1} scans missing)")
    logs = sorted(root.rglob("run_*.log"))
    print("\nrobot logs: " + (", ".join(str(p.relative_to(root)) for p in logs) or "none"))
    odd = [p for p in root.rglob("*") if p.suffix.lower() in (".tif", ".tiff") and not SCAN_NAME.match(p.name)
           and not SKIP_DIRS & set(p.relative_to(root).parts[:-1])]
    if odd:
        print(f"WARNING: {len(odd)} image file(s) not named <prefix>_<scanner>_<YYYYMMDD>_<HHMM>.tif[f], ignored: "
              + ", ".join(p.name for p in odd[:5]) + (" ..." if len(odd) > 5 else ""))


if __name__ == "__main__":
    main()
