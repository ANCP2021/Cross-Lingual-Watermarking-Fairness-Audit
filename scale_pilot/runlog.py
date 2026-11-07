#!/usr/bin/env python
import contextlib
import csv
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shim

EXCLUSIONS_PATH = os.path.join(shim.RESULTS_DIR, "exclusions.csv")
EXCLUSION_FIELDS = ["stage", "cell", "reason", "source"]


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message, command=None):
    print(f"[{_now()}] {message}" + (f"\n    $ {command}" if command else ""))


def metric(name, value, unit=""):
    print(f"[{_now()}] {name} = {value}{(' ' + unit) if unit else ''}")


def exclude(stage_name, cell, reason, source=""):
    os.makedirs(os.path.dirname(EXCLUSIONS_PATH) or ".", exist_ok=True)
    new = not os.path.exists(EXCLUSIONS_PATH)
    with open(EXCLUSIONS_PATH, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(EXCLUSION_FIELDS)
        w.writerow([stage_name, cell, reason, source])


@contextlib.contextmanager
def stage(name, command=None):
    log(f"START {name}", command)
    t0 = time.time()
    try:
        yield
    except Exception as e:
        dt = time.time() - t0
        log(f"FAILED {name} after {dt / 60:.1f} min — {type(e).__name__}: {e}")
        exclude(name, "(whole stage)", f"{type(e).__name__}: {e}", "runtime")
        raise
    dt = time.time() - t0
    log(f"END   {name} — {dt / 60:.1f} min ({dt:.0f} s)")


def selftest():
    import tempfile
    global EXCLUSIONS_PATH
    tmp = tempfile.mkdtemp(prefix="runlog_selftest_")
    EXCLUSIONS_PATH = os.path.join(tmp, "exclusions.csv")

    log("hello", command="python x.py --flag")
    metric("tokens_per_second", 412.5, "tok/s")
    exclude("detection", "XSIR/base/hin", "n_neg=143 < 200", "per_language_tpr.csv")
    try:
        with stage("boom"):
            raise RuntimeError("kaboom")
    except RuntimeError:
        pass

    with open(EXCLUSIONS_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["reason"] for r in rows][0] == "n_neg=143 < 200", rows
    assert any("kaboom" in r["reason"] and r["cell"] == "(whole stage)" for r in rows), rows
    print("  [ok] exclusions.csv gets one row per exclusion; stage() records failures")
    print("\nSelf-test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(selftest())
