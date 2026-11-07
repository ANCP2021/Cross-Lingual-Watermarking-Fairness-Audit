#!/usr/bin/env python
"""Qwen2.5-72B scale pilot — GE_2 exact shape-matched null tests."""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog
import shim

sys.path.insert(0, os.path.join(shim.REPO_ROOT, "family_sensitivity"))
import joint_family_test as joint_test
import ontarget_family_test as ontarget_test
import percell_family_test as percell_test


def concat_csv(paths, out_path):
    header, rows = None, []
    used = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            r = csv.reader(f)
            try:
                h = next(r)
            except StopIteration:
                continue
            if header is None:
                header = h
            elif h != header:
                raise SystemExit(
                    f"ERROR: schema mismatch between {used[0]} and {p}.\n"
                    f"  {used[0]}: {header}\n  {p}: {h}\n"
                    f"Both should come from the same pipeline script; do not merge by hand.")
            rows.extend(list(r))
            used.append(p)
    if header is None:
        raise SystemExit(f"ERROR: none of {paths} exist — cannot build {out_path}.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return len(rows), used


def filter_percell(in_path, out_path, generator, require_full_grid=False,
                   n_langs_full=11):
    kept, dropped = [], []
    with open(in_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for r in reader:
            if r.get("generator") != generator:
                continue
            if require_full_grid:
                try:
                    n = int(float(r.get("n_langs") or 0))
                except ValueError:
                    n = 0
                if n != n_langs_full:
                    dropped.append((r, n))
                    continue
            kept.append(r)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    return kept, dropped, fields


def flagships():
    out = []
    for metric in ("detection_tpr", "mauve"):
        for scheme, regime in shim.cells():
            out.append((metric, regime, scheme, shim.ALIAS))
    return out


def run_percell(tpr_csv, quality_csv, detection_summary_csv, quality_summary_csv, subset, output_dir, floor):
    saved = (percell_test.PUBLISHED_ANCHORS, percell_test.ANCHOR_KIND, percell_test.FLAGSHIP_CELLS)
    if subset == "all":
        percell_test.ANCHOR_KIND = "ge2_share"
    else:
        percell_test.PUBLISHED_ANCHORS = ontarget_test.ON_TARGET_ANCHORS
        percell_test.ANCHOR_KIND = "mean_tpr"
        percell_test.ANCHOR_ATOL_MEAN = 0.002
    percell_test.FLAGSHIP_CELLS = flagships()
    try:
        return percell_test.run(percell_test.parse_args([
            "--tpr_csv", tpr_csv, "--quality_csv", quality_csv,
            "--detection_summary_csv", detection_summary_csv, "--quality_summary_csv", quality_summary_csv,
            "--output_dir", output_dir, "--subset", subset,
            "--alpha", str(shim.ALPHA_HEADLINE),
            "--threshold_type", shim.THRESHOLD_TYPE, "--floor", str(floor),
        ]))
    finally:
        percell_test.PUBLISHED_ANCHORS, percell_test.ANCHOR_KIND, percell_test.FLAGSHIP_CELLS = saved


def run(args):
    for d in (shim.DERIVED_DIR, shim.RESULTS_DIR, shim.ANALYSIS_ALL,
              shim.ANALYSIS_OT, shim.ANALYSIS_JOINT):
        os.makedirs(d, exist_ok=True)

    tpr_combined = os.path.join(shim.DERIVED_DIR, "per_language_tpr_combined.csv")
    quality_combined = os.path.join(shim.DERIVED_DIR, "quality_per_language_combined.csv")
    n_tpr, used_tpr = concat_csv([shim.PAPER["tpr"], shim.OUT["tpr"]], tpr_combined)
    n_quality, used_quality = concat_csv([shim.PAPER["quality"], shim.OUT["quality"]], quality_combined)
    print(f"Derived inputs:\n  {tpr_combined}  ({n_tpr} rows from {len(used_tpr)} file(s))"
          f"\n  {quality_combined}  ({n_quality} rows from {len(used_quality)} file(s))")

    if shim.PAPER["tpr"] not in used_tpr:
        print("\nERROR: the 7B results/detection/per_language_tpr.csv is absent, so the "
              "anchor gate cannot validate against published numbers.\n"
              "       Provide results/detection/per_language_tpr.csv and re-run.",
              file=sys.stderr)
        return 1

    legs = [("all", shim.ANALYSIS_ALL), ("on_target_only", shim.ANALYSIS_OT)]
    percell_paths = {}
    for subset, out_dir in legs:
        print(f"\n=== per-cell exact null — subset={subset} ===")
        with runlog.stage(f"percell_family_test subset={subset}",
                          command=f"family_sensitivity/percell_family_test.py --subset {subset} "
                                  f"--output_dir {out_dir} (anchors re-pointed)"):
            rc = run_percell(tpr_combined, quality_combined, shim.PAPER["detection_summary"], shim.PAPER["quality_summary"],
                             subset, out_dir, args.floor)
        if rc != 0:
            print(f"\nPer-cell leg subset={subset} FAILED its validation gate — "
                  f"nothing downstream is written ({out_dir}).",
                  file=sys.stderr)
            runlog.exclude("ge_null", f"subset={subset}",
                           "anchor validation failed", "percell_family_test")
            return rc
        percell_paths[subset] = os.path.join(out_dir, "percell_results.csv")

    joint_inputs = {}
    for subset, path in percell_paths.items():
        all72 = os.path.join(shim.DERIVED_DIR, f"percell_{subset}_qwen72b.csv")
        kept, _, _ = filter_percell(path, all72, shim.ALIAS)
        joint_path = os.path.join(shim.DERIVED_DIR, f"percell_{subset}_qwen72b_full.csv")
        kept_full, dropped, _ = filter_percell(path, joint_path, shim.ALIAS,
                                               require_full_grid=True)
        joint_inputs[subset] = joint_path
        print(f"  subset={subset}: {len(kept)} 72B cells, "
              f"{len(kept_full)} with all 11 languages")
        for r, n in dropped:
            cell = f"{r['metric']}/{r['regime']}/{r['scheme']}"
            runlog.exclude("ge_null joint test", f"{cell} ({subset})",
                           f"n_langs={n} != 11 — a joint relabeling needs a shared "
                           f"language set (gate 4); per-cell null retained",
                           "percell_results.csv")
            print(f"    dropped from the JOINT test: {cell} (n_langs={n}) "
                  f"— per-cell null kept")

    if not any(os.path.getsize(p) > 0 for p in joint_inputs.values()):
        print("\nNo 72B cell survives to the joint test.", file=sys.stderr)
        return 1

    print("\n=== joint grid test (strata: detection/mauve x base/instruct) ===")
    with runlog.stage("joint_family_test",
                      command=f"family_sensitivity/joint_family_test.py --percell_all "
                              f"{joint_inputs['all']} --percell_ontarget "
                              f"{joint_inputs['on_target_only']}"):
        rc = joint_test.run(joint_test.parse_args([
            "--tpr_csv", tpr_combined, "--quality_csv", quality_combined,
            "--percell_all", joint_inputs["all"],
            "--percell_ontarget", joint_inputs["on_target_only"],
            "--output_dir", shim.ANALYSIS_JOINT,
            "--alpha", str(shim.ALPHA_HEADLINE),
            "--threshold_type", shim.THRESHOLD_TYPE,
        ]))
    if rc != 0:
        return rc

    src = os.path.join(shim.ANALYSIS_JOINT, "joint_results.csv")
    dst = os.path.join(shim.RESULTS_DIR, "null_test_joint.csv")
    if os.path.exists(src):
        with open(src, encoding="utf-8") as f_in, \
                open(dst, "w", encoding="utf-8", newline="") as f_out:
            f_out.write(f_in.read())
        print(f"\nWrote {dst}")

    for subset, out_dir in legs:
        src = os.path.join(shim.DERIVED_DIR, f"percell_{subset}_qwen72b.csv")
        dst = os.path.join(shim.RESULTS_DIR, f"null_test_percell_{subset}.csv")
        if os.path.exists(src):
            with open(src, encoding="utf-8") as f_in, \
                    open(dst, "w", encoding="utf-8", newline="") as f_out:
                f_out.write(f_in.read())
            print(f"Wrote {dst}")
    runlog.log("ge_null complete: per-cell (both subsets) + joint stratum test")
    return 0


def selftest():
    import tempfile
    print("Self-test: synthetic fixtures, no real data touched.")
    langs = list(percell_test.LANGUAGES)
    tmp = tempfile.mkdtemp(prefix="genull_selftest_")

    def vec(seed, shift=0.0):
        return {l: min(0.99, max(0.01, 0.12 + 0.07 * ((i * seed) % 9) + shift))
                for i, l in enumerate(langs)}

    paper_cells = [("qwen", "base", "KGW"), ("qwen", "instruct", "KGW")]
    pilot_cells = [(shim.ALIAS, regime, scheme) for scheme, regime in shim.cells()]
    vectors = {c: vec(i + 2) for i, c in enumerate(paper_cells + pilot_cells)}
    SHORT = (shim.ALIAS, "instruct", "XSIR")

    def write_tpr_fixture(path, cells, subset_list):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "variant", "watermark", "dataset", "subset", "alpha",
                        "threshold_type", "language", "tpr", "insufficient_samples"])
            for c in cells:
                model, regime, scheme = c
                ds = "flores" if regime == "base" else "aya"
                for sub in subset_list:
                    for l in langs:
                        if c == SHORT and l == "hin":
                            w.writerow([model, regime, scheme, ds, sub, "0.01",
                                        "global", l, "", "True"])
                            continue
                        w.writerow([model, regime, scheme, ds, sub, "0.01", "global",
                                    l, f"{vectors[c][l]:.6f}", "False"])

    def write_quality_fixture(path, cells, subset_list):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "variant", "watermark", "subset", "measurement",
                        "language", "family", "preservation_score"])
            for c in cells:
                model, regime, scheme = c
                for sub in subset_list:
                    for l in langs:
                        w.writerow([model, regime, scheme, sub, "mauve", l,
                                    percell_test.LANGUAGE_FAMILIES[l],
                                    f"{0.35 + vectors[c][l] / 2:.6f}"])

    paper_tpr = os.path.join(tmp, "paper_tpr.csv")
    paper_quality = os.path.join(tmp, "paper_quality.csv")
    pilot_tpr = os.path.join(tmp, "pilot_tpr.csv")
    pilot_quality = os.path.join(tmp, "pilot_quality.csv")
    subsets = ["all", "on_target_only"]
    write_tpr_fixture(paper_tpr, paper_cells, subsets)
    write_quality_fixture(paper_quality, paper_cells, subsets)
    write_tpr_fixture(pilot_tpr, pilot_cells, subsets)
    write_quality_fixture(pilot_quality, pilot_cells, subsets)

    comb = os.path.join(tmp, "combined_tpr.csv")
    n, used = concat_csv([paper_tpr, pilot_tpr], comb)
    assert len(used) == 2 and n == sum(
        sum(1 for _ in open(p, encoding="utf-8")) - 1 for p in (paper_tpr, pilot_tpr))
    print(f"  [ok] concat_csv merged {len(used)} files under one header ({n} rows)")
    bad = os.path.join(tmp, "bad.csv")
    with open(bad, "w", encoding="utf-8") as f:
        f.write("a,b\n1,2\n")
    try:
        concat_csv([paper_tpr, bad], os.path.join(tmp, "x.csv"))
        raise AssertionError("schema mismatch not detected")
    except SystemExit:
        print("  [ok] concat_csv refuses to merge mismatched schemas")

    saved_anchors, saved_ontarget = percell_test.PUBLISHED_ANCHORS, ontarget_test.ON_TARGET_ANCHORS
    saved_dirs = (shim.DERIVED_DIR, shim.ANALYSIS_ALL, shim.ANALYSIS_OT,
                  shim.ANALYSIS_JOINT, shim.RESULTS_DIR)
    saved_paper, saved_out = dict(shim.PAPER), dict(shim.OUT)
    saved_exc = runlog.EXCLUSIONS_PATH
    try:
        ge2, _b, _w, share = percell_test.decompose(vectors[("qwen", "base", "KGW")],
                                         percell_test.LANGUAGE_FAMILIES)
        percell_test.PUBLISHED_ANCHORS = [("detection_tpr", "base", "KGW", "qwen",
                                round(ge2, 3), round(share, 3))]
        v = vectors[("qwen", "instruct", "KGW")]
        ontarget_test.ON_TARGET_ANCHORS = [("detection_tpr", "instruct", "KGW", "qwen",
                                 round(sum(v.values()) / len(v), 3))]

        shim.DERIVED_DIR = os.path.join(tmp, "derived")
        shim.ANALYSIS_ALL = os.path.join(tmp, "an_all")
        shim.ANALYSIS_OT = os.path.join(tmp, "an_ot")
        shim.ANALYSIS_JOINT = os.path.join(tmp, "an_joint")
        shim.RESULTS_DIR = os.path.join(tmp, "results")
        shim.PAPER["tpr"], shim.PAPER["quality"] = paper_tpr, paper_quality
        shim.PAPER["detection_summary"] = shim.PAPER["quality_summary"] = os.path.join(tmp, "absent.csv")
        shim.OUT["tpr"], shim.OUT["quality"] = pilot_tpr, pilot_quality
        runlog.EXCLUSIONS_PATH = os.path.join(tmp, "exclusions.csv")
        for d in (shim.DERIVED_DIR, shim.RESULTS_DIR):
            os.makedirs(d, exist_ok=True)

        rc = run(argparse.Namespace(floor=1e-3))
        assert rc == 0, rc

        with open(os.path.join(shim.RESULTS_DIR, "null_test_joint.csv"),
                  encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows, "no joint rows"
        for r in rows:
            if r["n_cells"] and r["n_cells"] != "0":
                assert abs(float(r["null_mean"]) - 0.7) < 1e-9, r
                assert int(r["n_assignments"]) == 4620, r
        strata = {r["stratum"] for r in rows if r["n_cells"] not in ("", "0")}
        print(f"  [ok] joint test ran on 72B cells only; every null centred on "
              f"0.700 with 4620 assignments; strata {sorted(strata)}")

        with open(os.path.join(shim.DERIVED_DIR,
                               "percell_all_qwen72b_full.csv"), encoding="utf-8") as f:
            full = list(csv.DictReader(f))
        assert not any(r["scheme"] == "XSIR" and r["regime"] == "instruct"
                       and r["metric"] == "detection_tpr" for r in full)
        with open(os.path.join(shim.RESULTS_DIR, "null_test_percell_all.csv"),
                  encoding="utf-8") as f:
            allcells = list(csv.DictReader(f))
        short = next(r for r in allcells if r["scheme"] == "XSIR"
                     and r["regime"] == "instruct" and r["metric"] == "detection_tpr")
        assert short["n_langs"] == "10" and short["n_assignments"] != "4620"
        assert all(r["generator"] == shim.ALIAS for r in allcells)
        exc = open(runlog.EXCLUSIONS_PATH, encoding="utf-8").read()
        assert "n_langs=10 != 11" in exc
        print(f"  [ok] 10-language cell dropped from the joint test (logged), "
              f"per-cell null kept with {short['n_assignments']} exact assignments")

        percell_test.PUBLISHED_ANCHORS = [("detection_tpr", "base", "KGW", "qwen", 9.99, 0.5)]
        rc_bad = run(argparse.Namespace(floor=1e-3))
        assert rc_bad == 1, rc_bad
        print("  [ok] wrong anchors stop the run (exit 1), gate not bypassed")
    finally:
        percell_test.PUBLISHED_ANCHORS, ontarget_test.ON_TARGET_ANCHORS = saved_anchors, saved_ontarget
        (shim.DERIVED_DIR, shim.ANALYSIS_ALL, shim.ANALYSIS_OT,
         shim.ANALYSIS_JOINT, shim.RESULTS_DIR) = saved_dirs
        shim.PAPER.update(saved_paper)
        shim.OUT.update(saved_out)
        runlog.EXCLUSIONS_PATH = saved_exc
    print("\nSelf-test PASSED.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--floor", type=float, default=1e-3,
                   help="GE_2 floor for headline inclusion (default 1e-3)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    return selftest() if args.selftest else run(args)


if __name__ == "__main__":
    sys.exit(main())
