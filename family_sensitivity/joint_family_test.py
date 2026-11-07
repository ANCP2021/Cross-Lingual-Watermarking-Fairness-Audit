#!/usr/bin/env python
"""Grid-level joint exact test for the typological family labeling."""

import argparse
import csv
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "family_sensitivity"))

import percell_family_test as percell_test

NAN = float("nan")
TIE_TOL = 1e-12
IDENTITY_TOL = 1e-9
NULL_MEAN_TOL = 1e-9
EXPECTED_NULL_MEAN = 0.7

SUBSETS = ("all", "on_target_only")

STRATA = (
    ("detection-base",     "detection_tpr", "base",     True),
    ("detection-instruct", "detection_tpr", "instruct", True),
    ("mauve-base",         "mauve",         "base",     True),
    ("mauve-instruct",     "mauve",         "instruct", True),
    ("detection-pooled",   "detection_tpr", None,       False),
    ("mauve-pooled",       "mauve",         None,       False),
)

JOINT_FIELDS = [
    "subset", "stratum", "tier", "n_cells", "T_observed",
    "null_mean", "null_median", "null_p95", "exact_p", "exact_p_count",
    "holm_p", "n_cells_above_own_null_median", "n_assignments",
]


def load_percell(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["metric"], r["regime"], r["scheme"], r["generator"])] = r
    return out


def fnum(row, col):
    try:
        v = float(row.get(col, ""))
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def is_above_floor(row):
    return str(row.get("above_floor", "")).strip().lower() == "true"


def holm(pvals):
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def load_vectors(tpr_csv, quality_csv, subset, alpha, threshold_type):
    vecs = {}
    vecs.update(percell_test.load_detection_vectors(tpr_csv, subset, alpha, threshold_type))
    vecs.update(percell_test.load_mauve_vectors(quality_csv, subset))
    return {k: {l: x for l, x in v.items() if l in percell_test.LANGUAGE_FAMILIES}
            for k, v in vecs.items()}


def analyse_subset(subset, vectors, percell, assignments, identity_index, problems):
    langs = list(percell_test.LANGUAGES)

    members = {k: r for k, r in percell.items() if is_above_floor(r)}

    share_vec = {}
    max_identity_dev = 0.0
    for key in sorted(members, key=percell_test.cell_sort_key):
        v = vectors.get(key)
        if not v:
            problems.append(f"[{subset}] {'/'.join(key)} is above_floor in "
                            f"percell_results.csv but has no per-language vector")
            continue
        if len(v) != len(langs):
            problems.append(f"[{subset}] {'/'.join(key)} has {len(v)} languages, "
                            f"expected {len(langs)} — a joint relabeling needs a "
                            f"shared language set (gate 4)")
            continue
        shares = [percell_test.decompose(v, fam)[3] for fam in assignments]
        if not all(percell_test.isnum(s) for s in shares):
            problems.append(f"[{subset}] {'/'.join(key)} produced a non-finite share "
                            f"under some relabeling")
            continue
        recorded = fnum(members[key], "observed_between_share")
        dev = abs(shares[identity_index] - recorded) if recorded is not None else None
        if dev is None:
            problems.append(f"[{subset}] {'/'.join(key)} has no recorded "
                            f"observed_between_share to chain against (gate 1)")
        else:
            max_identity_dev = max(max_identity_dev, dev)
            if dev > IDENTITY_TOL:
                problems.append(f"[{subset}] {'/'.join(key)} identity share {shares[identity_index]:.12g} "
                                f"!= recorded {recorded:.12g} (Δ={dev:.2g} > {IDENTITY_TOL:g}, gate 1)")
        share_vec[key] = shares

    rows = []
    nulls = {}
    for name, metric, regime, primary in STRATA:
        cells = [k for k in sorted(share_vec, key=percell_test.cell_sort_key)
                 if k[0] == metric and (regime is None or k[1] == regime)]
        if not cells:
            rows.append({"subset": subset, "stratum": name,
                         "tier": "primary" if primary else "secondary",
                         "n_cells": 0, "n_assignments": len(assignments)})
            continue

        n = len(cells)
        joint = [sum(share_vec[k][j] for k in cells) / n for j in range(len(assignments))]
        t_obs = sum(share_vec[k][identity_index] for k in cells) / n

        if abs(joint[identity_index] - t_obs) > 1e-12:
            problems.append(f"[{subset}/{name}] identity entry of the joint null "
                            f"{joint[identity_index]:.12g} != T(identity) {t_obs:.12g} (gate 2)")
        null_mean = sum(joint) / len(joint)
        if abs(null_mean - EXPECTED_NULL_MEAN) > NULL_MEAN_TOL:
            problems.append(f"[{subset}/{name}] joint null mean {null_mean:.12f} != "
                            f"{EXPECTED_NULL_MEAN} (Δ={abs(null_mean - EXPECTED_NULL_MEAN):.2g} "
                            f"> {NULL_MEAN_TOL:g}, gate 3)")

        n_ge = sum(1 for t in joint if t >= t_obs - TIE_TOL)
        above_own = sum(1 for k in cells
                        if (fnum(percell[k], "observed_between_share") or -1)
                        > (fnum(percell[k], "null_median") or 2))

        rows.append({
            "subset": subset, "stratum": name,
            "tier": "primary" if primary else "secondary",
            "n_cells": n, "T_observed": t_obs,
            "null_mean": null_mean,
            "null_median": percell_test.percentile(joint, 0.5),
            "null_p95": percell_test.percentile(joint, 0.95),
            "exact_p": n_ge / len(assignments), "exact_p_count": n_ge,
            "n_cells_above_own_null_median": above_own,
            "n_assignments": len(assignments),
        })
        if primary:
            nulls[name] = joint

    prim = [r for r in rows if r["tier"] == "primary" and r.get("exact_p") is not None]
    adj = holm([r["exact_p"] for r in prim])
    for r, a in zip(prim, adj):
        r["holm_p"] = a

    return rows, nulls, max_identity_dev


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tpr_csv", default="results/detection/per_language_tpr.csv")
    p.add_argument("--quality_csv", default="results/quality/quality_per_language.csv")
    p.add_argument("--percell_all",
                   default="analysis/percell_family_test/percell_results.csv")
    p.add_argument("--percell_ontarget",
                   default="analysis/ontarget_family_test/percell_results.csv")
    p.add_argument("--output_dir", default="analysis/joint_family_test")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--threshold_type", default="global")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args(argv)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    null_dir = os.path.join(args.output_dir, "null_distributions")
    os.makedirs(null_dir, exist_ok=True)

    langs = list(percell_test.LANGUAGES)
    assignments = percell_test.shape_matched_assignments(langs, percell_test.registered_shape(langs))
    identity_index = next((i for i, fam in enumerate(assignments)
                           if percell_test.same_grouping(fam, percell_test.LANGUAGE_FAMILIES, langs)), -1)
    if identity_index < 0:
        sys.exit("ERROR: the registered labeling is not present in the enumeration.")
    print(f"Enumerated {len(assignments)} shape-preserving relabelings "
          f"(identity at index {identity_index})")

    percell_paths = {"all": args.percell_all, "on_target_only": args.percell_ontarget}
    all_rows, problems = [], []

    for subset in SUBSETS:
        pc_path = percell_paths[subset]
        percell = load_percell(pc_path)
        if not percell:
            sys.exit(f"ERROR: {pc_path} not found — run the per-cell analysis first.")
        vectors = load_vectors(args.tpr_csv, args.quality_csv, subset,
                               args.alpha, args.threshold_type)
        print(f"\nSubset {subset}: {len(percell)} cells in percell_results.csv, "
              f"{sum(1 for r in percell.values() if is_above_floor(r))} above floor, "
              f"{len(vectors)} vectors loaded")

        rows, nulls, _ = analyse_subset(subset, vectors, percell, assignments,
                                        identity_index, problems)

        for r in rows:
            holm_str = (percell_test.fmt(r["holm_p"], 3) if r.get("holm_p") is not None else "—")
            if r["n_cells"]:
                print(f"  {r['stratum']:<20} {r['tier']:<9} n={r['n_cells']:<3} "
                      f"T={r['T_observed']:.4f}  null_med={r['null_median']:.4f}  "
                      f"p={percell_test.fmt(r['exact_p'], 3):<9} holm={holm_str}")
            else:
                print(f"  {r['stratum']:<20} {r['tier']:<9} n=0  (no cells)")
        all_rows.extend(rows)

        for name, joint in nulls.items():
            out = os.path.join(null_dir, f"{subset}__{name.replace('-', '_')}.csv")
            with open(out, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["assignment_index", "group_3", "group_2", "singletons",
                            "T", "is_identity"])
                for j, fam in enumerate(assignments):
                    groups = defaultdict(list)
                    for l in langs:
                        groups[fam[l]].append(l)
                    sized = sorted(groups.values(), key=lambda g: (-len(g), g))
                    w.writerow([j, ",".join(sized[0]), ",".join(sized[1]),
                                ",".join(sorted(g[0] for g in sized if len(g) == 1)),
                                percell_test.fmt(joint[j], 12), j == identity_index])

    if problems:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    with open(os.path.join(args.output_dir, "joint_results.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(JOINT_FIELDS)
        for r in all_rows:
            w.writerow([percell_test.fmt(r.get(c), percell_test.CSV_PRECISION) for c in JOINT_FIELDS])
    print(f"\nWrote {args.output_dir}/joint_results.csv  ({len(all_rows)} rows)")

    return 0


def selftest():
    import tempfile
    print("Self-test: synthetic fixtures, no real data touched.")
    langs = list(percell_test.LANGUAGES)
    assignments = percell_test.shape_matched_assignments(langs, percell_test.registered_shape(langs))
    idx = next(i for i, f in enumerate(assignments)
               if percell_test.same_grouping(f, percell_test.LANGUAGE_FAMILIES, langs))

    vs = [{l: 0.1 + 0.09 * ((i * s) % 11) for i, l in enumerate(langs)} for s in (3, 5, 7)]
    mats = [[percell_test.decompose(v, f)[3] for f in assignments] for v in vs]
    for m in mats:
        assert abs(sum(m) / len(m) - 0.7) < 1e-9, sum(m) / len(m)
    joint = [sum(m[j] for m in mats) / len(mats) for j in range(len(assignments))]
    assert abs(sum(joint) / len(joint) - 0.7) < 1e-9
    print(f"  [ok] joint null mean = {sum(joint) / len(joint):.12f} (analytic 0.7)")

    aligned = [{l: (0.9 if percell_test.LANGUAGE_FAMILIES[l] == "Germanic"
                    else 0.2 if percell_test.LANGUAGE_FAMILIES[l] == "Romance"
                    else 0.4 + 0.04 * i) for i, l in enumerate(langs)} for _ in range(3)]
    singleton_vals = [v for l, v in aligned[0].items()
                      if percell_test.LANGUAGE_FAMILIES[l] not in ("Germanic", "Romance")]
    assert len(set(singleton_vals)) == len(singleton_vals), "singleton values collide"
    assert not ({0.9, 0.2} & set(singleton_vals)), "singleton collides with a family level"
    am = [[percell_test.decompose(v, f)[3] for f in assignments] for v in aligned]
    aj = [sum(m[j] for m in am) / len(am) for j in range(len(assignments))]
    t_obs = aj[idx]
    p = sum(1 for t in aj if t >= t_obs - TIE_TOL) / len(aj)
    assert abs(t_obs - 1.0) < 1e-12, t_obs
    assert abs(p - 1 / 4620) < 1e-12, p
    print(f"  [ok] perfectly aligned stratum: T=1.0, exact p={p:.6f} (= 1/4620, "
          f"uniquely optimal)")

    tied = [{**v, "zho": 0.9} for v in aligned]
    tm = [[percell_test.decompose(v, f)[3] for f in assignments] for v in tied]
    tj = [sum(m[j] for m in tm) / len(tm) for j in range(len(assignments))]
    p_tied = sum(1 for t in tj if t >= tj[idx] - TIE_TOL) / len(tj)
    assert abs(p_tied - 6 / 4620) < 1e-12, p_tied
    print(f"  [ok] tied optima counted, not broken: p={p_tied:.6f} (= 6/4620 when "
          f"zho ties the Germanic level)")

    adj = holm([0.001, 0.02, 0.04, 0.5])
    assert adj == sorted(adj), adj
    assert abs(adj[0] - 0.004) < 1e-12 and abs(adj[3] - 0.5) < 1e-12, adj
    print(f"  [ok] Holm: {[round(a, 4) for a in adj]}")

    tmp = tempfile.mkdtemp(prefix="joint_selftest_")
    cells = [("base", "KGW", "mistral"), ("base", "DIP", "qwen"),
             ("instruct", "KGW", "mistral"), ("instruct", "DIP", "qwen")]
    vecs = {c: {l: 0.1 + 0.07 * ((i * (n + 3)) % 11) for i, l in enumerate(langs)}
            for n, c in enumerate(cells)}
    tpr_path = os.path.join(tmp, "per_language_tpr.csv")
    with open(tpr_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "watermark", "dataset", "subset", "alpha",
                    "threshold_type", "language", "tpr", "insufficient_samples"])
        for (rg, sc, md), v in vecs.items():
            for sub in ("all", "on_target_only"):
                for l in langs:
                    w.writerow([md, rg, sc, "flores" if rg == "base" else "aya", sub,
                                "0.01", "global", l, f"{v[l]:.6f}", "False"])
    quality_path = os.path.join(tmp, "quality_per_language.csv")
    with open(quality_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "watermark", "subset", "measurement",
                    "language", "family", "preservation_score"])
        for (rg, sc, md), v in vecs.items():
            for sub in ("all", "on_target_only"):
                for l in langs:
                    w.writerow([md, rg, sc, sub, "mauve", l, percell_test.LANGUAGE_FAMILIES[l],
                                f"{v[l]:.6f}"])

    pc_paths = {}
    for sub in ("all", "on_target_only"):
        p_ = os.path.join(tmp, f"percell_{sub}.csv")
        pc_paths[sub] = p_
        with open(p_, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "regime", "scheme", "generator", "n_langs", "status",
                        "above_floor", "observed_between_share", "null_median", "p_exact"])
            for (rg, sc, md), v in vecs.items():
                for metric in ("detection_tpr", "mauve"):
                    total, _b, _w2, share = percell_test.decompose(v, percell_test.LANGUAGE_FAMILIES)
                    w.writerow([metric, rg, sc, md, 11, "ok", "True",
                                percell_test.fmt(share, 12), "0.7", "0.3"])

    out = os.path.join(tmp, "out")
    rc = run(parse_args(["--tpr_csv", tpr_path, "--quality_csv", quality_path,
                         "--percell_all", pc_paths["all"],
                         "--percell_ontarget", pc_paths["on_target_only"],
                         "--output_dir", out]))
    assert rc == 0, rc
    with open(os.path.join(out, "joint_results.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(STRATA) * 2, len(rows)
    for r in rows:
        if r["n_cells"] and r["n_cells"] != "0":
            assert abs(float(r["null_mean"]) - 0.7) < 1e-9, r
    prim = [r for r in rows if r["tier"] == "primary" and r["holm_p"]]
    assert prim and all(float(r["holm_p"]) >= float(r["exact_p"]) - 1e-15 for r in prim)
    assert len(os.listdir(os.path.join(out, "null_distributions"))) == 8
    with open(os.path.join(out, "null_distributions",
                           "all__detection_base.csv"), encoding="utf-8") as f:
        nd = list(csv.DictReader(f))
    assert len(nd) == 4620 and sum(1 for r in nd if r["is_identity"] == "True") == 1
    print(f"  [ok] end-to-end: {len(rows)} stratum rows, 8 null files, "
          f"4620-row null with exactly one identity -> {out}")

    bad = os.path.join(tmp, "percell_bad.csv")
    with open(pc_paths["all"], encoding="utf-8") as f:
        lines = list(csv.reader(f))
    lines[1][7] = "0.123456"
    with open(bad, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(lines)
    try:
        run(parse_args(["--tpr_csv", tpr_path, "--quality_csv", quality_path, "--percell_all", bad,
                        "--percell_ontarget", pc_paths["on_target_only"],
                        "--output_dir", os.path.join(tmp, "out_bad")]))
        raise AssertionError("gate 1 did not fire on a corrupted percell share")
    except SystemExit as e:
        assert e.code == 1, e.code
    print("  [ok] gate 1 fires (exit 1) when a percell share disagrees")
    print("\nSelf-test PASSED.")
    return 0


def main():
    args = parse_args()
    return selftest() if args.selftest else run(args)


if __name__ == "__main__":
    sys.exit(main())
