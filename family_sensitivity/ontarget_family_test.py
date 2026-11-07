#!/usr/bin/env python
"""On-target (LID-filtered) between-family sensitivity test."""

import argparse
import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "family_sensitivity"))

import percell_family_test as percell_test

SUBSET = "on_target_only"

ON_TARGET_ANCHORS = [
    ("detection_tpr", "base",     "Unigram", "gemma",   0.987),
    ("detection_tpr", "base",     "XSIR",    "qwen",    0.207),
    ("detection_tpr", "base",     "KGW",     "mistral", 0.995),
    ("detection_tpr", "base",     "SynthID", "mistral", 1.000),
    ("detection_tpr", "instruct", "Unigram", "mistral", 0.388),
    ("detection_tpr", "instruct", "DIP",     "mistral", 0.488),
    ("detection_tpr", "instruct", "XSIR",    "qwen",    0.047),
    ("detection_tpr", "instruct", "SynthID", "qwen",    0.941),
]

ON_TARGET_FLAGSHIPS = [
    ("detection_tpr", "instruct", "DIP",     "mistral"),
    ("detection_tpr", "instruct", "DIP",     "qwen"),
    ("detection_tpr", "instruct", "Unigram", "mistral"),
    ("detection_tpr", "instruct", "KGW",     "mistral"),
    ("detection_tpr", "base",     "XSIR",    "gemma"),
    ("mauve",         "instruct", "SynthID", "mistral"),
    ("detection_tpr", "base",     "Unigram", "gemma"),
    ("detection_tpr", "base",     "XSIR",    "qwen"),
]

LIKEFORLIKE_FIELDS = [
    "metric", "regime", "scheme", "generator",
    "n_langs_all", "n_langs_ontarget", "n_common", "common_langs", "dropped_langs",
    "share_all_full", "share_ontarget_full", "delta_naive",
    "ge2_all_common", "share_all_common", "null_median_all_common", "p_all_common",
    "ge2_ontarget_common", "share_ontarget_common", "null_median_ontarget_common",
    "p_ontarget_common",
    "delta_like_for_like", "delta_attributable_to_langset", "n_assignments_common",
]


def load_vectors(tpr_csv, quality_csv, subset, alpha, threshold_type):
    vecs = {}
    vecs.update(percell_test.load_detection_vectors(tpr_csv, subset, alpha, threshold_type))
    vecs.update(percell_test.load_mauve_vectors(quality_csv, subset))
    return {k: {l: x for l, x in v.items() if l in percell_test.LANGUAGE_FAMILIES}
            for k, v in vecs.items()}


def likeforlike_rows(all_vecs, ot_vecs):
    rows = []
    for key in sorted(set(all_vecs) & set(ot_vecs), key=percell_test.cell_sort_key):
        v_all, v_ot = all_vecs[key], ot_vecs[key]
        common = sorted(set(v_all) & set(v_ot))
        if set(v_all) == set(v_ot) and len(v_ot) == len(percell_test.LANGUAGES):
            continue
        if len(common) < percell_test.MIN_LANGS_FOR_FAIRNESS:
            rows.append({
                "metric": key[0], "regime": key[1], "scheme": key[2], "generator": key[3],
                "n_langs_all": len(v_all), "n_langs_ontarget": len(v_ot),
                "n_common": len(common), "common_langs": ",".join(common),
                "dropped_langs": ",".join(sorted(set(v_all) - set(v_ot))),
            })
            continue

        sub_all = {l: v_all[l] for l in common}
        sub_ot = {l: v_ot[l] for l in common}
        ge_all, _b1, _w1, sh_all = percell_test.decompose(sub_all, percell_test.LANGUAGE_FAMILIES)
        ge_ot, _b2, _w2, sh_ot = percell_test.decompose(sub_ot, percell_test.LANGUAGE_FAMILIES)
        null_all = percell_test.analysis2_null(sub_all, sh_all, ge_all)
        null_ot = percell_test.analysis2_null(sub_ot, sh_ot, ge_ot)

        _t, _b, _w, sh_all_full = percell_test.decompose(v_all, percell_test.LANGUAGE_FAMILIES)
        _t2, _b2, _w2, sh_ot_full = percell_test.decompose(v_ot, percell_test.LANGUAGE_FAMILIES)

        def sub(a, b):
            return (a - b) if (percell_test.isnum(a) and percell_test.isnum(b)) else None

        d_naive = sub(sh_ot_full, sh_all_full)
        d_lfl = sub(sh_ot, sh_all)
        rows.append({
            "metric": key[0], "regime": key[1], "scheme": key[2], "generator": key[3],
            "n_langs_all": len(v_all), "n_langs_ontarget": len(v_ot),
            "n_common": len(common), "common_langs": ",".join(common),
            "dropped_langs": ",".join(sorted(set(v_all) - set(v_ot))),
            "share_all_full": sh_all_full, "share_ontarget_full": sh_ot_full,
            "delta_naive": d_naive,
            "ge2_all_common": ge_all, "share_all_common": sh_all,
            "null_median_all_common": null_all["null_median"],
            "p_all_common": null_all["p_exact"],
            "ge2_ontarget_common": ge_ot, "share_ontarget_common": sh_ot,
            "null_median_ontarget_common": null_ot["null_median"],
            "p_ontarget_common": null_ot["p_exact"],
            "delta_like_for_like": d_lfl,
            "delta_attributable_to_langset": sub(d_naive, d_lfl),
            "n_assignments_common": null_all["n_assignments"],
        })
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tpr_csv", default="results/detection/per_language_tpr.csv")
    p.add_argument("--quality_csv", default="results/quality/quality_per_language.csv")
    p.add_argument("--detection_summary_csv", default="results/detection/detection_disparity.csv")
    p.add_argument("--quality_summary_csv", default="results/quality/quality_disparity.csv")
    p.add_argument("--output_dir", default="analysis/ontarget_family_test")
    p.add_argument("--subset", default=SUBSET)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--threshold_type", default="global")
    p.add_argument("--floor", type=float, default=1e-3)
    p.add_argument("--selftest", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.selftest:
        return selftest()

    percell_test.PUBLISHED_ANCHORS = ON_TARGET_ANCHORS
    percell_test.ANCHOR_KIND = "mean_tpr"
    percell_test.ANCHOR_ATOL_MEAN = 0.002
    percell_test.FLAGSHIP_CELLS = ON_TARGET_FLAGSHIPS

    print(f"On-target run: subset={args.subset}, alpha={args.alpha}, "
          f"threshold_type={args.threshold_type}")
    print(f"  anchors: {len(ON_TARGET_ANCHORS)} published on-target cell means, "
          f"tolerance {percell_test.ANCHOR_ATOL_MEAN}")

    rc = percell_test.run(percell_test.parse_args([
        "--tpr_csv", args.tpr_csv,
        "--quality_csv", args.quality_csv,
        "--detection_summary_csv", args.detection_summary_csv,
        "--quality_summary_csv", args.quality_summary_csv,
        "--output_dir", args.output_dir,
        "--subset", args.subset,
        "--alpha", str(args.alpha),
        "--threshold_type", args.threshold_type,
        "--floor", str(args.floor),
    ]))
    if rc != 0:
        print("\nOn-target run did not complete.", file=sys.stderr)
        return rc

    all_vecs = load_vectors(args.tpr_csv, args.quality_csv, "all", args.alpha, args.threshold_type)
    ot_vecs = load_vectors(args.tpr_csv, args.quality_csv, args.subset, args.alpha,
                           args.threshold_type)
    lfl = likeforlike_rows(all_vecs, ot_vecs)
    lfl_path = os.path.join(args.output_dir, "likeforlike_common_langs.csv")
    with open(lfl_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(LIKEFORLIKE_FIELDS)
        for r in lfl:
            w.writerow([percell_test.fmt(r.get(c), percell_test.CSV_PRECISION) for c in LIKEFORLIKE_FIELDS])
    print(f"Wrote {lfl_path}  ({len(lfl)} cells whose language set changed)")
    return 0


def selftest():
    import tempfile
    print("Self-test: synthetic on-target fixtures, no real data touched.")
    tmp = tempfile.mkdtemp(prefix="ontarget_selftest_")
    langs = list(percell_test.LANGUAGES)

    def vec(seed, shift=0.0):
        return {l: max(0.0, min(1.0, 0.15 + 0.06 * ((i * seed) % 7) + shift
                                + (0.25 if percell_test.LANGUAGE_FAMILIES[l] in ("Germanic", "Romance")
                                   else 0.0)))
                for i, l in enumerate(langs)}

    cells = [("base", "Unigram", "gemma"), ("base", "XSIR", "qwen"),
             ("instruct", "DIP", "mistral"), ("instruct", "KGW", "mistral")]
    v_all = {c: vec(i + 2) for i, c in enumerate(cells)}
    v_ot = {c: vec(i + 3, shift=0.05) for i, c in enumerate(cells)}
    SHORT = ("instruct", "DIP", "mistral")
    DROPPED = ("arb", "hin")

    tpr_path = os.path.join(tmp, "per_language_tpr.csv")
    with open(tpr_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "watermark", "dataset", "subset", "alpha",
                    "threshold_type", "language", "tpr", "insufficient_samples",
                    "reason_insufficient"])
        for cell in cells:
            regime, scheme, model = cell
            ds = "flores" if regime == "base" else "aya"
            for lang in langs:
                w.writerow([model, regime, scheme, ds, "all", "0.01", "global",
                            lang, f"{v_all[cell][lang]:.6f}", "False", ""])
                if cell == SHORT and lang in DROPPED:
                    w.writerow([model, regime, scheme, ds, SUBSET, "0.01", "global",
                                lang, "", "True", "n_pos=61 < 100"])
                else:
                    w.writerow([model, regime, scheme, ds, SUBSET, "0.01", "global",
                                lang, f"{v_ot[cell][lang]:.6f}", "False", ""])
                w.writerow([model, regime, scheme, ds, "all", "0.05", "global",
                            lang, "0.5", "False", ""])
    quality_path = os.path.join(tmp, "quality_per_language.csv")
    with open(quality_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "watermark", "subset", "measurement",
                    "language", "family", "preservation_score"])
        for cell in cells:
            regime, scheme, model = cell
            for lang in langs:
                w.writerow([model, regime, scheme, "all", "mauve", lang,
                            percell_test.LANGUAGE_FAMILIES[lang], f"{0.4 + v_all[cell][lang] / 2:.6f}"])
                w.writerow([model, regime, scheme, SUBSET, "mauve", lang,
                            percell_test.LANGUAGE_FAMILIES[lang], f"{0.4 + v_ot[cell][lang] / 2:.6f}"])

    def means(vectors, subset_langs=None):
        out = []
        for (rg, sc, md), v in vectors.items():
            vals = [x for l, x in v.items() if subset_langs is None
                    or l in subset_langs.get((rg, sc, md), langs)]
            out.append(("detection_tpr", rg, sc, md, round(sum(vals) / len(vals), 3)))
        return out

    ot_langsets = {SHORT: [l for l in langs if l not in DROPPED]}
    global ON_TARGET_ANCHORS, ON_TARGET_FLAGSHIPS
    saved_a, saved_f = ON_TARGET_ANCHORS, ON_TARGET_FLAGSHIPS
    ON_TARGET_ANCHORS = means(v_ot, ot_langsets)
    ON_TARGET_FLAGSHIPS = [("detection_tpr", "base", "Unigram", "gemma"),
                           ("detection_tpr", "base", "XSIR", "qwen")]

    all_dir = os.path.join(tmp, "all")
    ot_dir = os.path.join(tmp, "ot")
    try:
        percell_test.PUBLISHED_ANCHORS, percell_test.ANCHOR_KIND = means(v_all), "mean_tpr"
        percell_test.FLAGSHIP_CELLS = ON_TARGET_FLAGSHIPS
        rc_all = percell_test.run(percell_test.parse_args([
            "--tpr_csv", tpr_path, "--quality_csv", quality_path, "--subset", "all",
            "--detection_summary_csv", os.path.join(tmp, "x.csv"),
            "--quality_summary_csv", os.path.join(tmp, "y.csv"),
            "--output_dir", all_dir]))
        assert rc_all == 0, rc_all
        print("  [ok] mean_tpr anchor gate passes when the means match")

        percell_test.PUBLISHED_ANCHORS = [(m, rg, sc, md, v + 0.5)
                               for m, rg, sc, md, v in ON_TARGET_ANCHORS]
        rc_bad = percell_test.run(percell_test.parse_args([
            "--tpr_csv", tpr_path, "--quality_csv", quality_path, "--subset", SUBSET,
            "--detection_summary_csv", os.path.join(tmp, "x.csv"),
            "--quality_summary_csv", os.path.join(tmp, "y.csv"),
            "--output_dir", os.path.join(tmp, "bad")]))
        assert rc_bad == 1, rc_bad
        assert not os.path.exists(os.path.join(tmp, "bad", "percell_results.csv"))
        print("  [ok] mean_tpr anchor gate fails (exit 1) on wrong anchors")

        rc = main(["--tpr_csv", tpr_path, "--quality_csv", quality_path,
                   "--detection_summary_csv", os.path.join(tmp, "x.csv"),
                   "--quality_summary_csv", os.path.join(tmp, "y.csv"),
                   "--output_dir", ot_dir])
        assert rc == 0, rc
    finally:
        ON_TARGET_ANCHORS, ON_TARGET_FLAGSHIPS = saved_a, saved_f
        percell_test.ANCHOR_KIND = "ge2_share"

    with open(os.path.join(ot_dir, "likeforlike_common_langs.csv"), encoding="utf-8") as f:
        lfl_rows = list(csv.DictReader(f))
    assert len(lfl_rows) == 1, lfl_rows
    row = lfl_rows[0]
    assert row["scheme"] == SHORT[1] and row["generator"] == SHORT[2]
    assert row["n_langs_all"] == "11" and row["n_langs_ontarget"] == "9"
    assert row["n_common"] == "9", row["n_common"]
    assert sorted(row["dropped_langs"].split(",")) == sorted(DROPPED)
    assert row["n_assignments_common"] not in ("", "4620"), row["n_assignments_common"]
    assert row["share_all_full"] != row["share_all_common"]
    assert row["delta_naive"] != row["delta_like_for_like"]
    assert abs(float(row["delta_attributable_to_langset"])) > 0.01, row
    print(f"  [ok] like-for-like correction: 1 affected cell, common set 9 langs, "
          f"{row['n_assignments_common']} exact assignments (not 4620), "
          f"naive Δ {float(row['delta_naive']):+.4f} vs like-for-like Δ "
          f"{float(row['delta_like_for_like']):+.4f}")

    with open(os.path.join(ot_dir, "percell_results.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows and all(r["observed_between_share"] not in ("", None) for r in rows)
    print(f"  [ok] subset filter applied ({len(rows)} on-target cells, decoys excluded)")
    print("\nSelf-test PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
