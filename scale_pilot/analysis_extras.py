#!/usr/bin/env python
"""Qwen2.5-72B scale pilot — per-cell CSVs, bootstrap CIs and the threshold-type delta."""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog
import shim

BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 42
GE2_FLOOR = 1e-3
DI_THRESHOLD = 0.8

QUALITY_MEASUREMENTS = ["mauve", "bertscore_f1_rescaled", "ppl_pres_xglm-7.5b"]


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(raw):
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def truthy(raw):
    return str(raw).strip().lower() == "true"


def fmt(v, nd=6):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "" if v != v else f"{v:.{nd}g}"
    return str(v)


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([fmt(r.get(c)) for c in fields])
    return len(rows)


def bootstrap_mean_ci(values, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, level=0.95):
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n < 2:
        return None, None, None
    rng = random.Random(seed)
    means = []
    for _ in range(b):
        means.append(sum(rng.choice(vals) for _ in range(n)) / n)
    means.sort()
    lo_i = int((1.0 - level) / 2.0 * (b - 1) + 0.5)
    hi_i = int((1.0 + level) / 2.0 * (b - 1) + 0.5)
    return sum(vals) / n, means[lo_i], means[hi_i]


def di_below(by_lang, threshold=DI_THRESHOLD):
    langs = sorted(by_lang)
    if not langs:
        return []
    mx = max(by_lang[l] for l in langs)
    if mx <= 0:
        return list(langs)
    return [l for l in langs if by_lang[l] / mx < threshold]


def load_per_language_tpr(path, alpha, threshold_type):
    out = {}
    for r in read_csv(path):
        if r.get("model") != shim.ALIAS:
            continue
        if r.get("threshold_type") != threshold_type:
            continue
        if fnum(r.get("alpha")) != alpha:
            continue
        key = (r["watermark"], r["variant"], r["subset"])
        out.setdefault(key, {})[r["language"]] = r
    return out


def load_auc(path):
    out = {}
    for r in read_csv(path):
        if r.get("model") != shim.ALIAS:
            continue
        key = (r["watermark"], r["variant"], r["subset"])
        out.setdefault(key, {})[r["language"]] = r
    return out


def load_quality_per_language(path):
    out = {}
    for r in read_csv(path):
        if r.get("model") != shim.ALIAS:
            continue
        v = fnum(r.get("preservation_score"))
        if v is None:
            continue
        key = (r["watermark"], r["variant"], r["subset"], r["measurement"])
        out.setdefault(key, {})[r["language"]] = v
    return out


def load_summary(path, extra_key=None):
    out = {}
    for r in read_csv(path):
        if r.get("model") != shim.ALIAS:
            continue
        key = (r["watermark"], r["variant"], r["subset"])
        if extra_key:
            key = key + (r.get(extra_key, ""),)
        out[key] = r
    return out


PERLANG_DET_FIELDS = [
    "scheme", "regime", "subset", "language", "family",
    "tpr_global", "tpr_per_language", "auc",
    "n_pos", "n_neg", "threshold_global", "threshold_per_language",
    "empirical_fpr_global", "gap_std",
    "usable_global", "usable_per_language", "usable_auc", "reason_insufficient",
]


def build_perlanguage_detection(tpr_global, tpr_per_language, auc, families, log_exclusions=True):
    rows = []
    for key in sorted(set(tpr_global) | set(tpr_per_language)):
        scheme, regime, subset = key
        langs = sorted(set(tpr_global.get(key, {})) | set(tpr_per_language.get(key, {})))
        for lang in langs:
            rg = tpr_global.get(key, {}).get(lang, {})
            rl = tpr_per_language.get(key, {}).get(lang, {})
            ra = auc.get(key, {}).get(lang, {})
            ok_g = bool(rg) and not truthy(rg.get("insufficient_samples"))
            ok_l = bool(rl) and not truthy(rl.get("insufficient_samples"))
            ok_a = bool(ra) and not truthy(ra.get("insufficient_samples"))
            reason = (rg.get("reason_insufficient")
                      or rl.get("reason_insufficient")
                      or ra.get("reason_insufficient") or "")
            rows.append({
                "scheme": scheme, "regime": regime, "subset": subset,
                "language": lang, "family": families.get(lang, "?"),
                "tpr_global": fnum(rg.get("tpr")) if ok_g else None,
                "tpr_per_language": fnum(rl.get("tpr")) if ok_l else None,
                "auc": fnum(ra.get("auc")) if ok_a else None,
                "n_pos": rg.get("n_pos") or rl.get("n_pos"),
                "n_neg": rg.get("n_neg") or rl.get("n_neg"),
                "threshold_global": fnum(rg.get("threshold_value")),
                "threshold_per_language": fnum(rl.get("threshold_value")),
                "empirical_fpr_global": fnum(rg.get("empirical_fpr")),
                "gap_std": fnum(rg.get("gap_std")) or fnum(rl.get("gap_std")),
                "usable_global": ok_g, "usable_per_language": ok_l,
                "usable_auc": ok_a, "reason_insufficient": reason,
            })
            if log_exclusions and subset == "all" and not ok_g:
                runlog.exclude("detection (tau_global)",
                               f"{scheme}/{regime}/{lang}",
                               reason or "insufficient_samples",
                               "per_language_tpr.csv")
    return rows


PERCELL_DET_FIELDS = [
    "scheme", "regime", "subset", "n_langs",
    "mean_tpr_global", "mean_tpr_global_ci_lo", "mean_tpr_global_ci_hi",
    "min_tpr_global", "min_lang_global", "max_tpr_global", "max_lang_global",
    "di_count_below_80", "di_langs_below_80",
    "mean_auc", "min_auc", "min_auc_lang",
    "mean_tpr_per_language", "n_langs_common", "delta_global_minus_perlang",
    "n_langs_excluded", "excluded_langs",
    "bootstrap_B", "bootstrap_seed",
]


def build_percell_detection(perlang):
    by_cell = {}
    for r in perlang:
        by_cell.setdefault((r["scheme"], r["regime"], r["subset"]), []).append(r)

    rows = []
    for (scheme, regime, subset), rs in sorted(by_cell.items()):
        tpr_g = {r["language"]: r["tpr_global"] for r in rs if r["tpr_global"] is not None}
        tpr_l = {r["language"]: r["tpr_per_language"] for r in rs
                 if r["tpr_per_language"] is not None}
        aucs = {r["language"]: r["auc"] for r in rs if r["auc"] is not None}
        excluded = sorted(r["language"] for r in rs if r["tpr_global"] is None)

        mean_g, lo, hi = bootstrap_mean_ci(list(tpr_g.values()))

        common = sorted(set(tpr_g) & set(tpr_l))
        delta = mean_l_common = None
        if common:
            mean_g_common = sum(tpr_g[l] for l in common) / len(common)
            mean_l_common = sum(tpr_l[l] for l in common) / len(common)
            delta = mean_g_common - mean_l_common

        di = di_below(tpr_g)
        rows.append({
            "scheme": scheme, "regime": regime, "subset": subset,
            "n_langs": len(tpr_g),
            "mean_tpr_global": mean_g, "mean_tpr_global_ci_lo": lo,
            "mean_tpr_global_ci_hi": hi,
            "min_tpr_global": min(tpr_g.values()) if tpr_g else None,
            "min_lang_global": min(tpr_g, key=lambda l: tpr_g[l]) if tpr_g else "",
            "max_tpr_global": max(tpr_g.values()) if tpr_g else None,
            "max_lang_global": max(tpr_g, key=lambda l: tpr_g[l]) if tpr_g else "",
            "di_count_below_80": len(di), "di_langs_below_80": ",".join(di),
            "mean_auc": (sum(aucs.values()) / len(aucs)) if aucs else None,
            "min_auc": min(aucs.values()) if aucs else None,
            "min_auc_lang": min(aucs, key=lambda l: aucs[l]) if aucs else "",
            "mean_tpr_per_language": mean_l_common,
            "n_langs_common": len(common),
            "delta_global_minus_perlang": delta,
            "n_langs_excluded": len(excluded), "excluded_langs": ",".join(excluded),
            "bootstrap_B": BOOTSTRAP_B, "bootstrap_seed": BOOTSTRAP_SEED,
        })
    return rows


PERLANG_QUAL_FIELDS = ["scheme", "regime", "subset", "measurement",
                       "language", "family", "preservation_score"]

PERCELL_QUAL_FIELDS = [
    "scheme", "regime", "subset", "measurement", "n_langs",
    "mean", "mean_ci_lo", "mean_ci_hi", "min", "min_lang", "max", "max_lang",
    "di_count_below_80", "di_langs_below_80", "n_langs_missing", "missing_langs",
]


def build_quality(quality_by_cell, families):
    perlang, percell = [], []
    for key in sorted(quality_by_cell):
        scheme, regime, subset, measurement = key
        by_lang = quality_by_cell[key]
        for lang in sorted(by_lang):
            perlang.append({
                "scheme": scheme, "regime": regime, "subset": subset,
                "measurement": measurement, "language": lang,
                "family": families.get(lang, "?"),
                "preservation_score": by_lang[lang],
            })
        missing = sorted(set(shim.LANGUAGES) - set(by_lang))
        mean, lo, hi = bootstrap_mean_ci(list(by_lang.values()))
        di = di_below(by_lang) if measurement != "bertscore_f1_rescaled" else []
        percell.append({
            "scheme": scheme, "regime": regime, "subset": subset,
            "measurement": measurement, "n_langs": len(by_lang),
            "mean": mean, "mean_ci_lo": lo, "mean_ci_hi": hi,
            "min": min(by_lang.values()) if by_lang else None,
            "min_lang": min(by_lang, key=lambda l: by_lang[l]) if by_lang else "",
            "max": max(by_lang.values()) if by_lang else None,
            "max_lang": max(by_lang, key=lambda l: by_lang[l]) if by_lang else "",
            "di_count_below_80": len(di) if di or measurement != "bertscore_f1_rescaled" else "",
            "di_langs_below_80": ",".join(di),
            "n_langs_missing": len(missing), "missing_langs": ",".join(missing),
        })
        if missing and subset == "all":
            runlog.exclude(f"quality/{measurement}", f"{scheme}/{regime}",
                           f"languages below the 100-sample floor: {','.join(missing)}",
                           "quality_per_language.csv")
    return perlang, percell


GE2_FIELDS = [
    "metric", "scheme", "regime", "subset", "measurement", "n_langs", "n_families",
    "ge_2_total", "ge_2_between", "ge_2_within", "ge_2_between_share",
    "above_floor", "ge_2_floor",
]


def build_ge2(det_summary, qual_summary):
    rows = []
    for (scheme, regime, subset), r in sorted(det_summary.items()):
        total = fnum(r.get("ge_2_total"))
        rows.append({
            "metric": "detection_tpr", "scheme": scheme, "regime": regime,
            "subset": subset, "measurement": "tpr",
            "n_langs": r.get("n_langs"), "n_families": r.get("n_families"),
            "ge_2_total": total, "ge_2_between": fnum(r.get("ge_2_between")),
            "ge_2_within": fnum(r.get("ge_2_within")),
            "ge_2_between_share": fnum(r.get("ge_2_between_share")),
            "above_floor": total is not None and total >= GE2_FLOOR,
            "ge_2_floor": GE2_FLOOR,
        })
    for key, r in sorted(qual_summary.items()):
        scheme, regime, subset, measurement = key
        total = fnum(r.get("ge_2_total"))
        rows.append({
            "metric": "quality", "scheme": scheme, "regime": regime,
            "subset": subset, "measurement": measurement,
            "n_langs": r.get("n_langs"), "n_families": r.get("n_families"),
            "ge_2_total": total, "ge_2_between": fnum(r.get("ge_2_between")),
            "ge_2_within": fnum(r.get("ge_2_within")),
            "ge_2_between_share": fnum(r.get("ge_2_between_share")),
            "above_floor": total is not None and total >= GE2_FLOOR,
            "ge_2_floor": GE2_FLOOR,
        })
    return rows


def run(args):
    LANGUAGE_FAMILIES = shim.import_stage("quality_fairness", "quality_disparity").LANGUAGE_FAMILIES

    tpr_global = load_per_language_tpr(args.tpr_csv, args.alpha, "global")
    tpr_per_language = load_per_language_tpr(args.tpr_csv, args.alpha, "per_language")
    auc = load_auc(args.auc_csv)
    quality_by_cell = load_quality_per_language(args.quality_csv)
    det_summary = {k: v for k, v in load_summary(args.detection_summary_csv).items()}
    qual_summary = load_summary(args.quality_summary_csv, extra_key="measurement")

    if not tpr_global and not quality_by_cell:
        print(f"ERROR: no {shim.ALIAS} rows found in {args.tpr_csv} or {args.quality_csv}. "
              f"Run the detection/quality stages first.", file=sys.stderr)
        return 1

    det_summary = {
        k: v for k, v in det_summary.items()
        if fnum(v.get("alpha")) == args.alpha
        and v.get("threshold_type") == shim.THRESHOLD_TYPE
    }

    perlang_det = build_perlanguage_detection(tpr_global, tpr_per_language, auc, LANGUAGE_FAMILIES)
    percell_det = build_percell_detection(perlang_det)
    perlang_q, percell_q = build_quality(quality_by_cell, LANGUAGE_FAMILIES)
    ge2 = build_ge2(det_summary, qual_summary)

    out = shim.RESULTS_DIR
    n1 = write_csv(os.path.join(out, "perlanguage_detection.csv"),
                   PERLANG_DET_FIELDS, perlang_det)
    n2 = write_csv(os.path.join(out, "percell_detection.csv"),
                   PERCELL_DET_FIELDS, percell_det)
    n3 = write_csv(os.path.join(out, "perlanguage_quality.csv"),
                   PERLANG_QUAL_FIELDS, perlang_q)
    n4 = write_csv(os.path.join(out, "percell_quality.csv"),
                   PERCELL_QUAL_FIELDS, percell_q)
    n5 = write_csv(os.path.join(out, "percell_ge2.csv"), GE2_FIELDS, ge2)

    print(f"Wrote perlanguage_detection.csv  ({n1} rows)")
    print(f"Wrote percell_detection.csv      ({n2} rows)")
    print(f"Wrote perlanguage_quality.csv    ({n3} rows)")
    print(f"Wrote percell_quality.csv        ({n4} rows)")
    print(f"Wrote percell_ge2.csv            ({n5} rows)")

    print(f"\nHeadline (subset=all, alpha={args.alpha}, threshold_type=global)")
    print("=" * 88)
    print(f"  {'scheme':<9}{'regime':<10}{'n':>3}  {'meanTPR [95% CI]':<26}"
          f"{'minTPR':>8}{'DI<.8':>7}{'delta':>9}")
    for r in percell_det:
        if r["subset"] != "all":
            continue
        ci = (f"{r['mean_tpr_global']:.3f} [{r['mean_tpr_global_ci_lo']:.3f}, "
              f"{r['mean_tpr_global_ci_hi']:.3f}]"
              if r["mean_tpr_global"] is not None else "n/a")
        print(f"  {r['scheme']:<9}{r['regime']:<10}{r['n_langs']:>3}  {ci:<26}"
              f"{fmt(r['min_tpr_global'], 3):>8}{r['di_count_below_80']:>7}"
              f"{fmt(r['delta_global_minus_perlang'], 3):>9}")
    runlog.log(f"analysis_extras: {n2} detection cells, {n4} quality cells, "
               f"{n5} GE_2 rows (bootstrap B={BOOTSTRAP_B}, seed={BOOTSTRAP_SEED})")
    return 0


def selftest():
    import tempfile
    print("Self-test: synthetic fixtures, no real data touched.")
    langs = list(shim.LANGUAGES)

    vals = [0.1 * i for i in range(1, 12)]
    m1, lo1, hi1 = bootstrap_mean_ci(vals)
    m2, lo2, hi2 = bootstrap_mean_ci(vals)
    assert (m1, lo1, hi1) == (m2, lo2, hi2), "bootstrap not deterministic"
    assert abs(m1 - sum(vals) / len(vals)) < 1e-12
    assert lo1 < m1 < hi1, (lo1, m1, hi1)
    print(f"  [ok] bootstrap CI deterministic and brackets the mean: "
          f"{m1:.3f} [{lo1:.3f}, {hi1:.3f}]")

    m, lo, hi = bootstrap_mean_ci([0.5] * 11)
    assert lo == hi == m == 0.5, (m, lo, hi)
    print("  [ok] constant vector -> zero-width interval")

    assert len(di_below({l: 0.0 for l in langs})) == 11
    assert di_below({"a": 1.0, "b": 0.79}) == ["b"]
    assert di_below({"a": 1.0, "b": 0.80}) == []
    print("  [ok] four-fifths rule matches detection_disparity (strict <, all-zero => all)")

    tmp = tempfile.mkdtemp(prefix="extras_selftest_")
    saved_results, saved_exc = shim.RESULTS_DIR, runlog.EXCLUSIONS_PATH
    shim.RESULTS_DIR = os.path.join(tmp, "results")
    runlog.EXCLUSIONS_PATH = os.path.join(tmp, "exclusions.csv")
    try:
        tpr_path = os.path.join(tmp, "per_language_tpr.csv")
        with open(tpr_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "variant", "watermark", "dataset", "subset", "alpha",
                        "threshold_type", "language", "threshold_value", "tpr",
                        "n_pos", "n_neg", "empirical_fpr", "gap_std",
                        "insufficient_samples", "reason_insufficient"])
            for i, lang in enumerate(langs):
                bad = lang == "hin"
                for tt, base in (("global", 0.50), ("per_language", 0.55)):
                    w.writerow([shim.ALIAS, "base", "KGW", "flores", "all", "0.01", tt,
                                lang, "1.5",
                                "" if (bad and tt == "global") else f"{base + 0.02 * i:.4f}",
                                "500", "143" if bad else "500", "0.009", "3.1",
                                "True" if (bad and tt == "global") else "False",
                                "n_neg=143 < 200" if (bad and tt == "global") else ""])
                w.writerow([shim.ALIAS, "base", "KGW", "flores", "all", "0.05", "global",
                            lang, "1.5", "0.99", "500", "500", "0.04", "3.1", "False", ""])
                w.writerow(["qwen", "base", "KGW", "flores", "all", "0.01", "global",
                            lang, "1.5", "0.11", "500", "500", "0.009", "3.1", "False", ""])

        auc_csv = os.path.join(tmp, "auc.csv")
        with open(auc_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "variant", "watermark", "dataset", "subset", "language",
                        "n_pos", "n_neg", "auc", "insufficient_samples",
                        "reason_insufficient"])
            for i, lang in enumerate(langs):
                w.writerow([shim.ALIAS, "base", "KGW", "flores", "all", lang,
                            "500", "500", f"{0.80 + 0.01 * i:.4f}", "False", ""])

        quality_csv = os.path.join(tmp, "quality_per_language.csv")
        with open(quality_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "variant", "watermark", "subset", "measurement",
                        "language", "family", "preservation_score"])
            for i, lang in enumerate(langs):
                if lang == "arb":
                    continue
                w.writerow([shim.ALIAS, "base", "KGW", "all", "mauve", lang, "x",
                            f"{0.60 + 0.03 * i:.4f}"])

        empty = os.path.join(tmp, "empty.csv")
        open(empty, "w").close()

        args = argparse.Namespace(tpr_csv=tpr_path, auc_csv=auc_csv, quality_csv=quality_csv,
                                  detection_summary_csv=empty, quality_summary_csv=empty, alpha=0.01)
        assert run(args) == 0

        with open(os.path.join(shim.RESULTS_DIR, "percell_detection.csv"),
                  encoding="utf-8") as f:
            cells = list(csv.DictReader(f))
        cell = next(c for c in cells if c["subset"] == "all")
        assert cell["n_langs"] == "10", cell["n_langs"]
        assert cell["n_langs_excluded"] == "1" and cell["excluded_langs"] == "hin"
        assert cell["n_langs_common"] == "10", cell["n_langs_common"]
        d = float(cell["delta_global_minus_perlang"])
        assert abs(d - (-0.05)) < 1e-9, d
        print(f"  [ok] floor-excluded language dropped from the cell "
              f"(n_langs 10, delta on the common 10 = {d:+.3f})")

        exc = open(runlog.EXCLUSIONS_PATH, encoding="utf-8").read()
        assert "KGW/base/hin" in exc and "n_neg=143 < 200" in exc, exc
        assert "arb" in exc, exc
        print("  [ok] pipeline's own reason string reaches exclusions.csv")

        with open(os.path.join(shim.RESULTS_DIR, "perlanguage_detection.csv"),
                  encoding="utf-8") as f:
            pl = list(csv.DictReader(f))
        assert len(pl) == 11 and sum(1 for r in pl if r["tpr_global"] == "") == 1
        assert all(r["scheme"] == "KGW" for r in pl)
        print("  [ok] per-language rows kept for the excluded language, TPR blank")

        with open(os.path.join(shim.RESULTS_DIR, "percell_quality.csv"),
                  encoding="utf-8") as f:
            q = list(csv.DictReader(f))
        assert q[0]["n_langs"] == "10" and q[0]["missing_langs"] == "arb"
        print("  [ok] quality cell reports its missing language")
    finally:
        shim.RESULTS_DIR = saved_results
        runlog.EXCLUSIONS_PATH = saved_exc
    print("\nSelf-test PASSED.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tpr_csv", default=shim.OUT["tpr"])
    p.add_argument("--auc_csv", default=shim.OUT["per_language_auc"])
    p.add_argument("--quality_csv", default=shim.OUT["quality"])
    p.add_argument("--detection_summary_csv", default=shim.OUT["detection_summary"])
    p.add_argument("--quality_summary_csv", default=shim.OUT["quality_summary"])
    p.add_argument("--alpha", type=float, default=shim.ALPHA_HEADLINE)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    return selftest() if args.selftest else run(args)


if __name__ == "__main__":
    sys.exit(main())
