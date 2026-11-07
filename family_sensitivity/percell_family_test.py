#!/usr/bin/env python
"""Per-cell sensitivity test for the GE_2 between-family share."""

import argparse
import csv
import itertools
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from quality_fairness.quality_disparity import (
    LANGUAGE_FAMILIES, LANGUAGES, MIN_LANGS_FOR_FAIRNESS, ge_decompose,
)
from detection_fairness.detection_disparity import load_per_language_tpr


NAN = float("nan")

FLOOR_GE2 = 1e-3

MULTI_MEMBER_LANGS = ("eng", "nld", "fra", "spa", "por")

METRIC_ORDER    = ["detection_tpr", "mauve"]
REGIME_ORDER    = ["base", "instruct"]
SCHEME_ORDER    = ["KGW", "Unigram", "SynthID", "EXPEdit", "DIP", "XSIR"]
GENERATOR_ORDER = ["mistral", "gemma", "qwen"]

VARIANT_TO_REGIME = {"base": "base", "instruct": "instruct"}

PUBLISHED_ANCHORS = [
    ("detection_tpr", "instruct", "XSIR",    "qwen",    3.787, 0.999),
    ("detection_tpr", "instruct", "Unigram", "mistral", 0.297, 0.998),
    ("detection_tpr", "instruct", "KGW",     "mistral", 0.026, 0.953),
    ("detection_tpr", "base",     "XSIR",    "qwen",    0.493, 0.989),
]
ANCHOR_ATOL_GE2   = 1e-3
ANCHOR_ATOL_SHARE = 1e-3

ANCHOR_KIND = "ge2_share"
ANCHOR_ATOL_MEAN = 0.002

FLAGSHIP_CELLS = [
    ("detection_tpr", "instruct", "XSIR",    "qwen"),
    ("detection_tpr", "instruct", "Unigram", "mistral"),
    ("detection_tpr", "instruct", "DIP",     "mistral"),
    ("detection_tpr", "instruct", "DIP",     "qwen"),
    ("detection_tpr", "base",     "XSIR",    "qwen"),
    ("mauve",         "base",     "EXPEdit", "mistral"),
    ("mauve",         "base",     "EXPEdit", "gemma"),
    ("mauve",         "base",     "EXPEdit", "qwen"),
]

NO_SINGLETON_PARTITIONS = {
    "indo_european": {
        "Indo-European":     ["eng", "nld", "fra", "spa", "por", "hin"],
        "non-Indo-European": ["arb", "zho", "jpn", "tur", "vie"],
    },
    "latin_script": {
        "Latin-script":     ["eng", "nld", "fra", "spa", "por", "tur", "vie"],
        "non-Latin-script": ["zho", "jpn", "arb", "hin"],
    },
}

CLOSED_FORM_ATOL = 1e-9
IDENTITY_ATOL = 1e-12
INVARIANCE_RTOL = 1e-12


def isnum(x) -> bool:
    return isinstance(x, float) and x == x and not math.isinf(x)


def percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(v for v in values if isnum(v))
    if not vals:
        return NAN
    if len(vals) == 1:
        return vals[0]
    idx = q * (len(vals) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return vals[int(idx)]
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)


def median(values: Sequence[float]) -> float:
    return percentile(values, 0.5)


CSV_PRECISION = 12


def fmt(v, nd: int = 6) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if not (v == v):
            return ""
        return f"{v:.{nd}g}"
    return str(v)


def pct(v: float, nd: int = 1) -> str:
    return f"{100.0 * v:.{nd}f}%" if isnum(v) else "n/a"


def decompose(by_lang: Dict[str, float],
              fam_of: Dict[str, str]) -> Tuple[float, float, float, float]:
    families: Dict[str, List[float]] = defaultdict(list)
    for lang, value in by_lang.items():
        families[fam_of[lang]].append(value)
    total, between, within = ge_decompose(families, alpha=2)
    share = between / total if (isnum(total) and total > 0) else NAN
    return total, between, within, share


def between_share_closed_form(by_lang: Dict[str, float],
                              fam_of: Dict[str, str]) -> float:
    values = list(by_lang.values())
    n = len(values)
    if n == 0:
        return NAN
    mu = sum(values) / n
    ss_total = sum((v - mu) ** 2 for v in values)
    if ss_total <= 0:
        return NAN
    groups: Dict[str, List[float]] = defaultdict(list)
    for lang, value in by_lang.items():
        groups[fam_of[lang]].append(value)
    ss_within = 0.0
    for vs in groups.values():
        if len(vs) < 2:
            continue
        m = sum(vs) / len(vs)
        ss_within += sum((v - m) ** 2 for v in vs)
    return 1.0 - ss_within / ss_total


def expected_partition_count(sizes: Sequence[int]) -> int:
    n = sum(sizes)
    count = math.factorial(n)
    for s in sizes:
        count //= math.factorial(s)
    for mult in Counter(sizes).values():
        count //= math.factorial(mult)
    return count


def enumerate_shape_partitions(elements: Sequence[str],
                               sizes: Sequence[int]) -> List[Tuple[Tuple[str, ...], ...]]:
    elems = tuple(sorted(elements))
    sizes = sorted(sizes, reverse=True)
    if sum(sizes) != len(elems):
        raise ValueError(f"sizes {sizes} do not sum to {len(elems)} elements")

    def rec(remaining: Tuple[str, ...], remaining_sizes: List[int]):
        if not remaining_sizes:
            yield ()
            return
        head = remaining[0]
        seen: set = set()
        for i, size in enumerate(remaining_sizes):
            if size in seen:
                continue
            seen.add(size)
            others = remaining_sizes[:i] + remaining_sizes[i + 1:]
            for combo in itertools.combinations(remaining[1:], size - 1):
                group = (head,) + combo
                gset = set(group)
                rest = tuple(e for e in remaining if e not in gset)
                for tail in rec(rest, others):
                    yield (tuple(sorted(group)),) + tail

    parts = list(rec(elems, sizes))
    expected = expected_partition_count(sizes)
    if len(parts) != expected:
        raise AssertionError(
            f"enumeration produced {len(parts)} partitions, expected {expected}")
    return parts


_PARTITION_CACHE: Dict[Tuple[Tuple[str, ...], Tuple[int, ...]],
                       List[Dict[str, str]]] = {}


def shape_matched_assignments(langs: Sequence[str],
                              sizes: Sequence[int]) -> List[Dict[str, str]]:
    key = (tuple(sorted(langs)), tuple(sorted(sizes, reverse=True)))
    cached = _PARTITION_CACHE.get(key)
    if cached is None:
        parts = enumerate_shape_partitions(key[0], key[1])
        cached = [{lang: f"G{i}" for i, group in enumerate(part) for lang in group}
                  for part in parts]
        _PARTITION_CACHE[key] = cached
    return cached


def registered_shape(langs: Sequence[str]) -> List[int]:
    counts: Counter = Counter(LANGUAGE_FAMILIES[l] for l in langs)
    return sorted(counts.values(), reverse=True)


def same_grouping(fam_a: Dict[str, str], fam_b: Dict[str, str],
                  langs: Sequence[str]) -> bool:
    def canon(fam):
        groups: Dict[str, List[str]] = defaultdict(list)
        for l in langs:
            groups[fam[l]].append(l)
        return sorted(tuple(sorted(g)) for g in groups.values())
    return canon(fam_a) == canon(fam_b)


CellKey = Tuple[str, str, str, str]


def load_detection_vectors(path: str, subset: str, alpha: float,
                           threshold_type: str) -> Dict[CellKey, Dict[str, float]]:
    out: Dict[CellKey, Dict[str, float]] = defaultdict(dict)
    for row in load_per_language_tpr(path):
        if row["subset"] != subset or row["threshold_type"] != threshold_type:
            continue
        try:
            if abs(float(row["alpha"]) - alpha) > 1e-12:
                continue
        except ValueError:
            continue
        regime = VARIANT_TO_REGIME.get(row["variant"])
        if regime is None:
            continue
        out[("detection_tpr", regime, row["watermark"], row["model"])][row["language"]] = row["tpr"]
    return dict(out)


def load_mauve_vectors(path: str, subset: str) -> Dict[CellKey, Dict[str, float]]:
    out: Dict[CellKey, Dict[str, float]] = defaultdict(dict)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("measurement") != "mauve" or row.get("subset") != subset:
                continue
            score = row.get("preservation_score", "")
            if score == "":
                continue
            try:
                value = float(score)
            except ValueError:
                continue
            regime = VARIANT_TO_REGIME.get(row["variant"])
            if regime is None:
                continue
            out[("mauve", regime, row["watermark"], row["model"])][row["language"]] = value
    return dict(out)


def load_published_shares(path: str, metric: str, subset: str,
                          alpha: Optional[float],
                          threshold_type: Optional[str]) -> Dict[CellKey, Tuple[float, float]]:
    out: Dict[CellKey, Tuple[float, float]] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("subset") != subset:
                continue
            if metric == "mauve" and row.get("measurement") != "mauve":
                continue
            if alpha is not None:
                try:
                    if abs(float(row.get("alpha", "nan")) - alpha) > 1e-12:
                        continue
                except ValueError:
                    continue
            if threshold_type is not None and row.get("threshold_type") != threshold_type:
                continue
            regime = VARIANT_TO_REGIME.get(row.get("variant", ""))
            if regime is None:
                continue

            def num(col):
                raw = row.get(col, "")
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    return NAN

            out[(metric, regime, row["watermark"], row["model"])] = (
                num("ge_2_total"), num("ge_2_between_share"))
    return out


def analysis1_restricted(by_lang: Dict[str, float]) -> Tuple[float, float, bool]:
    langs = [l for l in MULTI_MEMBER_LANGS if l in by_lang]
    if len(langs) < 3:
        return NAN, NAN, True
    sub = {l: by_lang[l] for l in langs}
    total, _between, _within, share = decompose(sub, LANGUAGE_FAMILIES)
    unstable = (not isnum(total)) or total < FLOOR_GE2
    return total, share, unstable


def analysis2_null(by_lang: Dict[str, float], observed_share: float,
                   observed_total: float) -> dict:
    langs = sorted(by_lang)
    sizes = registered_shape(langs)
    assignments = shape_matched_assignments(langs, sizes)

    shares: List[float] = []
    totals: List[float] = []
    max_closed_form_dev = 0.0
    identity_share = NAN
    identity_index = -1

    for idx, fam_of in enumerate(assignments):
        total, _between, _within, share = decompose(by_lang, fam_of)
        shares.append(share)
        totals.append(total)
        if isnum(share) and isnum(total) and total >= FLOOR_GE2:
            cf = between_share_closed_form(by_lang, fam_of)
            if isnum(cf):
                max_closed_form_dev = max(max_closed_form_dev, abs(cf - share))
        if identity_index < 0 and same_grouping(fam_of, LANGUAGE_FAMILIES, langs):
            identity_index = idx
            identity_share = share

    valid = [s for s in shares if isnum(s)]
    finite_totals = [t for t in totals if isnum(t)]
    total_spread = (max(finite_totals) - min(finite_totals)) if finite_totals else NAN

    n_ge = sum(1 for s in valid if s >= observed_share - 1e-12) if isnum(observed_share) else 0
    p_exact = (n_ge / len(assignments)) if isnum(observed_share) else NAN

    return {
        "n_assignments":        len(assignments),
        "shape":                sizes,
        "null_median":          median(valid),
        "null_p95":             percentile(valid, 0.95),
        "null_min":             min(valid) if valid else NAN,
        "null_max":             max(valid) if valid else NAN,
        "p_exact":              p_exact,
        "p_exact_count":        n_ge,
        "identity_share":       identity_share,
        "identity_found":       identity_index >= 0,
        "identity_dev":         (abs(identity_share - observed_share)
                                 if isnum(identity_share) and isnum(observed_share) else NAN),
        "total_invariance_dev": total_spread,
        "total_ref":            observed_total,
        "closed_form_dev":      max_closed_form_dev,
        "shares":               shares,
        "assignments":          assignments,
    }


def analysis3_lofo(by_lang: Dict[str, float]) -> dict:
    per_drop: List[Tuple[str, float, float]] = []
    for drop in sorted(by_lang):
        sub = {l: v for l, v in by_lang.items() if l != drop}
        if len(sub) < MIN_LANGS_FOR_FAIRNESS:
            continue
        total, _between, _within, share = decompose(sub, LANGUAGE_FAMILIES)
        per_drop.append((drop, total, share))

    valid = sorted(((share, drop) for drop, _t, share in per_drop if isnum(share)))
    if not valid:
        return {"lofo_min_share": NAN, "lofo_median_share": NAN,
                "lofo_argmin_lang": "", "lofo_n_valid": 0, "per_drop": per_drop}
    return {
        "lofo_min_share":    valid[0][0],
        "lofo_argmin_lang":  valid[0][1],
        "lofo_median_share": median([s for s, _ in valid]),
        "lofo_n_valid":      len(valid),
        "per_drop":          per_drop,
    }


def analysis4_nosingleton(by_lang: Dict[str, float]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name, partition in NO_SINGLETON_PARTITIONS.items():
        fam_of = {l: fam for fam, langs in partition.items() for l in langs}
        langs = sorted(l for l in by_lang if l in fam_of)
        if len(langs) < MIN_LANGS_FOR_FAIRNESS:
            continue
        sub = {l: by_lang[l] for l in langs}
        total, _between, _within, share = decompose(sub, fam_of)

        sizes = sorted(Counter(fam_of[l] for l in langs).values(), reverse=True)
        assignments = shape_matched_assignments(langs, sizes)
        shares = [decompose(sub, fa)[3] for fa in assignments]
        valid = [s for s in shares if isnum(s)]
        n_ge = sum(1 for s in valid if s >= share - 1e-12) if isnum(share) else 0

        out[name] = {
            "n_langs":       len(langs),
            "sizes":         sizes,
            "ge2_total":     total,
            "between_share": share,
            "null_median":   median(valid),
            "null_p95":      percentile(valid, 0.95),
            "p_exact":       (n_ge / len(assignments)) if isnum(share) else NAN,
            "n_assignments": len(assignments),
            "above_floor":   isnum(total) and total >= FLOOR_GE2,
        }
    return out


PERCELL_FIELDS = [
    "metric", "regime", "scheme", "generator", "n_langs", "status", "above_floor",
    "observed_ge2", "observed_between", "observed_within", "observed_between_share",
    "restricted_ge2", "restricted_between_share", "restricted_unstable",
    "null_median", "null_p95", "null_min", "null_max",
    "p_exact", "p_exact_count", "n_assignments",
    "lofo_min_share", "lofo_median_share", "lofo_argmin_lang", "lofo_n_valid",
    "observed_gt_null_p95",
]

NOSINGLETON_FIELDS = [
    "metric", "regime", "scheme", "generator", "partition", "group_sizes",
    "n_langs", "ge2_total", "between_share", "null_median", "null_p95",
    "p_exact", "n_assignments", "above_floor",
]


def analyse_cell(key: CellKey, by_lang: Dict[str, float]) -> dict:
    metric, regime, scheme, generator = key
    total, between, within, share = decompose(by_lang, LANGUAGE_FAMILIES)

    if not isnum(total):
        status = "degenerate"
    elif total < FLOOR_GE2:
        status = "below_floor"
    else:
        status = "ok"

    row = {
        "metric": metric, "regime": regime, "scheme": scheme, "generator": generator,
        "n_langs": len(by_lang), "status": status,
        "above_floor": status == "ok",
        "observed_ge2": total, "observed_between": between,
        "observed_within": within, "observed_between_share": share,
    }

    r_total, r_share, r_unstable = analysis1_restricted(by_lang)
    row.update({"restricted_ge2": r_total, "restricted_between_share": r_share,
                "restricted_unstable": r_unstable})

    null = analysis2_null(by_lang, share, total)
    row.update({
        "null_median": null["null_median"], "null_p95": null["null_p95"],
        "null_min": null["null_min"], "null_max": null["null_max"],
        "p_exact": null["p_exact"], "p_exact_count": null["p_exact_count"],
        "n_assignments": null["n_assignments"],
        "observed_gt_null_p95": (isnum(share) and isnum(null["null_p95"])
                                 and share > null["null_p95"]),
    })

    lofo = analysis3_lofo(by_lang)
    row.update({k: lofo[k] for k in
                ("lofo_min_share", "lofo_median_share", "lofo_argmin_lang", "lofo_n_valid")})

    return row, null, analysis4_nosingleton(by_lang)


def validate(vectors: Dict[CellKey, Dict[str, float]],
             published: Dict[CellKey, Tuple[float, float]],
             metrics: Sequence[str]) -> dict:
    anchors = []
    if ANCHOR_KIND == "mean_tpr":
        for metric, regime, scheme, generator, exp_mean in PUBLISHED_ANCHORS:
            if metric not in metrics:
                continue
            key = (metric, regime, scheme, generator)
            by_lang = vectors.get(key)
            if not by_lang:
                anchors.append({"cell": key, "found": False, "passed": False,
                                "n_langs": 0, "expected_mean": exp_mean,
                                "actual_mean": NAN, "d_mean": NAN,
                                "full_grid": False})
                continue
            got = sum(by_lang.values()) / len(by_lang)
            d = abs(got - exp_mean)
            full_grid = len(by_lang) == len(LANGUAGES)
            anchors.append({"cell": key, "found": True, "n_langs": len(by_lang),
                            "expected_mean": exp_mean, "actual_mean": got,
                            "d_mean": d, "full_grid": full_grid,
                            "passed": d <= ANCHOR_ATOL_MEAN})
        cross = {"n_compared": 0, "max_dev_ge2": 0.0, "max_dev_share": 0.0,
                 "worst_ge2_cell": None, "worst_share_cell": None, "n_missing": 0}
        for key, (pub_ge2, pub_share) in sorted(published.items()):
            by_lang = vectors.get(key)
            if not by_lang:
                cross["n_missing"] += 1
                continue
            total, _b, _w, share = decompose(by_lang, LANGUAGE_FAMILIES)
            cross["n_compared"] += 1
            if isnum(total) and isnum(pub_ge2):
                dev = abs(total - pub_ge2)
                if dev > cross["max_dev_ge2"]:
                    cross["max_dev_ge2"], cross["worst_ge2_cell"] = dev, key
            if isnum(share) and isnum(pub_share):
                dev = abs(share - pub_share)
                if dev > cross["max_dev_share"]:
                    cross["max_dev_share"], cross["worst_share_cell"] = dev, key
        return {"anchors": anchors, "cross_check": cross,
                "all_passed": bool(anchors) and all(a["passed"] for a in anchors)}

    for metric, regime, scheme, generator, exp_ge2, exp_share in PUBLISHED_ANCHORS:
        if metric not in metrics:
            continue
        key = (metric, regime, scheme, generator)
        by_lang = vectors.get(key)
        if not by_lang:
            anchors.append({"cell": key, "found": False, "passed": False,
                            "expected_ge2": exp_ge2, "expected_share": exp_share,
                            "actual_ge2": NAN, "actual_share": NAN,
                            "d_ge2": NAN, "d_share": NAN, "n_langs": 0})
            continue
        total, _b, _w, share = decompose(by_lang, LANGUAGE_FAMILIES)
        d_ge2 = abs(total - exp_ge2) if isnum(total) else NAN
        d_share = abs(share - exp_share) if isnum(share) else NAN
        anchors.append({
            "cell": key, "found": True, "n_langs": len(by_lang),
            "expected_ge2": exp_ge2, "actual_ge2": total, "d_ge2": d_ge2,
            "expected_share": exp_share, "actual_share": share, "d_share": d_share,
            "passed": (isnum(d_ge2) and d_ge2 <= ANCHOR_ATOL_GE2
                       and isnum(d_share) and d_share <= ANCHOR_ATOL_SHARE),
        })

    cross = {"n_compared": 0, "max_dev_ge2": 0.0, "max_dev_share": 0.0,
             "worst_ge2_cell": None, "worst_share_cell": None, "n_missing": 0}
    for key, (pub_ge2, pub_share) in sorted(published.items()):
        by_lang = vectors.get(key)
        if not by_lang:
            cross["n_missing"] += 1
            continue
        total, _b, _w, share = decompose(by_lang, LANGUAGE_FAMILIES)
        cross["n_compared"] += 1
        if isnum(total) and isnum(pub_ge2):
            dev = abs(total - pub_ge2)
            if dev > cross["max_dev_ge2"]:
                cross["max_dev_ge2"], cross["worst_ge2_cell"] = dev, key
        if isnum(share) and isnum(pub_share):
            dev = abs(share - pub_share)
            if dev > cross["max_dev_share"]:
                cross["max_dev_share"], cross["worst_share_cell"] = dev, key

    return {"anchors": anchors, "cross_check": cross,
            "all_passed": bool(anchors) and all(a["passed"] for a in anchors)}


def cell_sort_key(key: CellKey):
    metric, regime, scheme, generator = key
    def rank(v, order):
        return order.index(v) if v in order else len(order)
    return (rank(metric, METRIC_ORDER), rank(regime, REGIME_ORDER),
            rank(scheme, SCHEME_ORDER), rank(generator, GENERATOR_ORDER),
            metric, regime, scheme, generator)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tpr_csv", default="results/detection/per_language_tpr.csv")
    p.add_argument("--quality_csv", default="results/quality/quality_per_language.csv")
    p.add_argument("--detection_summary_csv", default="results/detection/detection_disparity.csv")
    p.add_argument("--quality_summary_csv", default="results/quality/quality_disparity.csv")
    p.add_argument("--output_dir", default="analysis/percell_family_test")
    p.add_argument("--subset", default="all")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--threshold_type", default="global")
    p.add_argument("--floor", type=float, default=FLOOR_GE2,
                   help="GE_2 floor for headline inclusion (default 1e-3)")
    p.add_argument("--metrics", nargs="+", default=METRIC_ORDER, choices=METRIC_ORDER)
    p.add_argument("--skip_anchor_validation", action="store_true",
                   help="run the analysis even if the published anchors are absent "
                        "(diagnostic only)")
    p.add_argument("--selftest", action="store_true",
                   help="run the synthetic end-to-end self-test and exit")
    return p.parse_args(argv)


def run(args) -> int:
    global FLOOR_GE2
    FLOOR_GE2 = args.floor

    os.makedirs(args.output_dir, exist_ok=True)
    null_dir = os.path.join(args.output_dir, "null_distributions")
    os.makedirs(null_dir, exist_ok=True)

    vectors: Dict[CellKey, Dict[str, float]] = {}

    if "detection_tpr" in args.metrics:
        if not os.path.exists(args.tpr_csv):
            sys.exit(f"ERROR: {args.tpr_csv} not found (run detection_fairness/per_language_tpr.py).")
        vectors.update(load_detection_vectors(args.tpr_csv, args.subset, args.alpha,
                                              args.threshold_type))
    if "mauve" in args.metrics:
        if not os.path.exists(args.quality_csv):
            sys.exit(f"ERROR: {args.quality_csv} not found (run quality_fairness/quality_disparity.py).")
        vectors.update(load_mauve_vectors(args.quality_csv, args.subset))

    if not vectors:
        sys.exit("ERROR: no per-language vectors matched the requested "
                 f"subset/alpha/threshold_type ({args.subset}/{args.alpha}/{args.threshold_type}).")

    n_det = sum(1 for k in vectors if k[0] == "detection_tpr")
    n_mau = sum(1 for k in vectors if k[0] == "mauve")
    print(f"Loaded {len(vectors)} cells  (detection {n_det}, mauve {n_mau})")

    published: Dict[CellKey, Tuple[float, float]] = {}
    if "detection_tpr" in args.metrics:
        published.update(load_published_shares(args.detection_summary_csv, "detection_tpr",
                                               args.subset, args.alpha, args.threshold_type))
    if "mauve" in args.metrics:
        published.update(load_published_shares(args.quality_summary_csv, "mauve",
                                               args.subset, None, None))

    unknown = {l for v in vectors.values() for l in v} - set(LANGUAGE_FAMILIES)
    if unknown:
        print(f"  WARNING: dropping languages absent from LANGUAGE_FAMILIES: "
              f"{sorted(unknown)}", file=sys.stderr)
        vectors = {k: {l: x for l, x in v.items() if l in LANGUAGE_FAMILIES}
                   for k, v in vectors.items()}

    validation = validate(vectors, published, args.metrics)
    for a in validation["anchors"]:
        verdict = "PASS" if a["passed"] else ("NOT FOUND" if not a["found"] else "FAIL")
        if ANCHOR_KIND == "mean_tpr":
            print(f"  anchor {'/'.join(a['cell'])}: "
                  f"mean {fmt(a['actual_mean'], 6) or 'n/a'} vs {a['expected_mean']} "
                  f"(n_langs={a['n_langs']}) | {verdict}")
        else:
            print(f"  anchor {'/'.join(a['cell'])}: "
                  f"GE2 {fmt(a['actual_ge2'], 6) or 'n/a'} vs {a['expected_ge2']} | "
                  f"share {fmt(a['actual_share'], 6) or 'n/a'} vs {a['expected_share']} | "
                  f"{verdict}")
    print(f"  cross-check: {validation['cross_check']['n_compared']} cells, "
          f"max |Δ share| = {fmt(validation['cross_check']['max_dev_share'], 3)}")

    if not validation["all_passed"] and not args.skip_anchor_validation:
        print("\nVALIDATION FAILED — published anchors not reproduced; stopped. "
              "No analysis performed.", file=sys.stderr)
        return 1

    rows: List[dict] = []
    ns_rows: List[dict] = []
    inv = {"max_identity_dev": 0.0, "max_total_spread": 0.0, "max_closed_form_dev": 0.0,
           "n_identity_found": 0, "n_cells": 0, "n_assignments_full": 0}

    for key in sorted(vectors, key=cell_sort_key):
        by_lang = vectors[key]
        if len(by_lang) < MIN_LANGS_FOR_FAIRNESS:
            print(f"  skip {'/'.join(key)}: only {len(by_lang)} languages")
            continue
        row, null, ns = analyse_cell(key, by_lang)
        rows.append(row)
        inv["n_cells"] += 1
        if null["identity_found"]:
            inv["n_identity_found"] += 1
        if isnum(null["identity_dev"]):
            inv["max_identity_dev"] = max(inv["max_identity_dev"], null["identity_dev"])
        if isnum(null["total_invariance_dev"]):
            inv["max_total_spread"] = max(inv["max_total_spread"], null["total_invariance_dev"])
        inv["max_closed_form_dev"] = max(inv["max_closed_form_dev"], null["closed_form_dev"])
        if len(by_lang) == len(LANGUAGES):
            inv["n_assignments_full"] = null["n_assignments"]

        for name, res in ns.items():
            ns_rows.append({
                "metric": key[0], "regime": key[1], "scheme": key[2], "generator": key[3],
                "partition": name, "group_sizes": "+".join(str(s) for s in res["sizes"]),
                "n_langs": res["n_langs"], "ge2_total": res["ge2_total"],
                "between_share": res["between_share"], "null_median": res["null_median"],
                "null_p95": res["null_p95"], "p_exact": res["p_exact"],
                "n_assignments": res["n_assignments"], "above_floor": res["above_floor"],
            })

        if key in FLAGSHIP_CELLS:
            out = os.path.join(null_dir, "__".join(key) + ".csv")
            with open(out, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["assignment_index", "group_3", "group_2", "singletons",
                            "between_share", "is_identity"])
                for i, (fam_of, share) in enumerate(zip(null["assignments"], null["shares"])):
                    groups: Dict[str, List[str]] = defaultdict(list)
                    for lang in sorted(by_lang):
                        groups[fam_of[lang]].append(lang)
                    sized = sorted(groups.values(), key=lambda g: (-len(g), g))
                    g3 = ",".join(sized[0]) if sized and len(sized[0]) > 1 else ""
                    g2 = ",".join(sized[1]) if len(sized) > 1 and len(sized[1]) > 1 else ""
                    singles = ",".join(sorted(g[0] for g in sized if len(g) == 1))
                    w.writerow([i, g3, g2, singles, fmt(share, 12),
                                same_grouping(fam_of, LANGUAGE_FAMILIES, sorted(by_lang))])
            print(f"  wrote {out}")

        print(f"  {'/'.join(key):<44} GE2={fmt(row['observed_ge2'], 4):>10} "
              f"share={pct(row['observed_between_share']):>7} "
              f"null_med={pct(row['null_median']):>7} p={fmt(row['p_exact'], 3):>8} "
              f"{'' if row['above_floor'] else '[' + row['status'] + ']'}")

    problems = []
    if inv["max_identity_dev"] > IDENTITY_ATOL:
        problems.append(f"identity assignment deviates by {inv['max_identity_dev']:g} "
                        f"(> {IDENTITY_ATOL:g})")
    if inv["max_closed_form_dev"] > CLOSED_FORM_ATOL:
        problems.append(f"closed-form check deviates by {inv['max_closed_form_dev']:g} "
                        f"(> {CLOSED_FORM_ATOL:g})")
    if inv["n_identity_found"] != inv["n_cells"]:
        problems.append(f"identity assignment not located in "
                        f"{inv['n_cells'] - inv['n_identity_found']} cells")
    if problems:
        sys.exit("INVARIANT FAILURE: " + "; ".join(problems))

    percell_path = os.path.join(args.output_dir, "percell_results.csv")
    with open(percell_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(PERCELL_FIELDS)
        for r in rows:
            w.writerow([fmt(r.get(c), CSV_PRECISION) for c in PERCELL_FIELDS])
    print(f"Wrote {percell_path}  ({len(rows)} rows)")

    ns_path = os.path.join(args.output_dir, "nosingleton_partitions.csv")
    with open(ns_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(NOSINGLETON_FIELDS)
        for r in ns_rows:
            w.writerow([fmt(r.get(c), CSV_PRECISION) for c in NOSINGLETON_FIELDS])
    print(f"Wrote {ns_path}  ({len(ns_rows)} rows)")

    return 0


def selftest() -> int:
    print("Self-test: synthetic fixtures, no real data touched.")
    langs = list(LANGUAGES)

    parts = enumerate_shape_partitions(langs, registered_shape(langs))
    assert len(parts) == 4620, len(parts)
    assert len(set(tuple(sorted(p)) for p in parts)) == 4620, "duplicate partitions"
    print(f"  [ok] enumeration: {len(parts)} distinct shape-matched partitions "
          f"(C(11,3)·C(8,2) = 4620)")

    total, between, within, share = decompose(
        dict(zip(("a", "b", "c", "d"), (1.0, 2.0, 3.0, 4.0))),
        {"a": "G", "b": "G", "c": "H", "d": "I"})
    assert abs(total - 0.1) < 1e-12 and abs(within - 0.01) < 1e-12, (total, within)
    assert abs(share - 0.9) < 1e-12, share
    print("  [ok] decomposition matches the hand-computed GE_2 example (share 0.9)")

    aligned = {l: {"Germanic": 0.2, "Romance": 0.9}.get(LANGUAGE_FAMILIES[l], 0.1 * (i + 3))
               for i, l in enumerate(langs)}
    _t, _b, w_al, s_al = decompose(aligned, LANGUAGE_FAMILIES)
    assert abs(w_al) < 1e-15 and abs(s_al - 1.0) < 1e-12, (w_al, s_al)
    print("  [ok] family-aligned vector gives within=0, between-share=1")

    generic = {l: 0.05 + 0.07 * i + (0.11 if i % 3 == 0 else 0.0) for i, l in enumerate(langs)}
    obs_total, _b, _w, obs_share = decompose(generic, LANGUAGE_FAMILIES)
    null = analysis2_null(generic, obs_share, obs_total)
    assert null["n_assignments"] == 4620
    assert null["identity_found"] and null["identity_dev"] <= IDENTITY_ATOL, null["identity_dev"]
    assert null["total_invariance_dev"] <= abs(obs_total) * INVARIANCE_RTOL + 1e-15, \
        null["total_invariance_dev"]
    assert null["closed_form_dev"] <= CLOSED_FORM_ATOL, null["closed_form_dev"]
    assert 0.0 <= null["null_median"] <= 1.0 and 0.0 <= null["null_p95"] <= 1.0
    assert 0.0 < null["p_exact"] <= 1.0
    print(f"  [ok] null invariants: identity Δ={null['identity_dev']:.2e}, "
          f"total spread={null['total_invariance_dev']:.2e}, "
          f"closed-form Δ={null['closed_form_dev']:.2e}, "
          f"median={null['null_median']:.3f}, p95={null['null_p95']:.3f}, "
          f"p={null['p_exact']:.4f}")

    zero = {l: 0.0 for l in langs}
    t0, _b0, _w0, s0 = decompose(zero, LANGUAGE_FAMILIES)
    assert not isnum(t0) and not isnum(s0), (t0, s0)
    print("  [ok] all-zero vector yields undefined GE_2 (flagged degenerate)")

    for name, partition in NO_SINGLETON_PARTITIONS.items():
        covered = sorted(l for ls in partition.values() for l in ls)
        assert covered == sorted(langs), (name, covered)
        assert all(len(ls) >= 2 for ls in partition.values()), name
        sizes = sorted((len(ls) for ls in partition.values()), reverse=True)
        n_expected = expected_partition_count(sizes)
        assert len(shape_matched_assignments(langs, sizes)) == n_expected
        print(f"  [ok] partition '{name}': sizes {sizes}, {n_expected} shape-matched assignments")

    tmp = tempfile.mkdtemp(prefix="percell_selftest_")
    tpr_path = os.path.join(tmp, "per_language_tpr.csv")
    quality_path = os.path.join(tmp, "quality_per_language.csv")
    with open(tpr_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "watermark", "dataset", "subset", "alpha",
                    "threshold_type", "language", "tpr", "insufficient_samples"])
        for i, lang in enumerate(langs):
            w.writerow(["gemma", "base", "KGW", "flores", "all", "0.01", "global",
                        lang, f"{generic[lang]:.6f}", "False"])
            w.writerow(["gemma", "base", "KGW", "flores", "all", "0.05", "global",
                        lang, "0.5", "False"])
    with open(quality_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "watermark", "subset", "measurement",
                    "language", "family", "preservation_score"])
        for lang in langs:
            w.writerow(["gemma", "base", "KGW", "all", "mauve", lang,
                        LANGUAGE_FAMILIES[lang], f"{aligned[lang]:.6f}"])

    out_dir = os.path.join(tmp, "out")
    rc = run(parse_args([
        "--tpr_csv", tpr_path, "--quality_csv", quality_path,
        "--detection_summary_csv", os.path.join(tmp, "missing_d2.csv"),
        "--quality_summary_csv", os.path.join(tmp, "missing_q2.csv"),
        "--output_dir", out_dir, "--skip_anchor_validation",
    ]))
    assert rc == 0, rc
    with open(os.path.join(out_dir, "percell_results.csv"), encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert len(got) == 2, got
    det = next(r for r in got if r["metric"] == "detection_tpr")
    mau = next(r for r in got if r["metric"] == "mauve")
    assert det["n_langs"] == "11" and det["n_assignments"] == "4620"
    assert abs(float(det["observed_between_share"]) - obs_share) < 1e-9
    assert abs(float(mau["observed_between_share"]) - 1.0) < 1e-9
    assert float(mau["p_exact"]) > 0
    assert os.path.exists(os.path.join(out_dir, "nosingleton_partitions.csv"))
    print(f"  [ok] end-to-end run on synthetic CSVs -> {out_dir}")

    rc_fail = run(parse_args([
        "--tpr_csv", tpr_path, "--quality_csv", quality_path,
        "--detection_summary_csv", os.path.join(tmp, "missing_d2.csv"),
        "--quality_summary_csv", os.path.join(tmp, "missing_q2.csv"),
        "--output_dir", os.path.join(tmp, "out_fail"),
    ]))
    assert rc_fail == 1, rc_fail
    assert not os.path.exists(os.path.join(tmp, "out_fail", "percell_results.csv"))
    print("  [ok] failed anchors stop the run (exit 1, no results CSV)")

    global PUBLISHED_ANCHORS
    saved = PUBLISHED_ANCHORS
    truth_ge2, _b, _w, truth_share = decompose(generic, LANGUAGE_FAMILIES)
    PUBLISHED_ANCHORS = [("detection_tpr", "base", "KGW", "gemma",
                          round(truth_ge2, 3), round(truth_share, 3))]
    try:
        rc_pass = run(parse_args([
            "--tpr_csv", tpr_path, "--quality_csv", quality_path,
            "--detection_summary_csv", os.path.join(tmp, "missing_d2.csv"),
            "--quality_summary_csv", os.path.join(tmp, "missing_q2.csv"),
            "--output_dir", os.path.join(tmp, "out_pass"),
        ]))
    finally:
        PUBLISHED_ANCHORS = saved
    assert rc_pass == 0, rc_pass
    assert os.path.exists(os.path.join(tmp, "out_pass", "percell_results.csv"))
    print("  [ok] matching anchors pass validation and the run proceeds")

    print("\nSelf-test PASSED.")
    return 0


def main() -> int:
    args = parse_args()
    if args.selftest:
        return selftest()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
