import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quality_fairness.quality_disparity import (
    LANGUAGE_FAMILIES, ge_alpha, ge_decompose, GE0_EPS,
)

MIN_LANGS_FOR_FAIRNESS = 5
DI_THRESHOLD = 0.8
N_LANGS_TOTAL = 11
BOTTOM_QUINTILE_N = max(1, int(0.2 * N_LANGS_TOTAL))

DETECTION_SUMMARY_FIELDS = [
    "model", "variant", "watermark", "dataset", "subset", "alpha",
    "threshold_type",
    "n_langs", "n_families", "mean_tpr", "bottom_quintile_mean",
    "min_lang", "min_tpr", "max_lang", "max_tpr",
    "di_max_count_below_80", "di_max_langs_below_80",
    "di_eng_count_below_80", "di_eng_langs_below_80",
    "ge_2_total", "ge_2_between", "ge_2_within", "ge_2_between_share",
    "ge_0_total", "ge_0_between", "ge_0_within", "ge_0_between_share",
]


def load_per_language_tpr(path: str):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("insufficient_samples", "False").lower() == "true":
                continue
            tpr_str = row.get("tpr", "")
            if tpr_str == "" or tpr_str.lower() == "nan":
                continue
            try:
                tpr = float(tpr_str)
            except ValueError:
                continue
            yield {
                "model":          row["model"],
                "variant":        row["variant"],
                "watermark":      row["watermark"],
                "dataset":        row["dataset"],
                "subset":         row["subset"],
                "alpha":          row["alpha"],
                "threshold_type": row["threshold_type"],
                "language":       row["language"],
                "tpr":            tpr,
            }


def bottom_quintile_mean(values: List[float], n: int) -> float:
    if not values:
        return float("nan")
    k = min(n, len(values))
    return sum(sorted(values)[:k]) / k


def summarize_cell(rows: List[dict]) -> Optional[dict]:
    by_lang: Dict[str, float] = {r["language"]: r["tpr"] for r in rows}
    if len(by_lang) < MIN_LANGS_FOR_FAIRNESS:
        return None

    langs  = sorted(by_lang.keys())
    values = [by_lang[l] for l in langs]
    n      = len(values)
    mu     = sum(values) / n
    bq     = bottom_quintile_mean(values, BOTTOM_QUINTILE_N)

    max_v = max(values)
    min_v = min(values)
    max_lang = langs[values.index(max_v)]
    min_lang = langs[values.index(min_v)]

    if max_v > 0:
        di_max_below = [l for l in langs if (by_lang[l] / max_v) < DI_THRESHOLD]
    else:
        di_max_below = list(langs)

    eng_v = by_lang.get("eng")
    if eng_v is not None and eng_v > 0:
        di_eng_below = [l for l in langs if (by_lang[l] / eng_v) < DI_THRESHOLD]
        di_eng_count = len(di_eng_below)
    else:
        di_eng_below = []
        di_eng_count = ""

    families: Dict[str, List[float]] = defaultdict(list)
    for l in langs:
        families[LANGUAGE_FAMILIES[l]].append(by_lang[l])

    ge2_total, ge2_between, ge2_within = ge_decompose(families, alpha=2)

    families_eps = {fam: [max(v, GE0_EPS) for v in vs] for fam, vs in families.items()}
    ge0_total, ge0_between, ge0_within = ge_decompose(families_eps, alpha=0)

    def safe_share(num, den):
        if not (isinstance(den, float) and den == den) or den <= 0:
            return float("nan")
        return num / den

    return {
        "n_langs":               n,
        "n_families":            len(families),
        "mean_tpr":              mu,
        "bottom_quintile_mean":  bq,
        "min_lang":              min_lang,
        "min_tpr":               min_v,
        "max_lang":              max_lang,
        "max_tpr":               max_v,
        "di_max_count_below_80": len(di_max_below),
        "di_max_langs_below_80": ",".join(di_max_below),
        "di_eng_count_below_80": di_eng_count,
        "di_eng_langs_below_80": ",".join(di_eng_below),
        "ge_2_total":            ge2_total,
        "ge_2_between":          ge2_between,
        "ge_2_within":           ge2_within,
        "ge_2_between_share":    safe_share(ge2_between, ge2_total),
        "ge_0_total":            ge0_total,
        "ge_0_between":          ge0_between,
        "ge_0_within":           ge0_within,
        "ge_0_between_share":    safe_share(ge0_between, ge0_total),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tpr_csv",      default="results/detection/per_language_tpr.csv")
    p.add_argument("--summary_csv", default="results/detection/detection_disparity.csv")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.tpr_csv):
        sys.exit(f"ERROR: {args.tpr_csv} not found. Run detection_fairness/per_language_tpr.py first.")

    records = list(load_per_language_tpr(args.tpr_csv))
    if not records:
        sys.exit(f"No usable rows in {args.tpr_csv} after filtering insufficient_samples.")
    print(f"Loaded {len(records)} per-language TPR rows from {args.tpr_csv}")

    groups: Dict[Tuple[str, str, str, str, str, str, str], List[dict]] = defaultdict(list)
    for r in records:
        key = (r["model"], r["variant"], r["watermark"], r["dataset"],
               r["subset"], r["alpha"], r["threshold_type"])
        groups[key].append(r)

    summaries: List[dict] = []
    n_skipped = 0
    for key, rows in sorted(groups.items()):
        s = summarize_cell(rows)
        if s is None:
            n_skipped += 1
            continue
        model, variant, watermark, dataset, subset, alpha, ttype = key
        s.update({
            "model": model, "variant": variant, "watermark": watermark,
            "dataset": dataset, "subset": subset, "alpha": alpha,
            "threshold_type": ttype,
        })
        summaries.append(s)

    os.makedirs(os.path.dirname(args.summary_csv) or ".", exist_ok=True)
    with open(args.summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETECTION_SUMMARY_FIELDS)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k, "") for k in DETECTION_SUMMARY_FIELDS})

    print(f"Wrote {args.summary_csv}  ({len(summaries)} cells, "
          f"{n_skipped} cells skipped for n_langs < {MIN_LANGS_FOR_FAIRNESS})")

    print()
    print("Headline rollup (subset=all, alpha=0.01, threshold_type=global)")
    print("=" * 100)
    print(f"  {'variant':<9} {'model':<8} {'scheme':<8} | "
          f"{'meanTPR':>7} {'BQ2':>6} {'minTPR':>7} {'minLang':>7} | "
          f"{'DI<.8':>5} {'DIeng<.8':>8} | {'GE2':>6} {'GE2 b%':>7}")
    print(f"  {'-'*9} {'-'*8} {'-'*8}-+-{'-'*7} {'-'*6} {'-'*7} {'-'*7}-+-"
          f"{'-'*5} {'-'*8}-+-{'-'*6} {'-'*7}")
    for s in summaries:
        if (s["subset"] != "all" or s["alpha"] != "0.01"
                or s["threshold_type"] != "global"):
            continue
        di_eng_str = (str(s["di_eng_count_below_80"])
                      if s["di_eng_count_below_80"] != "" else "  --")
        ge2 = s["ge_2_total"]
        ge2_str = f"{ge2:>6.3f}" if isinstance(ge2, float) and ge2 == ge2 else "   -- "
        bs = s["ge_2_between_share"]
        if isinstance(ge2, float) and ge2 == ge2 and ge2 < 0.01:
            bs_str = "  -- "
        elif isinstance(bs, float) and bs == bs:
            bs_str = f"{bs:.0%}"
        else:
            bs_str = "  -- "
        print(f"  {s['variant']:<9} {s['model']:<8} {s['watermark']:<8} | "
              f"{s['mean_tpr']:>7.3f} {s['bottom_quintile_mean']:>6.3f} "
              f"{s['min_tpr']:>7.3f} {s['min_lang']:>7} | "
              f"{s['di_max_count_below_80']:>5} {di_eng_str:>8} | "
              f"{ge2_str} {bs_str:>7}")


if __name__ == "__main__":
    main()
