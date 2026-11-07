import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional


LANGUAGE_FAMILIES = {
    "eng": "Germanic",
    "nld": "Germanic",
    "fra": "Romance",
    "spa": "Romance",
    "por": "Romance",
    "hin": "Indic",
    "arb": "Semitic",
    "zho": "Sinitic",
    "jpn": "Japonic",
    "tur": "Turkic",
    "vie": "Austroasiatic",
}

LANGUAGES = sorted(LANGUAGE_FAMILIES.keys())
N_LANGS   = len(LANGUAGES)

BOTTOM_QUINTILE_N = max(1, int(0.2 * N_LANGS))

GE0_EPS = 1e-6

MIN_LANGS_FOR_FAIRNESS = 5


def load_mauve(path: str):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("insufficient_samples", "False").lower() == "true":
                continue
            score_str = row.get("mauve_score", "")
            if score_str == "":
                continue
            yield {
                "model":              row["model"],
                "variant":            row["variant"],
                "language":           row["language"],
                "watermark":          row["watermark"],
                "subset":             row["subset"],
                "measurement":        "mauve",
                "preservation_score": float(score_str),
            }


def load_ppl_ratio(path: str):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("insufficient_samples", "False").lower() == "true":
                continue
            score_str = row.get("log_ppl_ratio", "")
            if score_str == "":
                continue
            try:
                log_ratio = float(score_str)
            except ValueError:
                continue
            ref = row.get("reference_model", "")
            measurement = f"ppl_pres_{ref}" if ref else "ppl_preservation"
            yield {
                "model":              row["model"],
                "variant":            row["variant"],
                "language":           row["language"],
                "watermark":          row["watermark"],
                "subset":             row["subset"],
                "measurement":        measurement,
                "preservation_score": math.exp(-abs(log_ratio)),
            }


def load_bertscore(path: str):
    if not os.path.exists(path):
        return
    fields = (
        ("bertscore_f1_rescaled", "bertscore_f1_rescaled_mean"),
        ("bertscore_f1_raw",      "bertscore_f1_raw_mean"),
    )
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("insufficient_samples", "False").lower() == "true":
                continue
            for measurement, col in fields:
                score_str = row.get(col, "")
                if score_str == "":
                    continue
                try:
                    score = float(score_str)
                except ValueError:
                    continue
                yield {
                    "model":              row["model"],
                    "variant":            row["variant"],
                    "language":           row["language"],
                    "watermark":          row["watermark"],
                    "subset":             row["subset"],
                    "measurement":        measurement,
                    "preservation_score": score,
                }


QUALITY_PER_LANGUAGE_FIELDS = [
    "model", "variant", "watermark", "subset", "measurement",
    "language", "family", "preservation_score",
]


def write_per_language(path: str, rows: List[dict]):
    rows = sorted(rows, key=lambda r: (r["model"], r["variant"], r["watermark"],
                                       r["subset"], r["measurement"], r["language"]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QUALITY_PER_LANGUAGE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in QUALITY_PER_LANGUAGE_FIELDS})


def quintile_mean(values: List[float], n: int) -> float:
    if not values:
        return float("nan")
    return sum(sorted(values)[:n]) / n


def ge_alpha(values: List[float], alpha: float) -> float:
    n = len(values)
    if n == 0:
        return float("nan")
    mu = sum(values) / n
    if mu == 0:
        return float("nan")
    if alpha == 0:
        return sum(math.log(mu / v) for v in values) / n
    if alpha == 1:
        return sum((v / mu) * math.log(v / mu) for v in values if v > 0) / n
    return sum((v / mu) ** alpha - 1 for v in values) / (n * alpha * (alpha - 1))


def ge_decompose(values_by_family: Dict[str, List[float]], alpha: float):
    all_values = [v for vs in values_by_family.values() for v in vs]
    n  = len(all_values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mu = sum(all_values) / n
    if mu == 0:
        return float("nan"), float("nan"), float("nan")

    total = ge_alpha(all_values, alpha)

    if alpha == 0:
        between = sum(
            len(vs) * math.log(mu / (sum(vs) / len(vs)))
            for vs in values_by_family.values()
        ) / n
    elif alpha == 1:
        between = sum(
            (len(vs) / n) * (sum(vs) / len(vs) / mu) * math.log((sum(vs) / len(vs)) / mu)
            for vs in values_by_family.values() if sum(vs) > 0
        )
    else:
        acc = 0.0
        for vs in values_by_family.values():
            n_g  = len(vs)
            if n_g == 0:
                continue
            mu_g = sum(vs) / n_g
            acc += (n_g / n) * ((mu_g / mu) ** alpha - 1)
        between = acc / (alpha * (alpha - 1))

    within = total - between
    return total, between, within


Q_ROB_FIELDS = [
    "model", "variant", "watermark", "subset",
    "measurement_a", "measurement_b",
    "n_langs", "spearman_rho",
]


def _avg_ranks(values: List[float]) -> List[float]:
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    rx = _avg_ranks(xs)
    ry = _avg_ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx2 = sum((rx[i] - mx) ** 2 for i in range(n))
    dy2 = sum((ry[i] - my) ** 2 for i in range(n))
    if dx2 == 0 or dy2 == 0:
        return float("nan")
    return num / math.sqrt(dx2 * dy2)


def compute_q_robustness(per_language_records: List[dict]) -> List[dict]:
    by_cell: Dict[tuple, Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict))
    for r in per_language_records:
        cell = (r["model"], r["variant"], r["watermark"], r["subset"])
        by_cell[cell][r["measurement"]][r["language"]] = r["preservation_score"]

    out: List[dict] = []
    for (model, variant, watermark, subset), m_to_lang_score in sorted(by_cell.items()):
        measurements = sorted(m_to_lang_score.keys())
        if len(measurements) < 2:
            continue
        for i, ma in enumerate(measurements):
            for mb in measurements[i + 1:]:
                common = sorted(set(m_to_lang_score[ma]) & set(m_to_lang_score[mb]))
                if len(common) < MIN_LANGS_FOR_FAIRNESS:
                    continue
                xs = [m_to_lang_score[ma][l] for l in common]
                ys = [m_to_lang_score[mb][l] for l in common]
                out.append({
                    "model":         model,
                    "variant":       variant,
                    "watermark":     watermark,
                    "subset":        subset,
                    "measurement_a": ma,
                    "measurement_b": mb,
                    "n_langs":       len(common),
                    "spearman_rho":  spearman_rho(xs, ys),
                })
    return out


SUMMARY_FIELDS = [
    "model", "variant", "watermark", "subset", "measurement",
    "n_langs", "n_families", "mean",
    "min_lang", "min", "max_lang", "max", "bottom_quintile_mean",
    "di_max_count_below_80", "di_max_langs_below_80",
    "di_eng_count_below_80", "di_eng_langs_below_80",
    "ge_2_total", "ge_2_between", "ge_2_within", "ge_2_between_share",
    "ge_0_total", "ge_0_between", "ge_0_within", "ge_0_between_share",
]


def summarize_cell(rows: List[dict]) -> Optional[dict]:
    by_lang = {r["language"]: r["preservation_score"] for r in rows}
    if len(by_lang) < MIN_LANGS_FOR_FAIRNESS:
        return None

    langs  = sorted(by_lang.keys())
    values = [by_lang[l] for l in langs]
    n      = len(values)
    mu     = sum(values) / n

    max_v = max(values)
    min_v = min(values)
    max_lang = langs[values.index(max_v)]
    min_lang = langs[values.index(min_v)]

    di_max_below = [l for l in langs if (by_lang[l] / max_v) < 0.8] if max_v > 0 else langs

    eng_v = by_lang.get("eng")
    if eng_v is not None and eng_v > 0:
        di_eng_below       = [l for l in langs if (by_lang[l] / eng_v) < 0.8]
        di_eng_below_count = len(di_eng_below)
    else:
        di_eng_below       = []
        di_eng_below_count = ""

    families = defaultdict(list)
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
        "n_langs":              n,
        "n_families":           len(families),
        "mean":                 mu,
        "min_lang":             min_lang,
        "min":                  min_v,
        "max_lang":             max_lang,
        "max":                  max_v,
        "bottom_quintile_mean": quintile_mean(values, BOTTOM_QUINTILE_N),
        "di_max_count_below_80": len(di_max_below),
        "di_max_langs_below_80": ",".join(di_max_below),
        "di_eng_count_below_80": di_eng_below_count,
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
    p.add_argument("--mauve_csv",     default="results/quality/mauve.csv")
    p.add_argument("--bertscore_csv", default="results/quality/bertscore.csv")
    p.add_argument("--ppl_csv",       default="results/quality/ppl_ratio.csv")
    p.add_argument("--output_dir",    default="results/quality")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    per_language_records: List[dict] = []
    for r in load_mauve(args.mauve_csv):
        r["family"] = LANGUAGE_FAMILIES.get(r["language"], "?")
        per_language_records.append(r)
    n_mauve = len(per_language_records)

    for r in load_bertscore(args.bertscore_csv):
        r["family"] = LANGUAGE_FAMILIES.get(r["language"], "?")
        per_language_records.append(r)
    n_bertscore = len(per_language_records) - n_mauve

    for r in load_ppl_ratio(args.ppl_csv):
        r["family"] = LANGUAGE_FAMILIES.get(r["language"], "?")
        per_language_records.append(r)
    n_ppl = len(per_language_records) - n_mauve - n_bertscore

    if not per_language_records:
        sys.exit(f"No usable rows found in {args.mauve_csv}, "
                 f"{args.bertscore_csv}, or {args.ppl_csv}. Run "
                 f"quality_fairness/mauve_score.py, quality_fairness/bertscore_score.py, and/or "
                 f"quality_fairness/ppl_ratio.py first.")
    print(f"Loaded {n_mauve} mauve rows, {n_bertscore} bertscore rows, "
          f"{n_ppl} ppl rows")

    per_language_path = os.path.join(args.output_dir, "quality_per_language.csv")
    write_per_language(per_language_path, per_language_records)
    print(f"Wrote {per_language_path}  ({len(per_language_records)} rows)")

    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in per_language_records:
        key = (r["model"], r["variant"], r["watermark"], r["subset"], r["measurement"])
        groups[key].append(r)

    summaries: List[dict] = []
    n_skipped = 0
    for (model, variant, watermark, subset, measurement), rows in sorted(groups.items()):
        s = summarize_cell(rows)
        if s is None:
            n_skipped += 1
            continue
        s.update({
            "model":       model,
            "variant":     variant,
            "watermark":   watermark,
            "subset":      subset,
            "measurement": measurement,
        })
        summaries.append(s)

    summary_path = os.path.join(args.output_dir, "quality_disparity.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k, "") for k in SUMMARY_FIELDS})

    print(f"Wrote {summary_path}  ({len(summaries)} rows, "
          f"{n_skipped} cells skipped for n_langs < {MIN_LANGS_FOR_FAIRNESS})")

    qrob_rows = compute_q_robustness(per_language_records)
    qrob_path = os.path.join(args.output_dir, "quality_robustness.csv")
    with open(qrob_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=Q_ROB_FIELDS)
        w.writeheader()
        for r in qrob_rows:
            w.writerow({k: r[k] for k in Q_ROB_FIELDS})
    print(f"Wrote {qrob_path}  ({len(qrob_rows)} rows)")

    measurements_present = sorted({s["measurement"] for s in summaries
                                   if s["subset"] == "all"})
    for measurement in measurements_present:
        print()
        print(f"Headline (subset=all, measurement={measurement}, mean over models)")
        print("=" * 80)
        by_vw: Dict[tuple, List[dict]] = defaultdict(list)
        for s in summaries:
            if s["subset"] != "all" or s["measurement"] != measurement:
                continue
            by_vw[(s["variant"], s["watermark"])].append(s)
        if not by_vw:
            print("  (no cells)")
            continue

        print(f"  {'variant':<9} {'scheme':<8} | {'mean':>7} {'min':>7} {'BQ2':>7} | "
              f"{'DI<.8':>6} {'GE2 b%':>7}")
        print(f"  {'-'*9} {'-'*8}-+-{'-'*7} {'-'*7} {'-'*7}-+-{'-'*6} {'-'*7}")
        for (variant, watermark) in sorted(by_vw.keys()):
            ss   = by_vw[(variant, watermark)]
            mean = sum(s["mean"] for s in ss) / len(ss)
            mn   = sum(s["min"]  for s in ss) / len(ss)
            bq   = sum(s["bottom_quintile_mean"] for s in ss) / len(ss)
            di   = sum(s["di_max_count_below_80"] for s in ss) / len(ss)
            bs_vals = [s["ge_2_between_share"] for s in ss
                       if isinstance(s["ge_2_between_share"], float)
                       and s["ge_2_between_share"] == s["ge_2_between_share"]]
            bs   = sum(bs_vals) / len(bs_vals) if bs_vals else float("nan")
            bs_str = f"{bs:.0%}" if bs == bs else "  -- "
            print(f"  {variant:<9} {watermark:<8} | {mean:>7.3f} {mn:>7.3f} {bq:>7.3f} | "
                  f"{di:>6.1f} {bs_str:>7}")

    if qrob_rows:
        print()
        print("Cross-metric rank agreement (subset=all, mean Spearman across (model, variant, watermark) cells)")
        print("=" * 80)
        by_pair: Dict[tuple, List[float]] = defaultdict(list)
        for r in qrob_rows:
            if r["subset"] != "all":
                continue
            rho = r["spearman_rho"]
            if isinstance(rho, float) and rho == rho:
                by_pair[(r["measurement_a"], r["measurement_b"])].append(rho)
        for (ma, mb), rhos in sorted(by_pair.items()):
            mean_rho = sum(rhos) / len(rhos)
            print(f"  {ma:<24} vs {mb:<24}  rho={mean_rho:>+.3f}  (n_cells={len(rhos)})")


if __name__ == "__main__":
    main()
