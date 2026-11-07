import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

LANGUAGES  = ["eng", "fra", "spa", "por", "nld", "zho", "jpn", "arb", "hin", "tur", "vie"]
MODELS     = ["mistral", "gemma", "qwen"]
VARIANTS   = ["base", "instruct"]
WATERMARKS = ["KGW", "Unigram", "SynthID", "EXPEdit", "DIP", "XSIR"]
SUBSETS    = ["all", "on_target_only", "off_target_only"]
ALPHAS     = [0.01, 0.05]

VARIANT_DATASET = {"base": "flores", "instruct": "aya"}

LABEL_FOR_SUBSET = {
    "on_target_only":  "on_target",
    "off_target_only": "off_target",
    "code_mixed_only": "code_mixed",
    "uncertain_only":  "uncertain",
}

MIN_N_POS = 100
MIN_N_NEG_GLOBAL = 500

THRESHOLD_FIELDS = [
    "model", "variant", "watermark", "dataset", "subset", "alpha",
    "threshold_type", "language_or_global",
    "threshold_value", "n_neg", "empirical_fpr",
    "insufficient_samples", "reason_insufficient",
]

TPR_FIELDS = [
    "model", "variant", "watermark", "dataset", "subset", "alpha",
    "threshold_type", "language",
    "threshold_value", "tpr",
    "n_pos", "n_pos_above_threshold", "n_neg", "empirical_fpr",
    "wm_score_mean", "wm_score_std", "nwm_score_mean", "nwm_score_std",
    "gap_std",
    "insufficient_samples", "reason_insufficient",
]


def min_n_neg_per_lang(alpha: float) -> int:
    return 200 if alpha <= 0.01 else 100


def calibrate_threshold(nwm_scores: np.ndarray, alpha: float) -> Tuple[float, float]:
    if len(nwm_scores) == 0:
        return float("nan"), float("nan")
    threshold = float(np.quantile(nwm_scores, 1.0 - alpha, method="lower"))
    empirical_fpr = float(np.mean(nwm_scores > threshold))
    return threshold, empirical_fpr


def gap_std(wm: np.ndarray, nwm: np.ndarray) -> float:
    if len(wm) < 2 or len(nwm) < 2:
        return float("nan")
    pooled = math.sqrt((wm.var(ddof=1) + nwm.var(ddof=1)) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((wm.mean() - nwm.mean()) / pooled)


def mean_std(arr: np.ndarray) -> Tuple[float, float]:
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1))


ScoresIndex = Dict[Tuple[str, str, str, str, str], List[Tuple[float, str]]]


def load_scores_index(scores_csv: str) -> ScoresIndex:
    out: ScoresIndex = defaultdict(list)
    with open(scores_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                sc = float(row["score"])
            except (TypeError, ValueError):
                continue
            key = (row["model"], row["variant"], row["watermark"],
                   row["language"], row["side"])
            out[key].append((sc, row["lid_label"]))
    return dict(out)


def filter_scores(records: List[Tuple[float, str]], subset: str) -> np.ndarray:
    if subset == "all":
        return np.array([s for s, _ in records], dtype=np.float64)
    label = LABEL_FOR_SUBSET[subset]
    return np.array([s for s, lab in records if lab == label], dtype=np.float64)


def write_csv(path: str, fields: List[str], rows: List[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def compute_tpr_for_group(
    scores: ScoresIndex,
    model: str, variant: str, watermark: str,
    subset: str, alpha: float,
    languages: List[str],
) -> Tuple[List[dict], List[dict]]:
    dataset = VARIANT_DATASET[variant]
    threshold_rows: List[dict] = []
    tpr_rows: List[dict] = []

    wm_by_lang: Dict[str, np.ndarray] = {}
    nwm_by_lang: Dict[str, np.ndarray] = {}
    for lang in languages:
        wm_recs  = scores.get((model, variant, watermark, lang, "wm"), [])
        nwm_recs = scores.get((model, variant, watermark, lang, "nwm"), [])
        wm_by_lang[lang]  = filter_scores(wm_recs, subset)
        nwm_by_lang[lang] = filter_scores(nwm_recs, subset)

    n_neg_floor = min_n_neg_per_lang(alpha)
    for lang in languages:
        nwm = nwm_by_lang[lang]
        wm  = wm_by_lang[lang]
        n_neg, n_pos = len(nwm), len(wm)

        wm_mean, wm_sd   = mean_std(wm)
        nwm_mean, nwm_sd = mean_std(nwm)
        gs = gap_std(wm, nwm)

        if n_neg < n_neg_floor:
            tau = float("nan")
            emp_fpr = float("nan")
            insufficient = True
            reason = f"n_neg={n_neg} < {n_neg_floor}"
        else:
            tau, emp_fpr = calibrate_threshold(nwm, alpha)
            insufficient = False
            reason = ""

        threshold_rows.append({
            "model": model, "variant": variant, "watermark": watermark,
            "dataset": dataset, "subset": subset, "alpha": alpha,
            "threshold_type": "per_language", "language_or_global": lang,
            "threshold_value": tau, "n_neg": n_neg, "empirical_fpr": emp_fpr,
            "insufficient_samples": insufficient,
            "reason_insufficient": reason,
        })

        if insufficient or n_pos < MIN_N_POS:
            tpr = float("nan")
            n_above = 0
            reason_tpr = reason if insufficient else f"n_pos={n_pos} < {MIN_N_POS}"
            insufficient_tpr = True
        else:
            n_above = int(np.sum(wm > tau))
            tpr = n_above / n_pos
            reason_tpr = ""
            insufficient_tpr = False

        tpr_rows.append({
            "model": model, "variant": variant, "watermark": watermark,
            "dataset": dataset, "subset": subset, "alpha": alpha,
            "threshold_type": "per_language", "language": lang,
            "threshold_value": tau, "tpr": tpr,
            "n_pos": n_pos, "n_pos_above_threshold": n_above,
            "n_neg": n_neg, "empirical_fpr": emp_fpr,
            "wm_score_mean": wm_mean, "wm_score_std": wm_sd,
            "nwm_score_mean": nwm_mean, "nwm_score_std": nwm_sd,
            "gap_std": gs,
            "insufficient_samples": insufficient_tpr,
            "reason_insufficient": reason_tpr,
        })

    pooled_nwm = np.concatenate([nwm_by_lang[l] for l in languages]) \
        if any(len(nwm_by_lang[l]) for l in languages) else np.array([], dtype=np.float64)
    n_neg_global = len(pooled_nwm)

    if n_neg_global < MIN_N_NEG_GLOBAL:
        tau_glob = float("nan")
        emp_fpr_glob = float("nan")
        insufficient_glob = True
        reason_glob = f"pooled n_neg={n_neg_global} < {MIN_N_NEG_GLOBAL}"
    else:
        tau_glob, emp_fpr_glob = calibrate_threshold(pooled_nwm, alpha)
        insufficient_glob = False
        reason_glob = ""

    threshold_rows.append({
        "model": model, "variant": variant, "watermark": watermark,
        "dataset": dataset, "subset": subset, "alpha": alpha,
        "threshold_type": "global", "language_or_global": "global",
        "threshold_value": tau_glob, "n_neg": n_neg_global,
        "empirical_fpr": emp_fpr_glob,
        "insufficient_samples": insufficient_glob,
        "reason_insufficient": reason_glob,
    })

    for lang in languages:
        wm  = wm_by_lang[lang]
        nwm = nwm_by_lang[lang]
        n_neg_lang, n_pos = len(nwm), len(wm)

        wm_mean, wm_sd   = mean_std(wm)
        nwm_mean, nwm_sd = mean_std(nwm)
        gs = gap_std(wm, nwm)

        if insufficient_glob or n_pos < MIN_N_POS:
            tpr = float("nan")
            n_above = 0
            reason_tpr = reason_glob if insufficient_glob else f"n_pos={n_pos} < {MIN_N_POS}"
            insufficient_tpr = True
        else:
            n_above = int(np.sum(wm > tau_glob))
            tpr = n_above / n_pos
            reason_tpr = ""
            insufficient_tpr = False

        tpr_rows.append({
            "model": model, "variant": variant, "watermark": watermark,
            "dataset": dataset, "subset": subset, "alpha": alpha,
            "threshold_type": "global", "language": lang,
            "threshold_value": tau_glob, "tpr": tpr,
            "n_pos": n_pos, "n_pos_above_threshold": n_above,
            "n_neg": n_neg_lang, "empirical_fpr": emp_fpr_glob,
            "wm_score_mean": wm_mean, "wm_score_std": wm_sd,
            "nwm_score_mean": nwm_mean, "nwm_score_std": nwm_sd,
            "gap_std": gs,
            "insufficient_samples": insufficient_tpr,
            "reason_insufficient": reason_tpr,
        })

    return threshold_rows, tpr_rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores_csv",
                   default="results/detection/detection_scores.csv")
    p.add_argument("--thresholds_csv",
                   default="results/detection/thresholds.csv")
    p.add_argument("--tpr_csv",
                   default="results/detection/per_language_tpr.csv")
    p.add_argument("--models",     nargs="+", default=MODELS,     choices=MODELS)
    p.add_argument("--variants",   nargs="+", default=VARIANTS,   choices=VARIANTS)
    p.add_argument("--watermarks", nargs="+", default=WATERMARKS, choices=WATERMARKS)
    p.add_argument("--languages",  nargs="+", default=LANGUAGES,  choices=LANGUAGES)
    p.add_argument("--subsets",    nargs="+", default=SUBSETS,    choices=SUBSETS)
    p.add_argument("--alphas",     nargs="+", type=float, default=ALPHAS)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.scores_csv):
        sys.exit(f"ERROR: {args.scores_csv} not found. Run detection_fairness/build_score_table.py first.")

    print(f"Loading scores from {args.scores_csv}...")
    scores = load_scores_index(args.scores_csv)
    print(f"  loaded {sum(len(v) for v in scores.values())} score rows "
          f"across {len(scores)} (cell, side) keys")

    threshold_rows: List[dict] = []
    tpr_rows: List[dict] = []

    n_groups = (len(args.models) * len(args.variants) * len(args.watermarks)
                * len(args.subsets) * len(args.alphas))
    n_done = 0

    for variant in args.variants:
        for model in args.models:
            for watermark in args.watermarks:
                for subset in args.subsets:
                    for alpha in args.alphas:
                        thr, group_tpr_rows = compute_tpr_for_group(
                            scores, model, variant, watermark,
                            subset, alpha, args.languages,
                        )
                        threshold_rows.extend(thr)
                        tpr_rows.extend(group_tpr_rows)
                        n_done += 1

    print(f"\nProcessed {n_done}/{n_groups} groups.")
    print(f"  thresholds.csv rows: {len(threshold_rows)}")
    print(f"  per_language_tpr.csv rows: {len(tpr_rows)}")

    n_thr_insuf = sum(1 for r in threshold_rows if r["insufficient_samples"])
    n_tpr_insufficient  = sum(1 for r in tpr_rows if r["insufficient_samples"])
    print(f"  insufficient thresholds: {n_thr_insuf}")
    print(f"  insufficient TPR rows:   {n_tpr_insufficient}")

    write_csv(args.thresholds_csv, THRESHOLD_FIELDS, threshold_rows)
    write_csv(args.tpr_csv, TPR_FIELDS, tpr_rows)
    print(f"\nWrote thresholds -> {args.thresholds_csv}")
    print(f"Wrote per-language TPR -> {args.tpr_csv}")


if __name__ == "__main__":
    main()
