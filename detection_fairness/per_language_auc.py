import argparse
import csv
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

VARIANT_DATASET = {"base": "flores", "instruct": "aya"}

LABEL_FOR_SUBSET = {
    "on_target_only":  "on_target",
    "off_target_only": "off_target",
    "code_mixed_only": "code_mixed",
    "uncertain_only":  "uncertain",
}

MIN_N_PER_SIDE = 100

AUC_FIELDS = [
    "model", "variant", "watermark", "dataset", "subset", "language",
    "n_pos", "n_neg", "auc",
    "insufficient_samples", "reason_insufficient",
]


def _avg_ranks(values: np.ndarray) -> np.ndarray:
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def auc(wm: np.ndarray, nwm: np.ndarray) -> float:
    n_pos, n_neg = len(wm), len(nwm)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    combined = np.concatenate([wm, nwm])
    ranks = _avg_ranks(combined)
    sum_pos = ranks[:n_pos].sum()
    U = sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores_csv",
                   default="results/detection/detection_scores.csv")
    p.add_argument("--auc_csv",
                   default="results/detection/per_language_auc.csv")
    p.add_argument("--models",     nargs="+", default=MODELS,     choices=MODELS)
    p.add_argument("--variants",   nargs="+", default=VARIANTS,   choices=VARIANTS)
    p.add_argument("--watermarks", nargs="+", default=WATERMARKS, choices=WATERMARKS)
    p.add_argument("--languages",  nargs="+", default=LANGUAGES,  choices=LANGUAGES)
    p.add_argument("--subsets",    nargs="+", default=SUBSETS,    choices=SUBSETS)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.scores_csv):
        sys.exit(f"ERROR: {args.scores_csv} not found. Run detection_fairness/build_score_table.py first.")

    print(f"Loading scores from {args.scores_csv}...")
    scores = load_scores_index(args.scores_csv)
    print(f"  loaded {sum(len(v) for v in scores.values())} score rows "
          f"across {len(scores)} (cell, side) keys")

    rows: List[dict] = []
    n_insuf = 0
    for variant in args.variants:
        dataset = VARIANT_DATASET[variant]
        for model in args.models:
            for watermark in args.watermarks:
                for subset in args.subsets:
                    for lang in args.languages:
                        wm_recs  = scores.get((model, variant, watermark, lang, "wm"),  [])
                        nwm_recs = scores.get((model, variant, watermark, lang, "nwm"), [])
                        wm  = filter_scores(wm_recs,  subset)
                        nwm = filter_scores(nwm_recs, subset)
                        n_pos, n_neg = len(wm), len(nwm)

                        if min(n_pos, n_neg) < MIN_N_PER_SIDE:
                            a = float("nan")
                            insufficient = True
                            reason = f"min(n_pos,n_neg)={min(n_pos, n_neg)} < {MIN_N_PER_SIDE}"
                            n_insuf += 1
                        else:
                            a = auc(wm, nwm)
                            insufficient = False
                            reason = ""

                        rows.append({
                            "model": model, "variant": variant, "watermark": watermark,
                            "dataset": dataset, "subset": subset, "language": lang,
                            "n_pos": n_pos, "n_neg": n_neg, "auc": a,
                            "insufficient_samples": insufficient,
                            "reason_insufficient": reason,
                        })

    write_csv(args.auc_csv, AUC_FIELDS, rows)
    print(f"\nWrote {args.auc_csv}  ({len(rows)} rows, {n_insuf} insufficient)")

    print()
    print("Spot-check: gemma/base/Unigram, subset=all")
    print("=" * 60)
    print(f"  {'lang':<5} {'n_pos':>6} {'n_neg':>6} {'AUC':>6}")
    for r in rows:
        if (r["model"] == "gemma" and r["variant"] == "base"
                and r["watermark"] == "Unigram" and r["subset"] == "all"):
            auc_str = "NaN" if r["insufficient_samples"] else f"{r['auc']:.3f}"
            print(f"  {r['language']:<5} {r['n_pos']:>6} {r['n_neg']:>6} {auc_str:>6}")


if __name__ == "__main__":
    main()
