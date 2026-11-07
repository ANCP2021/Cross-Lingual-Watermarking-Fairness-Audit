import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.lid_subsets import load_subset, KNOWN_SUBSETS

LANGUAGES  = ["eng", "fra", "spa", "por", "nld", "zho", "jpn", "arb", "hin", "tur", "vie"]
MODELS     = ["mistral", "gemma", "qwen"]
VARIANTS   = ["base", "instruct"]
WATERMARKS = ["KGW", "Unigram", "SynthID", "EXPEdit", "DIP", "XSIR"]
DEFAULT_SUBSETS = ["all", "on_target_only", "off_target_only"]

VARIANT_DATASET = {"base": "flores", "instruct": "aya"}

CSV_FIELDS = [
    "model", "variant", "language", "watermark", "dataset", "subset",
    "reference_model",
    "n_pos", "n_neg", "n_truncated_wm", "n_truncated_nwm",
    "mean_nll_wm", "mean_nll_nwm", "log_ppl_ratio",
    "ppl_wm", "ppl_nwm", "ppl_ratio",
    "insufficient_samples", "min_samples_threshold",
    "computed_at",
]


def load_ppl_cache(path: str) -> Dict[str, Tuple[float, int]]:
    if not os.path.exists(path):
        return {}
    out: Dict[str, Tuple[float, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            mean_nll = r.get("mean_nll")
            n_eval   = r.get("n_eval_tokens", 0)
            if mean_nll is None or n_eval == 0:
                continue
            try:
                v = float(mean_nll)
            except (TypeError, ValueError):
                continue
            if v != v:
                continue
            out[str(r["prompt_id"])] = (v, int(r.get("n_truncated", 0)))
    return out


def existing_keys(path: str) -> Set[Tuple[str, str, str, str, str, str]]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.add((row["model"], row["variant"], row["language"],
                     row["watermark"], row["subset"], row["reference_model"]))
    return out


def append_row(path: str, row: dict):
    is_new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference_model", default="xglm-7.5b",
                   help="Reference model tag — names the subdir of --ppl_dir.")
    p.add_argument("--ppl_dir",     default="quality/ppl",
                   help="Stage-1 cache root. Cells live in {ppl_dir}/{ref_tag}/.")
    p.add_argument("--lid_dir",     default="lid_categorized")
    p.add_argument("--output_path", default="results/quality/ppl_ratio.csv")
    p.add_argument("--models",     default=",".join(MODELS))
    p.add_argument("--variants",   default=",".join(VARIANTS))
    p.add_argument("--watermarks", default=",".join(WATERMARKS))
    p.add_argument("--languages",  default=",".join(LANGUAGES))
    p.add_argument("--subsets",    default=",".join(DEFAULT_SUBSETS))
    p.add_argument("--min_samples", type=int, default=100,
                   help="Minimum n on each side; below this, mark insufficient and skip math.")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute (cell, subset, reference_model) tuples already in the CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    cache_root = os.path.join(args.ppl_dir, args.reference_model)
    if not os.path.isdir(cache_root):
        sys.exit(f"PPL cache dir not found: {cache_root}.  "
                 f"Run quality_fairness/ppl_nll.py --reference_model {args.reference_model} first.")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    models     = [s for s in args.models.split(",")     if s]
    variants   = [s for s in args.variants.split(",")   if s]
    watermarks = [s for s in args.watermarks.split(",") if s]
    languages  = [s for s in args.languages.split(",")  if s]
    subsets    = [s for s in args.subsets.split(",")    if s]

    bad = [s for s in subsets if s not in KNOWN_SUBSETS]
    if bad:
        sys.exit(f"Unknown subset(s): {bad}. Expected one of {sorted(KNOWN_SUBSETS)}.")

    done = set() if args.overwrite else existing_keys(args.output_path)
    if done:
        print(f"Skipping {len(done)} (cell, subset, ref) tuples already in {args.output_path}")

    cells = [(m, v, w, l)
             for m in models for v in variants for w in watermarks for l in languages]
    total = len(cells) * len(subsets)
    print(f"\n{total} (cell, subset) tuples to attempt for ref={args.reference_model}")

    n_computed = 0
    n_insuf    = 0
    n_skipped  = 0

    for (model, variant, watermark, lang) in cells:
        dataset = VARIANT_DATASET[variant]
        wm_path  = os.path.join(cache_root,
            f"{dataset}_{model}_{variant}_{watermark}_{lang}_ppl.jsonl")
        nwm_path = os.path.join(cache_root,
            f"{dataset}_{model}_{variant}_nowatermark_{lang}_ppl.jsonl")

        wm_cache  = load_ppl_cache(wm_path)
        nwm_cache = load_ppl_cache(nwm_path)
        if not wm_cache or not nwm_cache:
            n_skipped += len(subsets)
            continue

        for subset in subsets:
            key = (model, variant, lang, watermark, subset, args.reference_model)
            if key in done:
                continue

            try:
                wm_keep  = load_subset(model, variant, lang, watermark,     subset, args.lid_dir)
                nwm_keep = load_subset(model, variant, lang, "nowatermark", subset, args.lid_dir)
            except FileNotFoundError:
                n_skipped += 1
                continue

            wm_pairs  = [v for pid, v in wm_cache.items()  if pid in wm_keep]
            nwm_pairs = [v for pid, v in nwm_cache.items() if pid in nwm_keep]
            n_pos = len(wm_pairs)
            n_neg = len(nwm_pairs)
            n_trunc_wm  = sum(t for _, t in wm_pairs)
            n_trunc_nwm = sum(t for _, t in nwm_pairs)

            base_row = {
                "model": model, "variant": variant, "language": lang,
                "watermark": watermark, "dataset": dataset, "subset": subset,
                "reference_model": args.reference_model,
                "n_pos": n_pos, "n_neg": n_neg,
                "n_truncated_wm":  n_trunc_wm,
                "n_truncated_nwm": n_trunc_nwm,
                "min_samples_threshold": args.min_samples,
                "computed_at": datetime.now(timezone.utc).isoformat(),
            }

            if min(n_pos, n_neg) < args.min_samples:
                row = {**base_row,
                       "mean_nll_wm": "", "mean_nll_nwm": "", "log_ppl_ratio": "",
                       "ppl_wm": "", "ppl_nwm": "", "ppl_ratio": "",
                       "insufficient_samples": True}
                append_row(args.output_path, row)
                n_insuf += 1
                done.add(key)
                continue

            wm_nll  = [v[0] for v in wm_pairs]
            nwm_nll = [v[0] for v in nwm_pairs]
            mean_wm  = sum(wm_nll)  / n_pos
            mean_nwm = sum(nwm_nll) / n_neg
            log_ratio = mean_wm - mean_nwm
            row = {**base_row,
                   "mean_nll_wm":   mean_wm,
                   "mean_nll_nwm":  mean_nwm,
                   "log_ppl_ratio": log_ratio,
                   "ppl_wm":        math.exp(mean_wm),
                   "ppl_nwm":       math.exp(mean_nwm),
                   "ppl_ratio":     math.exp(log_ratio),
                   "insufficient_samples": False}
            append_row(args.output_path, row)
            n_computed += 1
            done.add(key)

    print(f"\nDone. computed={n_computed}  insufficient={n_insuf}  "
          f"skipped(missing)={n_skipped}")
    print(f"Results: {args.output_path}")


if __name__ == "__main__":
    main()
