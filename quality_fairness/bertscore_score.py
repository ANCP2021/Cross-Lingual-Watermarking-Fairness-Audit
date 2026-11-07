import argparse
import csv
import json
import os
import sys
import time
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

ENCODER_ID  = "xlm-roberta-large"
NUM_LAYERS  = 17

ISO3_TO_ISO1 = {
    "eng": "en", "fra": "fr", "spa": "es", "por": "pt", "nld": "nl",
    "zho": "zh", "jpn": "ja", "arb": "ar", "hin": "hi", "tur": "tr",
    "vie": "vi",
}

CSV_FIELDS = [
    "model", "variant", "language", "watermark", "dataset", "subset",
    "n_pairs", "n_wm_total", "n_nwm_total",
    "bertscore_f1_raw_mean", "bertscore_f1_raw_median", "bertscore_f1_raw_std",
    "bertscore_f1_rescaled_mean", "bertscore_f1_rescaled_median", "bertscore_f1_rescaled_std",
    "f1_baseline",
    "insufficient_samples",
    "encoder", "num_layers", "lang_code",
    "min_samples_threshold", "computed_at",
]


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def index_by_prompt_id(records: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for r in records:
        pid = r.get("prompt_id")
        if pid is not None:
            out[str(pid)] = r
    return out


def existing_keys(path: str) -> Set[Tuple[str, str, str, str, str]]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.add((row["model"], row["variant"], row["language"],
                     row["watermark"], row["subset"]))
    return out


def append_row(path: str, row: dict):
    is_new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def load_baselines(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: baselines file not found: {path}\n"
            f"Run: python quality_fairness/bertscore_baselines.py first."
        )
    out: Dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["language"]
            try:
                out[lang] = float(row["f1_baseline"])
            except (KeyError, ValueError):
                continue
    return out


def score_and_rescale(scorer, cands: List[str], refs: List[str], baseline: float):
    import torch
    P, R, F1 = scorer.score(cands, refs, verbose=False)
    raw_mean   = float(F1.mean())
    raw_median = float(torch.median(F1))
    raw_std    = float(F1.std(unbiased=False))

    F1_resc      = (F1 - baseline) / (1.0 - baseline)
    resc_mean   = float(F1_resc.mean())
    resc_median = float(torch.median(F1_resc))
    resc_std    = float(F1_resc.std(unbiased=False))

    return raw_mean, raw_median, raw_std, resc_mean, resc_median, resc_std


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",     default="outputs")
    p.add_argument("--lid_dir",       default="lid_categorized")
    p.add_argument("--baselines_path", default="results/quality/bertscore_baselines.csv")
    p.add_argument("--output_path",   default="results/quality/bertscore.csv")
    p.add_argument("--models",     default=",".join(MODELS))
    p.add_argument("--variants",   default=",".join(VARIANTS))
    p.add_argument("--watermarks", default=",".join(WATERMARKS))
    p.add_argument("--languages",  default=",".join(LANGUAGES))
    p.add_argument("--subsets",    default=",".join(DEFAULT_SUBSETS))
    p.add_argument("--min_samples", type=int, default=100,
                   help="Minimum n_pairs after subset filtering; below this, mark insufficient.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--overwrite",  action="store_true",
                   help="Recompute (cell, subset) tuples already present in the output CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        sys.exit(f"Input dir not found: {args.input_dir}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    try:
        import bert_score
        import torch
    except ImportError as e:
        sys.exit(f"ERROR: required package not installed: {e}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("ERROR: --device cuda but torch.cuda.is_available() is False.")

    models     = [s for s in args.models.split(",")     if s]
    variants   = [s for s in args.variants.split(",")   if s]
    watermarks = [s for s in args.watermarks.split(",") if s]
    languages  = [s for s in args.languages.split(",")  if s]
    subsets    = [s for s in args.subsets.split(",")    if s]

    bad = [s for s in subsets if s not in KNOWN_SUBSETS]
    if bad:
        sys.exit(f"Unknown subset(s): {bad}. Expected one of {sorted(KNOWN_SUBSETS)}.")
    bad_lang = [l for l in languages if l not in ISO3_TO_ISO1]
    if bad_lang:
        sys.exit(f"No ISO-1 mapping for language(s): {bad_lang}.")

    baselines = load_baselines(args.baselines_path)
    missing = [l for l in languages if l not in baselines]
    if missing:
        sys.exit(
            f"ERROR: missing baseline rows for language(s): {missing}\n"
            f"  in {args.baselines_path}\n"
            f"  Re-run bertscore_baselines.py to populate them."
        )
    for l in languages:
        b = baselines[l]
        if not (0.0 < b < 1.0):
            sys.exit(f"ERROR: baseline for {l} is {b}, expected in (0, 1). "
                     f"Inspect {args.baselines_path}.")
    print("Per-language F1 baselines:")
    for l in languages:
        print(f"  {l}: {baselines[l]:.4f}")

    done = set() if args.overwrite else existing_keys(args.output_path)
    if done:
        print(f"\nSkipping {len(done)} (cell, subset) tuples already in {args.output_path}")

    print(f"\nLoading BERTScorer (encoder={ENCODER_ID}, layer={NUM_LAYERS}, "
          f"rescale=False, device={args.device})")
    t_load = time.time()
    scorer = bert_score.BERTScorer(
        model_type=ENCODER_ID,
        num_layers=NUM_LAYERS,
        rescale_with_baseline=False,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"  loaded in {time.time() - t_load:.1f}s")

    cells = [(m, v, w, l)
             for m in models for v in variants for w in watermarks for l in languages]
    total       = len(cells) * len(subsets)
    n_done      = 0
    n_computed  = 0
    n_insuf     = 0
    n_skipped   = 0
    t0 = time.time()
    print(f"\n{total} (cell, subset) tuples to attempt")

    for (model, variant, watermark, lang) in cells:
        dataset = VARIANT_DATASET[variant]
        wm_path  = os.path.join(args.input_dir,
                                f"{dataset}_{model}_{variant}_{watermark}_{lang}.jsonl")
        nwm_path = os.path.join(args.input_dir,
                                f"{dataset}_{model}_{variant}_nowatermark_{lang}.jsonl")

        if not (os.path.exists(wm_path) and os.path.exists(nwm_path)):
            n_skipped += len(subsets)
            continue

        wm_by_pid  = index_by_prompt_id(load_jsonl(wm_path))
        nwm_by_pid = index_by_prompt_id(load_jsonl(nwm_path))
        n_wm_total, n_nwm_total = len(wm_by_pid), len(nwm_by_pid)

        paired_pids = sorted(set(wm_by_pid) & set(nwm_by_pid))
        if not paired_pids:
            n_skipped += len(subsets)
            continue

        baseline = baselines[lang]

        for subset in subsets:
            key = (model, variant, lang, watermark, subset)
            if key in done:
                n_done += 1
                continue

            if subset == "all":
                keep_pids = paired_pids
            else:
                try:
                    wm_keep  = load_subset(model, variant, lang, watermark,    subset, args.lid_dir)
                    nwm_keep = load_subset(model, variant, lang, "nowatermark", subset, args.lid_dir)
                except FileNotFoundError:
                    n_skipped += 1
                    continue
                keep_pids = [p for p in paired_pids if p in wm_keep and p in nwm_keep]

            cands, refs = [], []
            for pid in keep_pids:
                c = (wm_by_pid[pid].get("generated_text")  or "").strip()
                r = (nwm_by_pid[pid].get("generated_text") or "").strip()
                if c and r:
                    cands.append(c)
                    refs.append(r)
            n_pairs = len(cands)

            insufficient = n_pairs < args.min_samples
            row = {
                "model":                 model,
                "variant":               variant,
                "language":              lang,
                "watermark":             watermark,
                "dataset":               dataset,
                "subset":                subset,
                "n_pairs":               n_pairs,
                "n_wm_total":            n_wm_total,
                "n_nwm_total":           n_nwm_total,
                "bertscore_f1_raw_mean":      "",
                "bertscore_f1_raw_median":    "",
                "bertscore_f1_raw_std":       "",
                "bertscore_f1_rescaled_mean":   "",
                "bertscore_f1_rescaled_median": "",
                "bertscore_f1_rescaled_std":    "",
                "f1_baseline":           baseline,
                "insufficient_samples":  insufficient,
                "encoder":               ENCODER_ID,
                "num_layers":            NUM_LAYERS,
                "lang_code":             ISO3_TO_ISO1[lang],
                "min_samples_threshold": args.min_samples,
                "computed_at":           datetime.now(timezone.utc).isoformat(),
            }

            if insufficient:
                n_insuf += 1
            else:
                (raw_mean, raw_median, raw_std,
                 resc_mean, resc_median, resc_std) = score_and_rescale(
                    scorer, cands, refs, baseline,
                )
                row["bertscore_f1_raw_mean"]        = raw_mean
                row["bertscore_f1_raw_median"]      = raw_median
                row["bertscore_f1_raw_std"]         = raw_std
                row["bertscore_f1_rescaled_mean"]   = resc_mean
                row["bertscore_f1_rescaled_median"] = resc_median
                row["bertscore_f1_rescaled_std"]    = resc_std
                n_computed += 1

            append_row(args.output_path, row)
            done.add(key)
            n_done += 1

            if (n_computed + n_insuf) and (n_computed + n_insuf) % 10 == 0:
                rate = (n_computed + n_insuf) / (time.time() - t0 + 1e-9)
                print(f"  [{n_done}/{total}]  {rate:.2f} tuples/sec  "
                      f"computed={n_computed}  insufficient={n_insuf}  "
                      f"skipped={n_skipped}")

    print(f"\nDone. computed={n_computed}  insufficient={n_insuf}  "
          f"skipped(missing)={n_skipped}")
    print(f"Results: {args.output_path}")


if __name__ == "__main__":
    main()
