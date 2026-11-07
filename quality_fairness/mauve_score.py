import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Set, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.lid_subsets import load_subset, KNOWN_SUBSETS

LANGUAGES  = ["eng", "fra", "spa", "por", "nld", "zho", "jpn", "arb", "hin", "tur", "vie"]
MODELS     = ["mistral", "gemma", "qwen"]
VARIANTS   = ["base", "instruct"]
WATERMARKS = ["KGW", "Unigram", "SynthID", "EXPEdit", "DIP", "XSIR"]
SUBSETS    = ["all", "on_target_only", "off_target_only"]

VARIANT_DATASET = {"base": "flores", "instruct": "aya"}
ENCODER_TAG = "FacebookAI/xlm-roberta-large"

CSV_FIELDS = [
    "model", "variant", "language", "watermark", "dataset", "subset",
    "n_pos", "n_neg", "mauve_score", "insufficient_samples",
    "encoder", "min_samples_threshold", "seed", "computed_at",
]


def load_embeddings(path: str):
    if not os.path.exists(path):
        return None, None
    with np.load(path, allow_pickle=False) as data:
        pids = data["prompt_ids"].tolist()
        embs = data["embeddings"].astype(np.float32)
    return pids, embs


def filter_by_ids(prompt_ids: List[str], embeddings: np.ndarray, keep: Set[str]):
    mask = np.fromiter((pid in keep for pid in prompt_ids), dtype=bool, count=len(prompt_ids))
    return embeddings[mask]


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


def compute_mauve_for_cell(p_embs: np.ndarray, q_embs: np.ndarray,
                           device_id: int, seed: int) -> float:
    import mauve
    out = mauve.compute_mauve(
        p_features=p_embs,
        q_features=q_embs,
        device_id=device_id,
        seed=seed,
        verbose=False,
    )
    return float(out.mauve)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_dir", default="quality/embeddings")
    p.add_argument("--lid_dir",        default="lid_categorized")
    p.add_argument("--output_path",    default="results/quality/mauve.csv")
    p.add_argument("--models",     default=",".join(MODELS))
    p.add_argument("--variants",   default=",".join(VARIANTS))
    p.add_argument("--watermarks", default=",".join(WATERMARKS))
    p.add_argument("--languages",  default=",".join(LANGUAGES))
    p.add_argument("--subsets",    default=",".join(SUBSETS))
    p.add_argument("--min_samples", type=int, default=100,
                   help="Minimum n on each side; below this, mark insufficient and skip MAUVE.")
    p.add_argument("--device_id", type=int, default=-1,
                   help="MAUVE k-means device id; -1 for CPU. CPU is fast enough at this scale.")
    p.add_argument("--seed", type=int, default=25)
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute (cell, subset) tuples already present in the output CSV.")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.embeddings_dir):
        sys.exit(f"Embeddings dir not found: {args.embeddings_dir}. Run mauve_embed.py first.")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    try:
        import mauve
    except ImportError:
        sys.exit("ERROR: mauve-text not installed.  pip install mauve-text")

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
        print(f"Skipping {len(done)} (cell, subset) tuples already in {args.output_path}")

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
        dataset  = VARIANT_DATASET[variant]
        wm_path  = os.path.join(args.embeddings_dir,
                                f"{dataset}_{model}_{variant}_{watermark}_{lang}.npz")
        nwm_path = os.path.join(args.embeddings_dir,
                                f"{dataset}_{model}_{variant}_nowatermark_{lang}.npz")

        wm_pids,  wm_embs  = load_embeddings(wm_path)
        nwm_pids, nwm_embs = load_embeddings(nwm_path)

        if wm_embs is None or nwm_embs is None:
            n_skipped += len(subsets)
            continue

        for subset in subsets:
            key = (model, variant, lang, watermark, subset)
            if key in done:
                n_done += 1
                continue

            try:
                wm_keep  = load_subset(model, variant, lang, watermark,     subset, args.lid_dir)
                nwm_keep = load_subset(model, variant, lang, "nowatermark", subset, args.lid_dir)
            except FileNotFoundError:
                n_skipped += 1
                continue

            p_embs = filter_by_ids(wm_pids,  wm_embs,  wm_keep)
            q_embs = filter_by_ids(nwm_pids, nwm_embs, nwm_keep)
            n_pos, n_neg = len(p_embs), len(q_embs)

            insufficient = min(n_pos, n_neg) < args.min_samples
            if insufficient:
                score_cell: object = ""
                n_insuf += 1
            else:
                score_cell = compute_mauve_for_cell(p_embs, q_embs,
                                                    args.device_id, args.seed)
                n_computed += 1

            append_row(args.output_path, {
                "model":                 model,
                "variant":               variant,
                "language":              lang,
                "watermark":             watermark,
                "dataset":               dataset,
                "subset":                subset,
                "n_pos":                 n_pos,
                "n_neg":                 n_neg,
                "mauve_score":           score_cell,
                "insufficient_samples":  insufficient,
                "encoder":               ENCODER_TAG,
                "min_samples_threshold": args.min_samples,
                "seed":                  args.seed,
                "computed_at":           datetime.now(timezone.utc).isoformat(),
            })
            done.add(key)
            n_done += 1

            if (n_computed + n_insuf) and (n_computed + n_insuf) % 20 == 0:
                rate = (n_computed + n_insuf) / (time.time() - t0 + 1e-9)
                print(f"  [{n_done}/{total}]  {rate:.2f} tuples/sec  "
                      f"computed={n_computed}  insufficient={n_insuf}  skipped={n_skipped}")

    print(f"\nDone. computed={n_computed}  insufficient={n_insuf}  "
          f"skipped(missing)={n_skipped}")
    print(f"Results: {args.output_path}")


if __name__ == "__main__":
    main()
