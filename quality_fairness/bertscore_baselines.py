import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import List, Tuple

LANGUAGES  = ["eng", "fra", "spa", "por", "nld", "zho", "jpn", "arb", "hin", "tur", "vie"]
ENCODER_ID = "xlm-roberta-large"
NUM_LAYERS = 17

FLORES_TAGS = {
    "eng": ("eng", "Latn"),
    "fra": ("fra", "Latn"),
    "spa": ("spa", "Latn"),
    "por": ("por", "Latn"),
    "nld": ("nld", "Latn"),
    "zho": ("cmn", "Hans"),
    "jpn": ("jpn", "Jpan"),
    "arb": ("arb", "Arab"),
    "hin": ("hin", "Deva"),
    "tur": ("tur", "Latn"),
    "vie": ("vie", "Latn"),
}

ISO3_TO_ISO1 = {
    "eng": "en", "fra": "fr", "spa": "es", "por": "pt", "nld": "nl",
    "zho": "zh", "jpn": "ja", "arb": "ar", "hin": "hi", "tur": "tr",
    "vie": "vi",
}

CSV_FIELDS = [
    "language", "lang_code", "n_pairs",
    "p_baseline", "r_baseline", "f1_baseline",
    "encoder", "num_layers", "min_sentence_chars", "split", "seed", "computed_at",
]


def load_flores_records(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else [r for v in raw.values() for r in v]


def sentences_in_lang(records: List[dict], iso3: str, script: str,
                      splits: List[str], min_chars: int) -> List[str]:
    splits_set = set(splits)
    out = []
    for r in records:
        if (r.get("iso_639_3") == iso3
                and r.get("iso_15924") == script
                and r.get("split") in splits_set):
            text = (r.get("text") or "").strip()
            if len(text) >= min_chars:
                out.append(text)
    return out


def make_pairs(sentences: List[str], n_pairs: int, seed: int) -> Tuple[List[str], List[str], int]:
    rng = random.Random(seed)
    sents = sentences[:]
    rng.shuffle(sents)
    available_pairs = len(sents) // 2
    n = min(n_pairs, available_pairs)
    cands = [sents[2 * i]     for i in range(n)]
    refs  = [sents[2 * i + 1] for i in range(n)]
    return cands, refs, n


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--flores_path", default="data/flores_preprocessed_main.json")
    p.add_argument("--output_path", default="results/quality/bertscore_baselines.csv")
    p.add_argument("--languages",   default=",".join(LANGUAGES))
    p.add_argument("--n_pairs",     type=int, default=1000,
                   help="Target number of non-paired sentence pairs per language.")
    p.add_argument("--min_chars",   type=int, default=20,
                   help="Drop FLORES sentences shorter than this after strip().")
    p.add_argument("--split",       default="dev,devtest",
                   help="Comma-separated FLORES split(s) to pool from. Default "
                        "'dev,devtest' gives ~2000 sentences/lang -> ~1000 pairs.")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    try:
        import bert_score
        import torch
    except ImportError as e:
        sys.exit(f"ERROR: required package not installed: {e}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("ERROR: --device cuda but torch.cuda.is_available() is False.")

    languages = [s for s in args.languages.split(",") if s]
    bad = [l for l in languages if l not in FLORES_TAGS]
    if bad:
        sys.exit(f"Unknown language(s): {bad}")

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    if not splits:
        sys.exit("ERROR: --split must contain at least one FLORES split.")

    print(f"Loading FLORES records from {args.flores_path}")
    records = load_flores_records(args.flores_path)
    print(f"  {len(records)} total records (pooling splits: {splits})")

    print(f"Loading BERTScorer (encoder={ENCODER_ID}, layer={NUM_LAYERS}, rescale=False)")
    t_load = time.time()
    scorer = bert_score.BERTScorer(
        model_type=ENCODER_ID,
        num_layers=NUM_LAYERS,
        rescale_with_baseline=False,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"  loaded in {time.time() - t_load:.1f}s\n")

    rows = []
    t0 = time.time()
    for lang in languages:
        iso3, script = FLORES_TAGS[lang]
        sents = sentences_in_lang(records, iso3, script, splits, args.min_chars)
        if len(sents) < 2:
            print(f"[{lang}] only {len(sents)} sentences after filtering — skipping")
            continue

        cands, refs, n_pairs_actual = make_pairs(sents, args.n_pairs, args.seed)
        print(f"[{lang}] {len(sents)} sentences -> {n_pairs_actual} non-paired pairs")

        t_score = time.time()
        P, R, F1 = scorer.score(cands, refs, verbose=False)
        elapsed = time.time() - t_score

        p_mean  = float(P.mean())
        r_mean  = float(R.mean())
        f1_mean = float(F1.mean())
        print(f"[{lang}] baseline P={p_mean:.4f} R={r_mean:.4f} F1={f1_mean:.4f}  "
              f"({elapsed:.0f}s)")

        rows.append({
            "language":            lang,
            "lang_code":           ISO3_TO_ISO1[lang],
            "n_pairs":             n_pairs_actual,
            "p_baseline":          p_mean,
            "r_baseline":          r_mean,
            "f1_baseline":         f1_mean,
            "encoder":             ENCODER_ID,
            "num_layers":          NUM_LAYERS,
            "min_sentence_chars":  args.min_chars,
            "split":               "+".join(splits),
            "seed":                args.seed,
            "computed_at":         datetime.now(timezone.utc).isoformat(),
        })

    tmp_path = args.output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp_path, args.output_path)

    print(f"\nWrote {len(rows)} baselines to {args.output_path}")
    print(f"Total elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
