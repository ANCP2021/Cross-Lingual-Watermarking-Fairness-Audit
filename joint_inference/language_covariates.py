from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sampler import sample_flores, LANGUAGES


MODELS = ["mistral", "gemma", "qwen"]
VARIANTS = ["base", "instruct"]


JOSHI_TIER: Dict[str, int] = {
    "eng": 5,
    "fra": 4, "spa": 4, "por": 4, "zho": 4, "jpn": 4, "arb": 4,
    "nld": 3, "hin": 3, "tur": 3, "vie": 3,
}

SCRIPT: Dict[str, str] = {
    "eng": "Latin",  "fra": "Latin",  "spa": "Latin",  "por": "Latin",
    "nld": "Latin",  "tur": "Latin",  "vie": "Latin",
    "zho": "Han",
    "jpn": "Japanese-mixed",
    "arb": "Arabic",
    "hin": "Devanagari",
}

MORPHOLOGY: Dict[str, str] = {
    "eng": "isolating",
    "zho": "isolating",
    "vie": "isolating",
    "nld": "fusional",  "fra": "fusional", "spa": "fusional",
    "por": "fusional",  "hin": "fusional",
    "jpn": "agglutinative",
    "tur": "agglutinative",
    "arb": "templatic",
}


def compute_fertility(
    flores_path: str, num_samples: int, seed: int
) -> Dict[Tuple[str, str], float]:
    from loader import load_tokenizer_only
    print(f"[fertility] sampling {num_samples} parallel FLORES devtest sentences")
    sentences = sample_flores(flores_path, num_samples=num_samples, seed=seed)
    out: Dict[Tuple[str, str], float] = {}
    for model in MODELS:
        print(f"[fertility] loading tokenizer: {model}")
        tok = load_tokenizer_only(model, "base")
        for lang in LANGUAGES:
            texts = [t for _pid, t in sentences[lang]]
            counts = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
            out[(model, lang)] = sum(counts) / max(len(counts), 1)
            print(f"[fertility]   {model:8s} {lang}: {out[(model, lang)]:.2f} tok/sent")
    return out


def load_nwm_log_ppl(
    ppl_csv: str, reference_model: str, subset: str
) -> Dict[Tuple[str, str, str], float]:
    if not os.path.exists(ppl_csv):
        print(f"[entropy] missing {ppl_csv} — covariate will be empty", file=sys.stderr)
        return {}

    accum: Dict[Tuple[str, str, str], list] = defaultdict(list)
    with open(ppl_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("subset") != subset:
                continue
            if row.get("reference_model") != reference_model:
                continue
            v = row.get("mean_nll_nwm", "")
            if v == "" or v.lower() == "nan":
                continue
            try:
                val = float(v)
            except ValueError:
                continue
            key = (row["model"], row["variant"], row["language"])
            accum[key].append(val)
    return {k: sum(vs) / len(vs) for k, vs in accum.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--flores_path", default="data/flores_preprocessed_main.json")
    p.add_argument("--num_flores", type=int, default=1000,
                   help="Number of parallel devtest sentences (max ~1012).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ppl_csv", default="results/quality/ppl_ratio.csv")
    p.add_argument("--reference_model", default="xglm-7.5b")
    p.add_argument("--subset", default="all")
    p.add_argument("--out_path", default="results/joint/language_covariates.csv")
    return p.parse_args()


def main():
    args = parse_args()
    fertility = compute_fertility(args.flores_path, args.num_flores, args.seed)
    nwm_logppl = load_nwm_log_ppl(args.ppl_csv, args.reference_model, args.subset)

    rows = []
    for model in MODELS:
        for variant in VARIANTS:
            for lang in LANGUAGES:
                rows.append({
                    "model": model,
                    "variant": variant,
                    "language": lang,
                    "fertility":          fertility.get((model, lang), ""),
                    "nwm_log_ppl_xglm":   nwm_logppl.get((model, variant, lang), ""),
                    "joshi_tier":         JOSHI_TIER[lang],
                    "script":             SCRIPT[lang],
                    "morphology":         MORPHOLOGY[lang],
                })

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    fields = ["model", "variant", "language",
              "fertility", "nwm_log_ppl_xglm",
              "joshi_tier", "script", "morphology"]
    with open(args.out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_full = sum(1 for r in rows if r["fertility"] != "" and r["nwm_log_ppl_xglm"] != "")
    print(f"[done] {len(rows)} rows ({n_full} fully populated) → {args.out_path}")


if __name__ == "__main__":
    main()
