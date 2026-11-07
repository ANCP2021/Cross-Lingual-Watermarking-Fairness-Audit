from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SCHEMES = ["KGW", "Unigram", "SynthID", "EXPEdit", "DIP", "XSIR"]
MODELS = ["mistral", "gemma", "qwen"]
LANGUAGES = ["eng", "fra", "spa", "por", "nld", "zho", "jpn", "arb", "hin", "tur", "vie"]
VARIANT_DATASET = {"base": "flores", "instruct": "aya"}

DEFAULT_B = 1000
DEFAULT_SEED = 42

MIN_N_WM = 100
MIN_N_FOR_PAIRWISE = 100


def load_thresholds_global(path: str, subset: str, alpha: float) -> Dict[Tuple[str, str, str], float]:
    out: Dict[Tuple[str, str, str], float] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["threshold_type"] != "global":
                continue
            if r.get("subset") != subset:
                continue
            try:
                if abs(float(r["alpha"]) - alpha) > 1e-9:
                    continue
            except (KeyError, ValueError):
                continue
            if r.get("insufficient_samples", "False").lower() == "true":
                continue
            v = r.get("threshold_value", "")
            if v in ("", "nan"):
                continue
            try:
                out[(r["model"], r["variant"], r["watermark"])] = float(v)
            except ValueError:
                continue
    return out


def load_detection_scores(
    path: str, subset: str
) -> Dict[Tuple[str, str, str, str, str], List[Tuple[float, Optional[int], str]]]:
    keep_all = subset == "all"
    data: Dict[Tuple[str, str, str, str, str], List[Tuple[float, Optional[int], str]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lid = r.get("lid_label", "NA")
            if not keep_all and lid != subset:
                continue
            try:
                score = float(r["score"])
            except (KeyError, ValueError):
                continue
            tok_raw = r.get("token_count", "")
            tok: Optional[int]
            try:
                tok = int(tok_raw) if tok_raw not in ("", None) else None
            except ValueError:
                tok = None
            key = (r["model"], r["variant"], r["watermark"], r["language"], r["side"])
            data[key].append((score, tok, lid))
    return data


def load_ppl_cache(path: str) -> List[float]:
    if not os.path.exists(path):
        return []
    out: List[float] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = rec.get("mean_nll")
            if v is None:
                continue
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
    return out


def ppl_cache_path(ppl_dir: str, ref: str, dataset: str, model: str,
                   variant: str, watermark: str, lang: str) -> str:
    if watermark == "nowatermark":
        fname = f"{dataset}_{model}_{variant}_nowatermark_{lang}_ppl.jsonl"
    else:
        fname = f"{dataset}_{model}_{variant}_{watermark}_{lang}_ppl.jsonl"
    return os.path.join(ppl_dir, ref, fname)


def bootstrap_tpr(
    wm_scores: np.ndarray, tau: float, B: int, seed: int
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(wm_scores)
    tpr = float(np.mean(wm_scores > tau))
    samples = np.empty(B, dtype=np.float64)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(wm_scores[idx] > tau))
    return tpr, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap_log_ppl_ratio(
    wm_nlls: np.ndarray, nwm_nlls: np.ndarray, B: int, seed: int
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n_wm, n_nwm = len(wm_nlls), len(nwm_nlls)
    delta = float(wm_nlls.mean() - nwm_nlls.mean())
    samples = np.empty(B, dtype=np.float64)
    for i in range(B):
        idx_wm = rng.integers(0, n_wm, size=n_wm)
        idx_nwm = rng.integers(0, n_nwm, size=n_nwm)
        samples[i] = float(wm_nlls[idx_wm].mean() - nwm_nlls[idx_nwm].mean())
    return delta, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def bootstrap_delta_tpr(
    wm_a: np.ndarray, wm_b: np.ndarray, tau: float, B: int, seed: int
) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n_a, n_b = len(wm_a), len(wm_b)
    delta = float(np.mean(wm_a > tau) - np.mean(wm_b > tau))
    samples = np.empty(B, dtype=np.float64)
    for i in range(B):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        samples[i] = float(np.mean(wm_a[idx_a] > tau) - np.mean(wm_b[idx_b] > tau))
    p_pos = float(np.mean(samples >= 0))
    p_neg = float(np.mean(samples <= 0))
    p = min(2.0 * min(p_pos, p_neg), 1.0)
    return delta, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)), p


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> Tuple[List[float], List[bool]]:
    n = len(p_values)
    if n == 0:
        return [], []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * n
    running_min = 1.0
    for rank in range(n, 0, -1):
        orig_idx, p = indexed[rank - 1]
        a = p * n / rank
        running_min = min(running_min, a)
        adj[orig_idx] = running_min
    sig = [a < q for a in adj]
    return adj, sig


def length_stratified_tpr(
    scores: np.ndarray, toks: np.ndarray, tau: float, n_bins: int = 4
) -> Tuple[float, float, List[float], List[Tuple[float, float, int]]]:
    if len(scores) == 0:
        return float("nan"), float("nan"), [float("nan")] * n_bins, []
    tpr_raw = float(np.mean(scores > tau))
    qs = np.quantile(toks, np.linspace(0.0, 1.0, n_bins + 1))
    bin_tprs: List[float] = []
    bin_meta: List[Tuple[float, float, int]] = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        if i == n_bins - 1:
            mask = (toks >= lo) & (toks <= hi)
        else:
            mask = (toks >= lo) & (toks < hi)
        n_in = int(mask.sum())
        if n_in == 0:
            bin_tprs.append(float("nan"))
        else:
            bin_tprs.append(float(np.mean(scores[mask] > tau)))
        bin_meta.append((float(lo), float(hi), n_in))
    valid = [t for t in bin_tprs if t == t]
    tpr_bin_avg = float(np.mean(valid)) if valid else float("nan")
    return tpr_raw, tpr_bin_avg, bin_tprs, bin_meta


def task_tpr_ci(
    scores: Dict[Tuple[str, str, str, str, str], List[Tuple[float, Optional[int], str]]],
    thresholds: Dict[Tuple[str, str, str], float],
    regimes: List[str], subset: str, B: int, seed: int,
) -> List[dict]:
    rows = []
    for model in MODELS:
        for variant in regimes:
            for scheme in SCHEMES:
                tau = thresholds.get((model, variant, scheme))
                if tau is None:
                    continue
                for lang in LANGUAGES:
                    bucket = scores.get((model, variant, scheme, lang, "wm"), [])
                    wm = np.array([s for s, _t, _l in bucket], dtype=np.float64)
                    if len(wm) < MIN_N_WM:
                        rows.append({
                            "regime": variant, "subset": subset, "model": model,
                            "scheme": scheme, "language": lang, "tau": tau,
                            "tpr": float("nan"), "ci_lo": float("nan"),
                            "ci_hi": float("nan"), "n_wm": len(wm),
                            "insufficient": True,
                        })
                        continue
                    tpr, lo, hi = bootstrap_tpr(wm, tau, B, seed)
                    rows.append({
                        "regime": variant, "subset": subset, "model": model,
                        "scheme": scheme, "language": lang, "tau": tau,
                        "tpr": tpr, "ci_lo": lo, "ci_hi": hi, "n_wm": len(wm),
                        "insufficient": False,
                    })
    return rows


def task_ppl_ci(
    regimes: List[str], subset: str, ppl_dir: str, reference_model: str,
    B: int, seed: int,
) -> List[dict]:
    rows = []
    for model in MODELS:
        for variant in regimes:
            dataset = VARIANT_DATASET[variant]
            for scheme in SCHEMES:
                wm_dir = os.path.join(ppl_dir, reference_model)
                if not os.path.isdir(wm_dir):
                    print(f"[ppl_ci] missing dir {wm_dir} — skipping", file=sys.stderr)
                    return rows
                for lang in LANGUAGES:
                    wm_path = ppl_cache_path(ppl_dir, reference_model, dataset,
                                             model, variant, scheme, lang)
                    nwm_path = ppl_cache_path(ppl_dir, reference_model, dataset,
                                              model, variant, "nowatermark", lang)
                    wm = np.array(load_ppl_cache(wm_path), dtype=np.float64)
                    nwm = np.array(load_ppl_cache(nwm_path), dtype=np.float64)
                    if len(wm) < MIN_N_WM or len(nwm) < MIN_N_WM:
                        rows.append({
                            "regime": variant, "subset": subset, "model": model,
                            "scheme": scheme, "language": lang,
                            "reference_model": reference_model,
                            "log_ppl_ratio": float("nan"),
                            "ci_lo": float("nan"), "ci_hi": float("nan"),
                            "n_wm": len(wm), "n_nwm": len(nwm),
                            "insufficient": True,
                        })
                        continue
                    delta, lo, hi = bootstrap_log_ppl_ratio(wm, nwm, B, seed)
                    rows.append({
                        "regime": variant, "subset": subset, "model": model,
                        "scheme": scheme, "language": lang,
                        "reference_model": reference_model,
                        "log_ppl_ratio": delta, "ci_lo": lo, "ci_hi": hi,
                        "n_wm": len(wm), "n_nwm": len(nwm),
                        "insufficient": False,
                    })
    return rows


def task_paired_tpr(
    scores: Dict[Tuple[str, str, str, str, str], List[Tuple[float, Optional[int], str]]],
    thresholds: Dict[Tuple[str, str, str], float],
    regimes: List[str], subset: str, B: int, seed: int, q: float = 0.05,
) -> List[dict]:
    rows = []
    for model in MODELS:
        for variant in regimes:
            for scheme in SCHEMES:
                tau = thresholds.get((model, variant, scheme))
                if tau is None:
                    continue
                wm_by_lang: Dict[str, np.ndarray] = {}
                for lang in LANGUAGES:
                    bucket = scores.get((model, variant, scheme, lang, "wm"), [])
                    wm_by_lang[lang] = np.array([s for s, _t, _l in bucket], dtype=np.float64)

                pair_rows: List[dict] = []
                for la, lb in combinations(LANGUAGES, 2):
                    wm_a, wm_b = wm_by_lang[la], wm_by_lang[lb]
                    if len(wm_a) < MIN_N_FOR_PAIRWISE or len(wm_b) < MIN_N_FOR_PAIRWISE:
                        pair_rows.append({
                            "regime": variant, "subset": subset, "model": model,
                            "scheme": scheme, "lang_a": la, "lang_b": lb,
                            "delta": float("nan"), "ci_lo": float("nan"),
                            "ci_hi": float("nan"), "p_raw": float("nan"),
                            "p_bh": float("nan"), "significant": False,
                            "n_a": len(wm_a), "n_b": len(wm_b),
                            "insufficient": True,
                        })
                        continue
                    pair_seed = seed + abs(hash((model, variant, scheme, la, lb))) % (2**31)
                    delta, lo, hi, p = bootstrap_delta_tpr(wm_a, wm_b, tau, B, pair_seed)
                    pair_rows.append({
                        "regime": variant, "subset": subset, "model": model,
                        "scheme": scheme, "lang_a": la, "lang_b": lb,
                        "delta": delta, "ci_lo": lo, "ci_hi": hi,
                        "p_raw": p, "p_bh": float("nan"),
                        "significant": False, "n_a": len(wm_a), "n_b": len(wm_b),
                        "insufficient": False,
                    })
                idxs = [i for i, r in enumerate(pair_rows) if not r["insufficient"]]
                if idxs:
                    p_raws = [pair_rows[i]["p_raw"] for i in idxs]
                    adj, sig = benjamini_hochberg(p_raws, q=q)
                    for k, i in enumerate(idxs):
                        pair_rows[i]["p_bh"] = adj[k]
                        pair_rows[i]["significant"] = sig[k]
                rows.extend(pair_rows)
    return rows


def task_length_stratified(
    scores: Dict[Tuple[str, str, str, str, str], List[Tuple[float, Optional[int], str]]],
    thresholds: Dict[Tuple[str, str, str], float],
    regimes: List[str], subset: str,
) -> List[dict]:
    rows = []
    for model in MODELS:
        for variant in regimes:
            for scheme in SCHEMES:
                tau = thresholds.get((model, variant, scheme))
                if tau is None:
                    continue
                for lang in LANGUAGES:
                    bucket = scores.get((model, variant, scheme, lang, "wm"), [])
                    pairs = [(s, t) for s, t, _l in bucket if t is not None]
                    if len(pairs) < MIN_N_WM:
                        continue
                    s_arr = np.array([p[0] for p in pairs], dtype=np.float64)
                    t_arr = np.array([p[1] for p in pairs], dtype=np.int64)
                    tpr_raw, tpr_avg, bins, meta = length_stratified_tpr(
                        s_arr, t_arr, tau, n_bins=4
                    )
                    row = {
                        "regime": variant, "subset": subset, "model": model,
                        "scheme": scheme, "language": lang, "tau": tau,
                        "tpr_raw": tpr_raw, "tpr_bin_avg": tpr_avg,
                        "n_wm": len(pairs),
                        "agreement_abs_diff": (
                            abs(tpr_raw - tpr_avg) if tpr_avg == tpr_avg else float("nan")
                        ),
                    }
                    for i, (lo, hi, n_in) in enumerate(meta, start=1):
                        row[f"quartile{i}_tpr"] = bins[i - 1]
                        row[f"quartile{i}_tok_lo"] = lo
                        row[f"quartile{i}_tok_hi"] = hi
                        row[f"quartile{i}_n"] = n_in
                    rows.append(row)
    return rows


def write_csv(path: str, fieldnames: List[str], rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--ppl_dir", default="quality/ppl")
    p.add_argument("--out_dir", default="results/joint")
    p.add_argument("--regimes", nargs="+", default=["base", "instruct"])
    p.add_argument("--subset", default="all")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--bootstrap_b", type=int, default=DEFAULT_B)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--bh_q", type=float, default=0.05)
    p.add_argument("--reference_model", default="xglm-7.5b")
    p.add_argument("--tasks", nargs="+",
                   choices=["tpr_ci", "ppl_ci", "paired_tpr", "length_stratified", "all"],
                   default=["all"])
    return p.parse_args()


def main():
    args = parse_args()
    tasks = {"tpr_ci", "ppl_ci", "paired_tpr", "length_stratified"} if "all" in args.tasks else set(args.tasks)

    scores_csv = os.path.join(args.results_dir, "detection", "detection_scores.csv")
    thr_csv = os.path.join(args.results_dir, "detection", "thresholds.csv")
    if not os.path.exists(thr_csv):
        sys.exit(f"missing {thr_csv}")

    thresholds = load_thresholds_global(thr_csv, args.subset, args.alpha)
    print(f"[load] {len(thresholds)} thresholds (global, subset={args.subset}, α={args.alpha})")

    scores: Dict = {}
    if {"tpr_ci", "paired_tpr", "length_stratified"} & tasks:
        if not os.path.exists(scores_csv):
            sys.exit(f"missing {scores_csv}")
        scores = load_detection_scores(scores_csv, args.subset)
        print(f"[load] {len(scores)} (cell, side) buckets from detection_scores.csv")

    if "tpr_ci" in tasks:
        rows = task_tpr_ci(scores, thresholds, args.regimes, args.subset,
                           args.bootstrap_b, args.seed)
        write_csv(
            os.path.join(args.out_dir, "bootstrap_tpr_ci.csv"),
            ["regime", "subset", "model", "scheme", "language", "tau",
             "tpr", "ci_lo", "ci_hi", "n_wm", "insufficient"],
            rows,
        )
        print(f"[tpr_ci] {len(rows)} rows")

    if "ppl_ci" in tasks:
        rows = task_ppl_ci(args.regimes, args.subset, args.ppl_dir,
                           args.reference_model, args.bootstrap_b, args.seed)
        write_csv(
            os.path.join(args.out_dir, "bootstrap_ppl_ci.csv"),
            ["regime", "subset", "model", "scheme", "language", "reference_model",
             "log_ppl_ratio", "ci_lo", "ci_hi", "n_wm", "n_nwm", "insufficient"],
            rows,
        )
        print(f"[ppl_ci] {len(rows)} rows")

    if "paired_tpr" in tasks:
        rows = task_paired_tpr(scores, thresholds, args.regimes, args.subset,
                                args.bootstrap_b, args.seed, q=args.bh_q)
        write_csv(
            os.path.join(args.out_dir, "paired_tpr_tests.csv"),
            ["regime", "subset", "model", "scheme", "lang_a", "lang_b",
             "delta", "ci_lo", "ci_hi", "p_raw", "p_bh", "significant",
             "n_a", "n_b", "insufficient"],
            rows,
        )
        n_sig = sum(1 for r in rows if r["significant"])
        print(f"[paired_tpr] {len(rows)} rows, {n_sig} significant after BH(q={args.bh_q})")

    if "length_stratified" in tasks:
        rows = task_length_stratified(scores, thresholds, args.regimes, args.subset)
        write_csv(
            os.path.join(args.out_dir, "length_stratified_tpr.csv"),
            ["regime", "subset", "model", "scheme", "language", "tau",
             "tpr_raw", "tpr_bin_avg", "agreement_abs_diff", "n_wm",
             "quartile1_tpr", "quartile1_tok_lo", "quartile1_tok_hi", "quartile1_n",
             "quartile2_tpr", "quartile2_tok_lo", "quartile2_tok_hi", "quartile2_n",
             "quartile3_tpr", "quartile3_tok_lo", "quartile3_tok_hi", "quartile3_n",
             "quartile4_tpr", "quartile4_tok_lo", "quartile4_tok_hi", "quartile4_n"],
            rows,
        )
        print(f"[length_stratified] {len(rows)} rows")

    print(f"[done] outputs → {args.out_dir}")


if __name__ == "__main__":
    main()
