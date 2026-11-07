from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quality_fairness.quality_disparity import LANGUAGES


SCHEMES = ["KGW", "Unigram", "SynthID", "EXPEdit", "DIP", "XSIR"]
MODELS = ["mistral", "gemma", "qwen"]
REGIMES = ["base", "instruct"]
QUALITY_MEASUREMENTS = [
    "mauve", "bertscore_f1_rescaled",
    "ppl_pres_xglm-7.5b", "ppl_pres_mgpt-1.3b",
]
TARGETS = ["tpr"] + QUALITY_MEASUREMENTS

NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def load_tpr_model_average(path: str, subset: str, alpha: float) -> Dict[Tuple[str, str], Dict[str, float]]:
    accum: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
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
            v = r.get("tpr", "")
            if v in ("", "nan"):
                continue
            try:
                accum[(r["variant"], r["watermark"], r["language"])].append(float(v))
            except ValueError:
                continue
    out: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for (regime, scheme, lang), vals in accum.items():
        if vals:
            out[(regime, scheme)][lang] = sum(vals) / len(vals)
    return out


def load_quality_model_average(
    path: str, subset: str, measurements: List[str]
) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    accum: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("subset") != subset:
                continue
            m = r["measurement"]
            if m not in measurements:
                continue
            v = r.get("preservation_score", "")
            if v in ("", "nan"):
                continue
            try:
                accum[(r["variant"], r["watermark"], m, r["language"])].append(float(v))
            except ValueError:
                continue
    out: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
    for (regime, scheme, measurement, lang), vals in accum.items():
        if vals:
            out[(regime, scheme, measurement)][lang] = sum(vals) / len(vals)
    return out


def average_ranks_per_block(matrix: np.ndarray) -> np.ndarray:
    k, N = matrix.shape
    ranks = np.empty_like(matrix, dtype=np.float64)
    for j in range(N):
        col = matrix[:, j]
        order = np.argsort(-col, kind="mergesort")
        col_ranks = np.empty(k, dtype=np.float64)
        i = 0
        while i < k:
            j_end = i
            while j_end + 1 < k and col[order[j_end + 1]] == col[order[i]]:
                j_end += 1
            avg = (i + j_end) / 2.0 + 1.0
            for idx in range(i, j_end + 1):
                col_ranks[order[idx]] = avg
            i = j_end + 1
        ranks[:, j] = col_ranks
    return ranks


def chi2_sf(x: float, df: int) -> float:
    if x <= 0:
        return 1.0
    a = df / 2.0
    z = x / 2.0
    if z < a + 1.0:
        term = 1.0 / a
        s = term
        for n in range(1, 1000):
            term *= z / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-12:
                break
        log_p = math.log(s) - z + a * math.log(z) - math.lgamma(a + 1.0)
        p = math.exp(-z + a * math.log(z) - math.lgamma(a)) * s
        return max(0.0, min(1.0, 1.0 - p))
    b = z + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    q = math.exp(-z + a * math.log(z) - math.lgamma(a)) * h
    return max(0.0, min(1.0, q))


def friedman(matrix: np.ndarray) -> Tuple[float, int, float, np.ndarray]:
    k, N = matrix.shape
    ranks = average_ranks_per_block(matrix)
    R = ranks.sum(axis=1)
    chi2 = (12.0 / (N * k * (k + 1))) * float((R**2).sum()) - 3 * N * (k + 1)
    df = k - 1
    p = chi2_sf(chi2, df)
    return float(chi2), df, float(p), ranks.mean(axis=1)


def nemenyi_cd(k: int, N: int) -> float:
    if k not in NEMENYI_Q05:
        raise ValueError(f"k={k} outside the supported Nemenyi q05 table")
    q = NEMENYI_Q05[k]
    return q * math.sqrt(k * (k + 1) / (6.0 * N))


def plot_cd(
    scheme_names: List[str], mean_ranks: np.ndarray, cd: float,
    title: str, out_path: str,
):
    k = len(scheme_names)
    order = np.argsort(mean_ranks)
    sorted_names = [scheme_names[i] for i in order]
    sorted_ranks = [float(mean_ranks[i]) for i in order]

    fig_w = max(8.0, 1.0 + 1.4 * k)
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))
    rank_max = math.ceil(max(sorted_ranks) + 0.5)
    rank_min = max(1, math.floor(min(sorted_ranks) - 0.5))
    ax.set_xlim(rank_max + 0.6, rank_min - 0.6)
    ax.set_ylim(-1.6, 2.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.plot([rank_min, rank_max], [0, 0], "k-", linewidth=1.5)
    for r in range(rank_min, rank_max + 1):
        ax.plot([r, r], [-0.05, 0.05], "k-", linewidth=1.0)
        ax.text(r, 0.18, str(r), ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("mean rank (lower = better)")

    half = (k + 1) // 2
    line_y_left = 0.6
    line_y_right = 0.6
    label_step = 0.45

    for i, (name, r) in enumerate(zip(sorted_names, sorted_ranks)):
        if i < half:
            text_x = rank_max + 0.55
            text_y = line_y_left + (half - 1 - i) * label_step
            ax.plot([r, r], [0.05, text_y], "k-", linewidth=0.8)
            ax.plot([r, text_x], [text_y, text_y], "k-", linewidth=0.8)
            ax.text(text_x, text_y, f" {name}", ha="left", va="center", fontsize=10)
        else:
            text_x = rank_min - 0.55
            text_y = line_y_right + (i - half) * label_step
            ax.plot([r, r], [0.05, text_y], "k-", linewidth=0.8)
            ax.plot([r, text_x], [text_y, text_y], "k-", linewidth=0.8)
            ax.text(text_x, text_y, f"{name} ", ha="right", va="center", fontsize=10)

    bar_y = 1.85
    bar_x_lo = rank_min
    bar_x_hi = rank_min + cd
    ax.plot([bar_x_lo, bar_x_hi], [bar_y, bar_y], "k-", linewidth=2.0)
    ax.plot([bar_x_lo, bar_x_lo], [bar_y - 0.08, bar_y + 0.08], "k-", linewidth=1.5)
    ax.plot([bar_x_hi, bar_x_hi], [bar_y - 0.08, bar_y + 0.08], "k-", linewidth=1.5)
    ax.text((bar_x_lo + bar_x_hi) / 2, bar_y + 0.18,
            f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=10)

    bar_y_group = -0.35
    drawn = 0
    line_step = 0.18
    i = 0
    while i < k:
        j = i
        while j + 1 < k and (sorted_ranks[j + 1] - sorted_ranks[i]) < cd:
            j += 1
        if j > i:
            y = bar_y_group - drawn * line_step
            ax.plot([sorted_ranks[i] - 0.1, sorted_ranks[j] + 0.1], [y, y],
                    "k-", linewidth=4.0, solid_capstyle="round")
            drawn += 1
        i += 1

    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_matrix(
    target_data: Dict[str, Dict[str, float]],
) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
    schemes_used = [s for s in SCHEMES if s in target_data and target_data[s]]
    if not schemes_used:
        return None, [], []
    common = set.intersection(*(set(target_data[s].keys()) for s in schemes_used))
    langs_used = [l for l in LANGUAGES if l in common]
    if len(langs_used) < 5:
        return None, schemes_used, langs_used
    mat = np.array(
        [[target_data[s][l] for l in langs_used] for s in schemes_used],
        dtype=np.float64,
    )
    return mat, schemes_used, langs_used


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
    p.add_argument("--out_dir", default="results/joint")
    p.add_argument("--fig_dir", default="figures/joint")
    p.add_argument("--regimes", nargs="+", default=["base", "instruct"])
    p.add_argument("--subset", default="all")
    p.add_argument("--alpha", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()
    tpr_path = os.path.join(args.results_dir, "detection", "per_language_tpr.csv")
    quality_path = os.path.join(args.results_dir, "quality",   "quality_per_language.csv")
    if not os.path.exists(tpr_path):
        sys.exit(f"missing {tpr_path}")
    if not os.path.exists(quality_path):
        sys.exit(f"missing {quality_path}")

    tpr_data = load_tpr_model_average(tpr_path, args.subset, args.alpha)
    quality_data = load_quality_model_average(quality_path, args.subset, QUALITY_MEASUREMENTS)

    friedman_rows: List[dict] = []
    nemenyi_rows: List[dict] = []
    os.makedirs(args.fig_dir, exist_ok=True)

    for regime in args.regimes:
        for target in TARGETS:
            if target == "tpr":
                target_data = {
                    scheme: tpr_data.get((regime, scheme), {})
                    for scheme in SCHEMES
                }
            else:
                target_data = {
                    scheme: quality_data.get((regime, scheme, target), {})
                    for scheme in SCHEMES
                }

            mat, schemes_used, langs_used = build_matrix(target_data)
            if mat is None:
                print(f"[skip] regime={regime} target={target}: insufficient overlap "
                      f"(schemes={len(schemes_used)}, langs={len(langs_used)})")
                continue

            chi2, df, p, mean_ranks = friedman(mat)
            k = len(schemes_used)
            N = len(langs_used)
            cd = nemenyi_cd(k, N)

            friedman_rows.append({
                "regime": regime, "subset": args.subset, "target": target,
                "k_schemes": k, "n_blocks": N,
                "chi2": chi2, "df": df, "p": p,
                "cd": cd,
                "schemes": ",".join(schemes_used),
                "langs": ",".join(langs_used),
            })

            for (i, sa), (j, sb) in combinations(enumerate(schemes_used), 2):
                ra, rb = float(mean_ranks[i]), float(mean_ranks[j])
                nemenyi_rows.append({
                    "regime": regime, "subset": args.subset, "target": target,
                    "scheme_a": sa, "scheme_b": sb,
                    "mean_rank_a": ra, "mean_rank_b": rb,
                    "abs_rank_diff": abs(ra - rb),
                    "cd": cd, "significant": abs(ra - rb) > cd,
                })

            fig_path = os.path.join(args.fig_dir, f"critical_difference_{regime}_{target}.png")
            title = (f"Phase 5 §C3 — {regime}/{target}  "
                     f"χ²={chi2:.2f} df={df} p={p:.3g}  N={N} k={k}")
            plot_cd(schemes_used, mean_ranks, cd, title, fig_path)
            print(f"[ok] {regime}/{target}: χ²={chi2:.2f} p={p:.3g} → {fig_path}")

    write_csv(
        os.path.join(args.out_dir, "friedman_tests.csv"),
        ["regime", "subset", "target", "k_schemes", "n_blocks",
         "chi2", "df", "p", "cd", "schemes", "langs"],
        friedman_rows,
    )
    write_csv(
        os.path.join(args.out_dir, "nemenyi_tests.csv"),
        ["regime", "subset", "target", "scheme_a", "scheme_b",
         "mean_rank_a", "mean_rank_b", "abs_rank_diff", "cd", "significant"],
        nemenyi_rows,
    )
    print(f"[done] {len(friedman_rows)} Friedman tests, "
          f"{len(nemenyi_rows)} Nemenyi pairs → {args.out_dir}")


if __name__ == "__main__":
    main()
