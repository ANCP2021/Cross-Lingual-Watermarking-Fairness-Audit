from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quality_fairness.quality_disparity import LANGUAGES


SCHEMES = ["KGW", "Unigram", "SynthID", "EXPEdit", "DIP", "XSIR"]
MODELS = ["mistral", "gemma", "qwen"]
REGIMES = ["base", "instruct"]

CONTINUOUS_COVARS = ["fertility", "nwm_log_ppl_xglm", "joshi_tier"]
CATEGORICAL_COVARS = ["script", "morphology"]
ALL_COVARS = CONTINUOUS_COVARS + CATEGORICAL_COVARS

QUALITY_MEASUREMENTS = [
    "mauve",
    "bertscore_f1_rescaled",
    "ppl_pres_xglm-7.5b",
    "ppl_pres_mgpt-1.3b",
]
TARGETS = ["tpr"] + QUALITY_MEASUREMENTS

MIN_N_FOR_FIT = 5
BOOTSTRAP_B = 1000


def load_covariates(path: str) -> Dict[Tuple[str, str, str], dict]:
    out: Dict[Tuple[str, str, str], dict] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["model"], r["variant"], r["language"])
            row: dict = {}
            for c in CONTINUOUS_COVARS:
                v = r.get(c, "")
                row[c] = float(v) if v not in ("", None) else None
            for c in CATEGORICAL_COVARS:
                row[c] = r.get(c, "") or None
            out[key] = row
    return out


def load_per_language_tpr(
    path: str, subset: str, alpha: float
) -> Dict[Tuple[str, str, str, str], float]:
    out: Dict[Tuple[str, str, str, str], float] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["threshold_type"] != "global":
                continue
            if r["subset"] != subset:
                continue
            if abs(float(r["alpha"]) - alpha) > 1e-9:
                continue
            if r.get("insufficient_samples", "False").lower() == "true":
                continue
            v = r.get("tpr", "")
            if v == "" or v.lower() == "nan":
                continue
            out[(r["model"], r["variant"], r["watermark"], r["language"])] = float(v)
    return out


def load_quality_per_language(
    path: str, subset: str, measurements: List[str]
) -> Dict[str, Dict[Tuple[str, str, str, str], float]]:
    out: Dict[str, Dict[Tuple[str, str, str, str], float]] = {m: {} for m in measurements}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["subset"] != subset:
                continue
            m = r["measurement"]
            if m not in out:
                continue
            v = r.get("preservation_score", "")
            if v == "" or v.lower() == "nan":
                continue
            out[m][(r["model"], r["variant"], r["watermark"], r["language"])] = float(v)
    return out


def ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    ss_xx = float(((x - x_mean) ** 2).sum())
    ss_yy = float(((y - y_mean) ** 2).sum())
    if ss_xx == 0 or ss_yy == 0:
        return float("nan"), float("nan"), float("nan")
    slope = float(((x - x_mean) * (y - y_mean)).sum() / ss_xx)
    intercept = y_mean - slope * x_mean
    y_pred = intercept + slope * x
    ss_res = float(((y - y_pred) ** 2).sum())
    r2 = 1.0 - ss_res / ss_yy
    return slope, intercept, r2


def eta_squared(y: np.ndarray, groups: List[str]) -> float:
    y_mean = float(y.mean())
    ss_total = float(((y - y_mean) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    g_arr = np.array(groups)
    ss_between = 0.0
    for g in set(groups):
        mask = g_arr == g
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        ss_between += n_g * (float(y[mask].mean()) - y_mean) ** 2
    return ss_between / ss_total


def bootstrap_ci(
    fn, B: int, seed: int, n: int
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            v = fn(idx)
        except (ValueError, ZeroDivisionError):
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        samples.append(v)
    if len(samples) < 50:
        return float("nan"), float("nan")
    arr = np.array(samples)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def regression_rows(
    regime: str,
    subset: str,
    target_name: str,
    target_map: Dict[Tuple[str, str, str, str], float],
    covariates: Dict[Tuple[str, str, str], dict],
    seed: int,
) -> List[dict]:
    rows: List[dict] = []
    for scheme in SCHEMES:
        for model in MODELS:
            target_by_lang: Dict[str, float] = {}
            for lang in LANGUAGES:
                v = target_map.get((model, regime, scheme, lang))
                if v is not None:
                    target_by_lang[lang] = v
            if len(target_by_lang) < MIN_N_FOR_FIT:
                continue

            for cov in ALL_COVARS:
                kind = "continuous" if cov in CONTINUOUS_COVARS else "categorical"
                xs: List = []
                ys: List[float] = []
                langs_used: List[str] = []
                for lang in LANGUAGES:
                    if lang not in target_by_lang:
                        continue
                    cov_row = covariates.get((model, regime, lang))
                    if cov_row is None:
                        continue
                    cv = cov_row.get(cov)
                    if cv is None:
                        continue
                    xs.append(cv)
                    ys.append(target_by_lang[lang])
                    langs_used.append(lang)
                n = len(ys)
                if n < MIN_N_FOR_FIT:
                    continue
                y_arr = np.array(ys, dtype=float)

                if kind == "continuous":
                    x_arr = np.array(xs, dtype=float)
                    slope, intercept, r2 = ols(x_arr, y_arr)
                    ci_lo, ci_hi = bootstrap_ci(
                        fn=lambda idx, x=x_arr, y=y_arr: ols(x[idx], y[idx])[0],
                        B=BOOTSTRAP_B, seed=seed, n=n,
                    )
                    stat_value = slope
                    stat_label = "slope"
                else:
                    g = list(xs)
                    e2 = eta_squared(y_arr, g)
                    r2 = e2
                    intercept = float("nan")
                    ci_lo, ci_hi = bootstrap_ci(
                        fn=lambda idx, y=y_arr, g=g: eta_squared(y[idx], [g[i] for i in idx]),
                        B=BOOTSTRAP_B, seed=seed, n=n,
                    )
                    stat_value = e2
                    stat_label = "eta_squared"

                rows.append({
                    "regime": regime, "subset": subset,
                    "target": target_name, "scheme": scheme, "model": model,
                    "covariate": cov, "covariate_kind": kind,
                    "statistic": stat_label, "stat_value": stat_value,
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "intercept": intercept, "r_squared": r2,
                    "n": n, "langs_used": ",".join(sorted(langs_used)),
                })
    return rows


def plot_heatmap(
    regime: str, subset: str, rows: List[dict], out_path: str
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    by_target: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r["regime"] != regime or r["subset"] != subset:
            continue
        by_target[r["target"]].append(r)

    for ax, target in zip(axes, TARGETS):
        target_rows = by_target.get(target, [])
        accum: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for r in target_rows:
            v = r["r_squared"]
            if v != v:
                continue
            accum[(r["scheme"], r["covariate"])].append(float(v))
        grid = np.full((len(SCHEMES), len(ALL_COVARS)), np.nan)
        for i, sch in enumerate(SCHEMES):
            for j, cov in enumerate(ALL_COVARS):
                vals = accum.get((sch, cov), [])
                if vals:
                    grid[i, j] = float(np.mean(vals))
        im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(ALL_COVARS)))
        ax.set_xticklabels(ALL_COVARS, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(SCHEMES)))
        ax.set_yticklabels(SCHEMES, fontsize=9)
        ax.set_title(f"target = {target}", fontsize=10)
        for i in range(len(SCHEMES)):
            for j in range(len(ALL_COVARS)):
                v = grid[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if v < 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(TARGETS):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Phase 5 §B — Mechanism mean R² (across models)  regime={regime}  subset={subset}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir",  default="results")
    p.add_argument("--covariates_csv", default="results/joint/language_covariates.csv")
    p.add_argument("--out_dir",      default="results/joint")
    p.add_argument("--fig_dir",      default="figures/joint")
    p.add_argument("--regime", choices=REGIMES + ["both"], default="both")
    p.add_argument("--subset", default="all")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    regimes = REGIMES if args.regime == "both" else [args.regime]

    if not os.path.exists(args.covariates_csv):
        sys.exit(f"missing {args.covariates_csv}; run joint_inference/language_covariates.py first")

    covariates = load_covariates(args.covariates_csv)
    tpr_path = os.path.join(args.results_dir, "detection", "per_language_tpr.csv")
    quality_path = os.path.join(args.results_dir, "quality",   "quality_per_language.csv")
    if not os.path.exists(tpr_path):
        sys.exit(f"missing {tpr_path}")
    if not os.path.exists(quality_path):
        sys.exit(f"missing {quality_path}")

    tpr = load_per_language_tpr(tpr_path, subset=args.subset, alpha=args.alpha)
    quality_maps = load_quality_per_language(quality_path, subset=args.subset, measurements=QUALITY_MEASUREMENTS)

    target_maps: Dict[str, Dict[Tuple[str, str, str, str], float]] = {"tpr": tpr}
    for m in QUALITY_MEASUREMENTS:
        target_maps[m] = quality_maps[m]

    all_rows: List[dict] = []
    for regime in regimes:
        for target_name, target_map in target_maps.items():
            rows = regression_rows(
                regime=regime, subset=args.subset,
                target_name=target_name, target_map=target_map,
                covariates=covariates, seed=args.seed,
            )
            all_rows.extend(rows)
            print(f"[ok] regime={regime} target={target_name}: {len(rows)} regressions")

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "covariate_regressions.csv")
    fields = ["regime", "subset", "target", "scheme", "model",
              "covariate", "covariate_kind", "statistic", "stat_value",
              "ci_lo", "ci_hi", "intercept", "r_squared", "n", "langs_used"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"[done] {len(all_rows)} regression rows → {out_csv}")

    os.makedirs(args.fig_dir, exist_ok=True)
    for regime in regimes:
        fig_path = os.path.join(
            args.fig_dir, f"covariate_regressions_r2_{regime}_{args.subset}.png"
        )
        plot_heatmap(regime=regime, subset=args.subset, rows=all_rows, out_path=fig_path)
        print(f"[fig] {fig_path}")


if __name__ == "__main__":
    main()
