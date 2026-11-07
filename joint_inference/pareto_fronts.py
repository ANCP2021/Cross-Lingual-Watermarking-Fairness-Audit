from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quality_fairness.quality_disparity import LANGUAGE_FAMILIES, LANGUAGES

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


SCHEMES = ["KGW", "Unigram", "SynthID", "EXPEdit", "DIP", "XSIR"]
MODELS = ["mistral", "gemma", "qwen"]
REGIMES = ["base", "instruct"]

DEFAULT_MEASUREMENTS = [
    "mauve",
    "bertscore_f1_rescaled",
    "ppl_pres_xglm-7.5b",
    "ppl_pres_mgpt-1.3b",
]

FAMILY_COLORS = {
    "Germanic":      "#1f77b4",
    "Romance":       "#ff7f0e",
    "Indic":         "#2ca02c",
    "Semitic":       "#d62728",
    "Sinitic":       "#9467bd",
    "Japonic":       "#8c564b",
    "Turkic":        "#e377c2",
    "Austroasiatic": "#7f7f7f",
}

MODEL_MARKERS = {
    "mistral": "o",
    "gemma":   "s",
    "qwen":    "^",
}

DOMINANCE_EPS = 1e-9


def load_per_language_tpr(path: str, subset: str, alpha: float) -> Dict[Tuple[str, str, str, str], float]:
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
            tpr_str = r.get("tpr", "")
            if tpr_str == "" or tpr_str.lower() == "nan":
                continue
            key = (r["model"], r["variant"], r["watermark"], r["language"])
            out[key] = float(tpr_str)
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
            score_str = r.get("preservation_score", "")
            if score_str == "" or score_str.lower() == "nan":
                continue
            key = (r["model"], r["variant"], r["watermark"], r["language"])
            out[m][key] = float(score_str)
    return out


def mean_pairwise_distance(points_norm: List[Tuple[float, float]]) -> float:
    n = len(points_norm)
    if n < 2:
        return float("nan")
    total = 0.0
    for (x1, y1), (x2, y2) in combinations(points_norm, 2):
        total += math.hypot(x1 - x2, y1 - y2)
    return total / math.comb(n, 2)


def normalize_points(
    points: List[Tuple[float, float]], x_min: float, x_max: float, y_min: float, y_max: float
) -> List[Tuple[float, float]]:
    x_rng = x_max - x_min if x_max > x_min else 1.0
    y_rng = y_max - y_min if y_max > y_min else 1.0
    return [((x - x_min) / x_rng, (y - y_min) / y_rng) for x, y in points]


def language_means(
    points: List[Tuple[str, str, float, float]],
) -> Dict[str, Tuple[float, float]]:
    by_lang: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for _model, lang, tpr, q in points:
        by_lang[lang].append((tpr, q))
    out: Dict[str, Tuple[float, float]] = {}
    for lang, pairs in by_lang.items():
        n = len(pairs)
        out[lang] = (sum(p[0] for p in pairs) / n, sum(p[1] for p in pairs) / n)
    return out


def dominates(a: Dict[str, Tuple[float, float]], b: Dict[str, Tuple[float, float]]) -> bool:
    common = set(a.keys()) & set(b.keys())
    if not common:
        return False
    strict = False
    for lang in common:
        a_tpr, a_q = a[lang]
        b_tpr, b_q = b[lang]
        if a_tpr + DOMINANCE_EPS < b_tpr or a_q + DOMINANCE_EPS < b_q:
            return False
        if a_tpr > b_tpr + DOMINANCE_EPS or a_q > b_q + DOMINANCE_EPS:
            strict = True
    return strict


def plot_panels(
    regime: str,
    subset: str,
    measurement: str,
    scheme_to_points: Dict[str, List[dict]],
    spreads: Dict[str, float],
    pareto_optimal: set,
    x_min: float, x_max: float, y_min: float, y_max: float,
    out_path: str,
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    x_pad = 0.02 * max(x_max - x_min, 1e-3)
    y_pad = 0.02 * max(y_max - y_min, 1e-3)

    for ax, scheme in zip(axes, SCHEMES):
        pts = scheme_to_points.get(scheme, [])
        for p in pts:
            ax.scatter(
                p["tpr"], p["q"],
                color=FAMILY_COLORS.get(p["family"], "#000000"),
                marker=MODEL_MARKERS.get(p["model"], "o"),
                s=55, alpha=0.85, edgecolors="black", linewidths=0.4,
            )
        spread = spreads.get(scheme, float("nan"))
        spread_str = f"{spread:.3f}" if spread == spread else "n/a"
        pareto_mark = "*" if scheme in pareto_optimal else ""
        ax.set_title(
            f"{scheme}{pareto_mark}    spread={spread_str}    n={len(pts)}",
            fontsize=11,
        )
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("TPR @ FPR=0.01")
        ax.set_ylabel(f"Q ({measurement})")

    family_handles = [
        mpatches.Patch(color=color, label=fam) for fam, color in FAMILY_COLORS.items()
    ]
    model_handles = [
        mlines.Line2D([], [], color="black", marker=mk, linestyle="None",
                      markersize=8, label=model)
        for model, mk in MODEL_MARKERS.items()
    ]
    fig.legend(
        handles=family_handles + model_handles,
        loc="lower center", ncol=11, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Phase 5 §A — Joint Pareto: regime={regime}  subset={subset}  measurement={measurement}\n"
        f"* = Pareto-optimal scheme",
        fontsize=12, y=1.00,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: str, fieldnames: List[str], rows: Iterable[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--fig_dir",     default="figures/joint")
    p.add_argument("--out_dir",     default="results/joint")
    p.add_argument("--regime", choices=REGIMES + ["both"], default="both")
    p.add_argument("--subset", default="all")
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--measurements", nargs="+", default=DEFAULT_MEASUREMENTS)
    return p.parse_args()


def main():
    args = parse_args()
    regimes = REGIMES if args.regime == "both" else [args.regime]

    tpr_path = os.path.join(args.results_dir, "detection", "per_language_tpr.csv")
    quality_path = os.path.join(args.results_dir, "quality", "quality_per_language.csv")
    if not os.path.exists(tpr_path):
        sys.exit(f"missing input: {tpr_path}")
    if not os.path.exists(quality_path):
        sys.exit(f"missing input: {quality_path}")

    tpr_by_cell = load_per_language_tpr(tpr_path, subset=args.subset, alpha=args.alpha)
    quality_by_measurement = load_quality_per_language(quality_path, subset=args.subset, measurements=args.measurements)

    point_rows: List[dict] = []
    spread_rows: List[dict] = []
    dom_rows: List[dict] = []
    pareto_rows: List[dict] = []

    os.makedirs(args.fig_dir, exist_ok=True)

    for regime in regimes:
        for measurement in args.measurements:
            qmap = quality_by_measurement.get(measurement, {})
            if not qmap:
                print(f"[skip] regime={regime} measurement={measurement}: no quality_by_measurement rows")
                continue

            cloud: Dict[str, List[dict]] = {s: [] for s in SCHEMES}
            for (model, variant, watermark, language), tpr in tpr_by_cell.items():
                if variant != regime:
                    continue
                if watermark not in cloud:
                    continue
                q = qmap.get((model, variant, watermark, language))
                if q is None:
                    continue
                family = LANGUAGE_FAMILIES.get(language, "?")
                cloud[watermark].append({
                    "model": model, "language": language, "family": family,
                    "tpr": tpr, "q": q,
                })

            all_pts = [(p["tpr"], p["q"]) for pts in cloud.values() for p in pts]
            if not all_pts:
                print(f"[skip] regime={regime} measurement={measurement}: no joined points")
                continue
            x_min = min(x for x, _ in all_pts)
            x_max = max(x for x, _ in all_pts)
            y_min = min(y for _, y in all_pts)
            y_max = max(y for _, y in all_pts)

            spreads: Dict[str, float] = {}
            for scheme, pts in cloud.items():
                xy = [(p["tpr"], p["q"]) for p in pts]
                xy_norm = normalize_points(xy, x_min, x_max, y_min, y_max)
                spreads[scheme] = mean_pairwise_distance(xy_norm)
                spread_rows.append({
                    "regime": regime, "subset": args.subset, "measurement": measurement,
                    "scheme": scheme, "n_points": len(pts),
                    "n_langs": len({p["language"] for p in pts}),
                    "spread": spreads[scheme],
                })

            lang_means: Dict[str, Dict[str, Tuple[float, float]]] = {}
            for scheme, pts in cloud.items():
                tup_pts = [(p["model"], p["language"], p["tpr"], p["q"]) for p in pts]
                lang_means[scheme] = language_means(tup_pts)

            for a, b in combinations(SCHEMES, 2):
                a_d = dominates(lang_means[a], lang_means[b])
                b_d = dominates(lang_means[b], lang_means[a])
                verdict = a if a_d else (b if b_d else "neither")
                common = set(lang_means[a].keys()) & set(lang_means[b].keys())
                dom_rows.append({
                    "regime": regime, "subset": args.subset, "measurement": measurement,
                    "scheme_a": a, "scheme_b": b,
                    "dominates": verdict, "n_langs_compared": len(common),
                })

            pareto_optimal: set = set()
            for s in SCHEMES:
                dominated = any(
                    dominates(lang_means[other], lang_means[s])
                    for other in SCHEMES if other != s
                )
                if not dominated and lang_means[s]:
                    pareto_optimal.add(s)
                pareto_rows.append({
                    "regime": regime, "subset": args.subset, "measurement": measurement,
                    "scheme": s,
                    "is_pareto_optimal": s in pareto_optimal,
                })

            for scheme, pts in cloud.items():
                for p in pts:
                    point_rows.append({
                        "regime": regime, "subset": args.subset, "measurement": measurement,
                        "scheme": scheme, "model": p["model"],
                        "language": p["language"], "family": p["family"],
                        "tpr": p["tpr"], "q_value": p["q"],
                    })

            fig_path = os.path.join(
                args.fig_dir, f"pareto_{regime}_{measurement}_{args.subset}.png"
            )
            plot_panels(
                regime=regime, subset=args.subset, measurement=measurement,
                scheme_to_points=cloud, spreads=spreads,
                pareto_optimal=pareto_optimal,
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                out_path=fig_path,
            )
            n_total = sum(len(p) for p in cloud.values())
            print(
                f"[ok] regime={regime} measurement={measurement} subset={args.subset}: "
                f"{n_total} pts, pareto={sorted(pareto_optimal) or '(none)'} → {fig_path}"
            )

    write_csv(
        os.path.join(args.out_dir, "pareto_points.csv"),
        ["regime", "subset", "measurement", "scheme", "model",
         "language", "family", "tpr", "q_value"],
        point_rows,
    )
    write_csv(
        os.path.join(args.out_dir, "pareto_spread_metrics.csv"),
        ["regime", "subset", "measurement", "scheme",
         "n_points", "n_langs", "spread"],
        spread_rows,
    )
    write_csv(
        os.path.join(args.out_dir, "pareto_dominance.csv"),
        ["regime", "subset", "measurement", "scheme_a", "scheme_b",
         "dominates", "n_langs_compared"],
        dom_rows,
    )
    write_csv(
        os.path.join(args.out_dir, "pareto_optimal.csv"),
        ["regime", "subset", "measurement", "scheme", "is_pareto_optimal"],
        pareto_rows,
    )
    print(f"[done] wrote CSVs → {args.out_dir}")


if __name__ == "__main__":
    main()
