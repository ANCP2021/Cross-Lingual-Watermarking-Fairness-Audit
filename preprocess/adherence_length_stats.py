import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict


LANGUAGES  = ["eng", "fra", "spa", "por", "nld", "zho", "jpn", "arb", "hin", "tur", "vie"]
MODELS     = ["mistral", "gemma", "qwen"]
VARIANTS   = ["base", "instruct"]
WATERMARKS = ["KGW", "Unigram", "SynthID", "EXPEdit", "DIP", "XSIR", "nowatermark"]

SUBSETS = ["all", "on_target_only", "off_target_only"]

LID_FNAME_RE = re.compile(
    r"^(aya|flores)"
    r"_(mistral|gemma|qwen)"
    r"_(base|instruct)"
    r"_(KGW|Unigram|SynthID|EXPEdit|DIP|XSIR|nowatermark)"
    r"_(eng|fra|spa|por|nld|zho|jpn|arb|hin|tur|vie)"
    r"_lid_categorized\.jsonl$"
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def quantile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def length_stats(xs):
    n = len(xs)
    if n == 0:
        return dict(n=0,
                    mean=float("nan"), median=float("nan"),
                    q1=float("nan"), q3=float("nan"),
                    min=float("nan"), max=float("nan"),
                    std=float("nan"))
    return dict(
        n=n,
        mean=statistics.mean(xs),
        median=statistics.median(xs),
        q1=quantile(xs, 0.25),
        q3=quantile(xs, 0.75),
        min=min(xs),
        max=max(xs),
        std=statistics.stdev(xs) if n > 1 else 0.0,
    )


def aggregate(lid_dir):
    files = sorted(f for f in os.listdir(lid_dir) if f.endswith("_lid_categorized.jsonl"))
    if not files:
        sys.exit(f"No *_lid_categorized.jsonl in {lid_dir}.")

    adherence_rows = []
    length_rows    = []
    skipped        = []

    for fname in files:
        m = LID_FNAME_RE.match(fname)
        if not m:
            skipped.append(fname)
            continue
        dataset, model, variant, watermark, language = m.groups()
        path = os.path.join(lid_dir, fname)

        label_counts    = defaultdict(int)
        per_label_lens  = defaultdict(list)
        all_lens        = []
        n_total         = 0

        for rec in read_jsonl(path):
            label = rec.get("lid_label")
            tc    = rec.get("token_count") or 0
            if label is None:
                continue
            label_counts[label] += 1
            per_label_lens[label].append(tc)
            all_lens.append(tc)
            n_total += 1

        if n_total == 0:
            skipped.append(fname + " (empty)")
            continue

        n_on   = label_counts.get("on_target",  0)
        n_cm   = label_counts.get("code_mixed", 0)
        n_off  = label_counts.get("off_target", 0)
        n_unc  = label_counts.get("uncertain",  0)

        adherence_rows.append({
            "language":         language,
            "model":            model,
            "variant":          variant,
            "watermark":        watermark,
            "dataset":          dataset,
            "n_total":          n_total,
            "n_on_target":      n_on,
            "n_code_mixed":     n_cm,
            "n_off_target":     n_off,
            "n_uncertain":      n_unc,
            "on_target_rate":   n_on  / n_total,
            "code_mixed_rate":  n_cm  / n_total,
            "off_target_rate":  n_off / n_total,
            "uncertain_rate":   n_unc / n_total,
        })

        for subset in SUBSETS:
            if subset == "all":
                lens = all_lens
            elif subset == "on_target_only":
                lens = per_label_lens["on_target"]
            else:
                lens = per_label_lens["off_target"]
            length_rows.append({
                "language":  language,
                "model":     model,
                "variant":   variant,
                "watermark": watermark,
                "dataset":   dataset,
                "subset":    subset,
                **length_stats(lens),
            })

    return adherence_rows, length_rows, skipped


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_summary(adherence_rows, length_rows):
    nwm_adh = {(r["language"], r["model"], r["variant"]): r
               for r in adherence_rows if r["watermark"] == "nowatermark"}
    nwm_len = {(r["language"], r["model"], r["variant"]): r
               for r in length_rows
               if r["watermark"] == "nowatermark" and r["subset"] == "all"}

    headers = [f"{m[0]}/{v[:3]}" for m in MODELS for v in VARIANTS]

    def render(title, fmt_cell, by_key):
        print(f"\n{title}  (nowatermark, all-subset)")
        print(f"  {'lang':<5} | " + " | ".join(f"{h:>11}" for h in headers))
        print(f"  {'-'*5}-+-" + "-+-".join(f"{'-'*11}" for _ in headers))
        for lg in LANGUAGES:
            cells = []
            for model in MODELS:
                for variant in VARIANTS:
                    r = by_key.get((lg, model, variant))
                    cells.append(fmt_cell(r))
            print(f"  {lg:<5} | " + " | ".join(cells))

    render("Language adherence  on_target_rate",
           lambda r: f"{'-':>11}" if r is None else f"{r['on_target_rate']:>6.1%} ({r['n_total']:>3})",
           nwm_adh)

    render("Length stats        mean tokens",
           lambda r: f"{'-':>11}" if r is None else f"{r['mean']:>6.1f}  (n={r['n']:>3})",
           nwm_len)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lid_categorized_dir", default="lid_categorized")
    p.add_argument("--output_dir",          default="results/aggregate")
    p.add_argument("--no_summary", action="store_true",
                   help="Skip stdout summary tables; only write CSVs.")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.lid_categorized_dir):
        sys.exit(f"Categorized dir not found: {args.lid_categorized_dir}.\n"
                 f"Run preprocess/categorize_lid.py first.")
    os.makedirs(args.output_dir, exist_ok=True)

    adherence_rows, length_rows, skipped = aggregate(args.lid_categorized_dir)

    adh_path = os.path.join(args.output_dir, "adherence.csv")
    len_path = os.path.join(args.output_dir, "length_stats.csv")
    write_csv(adherence_rows, adh_path)
    write_csv(length_rows,    len_path)

    print(f"Wrote {adh_path}  ({len(adherence_rows)} rows)")
    print(f"Wrote {len_path}  ({len(length_rows)} rows)")

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for s in skipped[:10]:
            print(f"  {s}")
        if len(skipped) > 10:
            print(f"  ... ({len(skipped) - 10} more)")

    if not args.no_summary:
        print_summary(adherence_rows, length_rows)


if __name__ == "__main__":
    main()
