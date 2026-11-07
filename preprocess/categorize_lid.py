import argparse
import csv
import json
import os
import re
import sys

PROMPT_LANG_LABELS = {
    "eng": {"eng_Latn"},
    "fra": {"fra_Latn"},
    "spa": {"spa_Latn"},
    "por": {"por_Latn"},
    "nld": {"nld_Latn"},
    "zho": {"cmn_Hans", "cmn_Hant", "zho_Hans", "zho_Hant", "cmn_Hani", "zho_Hani", "wuu_Hani"},
    "jpn": {"jpn_Jpan"},
    "arb": {
        "arb_Arab", "ara_Arab",
        "ary_Arab",
        "arz_Arab",
        "ars_Arab",
        "acm_Arab",
        "aeb_Arab",
        "apc_Arab",
        "ajp_Arab",
    },
    "hin": {
        "hin_Deva",
        "anp_Deva",
        "bho_Deva",
        "mag_Deva",
        "awa_Deva",
    },
    "tur": {"tur_Latn"},
    "vie": {"vie_Latn"},
}

LID_FNAME_RE = re.compile(r"^(.+)_lid\.jsonl$")


def categorize_record(rec, t):
    target = PROMPT_LANG_LABELS.get(rec["language"], set())

    glotlid       = rec.get("glotlid")  or []
    fasttext_top3 = [x["lang"] for x in (rec.get("fasttext") or [])]
    sentences     = rec.get("sentences") or []
    token_count   = rec.get("token_count") or 0

    glotlid_top = glotlid[0] if glotlid else None

    if token_count < t["min_total_tokens"]:
        return "uncertain", _diag(glotlid_top, fasttext_top3, None, 0)
    if glotlid_top is None or glotlid_top["conf"] < t["conf_floor"]:
        return "uncertain", _diag(glotlid_top, fasttext_top3, None, 0)

    valid = [s for s in sentences if (s.get("n_chars") or 0) >= t["min_sentence_chars"]]
    if valid:
        on_target = sum(1 for s in valid if s.get("lang") in target)
        frac = on_target / len(valid)
    else:
        frac = 1.0 if glotlid_top["lang"] in target else 0.0

    glotlid_says_target    = glotlid_top["lang"] in target
    fasttext_no_contradict = any(label in target for label in fasttext_top3)

    if (glotlid_says_target
            and fasttext_no_contradict
            and frac >= t["on_target_ratio"]):
        return "on_target", _diag(glotlid_top, fasttext_top3, frac, len(valid))

    if (not glotlid_says_target
            and frac < t["code_mix_threshold"]):
        return "off_target", _diag(glotlid_top, fasttext_top3, frac, len(valid))

    if 0 < frac < t["on_target_ratio"]:
        return "code_mixed", _diag(glotlid_top, fasttext_top3, frac, len(valid))

    return "uncertain", _diag(glotlid_top, fasttext_top3, frac, len(valid))


def _diag(glot_top, ft_top3, frac, n_valid):
    return {
        "on_target_sentence_frac": None if frac is None else round(frac, 4),
        "valid_sentence_count":   n_valid,
        "glotlid_top":            glot_top,
        "fasttext_top3":          ft_top3,
    }


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def existing_ids(path):
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["prompt_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def categorize_file(in_path, out_path, thresholds, overwrite):
    if overwrite and os.path.exists(out_path):
        os.remove(out_path)
    seen = existing_ids(out_path)
    records = read_jsonl(in_path)
    pending = [r for r in records if r["prompt_id"] not in seen]
    if not pending:
        return 0

    with open(out_path, "a", encoding="utf-8") as f:
        for rec in pending:
            label, diag = categorize_record(rec, thresholds)
            out = {
                "prompt_id":        rec["prompt_id"],
                "language":         rec["language"],
                "dataset":          rec["dataset"],
                "model":            rec["model"],
                "variant":          rec["variant"],
                "watermark_method": rec.get("watermark_method"),
                "token_count":      rec.get("token_count"),
                "lid_label":        label,
                **diag,
                "thresholds":       thresholds,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return len(pending)


def run_sweep(args, lid_files):
    sweep_configs = []
    for cf in [0.3, 0.5, 0.7]:
        for ot in [0.7, 0.8, 0.9]:
            sweep_configs.append({
                "conf_floor":         cf,
                "on_target_ratio":    ot,
                "code_mix_threshold": args.code_mix_threshold,
                "min_total_tokens":   args.min_total_tokens,
                "min_sentence_chars": args.min_sentence_chars,
            })

    rows = []
    for cfg in sweep_configs:
        cfg_id = f"cf{cfg['conf_floor']}_ot{cfg['on_target_ratio']}"
        counts = {}
        for path in lid_files:
            for rec in read_jsonl(path):
                key = (rec["model"], rec["variant"], rec["language"],
                       rec.get("watermark_method") or "nowatermark")
                if key not in counts:
                    counts[key] = {"on_target": 0, "code_mixed": 0,
                                   "off_target": 0, "uncertain": 0}
                label, _ = categorize_record(rec, cfg)
                counts[key][label] += 1

        for (model, variant, lang, wm), labels in counts.items():
            n = sum(labels.values())
            rows.append({
                "config_id":       cfg_id,
                "conf_floor":      cfg["conf_floor"],
                "on_target_ratio": cfg["on_target_ratio"],
                "model":           model,
                "variant":         variant,
                "language":        lang,
                "watermark":       wm,
                "n_total":         n,
                "on_target_rate":  labels["on_target"] / max(1, n),
                "code_mixed_rate": labels["code_mixed"] / max(1, n),
                "off_target_rate": labels["off_target"] / max(1, n),
                "uncertain_rate": labels["uncertain"] / max(1, n),
            })

    os.makedirs(args.results_dir, exist_ok=True)
    out_csv = os.path.join(args.results_dir, "threshold_sweep.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nThreshold sweep written: {out_csv}  ({len(rows)} rows)")

    _print_stability(rows)


def _print_stability(rows):
    by_cfg = {}
    for r in rows:
        if r["watermark"] != "nowatermark":
            continue
        by_cfg.setdefault(r["config_id"], {})[
            (r["model"], r["variant"], r["language"])
        ] = r["on_target_rate"]

    if "cf0.5_ot0.8" not in by_cfg:
        return
    base = by_cfg["cf0.5_ot0.8"]
    keys = sorted(base.keys())
    base_vec = [base[k] for k in keys]

    print("\nRanking stability vs default (cf0.5_ot0.8), nowatermark cells:")
    print(f"  {'config':<14} {'spearman_rho':>14}")
    for cid in sorted(by_cfg.keys()):
        if cid == "cf0.5_ot0.8":
            continue
        vec = [by_cfg[cid].get(k, 0.0) for k in keys]
        print(f"  {cid:<14} {_spearman(base_vec, vec):>14.4f}")


def _spearman(a, b):
    if len(a) != len(b) or len(a) < 2:
        return float("nan")

    def ranks(xs):
        sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0] * len(xs)
        for rk, i in enumerate(sorted_idx):
            r[i] = rk
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(len(ra)))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((x - mb) ** 2 for x in rb) ** 0.5
    if da == 0 or db == 0:
        return float("nan")
    return num / (da * db)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lid_dir",     default="lid")
    p.add_argument("--out_dir",     default="lid_categorized")
    p.add_argument("--results_dir", default="results/preprocess")

    p.add_argument("--conf_floor",         type=float, default=0.5)
    p.add_argument("--on_target_ratio",    type=float, default=0.8)
    p.add_argument("--code_mix_threshold", type=float, default=0.2)
    p.add_argument("--min_total_tokens",   type=int,   default=20)
    p.add_argument("--min_sentence_chars", type=int,   default=20)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-categorize from scratch (delete existing categorized files first).")
    p.add_argument("--sweep", action="store_true",
                   help="Run threshold sensitivity sweep (writes threshold_sweep.csv).")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.lid_dir):
        sys.exit(f"LID dir not found: {args.lid_dir}.  Run preprocess/run_lid.py first.")
    os.makedirs(args.out_dir, exist_ok=True)

    lid_files = sorted(
        os.path.join(args.lid_dir, f) for f in os.listdir(args.lid_dir)
        if f.endswith("_lid.jsonl")
    )
    if not lid_files:
        sys.exit(f"No *_lid.jsonl files in {args.lid_dir}.")

    if args.sweep:
        run_sweep(args, lid_files)
        return

    thresholds = {
        "conf_floor":         args.conf_floor,
        "on_target_ratio":    args.on_target_ratio,
        "code_mix_threshold": args.code_mix_threshold,
        "min_total_tokens":   args.min_total_tokens,
        "min_sentence_chars": args.min_sentence_chars,
    }
    print(f"Categorizing with thresholds: {thresholds}")

    total = 0
    for in_path in lid_files:
        m = LID_FNAME_RE.match(os.path.basename(in_path))
        if not m:
            continue
        out_path = os.path.join(args.out_dir, f"{m.group(1)}_lid_categorized.jsonl")
        n = categorize_file(in_path, out_path, thresholds, args.overwrite)
        total += n
        print(f"  {os.path.basename(in_path)}  ->  {n} records")

    print(f"\nCategorized {total} records total -> {args.out_dir}/")


if __name__ == "__main__":
    main()
