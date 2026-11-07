import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.lid_subsets import load_labels

LANGUAGES  = ["eng", "fra", "spa", "por", "nld", "zho", "jpn", "arb", "hin", "tur", "vie"]
MODELS     = ["mistral", "gemma", "qwen"]
VARIANTS   = ["base", "instruct"]
WATERMARKS = ["KGW", "Unigram", "SynthID", "EXPEdit", "DIP", "XSIR"]
SIDES      = ["wm", "nwm"]

VARIANT_DATASET = {"base": "flores", "instruct": "aya"}

SCORE_FIELDS = [
    "model", "variant", "watermark", "dataset", "language", "side",
    "prompt_id", "score", "lid_label", "token_count",
]
COVERAGE_FIELDS = [
    "model", "variant", "watermark", "dataset", "language", "side",
    "n_detect", "n_lid", "n_joined",
    "n_score_missing_lid", "n_lid_missing_score", "status",
]


def detect_path(detection_dir: str, dataset: str, model: str, variant: str,
                watermark: str, language: str, side: str) -> str:
    if side == "wm":
        fname = f"{dataset}_{model}_{variant}_{watermark}_{language}_detect.jsonl"
    elif side == "nwm":
        fname = f"{dataset}_{model}_{variant}_{watermark}_nowatermark_{language}_detect.jsonl"
    else:
        raise ValueError(f"Unknown side {side!r}")
    return os.path.join(detection_dir, fname)


def load_score_records(path: str) -> Dict[str, Tuple[float, int]]:
    out: Dict[str, Tuple[float, int]] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("prompt_id")
            sc  = rec.get("score")
            if pid is None or sc is None:
                continue
            tok_raw = rec.get("token_count")
            try:
                tok = int(tok_raw) if tok_raw is not None else None
            except (TypeError, ValueError):
                tok = None
            out[str(pid)] = (float(sc), tok)
    return out


def existing_score_keys(path: str) -> Set[Tuple[str, str, str, str, str]]:
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.add((row["model"], row["variant"], row["watermark"],
                     row["language"], row["side"]))
    return out


def append_rows(path: str, fields: List[str], rows: List[dict]):
    if not rows:
        return
    is_new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            w.writeheader()
        w.writerows(rows)


def process_side(detection_dir: str, lid_dir: str,
                 scores_csv: str, coverage_csv: str,
                 dataset: str, model: str, variant: str,
                 watermark: str, language: str, side: str) -> str:
    det_path = detect_path(detection_dir, dataset, model, variant, watermark, language, side)
    score_map = load_score_records(det_path)

    lid_key = watermark if side == "wm" else "nowatermark"
    try:
        lid_map = load_labels(model, variant, language, lid_key, lid_dir=lid_dir)
    except FileNotFoundError:
        lid_map = {}

    n_det = len(score_map)
    n_lid = len(lid_map)

    rows: List[dict] = []
    n_score_missing_lid = 0
    for pid, (score, tok) in score_map.items():
        lab = lid_map.get(pid, "NA")
        if lab == "NA":
            n_score_missing_lid += 1
        rows.append({
            "model": model, "variant": variant, "watermark": watermark,
            "dataset": dataset, "language": language, "side": side,
            "prompt_id": pid, "score": score, "lid_label": lab,
            "token_count": tok if tok is not None else "",
        })
    n_lid_missing_score = sum(1 for pid in lid_map if pid not in score_map)

    if n_det == 0 and n_lid == 0:
        status = "missing_files"
    elif n_det == 0:
        status = "missing_detect"
    elif n_lid == 0:
        status = "missing_lid"
    elif n_score_missing_lid > 0 or n_lid_missing_score > 0:
        status = "partial_join"
    else:
        status = "ok"

    append_rows(scores_csv, SCORE_FIELDS, rows)
    append_rows(coverage_csv, COVERAGE_FIELDS, [{
        "model": model, "variant": variant, "watermark": watermark,
        "dataset": dataset, "language": language, "side": side,
        "n_detect": n_det, "n_lid": n_lid, "n_joined": len(rows),
        "n_score_missing_lid": n_score_missing_lid,
        "n_lid_missing_score": n_lid_missing_score,
        "status": status,
    }])
    return status


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detection_dir", default="detection")
    p.add_argument("--lid_dir", default="lid_categorized")
    p.add_argument("--scores_csv",   default="results/detection/detection_scores.csv")
    p.add_argument("--coverage_csv", default="results/detection/cells_loaded.csv")
    p.add_argument("--models",     nargs="+", default=MODELS,     choices=MODELS)
    p.add_argument("--variants",   nargs="+", default=VARIANTS,   choices=VARIANTS)
    p.add_argument("--watermarks", nargs="+", default=WATERMARKS, choices=WATERMARKS)
    p.add_argument("--languages",  nargs="+", default=LANGUAGES,  choices=LANGUAGES)
    p.add_argument("--sides",      nargs="+", default=SIDES,      choices=SIDES)
    p.add_argument("--overwrite", action="store_true",
                   help="Truncate output CSVs before running.")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = os.path.dirname(args.scores_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    if args.overwrite:
        for path in (args.scores_csv, args.coverage_csv):
            if os.path.exists(path):
                os.remove(path)

    already = existing_score_keys(args.scores_csv)
    if already:
        print(f"Resume mode: {len(already)} (model,variant,watermark,lang,side) "
              f"tuples already present in {args.scores_csv}")

    status_counts: Dict[str, int] = {}
    n_processed = 0
    n_skipped = 0

    for variant in args.variants:
        dataset = VARIANT_DATASET[variant]
        for model in args.models:
            for watermark in args.watermarks:
                for language in args.languages:
                    for side in args.sides:
                        key = (model, variant, watermark, language, side)
                        if key in already:
                            n_skipped += 1
                            continue
                        status = process_side(
                            args.detection_dir, args.lid_dir,
                            args.scores_csv, args.coverage_csv,
                            dataset, model, variant, watermark, language, side,
                        )
                        status_counts[status] = status_counts.get(status, 0) + 1
                        n_processed += 1

    print(f"\nProcessed: {n_processed}  skipped (already cached): {n_skipped}")
    if status_counts:
        print("Status breakdown for processed cells:")
        for s in sorted(status_counts):
            print(f"  {s:18s} {status_counts[s]}")
    print(f"\nScores   -> {args.scores_csv}")
    print(f"Coverage -> {args.coverage_csv}")


if __name__ == "__main__":
    main()
