import argparse
import json
import math
import os
import re
from datetime import datetime, timezone

import numpy as np

from loader import (
    load_tokenizer_only,
    build_transformers_config_for_detection,
    load_watermark,
)
from sampler import LANGUAGES

WATERMARK_CHOICES = ["KGW", "Unigram", "SynthID", "EXPEdit", "DIP", "XSIR"]
VARIANT_DATASET = {"base": "flores", "instruct": "aya"}

FILENAME_RE = re.compile(
    r'^(aya|flores)'
    r'_(mistral|gemma|qwen)'
    r'_(base|instruct)'
    r'_(KGW|Unigram|SynthID|EXPEdit|DIP|XSIR)'
    r'_(eng|fra|spa|por|nld|zho|jpn|arb|hin|tur|vie)'
    r'\.jsonl$'
)


def parse_args():
    parser = argparse.ArgumentParser(description="Watermark detection pipeline.")
    parser.add_argument(
        "input_file", nargs="?", default=None,
        help="Path to a single watermarked generation JSONL file. "
             "Mutually exclusive with --model/--variant/--watermark.",
    )
    parser.add_argument("--model",     choices=["mistral", "gemma", "qwen"])
    parser.add_argument("--variant",   choices=["base", "instruct"])
    parser.add_argument("--watermark", choices=WATERMARK_CHOICES)
    parser.add_argument("--input_dir", default="outputs",
                        help="Directory of generation files (cell mode).")
    parser.add_argument("--output_dir", default="detection")
    parser.add_argument(
        "--device", default="cpu",
        help="Device for detection (cpu or cuda). Must match the device used during "
             "generation to ensure identical green-list partitioning for hash-based "
             "watermarks (KGW, Unigram, DIP, etc.). Required for XSIR embedding models.",
    )
    args = parser.parse_args()

    cell_flags = (args.model, args.variant, args.watermark)
    if args.input_file is not None and any(cell_flags):
        raise SystemExit(
            "ERROR: pass either input_file OR --model/--variant/--watermark, not both."
        )
    if args.input_file is None and not all(cell_flags):
        raise SystemExit(
            "ERROR: must pass either input_file OR all three of --model/--variant/--watermark."
        )
    return args


def parse_filename(path: str):
    fname = os.path.basename(path)
    m = FILENAME_RE.match(fname)
    if not m:
        raise SystemExit(
            f"\nERROR: Cannot parse filename: {fname!r}\n"
            f"Expected format: {{dataset}}_{{model}}_{{variant}}_{{watermark}}_{{language}}.jsonl\n"
            f"  dataset:   aya | flores\n"
            f"  model:     mistral | gemma | qwen\n"
            f"  variant:   base | instruct\n"
            f"  watermark: KGW | Unigram | SynthID | EXPEdit | DIP | XSIR\n"
            f"  language:  eng | fra | spa | por | nld | zho | jpn | arb | hin | tur | vie\n"
            f"\nNote: pass a watermarked file, not a nowatermark file.\n"
        )
    dataset, model, variant, watermark, language = m.groups()
    return dataset, model, variant, watermark, language


def load_existing_ids(path: str) -> set:
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


def load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _expedit_score(wm_detector, xi_np, text: str) -> float:
    tokenizer = wm_detector.config.generation_tokenizer
    encoded = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).numpy()[0]
    if len(encoded) == 0:
        return 0.0
    test_result, _ = wm_detector.utils.one_run(encoded, xi_np)
    score = -float(test_result)
    if not math.isfinite(score):
        score = 1e10 if score > 0 else -1e10
    return score


def run_detection(records, existing_ids, out_path, wm_detector, is_wm_truth, label, watermark_name):
    pending = [r for r in records if r["prompt_id"] not in existing_ids]
    print(f"  {label}: {len(pending)} to detect ({len(existing_ids)} already done)")
    if not pending:
        return

    use_expedit_fastpath = watermark_name == "EXPEdit"
    if use_expedit_fastpath:
        expedit_xi_np = wm_detector.utils.xi.float().numpy().copy()
        max_below_one = np.nextafter(np.float32(1.0), np.float32(0.0))
        np.clip(expedit_xi_np, 0.0, max_below_one, out=expedit_xi_np)
    else:
        expedit_xi_np = None

    with open(out_path, "a", encoding="utf-8") as out_f:
        for record in pending:
            text = record.get("generated_text", "")

            if use_expedit_fastpath:
                score = _expedit_score(wm_detector, expedit_xi_np, text)
                is_pred = None
            else:
                detect_result = wm_detector.detect_watermark(text, return_dict=True)
                score = float(detect_result["score"])
                is_pred = bool(detect_result["is_watermarked"])

            out_record = {
                "prompt_id":            record["prompt_id"],
                "language":             record["language"],
                "dataset":              record["dataset"],
                "model":                record["model"],
                "variant":              record["variant"],
                "watermark_method":     record.get("watermark_method"),
                "generated_text":       text,
                "token_count":          record.get("token_count"),
                "watermark_config":     record.get("watermark_config"),
                "is_watermarked_truth": is_wm_truth,
                "score":                score,
                "is_watermarked_pred":  is_pred,
                "detection_timestamp":  datetime.now(timezone.utc).isoformat(),
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            out_f.flush()


def run_cell(dataset, model_alias, variant, watermark, language,
             wm_detector, input_dir, output_dir):
    wm_in  = os.path.join(input_dir, f"{dataset}_{model_alias}_{variant}_{watermark}_{language}.jsonl")
    nwm_in = os.path.join(input_dir, f"{dataset}_{model_alias}_{variant}_nowatermark_{language}.jsonl")

    wm_out  = os.path.join(output_dir, f"{dataset}_{model_alias}_{variant}_{watermark}_{language}_detect.jsonl")
    nwm_out = os.path.join(output_dir, f"{dataset}_{model_alias}_{variant}_{watermark}_nowatermark_{language}_detect.jsonl")

    print(f"\n[{language}]")
    if not os.path.exists(wm_in):
        print(f"  watermarked file not found, skipping: {os.path.basename(wm_in)}")
        return

    print(f"  Input: {os.path.basename(wm_in)}")
    wm_records = load_jsonl(wm_in)
    run_detection(wm_records, load_existing_ids(wm_out), wm_out, wm_detector, True, "watermarked", watermark)

    if not os.path.exists(nwm_in):
        print(f"  nowatermark file not found, skipping: {os.path.basename(nwm_in)}")
        return
    print(f"  Input: {os.path.basename(nwm_in)}")
    nwm_records = load_jsonl(nwm_in)
    run_detection(nwm_records, load_existing_ids(nwm_out), nwm_out, wm_detector, False, "nowatermark", watermark)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_file is not None:
        dataset, model, variant, watermark, language = parse_filename(args.input_file)
        input_dir = os.path.dirname(os.path.abspath(args.input_file))
        languages = [language]
    else:
        model     = args.model
        variant   = args.variant
        watermark = args.watermark
        dataset   = VARIANT_DATASET[variant]
        input_dir = args.input_dir
        languages = LANGUAGES

    print(f"Loading tokenizer: {model} / {variant}")
    tokenizer = load_tokenizer_only(model, variant)
    transformers_config = build_transformers_config_for_detection(tokenizer, model, variant, device=args.device)

    print(f"Loading watermark detector: {watermark}")
    wm_detector = load_watermark(watermark, transformers_config, model)

    for lang in languages:
        run_cell(dataset, model, variant, watermark, lang,
                 wm_detector, input_dir, args.output_dir)

    print("\nDetection complete.")


if __name__ == "__main__":
    main()
