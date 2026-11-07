import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch

from sampler import sample_flores, sample_aya, LANGUAGES
from loader import (
    load_model_and_tokenizer,
    build_transformers_config,
    load_watermark,
    GEN_CONFIG,
    get_watermark_config_path,
)

VALID_COMBOS = {("instruct", "aya"), ("base", "flores")}
WATERMARK_CHOICES = ["KGW", "Unigram", "SynthID", "EXPEdit", "DIP", "XSIR"]


def parse_args():
    parser = argparse.ArgumentParser(description="Watermarked text generation pipeline.")
    parser.add_argument("--model",       required=True, choices=["mistral", "gemma", "qwen"])
    parser.add_argument("--variant",     required=True, choices=["base", "instruct"])
    parser.add_argument("--dataset",     required=True, choices=["aya", "flores"])
    parser.add_argument("--watermark",   required=True, choices=WATERMARK_CHOICES)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir",  default="outputs")
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


def validate_combo(variant: str, dataset: str):
    if (variant, dataset) not in VALID_COMBOS:
        raise SystemExit(
            f"\nERROR: --variant '{variant}' + --dataset '{dataset}' is not a valid combination.\n"
            f"Valid combinations are:\n"
            f"  --variant instruct --dataset aya\n"
            f"  --variant base     --dataset flores\n"
        )


def output_path(output_dir, dataset, model, variant, watermark, language):
    fname = f"{dataset}_{model}_{variant}_{watermark}_{language}.jsonl"
    return os.path.join(output_dir, fname)


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


def prompt_seed(prompt_id: str, base_seed: int) -> int:
    id_bytes = prompt_id.encode()[:8].ljust(8, b"\x00")
    return (base_seed + int.from_bytes(id_bytes, "big")) % (2**31)


def apply_chat_template(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def strip_prompt(full_output: str, model_input: str, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer.encode(model_input, add_special_tokens=True)
        decoded_input = tokenizer.decode(ids, skip_special_tokens=True)
        if decoded_input and full_output.startswith(decoded_input):
            return full_output[len(decoded_input):]
        i = 0
        for a, b in zip(full_output, decoded_input):
            if a != b:
                break
            i += 1
        if i > 0:
            return full_output[i:].lstrip()

    if full_output.startswith(model_input):
        return full_output[len(model_input):]
    i = 0
    for a, b in zip(full_output, model_input):
        if a != b:
            break
        i += 1
    return full_output[i:].lstrip()


def main():
    args = parse_args()
    validate_combo(args.variant, args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Sampling {args.num_samples} prompts per language from {args.dataset}...")
    if args.dataset == "flores":
        samples = sample_flores(
            "data/flores_preprocessed_main.json", args.num_samples, args.seed
        )
    else:
        samples = sample_aya(
            "data/aya_dataset_preprocessed_main.json", args.num_samples, args.seed
        )

    print(f"\nLoading model: {args.model} / {args.variant}")
    model, tokenizer = load_model_and_tokenizer(args.model, args.variant)
    transformers_config = build_transformers_config(model, tokenizer)

    print(f"Loading watermark: {args.watermark}")
    watermark = load_watermark(args.watermark, transformers_config, args.model)

    with open(get_watermark_config_path(args.watermark, args.model), encoding="utf-8") as f:
        wm_config_dict = json.load(f)

    for lang in LANGUAGES:
        lang_samples = samples[lang]

        wm_path  = output_path(args.output_dir, args.dataset, args.model, args.variant, args.watermark, lang)
        nwm_path = output_path(args.output_dir, args.dataset, args.model, args.variant, "nowatermark", lang)

        existing_wm_ids  = load_existing_ids(wm_path)
        existing_nwm_ids = load_existing_ids(nwm_path)

        pending = [s for s in lang_samples if s[0] not in existing_wm_ids]
        print(f"\n[{lang}] {len(pending)} prompts to generate "
              f"({len(existing_wm_ids)} already done)")

        if not pending:
            continue

        with open(wm_path, "a", encoding="utf-8") as wm_f, \
             open(nwm_path, "a", encoding="utf-8") as nwm_f:
            for prompt_id, prompt_text in pending:
                if args.variant == "instruct":
                    model_input = apply_chat_template(tokenizer, prompt_text)
                else:
                    model_input = prompt_text

                pseed = prompt_seed(prompt_id, args.seed)

                torch.manual_seed(pseed)
                full_wm = watermark.generate_watermarked_text(model_input)
                generated_wm = strip_prompt(full_wm, model_input, tokenizer)

                wm_record = {
                    "prompt_id":          prompt_id,
                    "language":           lang,
                    "dataset":            args.dataset,
                    "model":              args.model,
                    "variant":            args.variant,
                    "watermark_method":   args.watermark,
                    "prompt_text":        prompt_text,
                    "model_input":        model_input,
                    "generated_text":     generated_wm,
                    "full_output":        full_wm,
                    "generation_config":  GEN_CONFIG,
                    "watermark_config":   wm_config_dict,
                    "token_count":        len(tokenizer.encode(generated_wm)),
                    "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                }
                wm_f.write(json.dumps(wm_record, ensure_ascii=False) + "\n")
                wm_f.flush()

                if prompt_id not in existing_nwm_ids:
                    torch.manual_seed(pseed)
                    full_nwm = watermark.generate_unwatermarked_text(model_input)
                    generated_nwm = strip_prompt(full_nwm, model_input, tokenizer)

                    nwm_record = {
                        "prompt_id":          prompt_id,
                        "language":           lang,
                        "dataset":            args.dataset,
                        "model":              args.model,
                        "variant":            args.variant,
                        "watermark_method":   "nowatermark",
                        "prompt_text":        prompt_text,
                        "model_input":        model_input,
                        "generated_text":     generated_nwm,
                        "full_output":        full_nwm,
                        "generation_config":  GEN_CONFIG,
                        "watermark_config":   None,
                        "token_count":        len(tokenizer.encode(generated_nwm)),
                        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    nwm_f.write(json.dumps(nwm_record, ensure_ascii=False) + "\n")
                    nwm_f.flush()
                    existing_nwm_ids.add(prompt_id)

        print(f"  [{lang}] done.")

    print("\nGeneration complete.")


if __name__ == "__main__":
    main()
