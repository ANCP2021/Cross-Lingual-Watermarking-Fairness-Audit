import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REFERENCE_MODELS: Dict[str, dict] = {
    "xglm-7.5b": {
        "hf_id": "facebook/xglm-7.5B",
        "ctx":   2048,
    },
    "mgpt-1.3b": {
        "hf_id": "ai-forever/mGPT",
        "ctx":   2048,
    },
    "mgpt-13b": {
        "hf_id": "ai-forever/mGPT-13B",
        "ctx":   2048,
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference_model",
                   choices=sorted(REFERENCE_MODELS.keys()),
                   default="xglm-7.5b",
                   help="Held-out reference LM. Primary is xglm-7.5b; "
                        "mgpt-{1.3b, 13b} available as cross-model robustness checks.")
    p.add_argument("--input_dir",  default="outputs")
    p.add_argument("--output_dir", default="quality/ppl",
                   help="Stage-1 cache root. Subdir is keyed by reference_model tag.")
    p.add_argument("--pattern", default="*.jsonl",
                   help="Glob over --input_dir to filter input files.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype",  default="float16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--smoke_test", type=int, default=0,
                   help="If >0, score only the first N pending records per file.")
    return p.parse_args()


def load_existing_ids(path: str) -> Set[str]:
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


def load_input(path: str) -> List[dict]:
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


@torch.no_grad()
def score_record(model, tokenizer, prompt_text: str, generated_text: str,
                 ctx_len: int, device: str) -> Tuple[float, int, int]:
    if not generated_text:
        return float("nan"), 0, 0

    prompt_ids = tokenizer.encode(prompt_text or "", add_special_tokens=True)
    gen_ids    = tokenizer.encode(generated_text,    add_special_tokens=False)
    n_gen = len(gen_ids)
    if n_gen == 0:
        return float("nan"), 0, 0

    truncated = 0
    if len(prompt_ids) + n_gen > ctx_len:
        keep_prompt = max(0, ctx_len - n_gen)
        prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt > 0 else []
        if n_gen > ctx_len:
            gen_ids = gen_ids[-ctx_len:]
            n_gen = ctx_len
        truncated = 1

    input_ids = torch.tensor(prompt_ids + gen_ids, dtype=torch.long,
                             device=device).unsqueeze(0)
    n_prompt  = len(prompt_ids)

    logits = model(input_ids=input_ids).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
    nll = -log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    start = max(0, n_prompt - 1)
    nll_gen = nll[:, start:]
    n_eval  = nll_gen.numel()
    if n_eval == 0:
        return float("nan"), 0, truncated
    return nll_gen.mean().item(), int(n_eval), truncated


def main():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        sys.exit(f"Input dir not found: {args.input_dir}")

    cfg = REFERENCE_MODELS[args.reference_model]
    out_root = os.path.join(args.output_dir, args.reference_model)
    os.makedirs(out_root, exist_ok=True)

    input_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not input_files:
        sys.exit(f"No files match {args.pattern!r} in {args.input_dir}")
    print(f"Reference model: {cfg['hf_id']}  (ctx={cfg['ctx']})")
    print(f"Output cache:    {out_root}")
    print(f"Input files:     {len(input_files)}")

    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    dtype = {"float32": torch.float32,
             "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    print(f"Loading model in {args.dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    print(f"Model loaded.  device={args.device}\n")

    t0      = time.time()
    total   = 0
    files_skipped = 0
    files_done    = 0

    for in_path in input_files:
        fname = os.path.basename(in_path)
        out_name = fname[:-len(".jsonl")] + "_ppl.jsonl"
        out_path = os.path.join(out_root, out_name)

        records  = load_input(in_path)
        existing = load_existing_ids(out_path)
        pending  = [r for r in records if r["prompt_id"] not in existing]
        if args.smoke_test > 0:
            pending = pending[:args.smoke_test]

        print(f"[{fname}]  {len(pending)} to score "
              f"({len(existing)} done, {len(records)} total)")
        if not pending:
            files_skipped += 1
            continue

        with open(out_path, "a", encoding="utf-8") as out_f:
            for rec in pending:
                prompt_text    = rec.get("prompt_text",    "") or ""
                generated_text = rec.get("generated_text", "") or ""
                mean_nll, n_eval, trunc = score_record(
                    model, tokenizer, prompt_text, generated_text,
                    cfg["ctx"], args.device,
                )
                out_rec = {
                    "prompt_id":     rec["prompt_id"],
                    "mean_nll":      None if mean_nll != mean_nll else mean_nll,
                    "n_eval_tokens": n_eval,
                    "n_truncated":   trunc,
                }
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                out_f.flush()
                total += 1

        files_done += 1
        elapsed = time.time() - t0
        rate    = total / elapsed if elapsed > 0 else 0.0
        print(f"  done.  scored={total}  rate={rate:.2f}/s  elapsed={elapsed:.1f}s")

    print(f"\nFinished. {total} generations scored across {files_done} files "
          f"({files_skipped} files already complete). ref_model={args.reference_model}")
    print(f"Output:   {out_root}")


if __name__ == "__main__":
    main()
