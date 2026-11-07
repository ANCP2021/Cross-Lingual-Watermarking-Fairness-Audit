import argparse
import fnmatch
import json
import os
import re
import sys
import time
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

GEN_FNAME_RE = re.compile(
    r"^(aya|flores)"
    r"_(mistral|gemma|qwen)"
    r"_(base|instruct)"
    r"_(KGW|Unigram|SynthID|EXPEdit|DIP|XSIR|nowatermark)"
    r"_(eng|fra|spa|por|nld|zho|jpn|arb|hin|tur|vie)"
    r"\.jsonl$"
)

ENCODER_ID = "FacebookAI/xlm-roberta-large"
HIDDEN_DIM = 1024
MAX_LEN = 512


def load_jsonl(path: str):
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


def load_existing(npz_path: str):
    if not os.path.exists(npz_path):
        return set(), None, None, None
    with np.load(npz_path, allow_pickle=False) as data:
        pids  = data["prompt_ids"]
        embs  = data["embeddings"]
        trunc = data["truncated"]
    return set(pids.tolist()), pids, embs, trunc


def mean_pool(last_hidden_state, attention_mask):
    mask    = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed  = (last_hidden_state * mask).sum(dim=1)
    counts  = mask.sum(dim=1).clamp(min=1)
    return summed / counts


@torch.no_grad()
def encode_batch(texts: List[str], tokenizer, model, device):
    untrunc    = tokenizer(texts, padding=False, truncation=False, add_special_tokens=True)
    lengths    = [len(ids) for ids in untrunc["input_ids"]]
    truncated  = np.array([L > MAX_LEN for L in lengths], dtype=bool)

    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LEN,
        add_special_tokens=True, return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
    return pooled.detach().cpu().to(torch.float16).numpy(), truncated


def process_file(in_path, out_path, tokenizer, model, device,
                 batch_size, smoke_n, report_every):
    existing_ids, old_pids, old_embs, old_trunc = load_existing(out_path)

    records = load_jsonl(in_path)
    pending = [r for r in records if r["prompt_id"] not in existing_ids]
    if smoke_n is not None:
        pending = pending[:smoke_n]

    if not pending:
        print(f"  {len(existing_ids)} already done, nothing pending")
        return 0

    print(f"  {len(pending)} pending ({len(existing_ids)} already done)")

    new_pids:  List[str]        = []
    new_embs:  List[np.ndarray] = []
    new_trunc: List[np.ndarray] = []

    t0 = time.time()
    n_batches = (len(pending) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        batch = pending[bi * batch_size : (bi + 1) * batch_size]
        texts = [r.get("generated_text", "") or "" for r in batch]
        embs, trunc = encode_batch(texts, tokenizer, model, device)

        new_pids.extend(r["prompt_id"] for r in batch)
        new_embs.append(embs)
        new_trunc.append(trunc)

        if (bi + 1) % report_every == 0:
            done = min((bi + 1) * batch_size, len(pending))
            rate = done / (time.time() - t0 + 1e-9)
            print(f"    {done}/{len(pending)}  ({rate:.1f} rec/s)")

    new_pids_arr  = np.array(new_pids)
    new_embs_arr  = np.concatenate(new_embs,  axis=0)
    new_trunc_arr = np.concatenate(new_trunc, axis=0)

    if old_pids is not None:
        all_pids  = np.concatenate([old_pids,  new_pids_arr])
        all_embs  = np.concatenate([old_embs,  new_embs_arr],  axis=0)
        all_trunc = np.concatenate([old_trunc, new_trunc_arr])
    else:
        all_pids, all_embs, all_trunc = new_pids_arr, new_embs_arr, new_trunc_arr

    np.savez(out_path,
             prompt_ids = all_pids,
             embeddings = all_embs,
             truncated  = all_trunc)

    n_trunc = int(new_trunc_arr.sum())
    if n_trunc > 0:
        pct = 100.0 * n_trunc / len(new_trunc_arr)
        print(f"    truncated to {MAX_LEN}: {n_trunc}/{len(new_trunc_arr)} ({pct:.1f}%)")
    return len(pending)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",   default="outputs")
    p.add_argument("--output_dir",  default="quality/embeddings")
    p.add_argument("--pattern",     default="*",
                   help="Glob pattern for input filenames (e.g. 'flores_gemma_*').")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--smoke_test",  type=int, default=None,
                   help="Encode only the first N records per file.")
    p.add_argument("--report_every", type=int, default=4,
                   help="Print progress every N batches.")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        sys.exit(f"Input dir not found: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading encoder: {ENCODER_ID}")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(ENCODER_ID, torch_dtype=dtype)
    model.to(args.device)
    model.eval()

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".jsonl"))
    files = [f for f in files if fnmatch.fnmatch(f, args.pattern)]
    if not files:
        sys.exit(f"No files in {args.input_dir} matching {args.pattern!r}.")

    print(f"\nProcessing {len(files)} file(s) from {args.input_dir}/  ->  {args.output_dir}/")

    total = 0
    for fname in files:
        if not GEN_FNAME_RE.match(fname):
            print(f"\nSkipping unparsed filename: {fname}")
            continue
        in_path  = os.path.join(args.input_dir,  fname)
        out_path = os.path.join(args.output_dir, fname.replace(".jsonl", ".npz"))
        print(f"\n{fname}")
        total += process_file(in_path, out_path, tokenizer, model, args.device,
                              args.batch_size, args.smoke_test, args.report_every)

    print(f"\nEncoded {total} new records total -> {args.output_dir}/")


if __name__ == "__main__":
    main()
