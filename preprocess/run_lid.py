import argparse
import fnmatch
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import List


LANGUAGES = ["eng", "fra", "spa", "por", "nld", "zho", "jpn", "arb", "hin", "tur", "vie"]

GEN_FNAME_RE = re.compile(
    r"^(aya|flores)"
    r"_(mistral|gemma|qwen)"
    r"_(base|instruct)"
    r"_(KGW|Unigram|SynthID|EXPEdit|DIP|XSIR|nowatermark)"
    r"_(eng|fra|spa|por|nld|zho|jpn|arb|hin|tur|vie)"
    r"\.jsonl$"
)

GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_FILE = "model_v3.bin"

FASTTEXT_LID_REPO = "facebook/fasttext-language-identification"
FASTTEXT_LID_FILE = "model.bin"


def load_glotlid():
    import fasttext
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=GLOTLID_REPO, filename=GLOTLID_FILE)
    print(f"  GlotLID:  {path}")
    return fasttext.load_model(path)


def load_fasttext_lid():
    import fasttext
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=FASTTEXT_LID_REPO, filename=FASTTEXT_LID_FILE)
    print(f"  fastText: {path}")
    return fasttext.load_model(path)


SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[。！？।؟])(?=.)")


class SentenceSplitter:
    def split(self, text: str, prompt_lang: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        return [s.strip() for s in SENT_BOUNDARY.split(text) if s.strip()]


def topk_predict(model, text: str, k: int = 3):
    text = (text or "").replace("\n", " ").strip()
    if not text:
        return []
    labels, probs = model.predict(text, k=k)
    out = []
    for label, prob in zip(labels, probs):
        out.append({"lang": label.replace("__label__", ""), "conf": float(prob)})
    return out


def lid_for_text(text: str, prompt_lang: str, glotlid, fasttext_lid, splitter):
    sentences = splitter.split(text, prompt_lang)
    sentence_records = []
    for s in sentences:
        top = topk_predict(glotlid, s, k=1)
        sentence_records.append({
            "text":    s,
            "lang":    top[0]["lang"] if top else None,
            "conf":    top[0]["conf"] if top else None,
            "n_chars": len(s),
            "n_words": len(s.split()),
        })
    return {
        "glotlid":   topk_predict(glotlid,      text, k=3),
        "fasttext":  topk_predict(fasttext_lid, text, k=3),
        "sentences": sentence_records,
    }


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="outputs")
    p.add_argument("--lid_dir",   default="lid")
    p.add_argument("--pattern", default="*",
                   help="Glob pattern for input filenames (e.g. 'flores_gemma_*').")
    p.add_argument("--smoke_test", type=int, default=None,
                   help="Process only the first N records per file.")
    p.add_argument("--report_every", type=int, default=200,
                   help="Print progress every N records.")
    return p.parse_args()


def process_file(in_path, out_path, prompt_lang, glotlid, fasttext_lid, splitter,
                 smoke_n=None, report_every=200):
    records = load_jsonl(in_path)
    existing = load_existing_ids(out_path)
    pending = [r for r in records if r["prompt_id"] not in existing]
    if smoke_n is not None:
        pending = pending[:smoke_n]

    if not pending:
        print(f"  {len(existing)} already done, nothing pending")
        return

    print(f"  {len(pending)} pending ({len(existing)} already done)")
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as out_f:
        for i, rec in enumerate(pending, 1):
            text = rec.get("generated_text", "")
            lid = lid_for_text(text, prompt_lang, glotlid, fasttext_lid, splitter)
            out_record = {
                "prompt_id":        rec["prompt_id"],
                "language":         rec["language"],
                "dataset":          rec["dataset"],
                "model":            rec["model"],
                "variant":          rec["variant"],
                "watermark_method": rec.get("watermark_method"),
                "token_count":      rec.get("token_count"),
                "glotlid":          lid["glotlid"],
                "fasttext":         lid["fasttext"],
                "sentences":        lid["sentences"],
                "lid_timestamp":    datetime.now(timezone.utc).isoformat(),
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            out_f.flush()
            if i % report_every == 0:
                rate = i / (time.time() - t0 + 1e-9)
                print(f"    {i}/{len(pending)}  ({rate:.1f} rec/s)")


def main():
    args = parse_args()
    os.makedirs(args.lid_dir, exist_ok=True)

    print("Loading LID models...")
    glotlid = load_glotlid()
    fasttext_lid = load_fasttext_lid()
    print("Initializing sentence splitter...")
    splitter = SentenceSplitter()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"Input dir not found: {args.input_dir}")

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".jsonl"))
    files = [f for f in files if fnmatch.fnmatch(f, args.pattern)]
    if not files:
        sys.exit(f"No files in {args.input_dir} matching {args.pattern!r}.")

    print(f"\nProcessing {len(files)} file(s) from {args.input_dir}/  ->  {args.lid_dir}/")

    for fname in files:
        m = GEN_FNAME_RE.match(fname)
        if not m:
            print(f"\nSkipping unparsed filename: {fname}")
            continue
        prompt_lang = m.group(5)
        in_path  = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.lid_dir, fname.replace(".jsonl", "_lid.jsonl"))
        print(f"\n{fname}  (prompt_lang={prompt_lang})")
        process_file(in_path, out_path, prompt_lang, glotlid, fasttext_lid, splitter,
                     args.smoke_test, args.report_every)

    print("\nLID preprocessing complete.")


if __name__ == "__main__":
    main()
