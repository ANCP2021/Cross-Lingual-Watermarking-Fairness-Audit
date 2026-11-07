#!/usr/bin/env python
"""Qwen2.5-72B scale pilot — pre-flight gates. Run before any other stage."""

import argparse
import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shim

FLORES_PATH = os.path.join(shim.REPO_ROOT, "data", "flores_preprocessed_main.json")
AYA_PATH    = os.path.join(shim.REPO_ROOT, "data", "aya_dataset_preprocessed_main.json")

REQUIRED_FLORES_TAGS = {
    "eng_Latn", "fra_Latn", "spa_Latn", "por_Latn", "nld_Latn", "cmn_Hans",
    "jpn_Jpan", "arb_Arab", "hin_Deva", "tur_Latn", "vie_Latn",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_flores(path, split="devtest", n_needed=shim.NUM_SAMPLES):
    out = {"path": path, "ok": False}
    if not os.path.exists(path):
        out["error"] = "file not found"
        return out
    out["sha256"] = sha256_file(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    records = raw if isinstance(raw, list) else [r for v in raw.values() for r in v]
    records = [r for r in records if r.get("split") == split]

    by_id = collections.defaultdict(set)
    for r in records:
        by_id[r["id"]].add(f"{r.get('iso_639_3')}_{r.get('iso_15924')}")
    present = set().union(*by_id.values()) if by_id else set()

    missing = sorted(REQUIRED_FLORES_TAGS - present)
    complete = [i for i, tags in by_id.items() if REQUIRED_FLORES_TAGS <= tags]

    out.update({
        "split": split, "n_records": len(records), "n_ids": len(by_id),
        "n_tags_present": len(present), "missing_tags": missing,
        "n_ids_with_all_tags": len(complete),
    })
    if missing:
        out["error"] = (f"missing language tag(s) {missing} — sampler.sample_flores "
                        f"raises ValueError before sampling. This file cannot "
                        f"reproduce the paper's base-regime prompts.")
        return out
    if len(complete) < n_needed:
        out["error"] = (f"only {len(complete)} ids carry all 11 tags; "
                        f"{n_needed} required")
        return out
    out["ok"] = True
    return out


def gate_aya(path, n_needed=shim.NUM_SAMPLES):
    out = {"path": path, "ok": False}
    if not os.path.exists(path):
        out["error"] = "file not found"
        return out
    out["sha256"] = sha256_file(path)
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    counts = collections.Counter(r.get("language_code") for r in records)
    short = {l: counts.get(l, 0) for l in shim.LANGUAGES if counts.get(l, 0) < n_needed}
    out.update({"n_records": len(records),
                "per_language": {l: counts.get(l, 0) for l in shim.LANGUAGES},
                "short_languages": short})
    if short:
        out["error"] = f"languages below {n_needed} records: {short}"
        return out
    out["ok"] = True
    return out


def gate_xsir_mapping():
    path = os.path.join(shim.REPO_ROOT, shim.XSIR_MAPPING)
    out = {"path": path, "ok": False}
    if not os.path.exists(path):
        out["error"] = ("XSIR mapping absent. XSIR would silently GENERATE a new "
                        "one, producing a different clustering than the 7B cells.")
        return out
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    out["len"] = len(mapping)
    out["n_clusters"] = len(set(mapping))
    if len(mapping) != shim.XSIR_MAPPING_LEN:
        out["error"] = f"mapping has {len(mapping)} entries, expected {shim.XSIR_MAPPING_LEN}"
        return out
    out["ok"] = True
    return out


def gate_tokenizer():
    out = {"ok": False}
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as e:
        out["error"] = f"transformers unavailable: {e}"
        return out

    sizes, vocab = {}, {}
    for variant, model_id in shim.MODEL_IDS.items():
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            out["error"] = f"could not load {model_id}: {e}"
            return out
        sizes[variant] = len(tok)
        vocab[variant] = getattr(cfg, "vocab_size", None)

    out.update({"tokenizer_len": sizes, "config_vocab_size": vocab,
                "expected_tokenizer_len": shim.XSIR_MAPPING_LEN})
    bad = {v: n for v, n in sizes.items() if n != shim.XSIR_MAPPING_LEN}
    if bad:
        out["error"] = (f"72B tokenizer length {bad} != 7B XSIR mapping length "
                        f"{shim.XSIR_MAPPING_LEN}. Reusing the 7B mapping is NOT "
                        f"valid; regenerating would change the clustering and "
                        f"break 'identical hyperparameters/keys'. Stopping.")
        return out
    out["ok"] = True
    return out


def soft_checks():
    out = {}
    for name, path in shim.PAPER.items():
        out[name] = {"path": path, "present": os.path.exists(path)}
    return out


def run(args):
    report = {"gates": {}, "soft": {}}
    report["gates"]["flores"] = gate_flores(args.flores)
    report["gates"]["aya"] = gate_aya(args.aya)
    report["gates"]["xsir_mapping"] = gate_xsir_mapping()
    if args.check_tokenizer:
        report["gates"]["tokenizer"] = gate_tokenizer()
    report["soft"] = soft_checks()

    print("Pre-flight gates")
    print("=" * 72)
    failed = []
    for name, g in report["gates"].items():
        status = "PASS" if g.get("ok") else "FAIL"
        print(f"  [{status}] {name}")
        if g.get("sha256"):
            print(f"           sha256 {g['sha256'][:16]}…")
        if not g.get("ok"):
            print(f"           {g.get('error', 'unknown failure')}")
            failed.append(name)

    print("\nSoft checks (absence is reported, never worked around)")
    for name, s in report["soft"].items():
        print(f"  [{'ok ' if s['present'] else 'MISSING'}] {name}: {s['path']}")
    missing_soft = [n for n, s in report["soft"].items() if not s["present"]]
    if missing_soft:
        print(f"\n  NOTE: {len(missing_soft)} stored 7B/baseline input(s) missing "
              f"({', '.join(missing_soft)}).")
        print("        - bertscore_baselines missing blocks the BERTScore quality stage.")
        print("        - tpr missing => ge_null.py stops: its anchor gate needs the 7B rows.")
        print("        - detection_summary/quality_summary missing => the optional published-summary cross-check is skipped.")

    if not args.check_tokenizer:
        print("\n  NOTE: tokenizer gate G3b not run. Pass --check_tokenizer before")
        print("        generation — XSIR mapping reuse depends on it.")

    if failed:
        print(f"\n[preflight] FAIL — {', '.join(failed)}. Fix before generating.",
              file=sys.stderr)
        return 1
    print("\n[preflight] OK")
    return 0


def selftest():
    import tempfile
    print("Self-test: synthetic fixtures, no real data touched.")
    tmp = tempfile.mkdtemp(prefix="preflight_selftest_")

    good = os.path.join(tmp, "flores_good.json")
    recs = []
    for rid in range(600):
        for tag in REQUIRED_FLORES_TAGS:
            iso, script = tag.split("_")
            recs.append({"id": rid, "iso_639_3": iso, "iso_15924": script,
                         "split": "devtest", "text": "x"})
    json.dump(recs, open(good, "w", encoding="utf-8"))
    g = gate_flores(good)
    assert g["ok"] and g["n_ids_with_all_tags"] == 600, g
    print(f"  [ok] complete FLORES passes ({g['n_ids_with_all_tags']} full ids)")

    bad = os.path.join(tmp, "flores_bad.json")
    json.dump([r for r in recs if r["iso_639_3"] != "cmn"], open(bad, "w", encoding="utf-8"))
    g = gate_flores(bad)
    assert not g["ok"] and g["missing_tags"] == ["cmn_Hans"], g
    print("  [ok] FLORES missing cmn_Hans fails with the tag named")

    few = os.path.join(tmp, "flores_few.json")
    json.dump([r for r in recs if r["id"] < 100], open(few, "w", encoding="utf-8"))
    g = gate_flores(few)
    assert not g["ok"] and "only 100 ids" in g["error"], g
    print("  [ok] FLORES with <500 complete ids fails")

    aya_ok = os.path.join(tmp, "aya_ok.json")
    json.dump([{"language_code": l, "inputs": "x"}
               for l in shim.LANGUAGES for _ in range(500)],
              open(aya_ok, "w", encoding="utf-8"))
    assert gate_aya(aya_ok)["ok"]
    aya_bad = os.path.join(tmp, "aya_bad.json")
    json.dump([{"language_code": l, "inputs": "x"}
               for l in shim.LANGUAGES for _ in range(499)],
              open(aya_bad, "w", encoding="utf-8"))
    g = gate_aya(aya_bad)
    assert not g["ok"] and len(g["short_languages"]) == 11, g
    print("  [ok] AYA gate fires at 499 records/language, passes at 500")

    assert len(gate_flores(good)["sha256"]) == 64
    print("  [ok] sha256 recorded for the prompt-pairing audit trail")

    print("\nSelf-test PASSED.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--flores", default=FLORES_PATH)
    p.add_argument("--aya", default=AYA_PATH)
    p.add_argument("--check_tokenizer", action="store_true",
                   help="also run G3b (loads the 72B tokenizer + config)")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    return selftest() if args.selftest else run(args)


if __name__ == "__main__":
    sys.exit(main())
