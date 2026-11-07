#!/usr/bin/env python
"""Qwen2.5-72B scale pilot — GPU stages: generation, LID, detection, quality."""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog
import shim

BUDGET_HOURS = float(os.environ.get("BUDGET_HOURS", "72"))
N_GPUS = int(os.environ.get("N_GPUS", "4"))


def scan_jsonl(path):
    if not os.path.exists(path):
        return 0, 0
    ids, toks = set(), 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "prompt_id" in rec:
                ids.add(rec["prompt_id"])
            tc = rec.get("token_count")
            if isinstance(tc, int):
                toks += tc
    return len(ids), toks


def scan_tree(languages=None, variants=None, schemes=None):
    n_gen = n_tok = 0
    for variant in (variants or shim.VARIANTS):
        dataset = shim.VARIANT_DATASET[variant]
        for lang in (languages or shim.LANGUAGES):
            for wm in list(schemes or shim.SCHEMES) + ["nowatermark"]:
                path = os.path.join(
                    shim.OUTPUTS_DIR,
                    f"{dataset}_{shim.ALIAS}_{variant}_{wm}_{lang}.jsonl")
                g, t = scan_jsonl(path)
                n_gen += g
                n_tok += t
    return n_gen, n_tok


def expected_totals():
    n_wm = len(shim.SCHEMES) * len(shim.VARIANTS) * len(shim.LANGUAGES) * shim.NUM_SAMPLES
    n_nwm = len(shim.VARIANTS) * len(shim.LANGUAGES) * shim.NUM_SAMPLES
    return n_wm, n_nwm


def cmd_pilot(args):
    shim.ensure_dirs()
    lang, variant = args.language, args.variant

    before_gen, before_tok = scan_tree([lang], [variant])
    t0 = time.time()
    for scheme in shim.SCHEMES:
        with runlog.stage(f"pilot generate {variant}/{scheme}/{lang}",
                          command=(f"generate.py --model {shim.ALIAS} --variant {variant} "
                                   f"--dataset {shim.VARIANT_DATASET[variant]} "
                                   f"--watermark {scheme} --num_samples {shim.NUM_SAMPLES} "
                                   f"--seed {shim.SEED}  [languages={lang}]")):
            shim.run_generate(variant, scheme, languages=[lang],
                              num_samples=args.num_samples)
    elapsed = time.time() - t0
    after_gen, after_tok = scan_tree([lang], [variant])

    d_gen, d_tok = after_gen - before_gen, after_tok - before_tok
    if d_gen <= 0 or elapsed <= 0:
        print("Pilot produced no new records (already complete?). "
              "Delete the pilot language's files to re-measure, or pass --force_budget_ok.",
              file=sys.stderr)
        if not args.force_budget_ok:
            return 1
        return 0

    tok_s = d_tok / elapsed
    gen_s = d_gen / elapsed
    mean_tok = d_tok / d_gen

    n_wm, n_nwm = expected_totals()
    total_gen = n_wm + n_nwm
    remaining_gen = max(0, total_gen - after_gen)
    projected_h = (remaining_gen / gen_s) / 3600.0
    elapsed_h = elapsed / 3600.0

    print("\nThroughput pilot")
    print("=" * 72)
    print(f"  pilot cell            {variant}/{lang}, schemes {', '.join(shim.SCHEMES)}")
    print(f"  N_GPUS                {N_GPUS}")
    print(f"  generations produced  {d_gen}  ({d_tok} tokens, mean {mean_tok:.1f} tok/gen)")
    print(f"  wall-clock            {elapsed / 60:.1f} min")
    print(f"  throughput            {tok_s:.1f} tok/s   ({gen_s * 3600:.0f} generations/h)")
    print(f"  grid total            {total_gen} generations "
          f"({n_wm} watermarked + {n_nwm} shared unwatermarked)")
    print(f"  already on disk       {after_gen}")
    print(f"  PROJECTED REMAINING   {projected_h:.1f} h  "
          f"(+{elapsed_h:.1f} h already spent)")
    print(f"  BUDGET_HOURS          {BUDGET_HOURS}")

    runlog.metric("pilot_tokens_per_second", f"{tok_s:.1f}", "tok/s")
    runlog.metric("pilot_generations_per_hour", f"{gen_s * 3600:.0f}")
    runlog.metric("pilot_mean_tokens_per_generation", f"{mean_tok:.1f}")
    runlog.metric("projected_remaining_hours", f"{projected_h:.1f}", "h")
    runlog.metric("BUDGET_HOURS", BUDGET_HOURS, "h")
    runlog.metric("N_GPUS", N_GPUS)

    if projected_h > BUDGET_HOURS and not args.force_budget_ok:
        msg = (f"projected {projected_h:.1f} h > BUDGET_HOURS {BUDGET_HOURS} — "
               f"stopping before the full grid, per spec")
        print(f"\n[pilot] STOP — {msg}", file=sys.stderr)
        runlog.log(f"STOP {msg}")
        runlog.exclude("generate", "(full grid)", msg, "throughput gate")
        return 2
    print("\n[pilot] within budget — proceed to `generate`")
    runlog.log("pilot within budget; proceeding to full generation")
    return 0


def cmd_generate(args):
    shim.ensure_dirs()
    for variant in (args.variants or shim.VARIANTS):
        for scheme in (args.schemes or shim.SCHEMES):
            with runlog.stage(f"generate {variant}/{scheme}",
                              command=(f"generate.py --model {shim.ALIAS} --variant {variant} "
                                       f"--dataset {shim.VARIANT_DATASET[variant]} "
                                       f"--watermark {scheme} "
                                       f"--num_samples {shim.NUM_SAMPLES} --seed {shim.SEED}")):
                shim.run_generate(variant, scheme, languages=args.languages,
                                  num_samples=args.num_samples)
    return cmd_verify(args)


def cmd_verify(args):
    problems = []
    for variant in (args.variants or shim.VARIANTS):
        dataset = shim.VARIANT_DATASET[variant]
        for lang in (args.languages or shim.LANGUAGES):
            for wm in list(args.schemes or shim.SCHEMES) + ["nowatermark"]:
                path = os.path.join(
                    shim.OUTPUTS_DIR,
                    f"{dataset}_{shim.ALIAS}_{variant}_{wm}_{lang}.jsonl")
                n, _ = scan_jsonl(path)
                if n != shim.NUM_SAMPLES:
                    problems.append((f"{wm}/{variant}/{lang}", n, path))

    print(f"\nCount verification ({shim.NUM_SAMPLES} expected per file)")
    print("=" * 72)
    if not problems:
        print("  all files complete")
        runlog.log(f"verify: all generation files at {shim.NUM_SAMPLES} records")
        return 0
    for cell, n, path in problems:
        print(f"  {cell:<28} {n} records  ({os.path.basename(path)})")
        runlog.exclude("generate", cell, f"{n}/{shim.NUM_SAMPLES} records", "verify")
    print(f"\n[verify] {len(problems)} incomplete file(s) — logged to exclusions.csv.")
    print("         Re-run `generate` to resume (resumable at prompt_id granularity).")
    return 1


def cmd_lid(args):
    with runlog.stage("LID inference (GlotLID v3 + fastText LID-218)",
                      command=f"preprocess/run_lid.py --input_dir {shim.OUTPUTS_DIR} "
                              f"--lid_dir {shim.LID_DIR}"):
        shim.stage_lid()
    with runlog.stage("LID categorization (conf_floor 0.5, on_target_ratio 0.8)",
                      command=f"preprocess/categorize_lid.py --lid_dir {shim.LID_DIR} "
                              f"--out_dir {shim.LID_CAT_DIR} --conf_floor 0.5 "
                              f"--on_target_ratio 0.8"):
        shim.stage_categorize()
    with runlog.stage("adherence + length stats"):
        shim.stage_adherence_length_stats()
    return 0


def cmd_detect(args):
    for variant in (args.variants or shim.VARIANTS):
        for scheme in (args.schemes or shim.SCHEMES):
            with runlog.stage(f"detect {variant}/{scheme}",
                              command=(f"detect.run_cell(model={shim.ALIAS}, "
                                       f"variant={variant}, watermark={scheme}, "
                                       f"device={args.device})")):
                shim.run_detect(variant, scheme, device=args.device)
    return 0


def cmd_aggregate_detection(args):
    with runlog.stage("detection score table (build_score_table)"):
        shim.stage_build_score_table()
    with runlog.stage("per-language TPR (per_language_tpr)"):
        shim.stage_per_language_tpr()
    with runlog.stage("per-language AUC (per_language_auc)"):
        shim.stage_per_language_auc()
    with runlog.stage("detection disparity summary (detection_disparity)"):
        shim.stage_detection_disparity()
    return 0


def cmd_quality(args):
    with runlog.stage("MAUVE stage 1 — xlm-roberta-large embeddings"):
        shim.stage_mauve_embed(device=args.device)
    with runlog.stage("MAUVE stage 2"):
        shim.stage_mauve_score()
    if os.path.exists(shim.PAPER["bertscore_baselines"]):
        with runlog.stage("BERTScore (stored FLORES baselines reused)"):
            shim.stage_bertscore_score(device=args.device)
    else:
        msg = f"baselines absent at {shim.PAPER['bertscore_baselines']}"
        print(f"[quality] SKIP BERTScore — {msg}", file=sys.stderr)
        runlog.exclude("quality/bertscore", "(all cells)", msg, "preflight soft check")
    with runlog.stage(f"PPL stage 1 — {args.reference_model}"):
        shim.stage_ppl_nll(args.reference_model, device=args.device)
    with runlog.stage(f"PPL stage 2 — {args.reference_model}"):
        shim.stage_ppl_ratio(args.reference_model)
    with runlog.stage("quality disparity summary (quality_disparity)"):
        shim.stage_quality_disparity()
    return 0


def selftest():
    import tempfile
    print("Self-test: synthetic fixtures, no model loaded.")
    tmp = tempfile.mkdtemp(prefix="stages_selftest_")
    saved_out = shim.OUTPUTS_DIR
    saved_exc = runlog.EXCLUSIONS_PATH
    shim.OUTPUTS_DIR = os.path.join(tmp, "outputs")
    runlog.EXCLUSIONS_PATH = os.path.join(tmp, "exclusions.csv")
    os.makedirs(shim.OUTPUTS_DIR, exist_ok=True)
    try:
        def write(variant, wm, lang, n, tok=200):
            ds = shim.VARIANT_DATASET[variant]
            p = os.path.join(shim.OUTPUTS_DIR,
                             f"{ds}_{shim.ALIAS}_{variant}_{wm}_{lang}.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                for i in range(n):
                    f.write(json.dumps({"prompt_id": str(i), "token_count": tok}) + "\n")
            return p

        p = write("base", "KGW", "eng", shim.NUM_SAMPLES)
        n, t = scan_jsonl(p)
        assert n == 500 and t == 100_000, (n, t)
        print(f"  [ok] scan_jsonl: {n} unique ids, {t} tokens")

        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt_id": "0", "token_count": 200}) + "\n")
        n2, _ = scan_jsonl(p)
        assert n2 == 500, n2
        print("  [ok] duplicate prompt_id does not inflate the count")

        write("base", "SynthID", "eng", 499)
        args = argparse.Namespace(variants=["base"], schemes=["KGW", "SynthID"],
                                  languages=["eng"])
        assert cmd_verify(args) == 1
        exc = open(runlog.EXCLUSIONS_PATH, encoding="utf-8").read()
        assert "SynthID/base/eng" in exc and "499/500" in exc, exc
        print("  [ok] verify fails on a short cell and logs it to exclusions.csv")
        assert runlog.EXCLUSIONS_PATH.startswith(tmp)

        n_wm, n_nwm = expected_totals()
        assert (n_wm, n_nwm) == (3 * 2 * 11 * 500, 2 * 11 * 500), (n_wm, n_nwm)
        assert n_wm / n_nwm == 3.0
        print(f"  [ok] grid totals: {n_wm} WM + {n_nwm} NWM = {n_wm + n_nwm}; "
              f"WM:NWM = 3:1 matches the pilot blend")

        g, tk = scan_tree(["eng"], ["base"], ["KGW", "SynthID"])
        assert g == 500 + 499 and tk == (500 + 499 + 1) * 200, (g, tk)
        print(f"  [ok] scan_tree aggregates: {g} generations, {tk} tokens")
    finally:
        shim.OUTPUTS_DIR = saved_out
        runlog.EXCLUSIONS_PATH = saved_exc
    print("\nSelf-test PASSED.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--selftest", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--variants", nargs="+", choices=shim.VARIANTS)
        sp.add_argument("--schemes", nargs="+", choices=shim.SCHEMES)
        sp.add_argument("--languages", nargs="+", choices=shim.LANGUAGES,
                        help="scope the stage to a subset (resume / parallel jobs)")
        sp.add_argument("--num_samples", type=int, default=shim.NUM_SAMPLES)
        return sp

    sp = common(sub.add_parser("pilot"))
    sp.add_argument("--language", default="eng", choices=shim.LANGUAGES)
    sp.add_argument("--variant", default="base", choices=shim.VARIANTS)
    sp.add_argument("--force_budget_ok", action="store_true",
                    help="proceed even if the projection exceeds BUDGET_HOURS")
    sp.set_defaults(func=cmd_pilot)

    common(sub.add_parser("generate")).set_defaults(func=cmd_generate)
    common(sub.add_parser("verify")).set_defaults(func=cmd_verify)
    common(sub.add_parser("lid")).set_defaults(func=cmd_lid)

    sp = common(sub.add_parser("detect"))
    sp.add_argument("--device", default="cuda")
    sp.set_defaults(func=cmd_detect)

    sub.add_parser("aggregate-detection").set_defaults(func=cmd_aggregate_detection)

    sp = common(sub.add_parser("quality"))
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--reference_model", default="xglm-7.5b")
    sp.set_defaults(func=cmd_quality)

    args = p.parse_args()
    if args.selftest:
        return selftest()
    if not getattr(args, "func", None):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
