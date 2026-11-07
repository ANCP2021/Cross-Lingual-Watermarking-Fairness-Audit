#!/usr/bin/env python
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


ALIAS = "qwen72b"
BASELINE_ALIAS = "qwen"

MODEL_IDS = {
    "base":     "Qwen/Qwen2.5-72B",
    "instruct": "Qwen/Qwen2.5-72B-Instruct",
}

SCHEMES   = ["KGW", "SynthID", "XSIR"]
VARIANTS  = ["base", "instruct"]
LANGUAGES = ["eng", "nld", "fra", "spa", "por", "tur", "vie", "hin", "arb", "zho", "jpn"]
VARIANT_DATASET = {"base": "flores", "instruct": "aya"}

NUM_SAMPLES = 500
SEED = 42
ALPHA_HEADLINE = 0.01
THRESHOLD_TYPE = "global"
SUBSETS = ["all", "on_target_only"]

XSIR_MAPPING = "MarkLLM/watermark/xsir/mapping/300_mapping_qwen2.5-7b.json"
XSIR_MAPPING_LEN = 151665


OUTPUTS_DIR    = os.path.join(REPO_ROOT, "outputs_qwen72b")
LID_DIR        = os.path.join(REPO_ROOT, "lid_qwen72b")
LID_CAT_DIR    = os.path.join(REPO_ROOT, "lid_categorized_qwen72b")
DETECTION_DIR  = os.path.join(REPO_ROOT, "detection_qwen72b")
EMBEDDINGS_DIR = os.path.join(REPO_ROOT, "quality_qwen72b", "embeddings")
PPL_DIR        = os.path.join(REPO_ROOT, "quality_qwen72b", "ppl")

RESULTS_DIR    = os.path.join(REPO_ROOT, "results", "qwen72b")
DERIVED_DIR    = os.path.join(RESULTS_DIR, "derived")
ANALYSIS_ALL   = os.path.join(REPO_ROOT, "analysis", "qwen72b_percell_family_test")
ANALYSIS_OT    = os.path.join(REPO_ROOT, "analysis", "qwen72b_ontarget_family_test")
ANALYSIS_JOINT = os.path.join(REPO_ROOT, "analysis", "qwen72b_joint_family_test")

PAPER = {
    "tpr":       os.path.join(REPO_ROOT, "results", "detection", "per_language_tpr.csv"),
    "per_language_auc":    os.path.join(REPO_ROOT, "results", "detection", "per_language_auc.csv"),
    "detection_summary": os.path.join(REPO_ROOT, "results", "detection", "detection_disparity.csv"),
    "quality":   os.path.join(REPO_ROOT, "results", "quality", "quality_per_language.csv"),
    "quality_summary": os.path.join(REPO_ROOT, "results", "quality", "quality_disparity.csv"),
    "mauve":     os.path.join(REPO_ROOT, "results", "quality", "mauve.csv"),
    "bertscore_baselines": os.path.join(REPO_ROOT, "results", "quality", "bertscore_baselines.csv"),
}

OUT = {
    "scores":    os.path.join(RESULTS_DIR, "detection_scores.csv"),
    "coverage":  os.path.join(RESULTS_DIR, "cells_loaded.csv"),
    "thresholds": os.path.join(RESULTS_DIR, "thresholds.csv"),
    "tpr":       os.path.join(RESULTS_DIR, "per_language_tpr.csv"),
    "per_language_auc":    os.path.join(RESULTS_DIR, "per_language_auc.csv"),
    "detection_summary": os.path.join(RESULTS_DIR, "detection_disparity.csv"),
    "mauve":     os.path.join(RESULTS_DIR, "mauve.csv"),
    "bertscore": os.path.join(RESULTS_DIR, "bertscore.csv"),
    "ppl":       os.path.join(RESULTS_DIR, "ppl_ratio.csv"),
    "quality":   os.path.join(RESULTS_DIR, "quality_per_language.csv"),
    "quality_summary": os.path.join(RESULTS_DIR, "quality_disparity.csv"),
}

def all_dirs():
    return [OUTPUTS_DIR, LID_DIR, LID_CAT_DIR, DETECTION_DIR, EMBEDDINGS_DIR,
            PPL_DIR, RESULTS_DIR, DERIVED_DIR]


def ensure_dirs():
    for d in all_dirs():
        os.makedirs(d, exist_ok=True)


_REGISTERED = False


def import_stage(package: str, module: str):
    import importlib

    return importlib.import_module(f"{package}.{module}")


def register(dtype_name: str = "float16"):
    global _REGISTERED
    if _REGISTERED:
        return
    import torch
    import loader

    loader.MODEL_REGISTRY[ALIAS] = {
        "base":         MODEL_IDS["base"],
        "instruct":     MODEL_IDS["instruct"],
        "xsir_config":  "watermark_configs/XSIR_qwen.json",
        "dtype":        {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name],
    }
    _REGISTERED = True


def run_module(module, argv, patch_models=False):
    saved = sys.argv
    sys.argv = [getattr(module, "__name__", "stage")] + [str(a) for a in argv]
    try:
        if patch_models and hasattr(module, "MODELS") and ALIAS not in module.MODELS:
            module.MODELS.append(ALIAS)
        return module.main()
    finally:
        sys.argv = saved


def run_generate(variant, watermark, languages=None, num_samples=NUM_SAMPLES):
    register()
    import generate

    ns = argparse.Namespace(
        model=ALIAS, variant=variant, dataset=VARIANT_DATASET[variant],
        watermark=watermark, num_samples=num_samples,
        output_dir=OUTPUTS_DIR, seed=SEED,
    )
    saved_parse, saved_langs = generate.parse_args, generate.LANGUAGES
    generate.parse_args = lambda: ns
    if languages is not None:
        generate.LANGUAGES = list(languages)
    try:
        generate.main()
    finally:
        generate.parse_args, generate.LANGUAGES = saved_parse, saved_langs


def run_detect(variant, watermark, languages=None, device="cuda"):
    register()
    import detect
    from loader import (load_tokenizer_only, build_transformers_config_for_detection,
                        load_watermark)

    dataset = VARIANT_DATASET[variant]
    tokenizer = load_tokenizer_only(ALIAS, variant)
    cfg = build_transformers_config_for_detection(tokenizer, ALIAS, variant, device=device)
    detector = load_watermark(watermark, cfg, ALIAS)

    for lang in (languages or LANGUAGES):
        detect.run_cell(dataset, ALIAS, variant, watermark, lang,
                        detector, OUTPUTS_DIR, DETECTION_DIR)


def stage_lid():
    run_lid = import_stage("preprocess", "run_lid")
    run_module(run_lid, ["--input_dir", OUTPUTS_DIR, "--lid_dir", LID_DIR])


def stage_categorize():
    categorize_lid = import_stage("preprocess", "categorize_lid")
    run_module(categorize_lid, [
        "--lid_dir", LID_DIR, "--out_dir", LID_CAT_DIR,
        "--results_dir", os.path.join(RESULTS_DIR, "preprocess"),
        "--conf_floor", "0.5", "--on_target_ratio", "0.8",
    ])


def stage_adherence_length_stats():
    adherence_length_stats = import_stage("preprocess", "adherence_length_stats")
    run_module(adherence_length_stats, ["--lid_categorized_dir", LID_CAT_DIR,
                           "--output_dir", os.path.join(RESULTS_DIR, "aggregate")])


def stage_build_score_table():
    build_score_table = import_stage("detection_fairness", "build_score_table")
    run_module(build_score_table, [
        "--detection_dir", DETECTION_DIR, "--lid_dir", LID_CAT_DIR,
        "--scores_csv", OUT["scores"], "--coverage_csv", OUT["coverage"],
        "--models", ALIAS, "--watermarks", *SCHEMES,
    ], patch_models=True)


def stage_per_language_tpr():
    per_language_tpr = import_stage("detection_fairness", "per_language_tpr")
    run_module(per_language_tpr, [
        "--scores_csv", OUT["scores"], "--thresholds_csv", OUT["thresholds"],
        "--tpr_csv", OUT["tpr"], "--models", ALIAS, "--watermarks", *SCHEMES,
        "--subsets", *SUBSETS,
    ], patch_models=True)


def stage_per_language_auc():
    per_language_auc = import_stage("detection_fairness", "per_language_auc")
    run_module(per_language_auc, [
        "--scores_csv", OUT["scores"], "--auc_csv", OUT["per_language_auc"],
        "--models", ALIAS, "--watermarks", *SCHEMES, "--subsets", *SUBSETS,
    ], patch_models=True)


def stage_detection_disparity():
    detection_disparity = import_stage("detection_fairness", "detection_disparity")
    run_module(detection_disparity, ["--tpr_csv", OUT["tpr"], "--summary_csv", OUT["detection_summary"]])


def stage_mauve_embed(device="cuda"):
    mauve_embed = import_stage("quality_fairness", "mauve_embed")
    run_module(mauve_embed, ["--input_dir", OUTPUTS_DIR,
                             "--output_dir", EMBEDDINGS_DIR, "--device", device])


def stage_mauve_score():
    mauve_score = import_stage("quality_fairness", "mauve_score")
    run_module(mauve_score, [
        "--embeddings_dir", EMBEDDINGS_DIR, "--lid_dir", LID_CAT_DIR,
        "--output_path", OUT["mauve"], "--models", ALIAS,
        "--watermarks", ",".join(SCHEMES), "--subsets", ",".join(SUBSETS),
    ])


def stage_bertscore_score(device="cuda"):
    bertscore_score = import_stage("quality_fairness", "bertscore_score")
    run_module(bertscore_score, [
        "--input_dir", OUTPUTS_DIR, "--lid_dir", LID_CAT_DIR,
        "--baselines_path", PAPER["bertscore_baselines"], "--output_path", OUT["bertscore"],
        "--models", ALIAS, "--watermarks", ",".join(SCHEMES),
        "--subsets", ",".join(SUBSETS), "--device", device,
    ])


def stage_ppl_nll(reference_model="xglm-7.5b", device="cuda"):
    ppl_nll = import_stage("quality_fairness", "ppl_nll")
    run_module(ppl_nll, ["--reference_model", reference_model,
                             "--input_dir", OUTPUTS_DIR, "--output_dir", PPL_DIR,
                             "--device", device])


def stage_ppl_ratio(reference_model="xglm-7.5b"):
    ppl_ratio = import_stage("quality_fairness", "ppl_ratio")
    run_module(ppl_ratio, [
        "--reference_model", reference_model, "--ppl_dir", PPL_DIR,
        "--lid_dir", LID_CAT_DIR, "--output_path", OUT["ppl"],
        "--models", ALIAS, "--watermarks", ",".join(SCHEMES),
        "--subsets", ",".join(SUBSETS),
    ])


def stage_quality_disparity():
    quality_disparity = import_stage("quality_fairness", "quality_disparity")
    run_module(quality_disparity, ["--mauve_csv", OUT["mauve"], "--bertscore_csv", OUT["bertscore"],
                       "--ppl_csv", OUT["ppl"], "--output_dir", RESULTS_DIR])


def cells():
    for variant in VARIANTS:
        for scheme in SCHEMES:
            yield scheme, variant
