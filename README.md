# Cross-Lingual-Watermarking-Fairness-Audit

A fairness/bias evaluation framework for LLM text-watermarking schemes across languages. Submitted to the NeurIPS Datasets and Benchmarks track.

> **Research question.** Do watermarking algorithms maintain consistent *detectability* and *text quality* across languages — particularly low-resource and non-Latin-script ones?

The pipeline produces, per `(language × watermark × generator × regime)` cell of the experimental grid, a set of detection-fairness and quality-fairness measurements that are pooled into headline disparate-impact, minimax, and Generalized-Entropy fairness aggregates.

## Table of contents

- [Experimental grid](#experimental-grid)
- [Repo layout](#repo-layout)
- [Setup](#setup)
- [Pipeline](#pipeline)
- [Common operational tips](#common-operational-tips)
- [Output schemas](#output-schemas)
- [Headline findings](#headline-findings)
- [Known limitations](#known-limitations)
- [Citation](#citation)

## Experimental grid

| Axis | Values |
| --- | --- |
| Languages (11) | `eng`, `fra`, `spa`, `por`, `nld`, `zho`, `jpn`, `arb`, `hin`, `tur`, `vie` |
| Watermarks (6) | `KGW`, `Unigram`, `SynthID`, `EXPEdit`, `DIP`, `XSIR` |
| Generators (3) | Mistral-NeMo-12B, Gemma-3-4B, Qwen2.5-7B (each in `base` and `instruct`) |
| Regimes (2) | `(base, FLORES)` translated continuations; `(instruct, AYA)` native-speaker prompts |
| Samples per cell | 500 |

A *cell* is `(language, watermark, model, regime)`. The matched unwatermarked baseline is shared across watermarks within `(language, model, regime)`.

Generation hyperparameters are identical across all models, methods, and languages: `temperature=0.7`, `max_new_tokens=200`, `min_new_tokens=100`, `do_sample=True` (see `loader.py:GEN_CONFIG`).

## Repo layout

```text
generate.py                  end-to-end generation CLI
detect.py                    detection CLI (single-file or whole-cell)
loader.py                    model + watermark loaders, GEN_CONFIG, MODEL_REGISTRY
sampler.py                   FLORES / AYA prompt samplers; LANGUAGES list

watermark_configs/           per-scheme JSON; XSIR has per-model variants
MarkLLM/                     vendored upstream watermarking toolkit

data/
  pull_dataset.py                       HuggingFace → JSON for AYA + FLORES+
  aya_dataset.json                      raw AYA
  flores_plus.json                      raw FLORES+
  aya_dataset_preprocessed_main.json    11-language filter
  flores_preprocessed_main.json         11-language filter

preprocess_data/
  filter_aya.py                 filters raw AYA → 11 languages
  filter_flores.py              filters raw FLORES+ → 11 languages

preprocess/                  language identification (LID) + generation stats
  run_lid.py                 LID inference (GlotLID + fastText LID-218)
  categorize_lid.py          LID signals → categorical lid_label
  adherence_length_stats.py  per-cell language adherence + length stats
  lid_subsets.py             load_subset() helper used by downstream scripts

detection_fairness/            detection fairness analyses
  build_score_table.py         cache builder: detection × lid_label join
  per_language_tpr.py          per-language TPR @ empirical FPR (per-language + global τ)
  per_language_auc.py          per-language AUC (robustness companion to the TPR)
  detection_disparity.py       disparate impact + minimax + GE family decomposition

quality_fairness/              quality fairness — three two-stage paradigms
  mauve_embed.py               MAUVE stage 1 (GPU): xlm-r embeddings
  mauve_score.py               MAUVE stage 2 (CPU)
  bertscore_baselines.py       per-lang baselines on FLORES+ pairs
  bertscore_score.py           BERTScore per cell (GPU)
  ppl_nll.py                   PPL stage 1 (GPU): per-generation NLL
  ppl_ratio.py                 PPL stage 2 (CPU)
  quality_disparity.py         per-language preservation, disparate impact, minimax,
                               GE family decomposition, cross-metric rank agreement

joint_inference/               detection–quality trade-offs, covariates, statistical tests
  pareto_fronts.py             detection–quality Pareto fronts
  language_covariates.py       per-language covariate table
  covariate_regressions.py     single-covariate regressions (n=11/group)
  bootstrap_tests.py           bootstrap CIs, paired TPR tests, length-stratified TPR
  critical_difference.py       Friedman + Nemenyi critical-difference tests

family_sensitivity/            sensitivity of the GE₂ between-family share
  percell_family_test.py       per-cell exact shape-matched null, restricted
                               subset, leave-one-language-out (subset=all)
  ontarget_family_test.py      same analysis on the on-target (LID-filtered) subset
  joint_family_test.py         grid-level joint exact test (one relabeling per stratum)

scale_pilot/                   Qwen2.5-72B scale pilot (mirrors the Qwen2.5-7B cells)
  shim.py                      config, qwen72b registry entry, in-process stage runners
  preflight.py                 data-file, XSIR-mapping and tokenizer-parity gates
  stages.py                    pilot, generate, verify, lid, detect,
                               aggregate-detection, quality
  analysis_extras.py           per-cell / per-language CSVs, bootstrap CI, Δ(τ_g − τ_l)
  ge_null.py                   per-cell + joint exact nulls for the 72B cells
  runlog.py                    console stage log + exclusions.csv

requirements.txt             pinned Python deps
```

Pipeline outputs are written to a parallel set of directories that are **not** checked in (each is large; everything is reproducible from the scripts):

```text
outputs/                     generation JSONLs, one per cell
detection/                   detection JSONLs
lid/, lid_categorized/       LID outputs (raw and categorized)
quality/embeddings/          xlm-roberta-large embeddings (fp16, mean-pooled)
quality/ppl/{ref_tag}/       per-generation NLL caches

results/aggregate/           adherence.csv, length_stats.csv
results/detection/           detection_scores.csv, thresholds.csv,
                             per_language_tpr.csv, per_language_auc.csv,
                             detection_disparity.csv
results/quality/             mauve.csv, bertscore.csv, bertscore_baselines.csv,
                             ppl_ratio.csv, quality_per_language.csv,
                             quality_disparity.csv, quality_robustness.csv
results/joint/               language_covariates.csv, covariate_regressions.csv,
                             pareto_points.csv, pareto_optimal.csv,
                             pareto_dominance.csv, pareto_spread_metrics.csv,
                             bootstrap_tpr_ci.csv, bootstrap_ppl_ci.csv,
                             paired_tpr_tests.csv, length_stratified_tpr.csv,
                             friedman_tests.csv, nemenyi_tests.csv
figures/joint/               Pareto-front and critical-difference figures

analysis/percell_family_test/             between-family sensitivity, subset=all
analysis/ontarget_family_test/            between-family sensitivity, on-target subset
analysis/joint_family_test/               grid-level joint exact test

outputs_qwen72b/, lid_qwen72b/,           72B pilot generation / LID / detection /
lid_categorized_qwen72b/,                 quality trees (same layout and filename
detection_qwen72b/, quality_qwen72b/      grammar as the 7B trees above)
results/qwen72b/                          72B pilot result CSVs, exclusions.csv
analysis/qwen72b_percell_family_test/,    72B per-cell and joint exact nulls
analysis/qwen72b_ontarget_family_test/,
analysis/qwen72b_joint_family_test/
```

## Setup

### Environment

A CUDA GPU is required for generation, detection of model-conditioned schemes, and the GPU stages of quality fairness (`mauve_embed.py`, `bertscore_baselines.py`, `bertscore_score.py`, `ppl_nll.py`). LID categorization, detection fairness, the CPU quality stages, joint inference and the family-sensitivity tests run on CPU.

```bash
# Python 3.10+; CUDA 12.4 build of torch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# spaCy English model (used by some preprocessing utilities)
python -m spacy download en_core_web_md   # already pinned in requirements

# HuggingFace login (required for gated Mistral / Gemma weights)
huggingface-cli login
```

The project pins `torch==2.6.0+cu124` and `transformers==5.6.2`; older Torch builds will not match Gemma-3's bf16 path. The MarkLLM watermarking toolkit is vendored under `MarkLLM/` and loaded via the project-root `loader.py` — no separate install step.

### Models

Pulled lazily by `transformers` on first use:

| Alias | Base | Instruct | dtype |
| --- | --- | --- | --- |
| `mistral` | `mistralai/Mistral-Nemo-Base-2407` | `mistralai/Mistral-Nemo-Instruct-2407` | fp16 |
| `gemma` | `google/gemma-3-4b-pt` | `google/gemma-3-4b-it` | bf16 |
| `qwen` | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-7B-Instruct` | fp16 |

> **Note.** Gemma-3 must run in bf16 — fp16 overflows on its 262K-token logit distributions and silently produces NaN probabilities (see comment in `loader.py:MODEL_REGISTRY`).

### Data

```bash
# 1. Pull the raw datasets.
python data/pull_dataset.py --dataset CohereLabs/aya_dataset       --output data/aya_dataset.json
python data/pull_dataset.py --dataset openlanguagedata/flores_plus --output data/flores_plus.json

# 2. Filter to the 11 target languages.
python preprocess_data/filter_aya.py
python preprocess_data/filter_flores.py
# → data/aya_dataset_preprocessed_main.json
# → data/flores_preprocessed_main.json
```

Both filter scripts have hardcoded paths and the 11-language list at the top — edit those constants if you want a different language set.

## Pipeline

Each stage reads from the CSV / JSONL caches produced by earlier stages, so every stage can be re-run independently. Every analysis writes CSV files only; progress and validation results are printed to the console.

### Generation and detection

Generation runs one `(model, variant, dataset, watermark)` quadruple at a time and emits 11 watermarked + 11 unwatermarked JSONLs (one per language). Resumable at `prompt_id` granularity; nowatermark output is cached and shared across all watermarks within the same `(model, variant, dataset, lang)`.

```bash
# Smoke test: 5 prompts/language to verify the path works.
python generate.py --model gemma --variant base --dataset flores \
                   --watermark KGW --num_samples 5

# Full run (all 6 watermarks × 3 models × 2 regimes = 36 invocations).
python generate.py --model mistral --variant base     --dataset flores --watermark KGW --num_samples 500
python generate.py --model mistral --variant instruct --dataset aya    --watermark KGW --num_samples 500
# ... repeat for {Unigram, SynthID, EXPEdit, DIP, XSIR} × {gemma, qwen} × {base, instruct}
```

Valid `(variant, dataset)` combos: `(base, flores)` and `(instruct, aya)`.

Detection auto-discovers the 11 language files in `--input_dir` and runs in a single Python process so the tokenizer and detector load once. Pass `--device cuda` whenever the watermark is model-conditioned (XSIR embedding model in particular); KGW, Unigram, and DIP work on CPU but must match the device used at generation time so the green list partitions identically.

```bash
python detect.py --model gemma --variant base --watermark KGW --device cuda
# or single-file:
python detect.py outputs/flores_gemma_base_KGW_eng.jsonl --device cuda
```

### Preprocessing — LID, adherence, length stats

```bash
python preprocess/run_lid.py                  # CPU is fine
python preprocess/categorize_lid.py           # writes lid_categorized/
python preprocess/adherence_length_stats.py   # writes results/aggregate/{adherence,length_stats}.csv
```

`run_lid.py` runs GlotLID + fastText LID-218 on every generation. `categorize_lid.py` turns the raw signals into a categorical `lid_label ∈ {on_target, off_target, code_mixed, uncertain}` using thresholds you can re-sweep cheaply. `adherence_length_stats.py` produces per-cell language adherence and per-cell length stats in one pass.

### Detection fairness (`detection_fairness/`)

CPU-only. Each step depends only on prior outputs in `results/detection/`.

```bash
python detection_fairness/build_score_table.py     # cache: detection × lid_label join
python detection_fairness/per_language_tpr.py      # TPR@FPR=0.01 (and 0.05) per language
python detection_fairness/per_language_auc.py      # per-language AUC (robustness companion)
python detection_fairness/detection_disparity.py   # disparate impact + minimax + GE family decomp.
```

Conventions:

- **Empirical thresholds.** Calibrated from the matched NWM distribution per cell (`np.quantile(method='lower')` so empirical FPR ≤ α). MarkLLM's hardcoded `is_watermarked` boolean is ignored to keep schemes comparable across models and languages.
- **Set-vs-set subset filtering.** WM and NWM scores are filtered independently by their own LID labels.
- **Insufficient-data floors.** Per-language threshold skipped at α=0.01 if `n_neg < 200`; TPR flagged insufficient if `n_pos < 100`.

### Quality fairness (`quality_fairness/`)

Three quality paradigms, each two-stage. Stage 1 is a heavy GPU pass that is cached so stage 2 (and any subset / sample-size sweep) is cheap.

```bash
# (a) MAUVE — distributional similarity
python quality_fairness/mauve_embed.py   # GPU
python quality_fairness/mauve_score.py   # CPU

# (b) BERTScore — paired surface-form quality
python quality_fairness/bertscore_baselines.py   # GPU
python quality_fairness/bertscore_score.py       # GPU

# (c) Reference-model PPL — token-level likelihood under a held-out LM
python quality_fairness/ppl_nll.py --reference_model xglm-7.5b   # GPU
python quality_fairness/ppl_nll.py --reference_model mgpt-1.3b   # GPU
python quality_fairness/ppl_ratio.py --reference_model xglm-7.5b
python quality_fairness/ppl_ratio.py --reference_model mgpt-1.3b

# (d) Per-language preservation + disparity summary + cross-metric agreement
python quality_fairness/quality_disparity.py
```

Notes:

- MAUVE uses `xlm-roberta-large` with mean-pooling over non-pad tokens (more stable than `[CLS]` on paragraph-length text). N=500 per side is below MAUVE's recommended 5000+; cross-cell *rankings* are stable but absolute values are noisier.
- BERTScore baselines are computed locally on FLORES+ dev+devtest pairs rather than read from `bert-score`'s shipped TSVs because the shipped tables don't cover all 11 of our languages on `xlm-roberta-large` layer 17.
- PPL preservation transform is `exp(-|log(PPL_wm / PPL_nwm)|) ∈ (0, 1]` so the four-fifths (disparate-impact) ratios are always meaningful.
- Pairing semantics: MAUVE and PPL are set-vs-set; BERTScore is paired (both sides must be in the LID subset).
- Insufficient-samples floor on every measurement: `min(|T+|, |T-|) < 100` skipped.

### Joint analysis, mechanism, inference (`joint_inference/`)

Reads only from `results/detection/*.csv` and `results/quality/*.csv` (plus `quality/ppl/` for the PPL bootstrap CIs). CPU only.

```bash
# Detection–quality Pareto fronts
python joint_inference/pareto_fronts.py --regime base --subset all

# Per-language covariate table + single-covariate regressions
python joint_inference/language_covariates.py
python joint_inference/covariate_regressions.py --regime base --subset all

# Bootstrap CIs + paired ΔTPR tests + length-stratified TPR
python joint_inference/bootstrap_tests.py --tasks tpr_ci ppl_ci paired_tpr length_stratified

# Friedman χ² + Nemenyi critical-difference tests
python joint_inference/critical_difference.py
```

Headline settings: `regime=base`, `subset=all`, `α=0.01`, `threshold_type=global`. Instruct goes in the appendix.

### Between-family sensitivity (`family_sensitivity/`)

Tests whether the GE₂ between-family share reported by `detection_disparity.py` / `quality_disparity.py` is driven by the shape of the family partition (six of the eight families are singletons) rather than by typology. Reads `results/detection/per_language_tpr.csv` and `results/quality/quality_per_language.csv` (plus `detection_disparity.csv` / `quality_disparity.csv` as an optional cross-check; flags `--tpr_csv`, `--quality_csv`, `--detection_summary_csv`, `--quality_summary_csv`). CPU + stdlib only; seconds per run.

```bash
# Per-cell analyses on subset=all: restricted {Germanic, Romance} subset, exact
# shape-matched permutation null, leave-one-language-out, no-singleton partitions
python family_sensitivity/percell_family_test.py

# Same analyses on the on-target (LID-filtered) subset, plus a like-for-like
# recomputation of both subsets on their common language set
python family_sensitivity/ontarget_family_test.py

# Grid-level joint test: one relabeling applied to every cell of a stratum.
# Needs both per-cell runs above.
python family_sensitivity/joint_family_test.py

# Self-tests on synthetic fixtures (no data needed)
python family_sensitivity/percell_family_test.py --selftest
python family_sensitivity/ontarget_family_test.py --selftest
python family_sensitivity/joint_family_test.py --selftest
```

Notes:

- **Same math as the published tables.** `ge_decompose` and `LANGUAGE_FAMILIES` are imported from `quality_fairness/quality_disparity.py`; detection vectors are loaded through `detection_fairness/detection_disparity.py:load_per_language_tpr`, so the `insufficient_samples` filter matches the detection disparity summary.
- **Anchor gate.** Each run first reproduces published anchors from the stored per-language vectors and exits 1 on any mismatch (`ontarget_family_test.py` checks the on-target cell mean TPRs instead). `--skip_anchor_validation` is for diagnostics only.
- **Exact, deterministic nulls.** All C(11,3)·C(8,2) = 4,620 shape-preserving relabelings are enumerated, with no sampling and no RNG; p = #{share ≥ observed} / 4,620 with the identity counted. Cells that lose languages to sample-size floors re-derive the shape from the languages present, so the enumeration stays exhaustive.
- **Floor.** Cells with GE₂ < 10⁻³ are computed but flagged `above_floor=False` and excluded from headline counts.
- **Joint test.** Statistic = unweighted mean between-family share over a stratum's cells; every stratum's null has mean exactly 0.7 by construction. Holm adjustment across the four primary strata (detection / MAUVE × base / instruct).
- **Outputs (CSV only).** `percell_results.csv`, `nosingleton_partitions.csv` and `null_distributions/*.csv`; the on-target run adds `likeforlike_common_langs.csv`; the joint test writes `joint_results.csv` and `null_distributions/*.csv`.

### Qwen2.5-72B scale pilot (`scale_pilot/`)

Re-runs the Qwen2.5-7B cells for `KGW`, `SynthID` and `XSIR` at Qwen2.5-72B, with model size as the only intended change: the same prompts, per-prompt seeds, scheme keys, `GEN_CONFIG` and float16 dtype, and no batching. The `qwen72b` alias is registered at import time by `shim.py`, which drives the published pipeline scripts in-process, so no pipeline file is modified. Every stage writes to its own `*_qwen72b` tree or `results/qwen72b/`. Both regimes are supported: `base` (FLORES+) and `instruct` (AYA).

Generation needs 4×80 GB GPUs on one node (`device_map="auto"`). Detection and quality need one GPU. LID, detection aggregation and step 4 run on CPU.

```bash
# 0. Gates: data files, XSIR mapping reuse, 72B tokenizer parity with 7B
python scale_pilot/preflight.py --check_tokenizer

# 1. Throughput probe (eng/base, all three schemes). Exits 2 if the projected
#    wall-clock exceeds BUDGET_HOURS (env, default 72; N_GPUS default 4).
#    --force_budget_ok proceeds anyway.
python scale_pilot/stages.py pilot

# 2. Generation, in the order KGW -> SynthID -> XSIR. KGW must finish first
#    because it writes the shared unwatermarked set.
python scale_pilot/stages.py generate
python scale_pilot/stages.py verify          # 500 unique prompt_ids per file (generate also runs it)

# 3. LID -> detection -> detection aggregation -> quality
python scale_pilot/stages.py lid
python scale_pilot/stages.py detect
python scale_pilot/stages.py aggregate-detection   # build_score_table -> per_language_tpr -> per_language_auc -> detection_disparity
python scale_pilot/stages.py quality               # MAUVE, BERTScore, PPL (XGLM-7.5B), quality disparity

# 4. Per-cell / per-language CSVs and exact nulls (CPU)
python scale_pilot/analysis_extras.py
python scale_pilot/ge_null.py

# Self-tests on synthetic fixtures (no data, no GPU)
python scale_pilot/preflight.py --selftest
python scale_pilot/runlog.py
python scale_pilot/stages.py --selftest
python scale_pilot/analysis_extras.py --selftest
python scale_pilot/ge_null.py --selftest
```

Notes:

- **Scoping.** `generate`, `verify` and `detect` take `--variants`, `--schemes` and `--languages`, for example `generate --variants base --schemes KGW` to split generation into separate jobs (`pilot` takes `--variant` / `--language`). Every stage is resumable.
- **XSIR mapping.** The 7B mapping `300_mapping_qwen2.5-7b.json` is reused, because the mapping is built from the tokenizer, not the weights. `preflight.py --check_tokenizer` asserts that the 72B tokenizer has the same length (151,665) and stops otherwise.
- **Stored 7B inputs.** `quality` reuses `results/quality/bertscore_baselines.csv`. `ge_null.py` concatenates the stored 7B `per_language_tpr.csv` / `quality_per_language.csv` with the 72B rows, so the `family_sensitivity/` anchor gate still validates against published numbers. It then restricts the joint test to the 72B cells.
- **Statistics added on top of the pipeline outputs** (`analysis_extras.py`): a 95% percentile bootstrap CI of mean TPR over the 11-language vector (B=1000, seed 42), and Δ = meanTPR(τ_global) − meanTPR(τ_per-language) over the languages usable under both threshold types. Floor-excluded rows are never imputed; each one is listed in `exclusions.csv` with the pipeline's own reason string.
- **Outputs** (`results/qwen72b/`): the pipeline CSVs (`detection_scores.csv`, `thresholds.csv`, `per_language_tpr.csv`, `per_language_auc.csv`, `detection_disparity.csv`, `mauve.csv`, `bertscore.csv`, `ppl_ratio.csv`, `quality_per_language.csv`, `quality_disparity.csv`), `perlanguage_{detection,quality}.csv`, `percell_{detection,quality,ge2}.csv`, `null_test_percell_{all,on_target_only}.csv`, `null_test_joint.csv`, `exclusions.csv`. Pre-flight gate results and stage timings are printed to the console.

## Common operational tips

- **Smoke tests.** Every preprocessing and quality script accepts `--smoke_test N` (process N records per file) and most accept `--pattern <glob>` or per-axis filters (`--models`, `--watermarks`, `--languages`, …). Use these to validate a single cell before submitting a full-grid run.
- **Resumability.** Generation, detection, LID, encoding, and PPL are resumable at `prompt_id` granularity — re-running on top of a partial output skips finished records.
- **Filename grammar.** `{aya|flores}_{mistral|gemma|qwen}_{base|instruct}_{KGW|Unigram|SynthID|EXPEdit|DIP|XSIR|nowatermark}_{lang}.jsonl` — preserved across `outputs/`, `detection/`, and `lid_categorized/`. Detection adds a `_detect.jsonl` suffix; LID adds `_lid.jsonl` / `_lid_categorized.jsonl`.
- **Detection device.** Pass `--device cuda` for any model-conditioned scheme (XSIR especially). For hash-based schemes (KGW, Unigram, DIP) the device must match generation; CPU detection is supported but slow.
- **EXPEdit fast-path.** `detect.py` skips MarkLLM's per-text 100-sample permutation null and stores the raw `-test_result` (rank-equivalent to `-p_val` at fixed length). Empirical-FPR thresholds make the per-text null redundant; the fast path is much faster.

## Output schemas

### Generation record (`outputs/*.jsonl`)

```json
{
  "prompt_id": "12345",
  "language": "eng",
  "dataset": "flores",
  "model": "gemma",
  "variant": "base",
  "watermark_method": "KGW",
  "prompt_text": "...",
  "model_input": "...",
  "generated_text": "...",
  "full_output": "...",
  "generation_config": {"temperature": 0.7, "max_new_tokens": 200},
  "watermark_config": {"algorithm_name": "KGW", "gamma": 0.5},
  "token_count": 187,
  "generation_timestamp": "..."
}
```

### Detection record (`detection/*_detect.jsonl`)

```json
{
  "prompt_id": "12345",
  "score": 9.62,
  "is_watermarked_truth": true,
  "is_watermarked_pred": true,
  "token_count": 187,
  "detection_timestamp": "..."
}
```

### LID record (`lid/*_lid.jsonl` → `lid_categorized/*_lid_categorized.jsonl`)

Raw record stores GlotLID and fastText top-3 plus per-sentence labels; categorized record adds `lid_label` and the threshold config.

## Headline findings

> Subset = all, α = 0.01, threshold_type = global.

### Detection

- **Base regime is fair** for the four "strong" schemes (DIP, EXPEdit, KGW, SynthID) across all three generators — no language falls below a disparate impact of 0.8 (DI<.8 count = 0) on any cell. The two outliers are `base/qwen/XSIR` (mean TPR 0.215, DI<.8=10) and `base/gemma/Unigram` (mean TPR 0.000, DI<.8=11; saturation, see below).
- **Instruct regime collapses across schemes** — 17 of 18 instruct cells have mean TPR 0.04–0.69. The single survivor is **SynthID on Mistral and Qwen** (mean TPR 0.91 / 0.94, DI<.8=1).
- **Unigram static-green-list saturation.** On `gemma/base/Unigram`, Hindi and Vietnamese have TPR@FPR=0.01 = 0 *despite* clean WM/NWM separation in the body (AUC 0.97 / 0.91). Why: ≥5/500 NWM generations hit Unigram's theoretical max z = √T ≈ 14.142 — a tokenizer × language interaction that lets unwatermarked text systematically saturate the score function. KGW's per-position hashed green list rotates each token, so it doesn't suffer this. Real fairness finding, not a bug.
- **Between-family share of GE₂ = 77–100%** on disparity-ful detection cells. Same conclusion as the quality analysis (67–92%) — the disparity is *structural* (script, morphology, tokenizer interactions), not idiosyncratic.

### Quality

- **MAUVE rankings** (base, mean across models): DIP 0.95 > KGW 0.90 > XSIR 0.80 > Unigram 0.71 > SynthID 0.16 > EXPEdit 0.05. The "distortion-free" claims hold *only* for DIP at the distributional level; SynthID and EXPEdit visibly distort on base.
- **Cross-measurement rank inversion.** MAUVE and BERTScore agree on the top-half ranking; PPL inverts it (XSIR > Unigram > KGW > DIP). Both XGLM and mGPT agree on the inverted ranking. Mechanism: green-list-biased schemes shift each *token's* probability less under a held-out reference than DIP does, even though DIP's *aggregate* embedding distribution looks closer to NWM. Distributional similarity is not token-level likelihood.
- **Floor languages.** Hindi recurs as the floor across multiple schemes; Arabic emerges as a co-floor on the heavy-distorter schemes (SynthID, EXPEdit).

## Known limitations

- **n=500 per cell** is below MAUVE's recommended 5000+. We rely on the cross-cell *ranking* being stable rather than absolute MAUVE values being ground-truth.
- **Instruct-regime entropy collapse** depresses both watermark signal *and* distributional distortion. We report instruct as a separate axis; do **not** pool base and instruct cells in headline numbers.

## Citation

If you use this benchmark, please cite the underlying datasets:

- **AYA dataset** — `CohereLabs/aya_dataset` (HuggingFace).
- **FLORES+** — Open Language Data Initiative, `openlanguagedata/flores_plus` (HuggingFace).
- **MarkLLM** — vendored upstream toolkit under `MarkLLM/` (see `MarkLLM/LICENSE` and `MarkLLM/README.md`).
