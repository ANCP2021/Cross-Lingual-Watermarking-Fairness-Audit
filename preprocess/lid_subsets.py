import json
import os
from typing import Dict, Set

VARIANT_DATASET = {"base": "flores", "instruct": "aya"}

KNOWN_SUBSETS = {"all",
                 "on_target_only", "off_target_only",
                 "code_mixed_only", "uncertain_only"}

_LABEL_FOR_SUBSET = {
    "on_target_only":  "on_target",
    "off_target_only": "off_target",
    "code_mixed_only": "code_mixed",
    "uncertain_only":  "uncertain",
}


def categorized_path(model: str, variant: str, language: str, watermark: str,
                     lid_dir: str = "lid_categorized") -> str:
    if variant not in VARIANT_DATASET:
        raise ValueError(f"Unknown variant {variant!r}; expected base or instruct.")
    dataset = VARIANT_DATASET[variant]
    fname = f"{dataset}_{model}_{variant}_{watermark}_{language}_lid_categorized.jsonl"
    return os.path.join(lid_dir, fname)


def load_labels(model: str, variant: str, language: str, watermark: str,
                lid_dir: str = "lid_categorized") -> Dict[str, str]:
    path = categorized_path(model, variant, language, watermark, lid_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No categorized file for cell ({model}/{variant}/{language}/{watermark}): "
            f"{path}. Run preprocess/categorize_lid.py first."
        )
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid   = rec.get("prompt_id")
            label = rec.get("lid_label")
            if pid is None or label is None:
                continue
            out[str(pid)] = label
    return out


def load_subset(model: str, variant: str, language: str, watermark: str,
                subset: str, lid_dir: str = "lid_categorized") -> Set[str]:
    if subset not in KNOWN_SUBSETS:
        raise ValueError(
            f"Unknown subset {subset!r}; expected one of {sorted(KNOWN_SUBSETS)}."
        )
    labels = load_labels(model, variant, language, watermark, lid_dir)
    if subset == "all":
        return set(labels.keys())
    target_label = _LABEL_FOR_SUBSET[subset]
    return {pid for pid, lab in labels.items() if lab == target_label}
