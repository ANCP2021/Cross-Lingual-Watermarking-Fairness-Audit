import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

VALID_ANNOTATION_TYPES = {"original-annotations", "re-annotations"}
ANNOTATION_MODE_TO_TYPES = {
    "original-annotations": {"original-annotations"},
    "re-annotations": {"re-annotations"},
    "both": {"original-annotations", "re-annotations"},
}

INPUT_PATH = "data/aya_dataset.json"
OUTPUT_PATH = "data/aya_dataset_preprocessed_main.json"
LANGUAGE_CODES = ["eng", "fra", "spa", "por", "nld", "zho", "jpn", "arb", "hin", "tur", "vie"]
ANNOTATION_MODE = "original-annotations"

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def iter_records(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(data, dict):
        for value in data.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict):
                    yield item

def filter_records(
    data: Any, allowed_language_codes: Set[str], allowed_annotation_types: Set[str]
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for item in iter_records(data):
        language_code = item.get("language_code")
        annotation_type = item.get("annotation_type")
        if language_code not in allowed_language_codes:
            continue
        if annotation_type not in allowed_annotation_types:
            continue
        filtered.append(item)
    return filtered

def validate_annotation_types(records: Iterable[Dict[str, Any]]) -> None:
    seen_types = {record.get("annotation_type") for record in records}
    invalid = sorted(
        annotation_type
        for annotation_type in seen_types
        if annotation_type and annotation_type not in VALID_ANNOTATION_TYPES
    )
    if invalid:
        print(
            "Warning: found unexpected annotation_type values in input:",
            ", ".join(invalid),
        )

def save_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

def main() -> None:
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    data = load_json(input_path)
    all_records = list(iter_records(data))
    validate_annotation_types(all_records)

    if ANNOTATION_MODE not in ANNOTATION_MODE_TO_TYPES:
        raise ValueError(
            "ANNOTATION_MODE must be one of: "
            f"{', '.join(sorted(ANNOTATION_MODE_TO_TYPES.keys()))}"
        )
    allowed_language_codes = {code.strip() for code in LANGUAGE_CODES if code.strip()}
    if not allowed_language_codes:
        raise ValueError("LANGUAGE_CODES is empty. Add at least one language code.")
    allowed_annotation_types = ANNOTATION_MODE_TO_TYPES[ANNOTATION_MODE]

    filtered_records = filter_records(
        data=data,
        allowed_language_codes=allowed_language_codes,
        allowed_annotation_types=allowed_annotation_types,
    )
    save_json(filtered_records, output_path)

    print(f"Saved {len(filtered_records)} records to {output_path.resolve()}")
    print(f"Languages filter: {sorted(allowed_language_codes)}")
    print(f"Annotation types: {sorted(allowed_annotation_types)}")

if __name__ == "__main__":
    main()
