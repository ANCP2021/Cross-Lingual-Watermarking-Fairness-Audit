import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

INPUT_PATH = "data/flores_plus.json"
OUTPUT_PATH = "data/flores_preprocessed_main.json"
LANGUAGE_TAGS = [
    "eng_Latn", "fra_Latn", "spa_Latn", "por_Latn", 
    "nld_Latn", "cmn_Hans", "jpn_Jpan", "arb_Arab",
    "hin_Deva", "tur_Latn", "vie_Latn"
]

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

def get_language_tag(record: Dict[str, Any]) -> str:
    iso_639_3 = record.get("iso_639_3")
    iso_15924 = record.get("iso_15924")
    if not iso_639_3 or not iso_15924:
        return ""
    return f"{iso_639_3}_{iso_15924}"

def filter_records(data: Any, allowed_language_tags: Set[str]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for item in iter_records(data):
        language_tag = get_language_tag(item)
        if language_tag in allowed_language_tags:
            filtered.append(item)
    return filtered

def save_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

def main() -> None:
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    data = load_json(input_path)
    allowed_language_tags = {tag.strip() for tag in LANGUAGE_TAGS if tag.strip()}
    if not allowed_language_tags:
        raise ValueError("LANGUAGE_TAGS is empty. Add at least one language tag.")
    filtered_records = filter_records(data=data, allowed_language_tags=allowed_language_tags)
    save_json(filtered_records, output_path)

    print(f"Saved {len(filtered_records)} records to {output_path.resolve()}")
    print(f"Languages filter: {sorted(allowed_language_tags)}")

if __name__ == "__main__":
    main()
