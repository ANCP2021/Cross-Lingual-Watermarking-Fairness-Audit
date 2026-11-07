import argparse
import json
from pathlib import Path
from collections import Counter
from collections import defaultdict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print per-language counts from a dataset JSON (optionally with annotation_type breakdown)."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        default="aya_dataset.json",
        help="Path to the JSON dataset file to analyze.",
    )
    parser.add_argument(
        "--dataset",
        choices=["aya", "flores"],
        default="aya",
        help="Dataset schema to use for field mapping.",
    )
    parser.add_argument(
        "--annotation-type",
        default=None,
        help="Filter by annotation type (Aya only). Value is validated from data dynamically.",
    )
    parser.add_argument(
        "--language",
        dest="language",
        action="store_true",
        help="Print language counts.",
    )
    parser.add_argument(
        "--language-code",
        dest="language_code",
        action="store_true",
        help="Print language-code counts.",
    )
    parser.add_argument(
        "--language-name",
        dest="language_name",
        default=None,
        help="Return language code(s) associated with a specific language value.",
    )
    parser.add_argument(
        "--code-name",
        dest="code_name",
        default=None,
        help="Return language value(s) associated with a specific language code.",
    )
    parser.add_argument(
        "--show-annotations",
        action="store_true",
        help="Always print annotation_type breakdown (even if only one type exists).",
    )
    return parser.parse_args()

def load_data(input_path: Path):
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_dataset_fields(item: dict, dataset_name: str):
    if dataset_name == "aya":
        language = item.get("language")
        language_code = item.get("language_code")
        annotation_type = item.get("annotation_type")
        return language, language_code, annotation_type

    iso_639_3 = item.get("iso_639_3")
    iso_15924 = item.get("iso_15924")
    if iso_639_3 and iso_15924:
        language = f"{iso_639_3}_{iso_15924}"
    else:
        language = iso_639_3
    language_code = iso_639_3
    return language, language_code, None

def main() -> None:
    args = parse_args()
    data = load_data(Path(args.input_path))
    records = []

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                records.extend(value)
    elif isinstance(data, list):
        records = data

    if args.dataset != "aya" and args.annotation_type:
        raise ValueError("--annotation-type is only supported when --dataset aya is used.")

    available_annotation_types = {
        item.get("annotation_type")
        for item in records
        if isinstance(item, dict) and item.get("annotation_type")
    }
    if args.dataset == "aya" and args.annotation_type and args.annotation_type not in available_annotation_types:
        available_values = ", ".join(sorted(available_annotation_types)) or "(none found)"
        raise ValueError(
            f"Unknown --annotation-type '{args.annotation_type}'. Available values: {available_values}"
        )

    language_counter = Counter()
    language_code_counter = Counter()
    annotation_type_breakdown = defaultdict(Counter)
    annotation_type_breakdown_by_code = defaultdict(Counter)
    annotation_types_seen = set()
    language_to_codes = defaultdict(Counter)
    code_to_languages = defaultdict(Counter)
    for item in records:
        if not isinstance(item, dict):
            continue

        language, language_code, annotation_type = get_dataset_fields(item, args.dataset)
        if not language:
            continue

        if args.annotation_type and annotation_type != args.annotation_type:
            continue
        language_counter[language] += 1

        if language_code:
            language_code_counter[language_code] += 1
            language_to_codes[language][language_code] += 1
            code_to_languages[language_code][language] += 1

        if annotation_type:
            annotation_types_seen.add(annotation_type)
            annotation_type_breakdown[language][annotation_type] += 1
            if language_code:
                annotation_type_breakdown_by_code[language_code][annotation_type] += 1

    if args.language_name:
        codes_for_language = language_to_codes.get(args.language_name, Counter())
        if not codes_for_language:
            print(f"No language codes found for language '{args.language_name}'.")
            return
        print(f"Language codes for {args.language_name}:")
        for code, count in codes_for_language.most_common():
            print(f"{code}: {count}")
        return

    if args.code_name:
        languages_for_code = code_to_languages.get(args.code_name, Counter())
        if not languages_for_code:
            print(f"No languages found for language code '{args.code_name}'.")
            return
        print(f"Languages for {args.code_name}:")
        for language, count in languages_for_code.most_common():
            print(f"{language}: {count}")
        return

    show_annotation_breakdown = args.show_annotations
    annotation_labels = sorted(annotation_types_seen)

    print_language_counts = args.language or not args.language_code
    print_language_code_counts = args.language_code or not args.language

    if print_language_counts:
        for language, count in language_counter.most_common():
            print(f"{language}: {count}")
            if show_annotation_breakdown:
                for label in annotation_labels:
                    print(f"  {label}: {annotation_type_breakdown[language].get(label, 0)}")

    if print_language_code_counts and language_code_counter:
        if print_language_counts:
            print("\nLanguage code counts:")
        for code, count in language_code_counter.most_common():
            print(f"{code}: {count}")
            if show_annotation_breakdown:
                for label in annotation_labels:
                    print(f"  {label}: {annotation_type_breakdown_by_code[code].get(label, 0)}")

if __name__ == "__main__":
    main()
