import argparse
import json
from pathlib import Path
from typing import Any, List, Union

from datasets import Dataset, DatasetDict, load_dataset

FLORES_LANGUAGE_CONFIGS = [
    "arb_Arab",
    "eng_Latn",
    "fra_Latn",
    "hin_Deva",
    "jpn_Jpan",
    "nld_Latn",
    "por_Latn",
    "spa_Latn",
    "tur_Latn",
    "vie_Latn",
    "cmn_Hans",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset and save it as JSON."
    )
    parser.add_argument(
        "--dataset",
        default="CohereLabs/aya_dataset",
        choices=["CohereLabs/aya_dataset", "openlanguagedata/flores_plus"],
        help="Dataset to pull.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split to download (e.g. train, dev, devtest).",
    )
    parser.add_argument(
        "--output",
        default="aya_dataset.json",
        help="Path to output JSON file.",
    )
    return parser.parse_args()


def save_dataset_to_json(payload: Any, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def format_raw_dataset(dataset_obj: Union[Dataset, DatasetDict]) -> Any:
    if isinstance(dataset_obj, DatasetDict):
        return {split_name: split_ds.to_list() for split_name, split_ds in dataset_obj.items()}
    return dataset_obj.to_list()


def pull_flores_plus(target_split: str = None) -> List[dict]:
    all_records: List[dict] = []

    for config in FLORES_LANGUAGE_CONFIGS:
        iso_639_3, iso_15924 = config.split("_", 1)
        print(f"  Pulling config: {config} … ", end="", flush=True)
        try:
            if target_split:
                ds = load_dataset(
                    "openlanguagedata/flores_plus", config, split=target_split
                )
                raw_records = ds.to_list()
                for r in raw_records:
                    r.setdefault("iso_639_3", iso_639_3)
                    r.setdefault("iso_15924", iso_15924)
                    r.setdefault("split", target_split)
                all_records.extend(raw_records)
            else:
                ds = load_dataset("openlanguagedata/flores_plus", config)
                for split_name, split_ds in ds.items():
                    for r in split_ds:
                        rec = dict(r)
                        rec.setdefault("iso_639_3", iso_639_3)
                        rec.setdefault("iso_15924", iso_15924)
                        rec.setdefault("split", split_name)
                        all_records.append(rec)
            print(f"ok ({len([r for r in all_records if r.get('iso_639_3') == iso_639_3])} records so far for {iso_639_3})")
        except Exception as exc:
            print(f"FAILED — {exc}")
            print(f"  WARNING: Skipping '{config}'. Re-run or check the HuggingFace config name.")

    return all_records


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    if args.dataset == "openlanguagedata/flores_plus":
        print("Pulling FLORES+ per-language config (11 configs)…")
        payload = pull_flores_plus(args.split)
        langs_present = {r.get("iso_639_3") for r in payload}
        print(f"\nTotal records: {len(payload)}")
        print(f"Languages present: {sorted(langs_present)}")
        missing_configs = [
            c for c in FLORES_LANGUAGE_CONFIGS
            if c.split("_", 1)[0] not in langs_present
        ]
        if missing_configs:
            print(f"WARNING: Missing configs: {missing_configs}")
    else:
        if args.split:
            dataset_obj = load_dataset(args.dataset, split=args.split)
        else:
            dataset_obj = load_dataset(args.dataset)
        payload = format_raw_dataset(dataset_obj)

    save_dataset_to_json(payload, output_path)
    print(f"Saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
