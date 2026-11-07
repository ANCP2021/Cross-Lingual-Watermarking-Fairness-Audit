import json
import random
from typing import Dict, List, Tuple

LANGUAGES = ["eng", "fra", "spa", "por", "nld", "zho", "jpn", "arb", "hin", "tur", "vie"]

FLORES_LANGUAGE_TAGS = {
    "eng": "eng_Latn",
    "fra": "fra_Latn",
    "spa": "spa_Latn",
    "por": "por_Latn",
    "nld": "nld_Latn",
    "zho": "cmn_Hans",
    "jpn": "jpn_Jpan",
    "arb": "arb_Arab",
    "hin": "hin_Deva",
    "tur": "tur_Latn",
    "vie": "vie_Latn",
}


def sample_flores(
    path: str, num_samples: int, seed: int = 42, split: str = "devtest"
) -> Dict[str, List[Tuple[str, str]]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    records = raw if isinstance(raw, list) else [r for v in raw.values() for r in v]

    records = [r for r in records if r.get("split") == split]

    by_id: Dict[int, Dict[str, str]] = {}
    for rec in records:
        lang_tag = f"{rec.get('iso_639_3')}_{rec.get('iso_15924')}"
        rid = rec["id"]
        if rid not in by_id:
            by_id[rid] = {}
        by_id[rid][lang_tag] = rec["text"]

    required = set(FLORES_LANGUAGE_TAGS.values())
    present = set()
    for langs in by_id.values():
        present.update(langs.keys())
    missing = required - present
    if missing:
        raise ValueError(
            f"The following required language tags are absent from the FLORES '{split}' data:\n"
            f"  {sorted(missing)}\n"
            f"Re-pull the dataset and re-run filter_flores.py to include them."
        )

    valid_ids = [rid for rid, langs in by_id.items() if required.issubset(langs.keys())]

    if len(valid_ids) < num_samples:
        raise ValueError(
            f"Only {len(valid_ids)} FLORES '{split}' IDs have all {len(required)} languages; "
            f"requested {num_samples}."
        )

    rng = random.Random(seed)
    sampled_ids = rng.sample(valid_ids, num_samples)

    result: Dict[str, List[Tuple[str, str]]] = {lang: [] for lang in LANGUAGES}
    for rid in sampled_ids:
        for lang_code, lang_tag in FLORES_LANGUAGE_TAGS.items():
            result[lang_code].append((str(rid), by_id[rid][lang_tag]))

    return result


def sample_aya(
    path: str, num_samples: int, seed: int = 42
) -> Dict[str, List[Tuple[str, str]]]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    by_lang: Dict[str, List[Tuple[str, str]]] = {}
    for idx, rec in enumerate(records):
        lang = rec.get("language_code")
        if lang not in by_lang:
            by_lang[lang] = []
        by_lang[lang].append((str(idx), rec["inputs"]))

    result: Dict[str, List[Tuple[str, str]]] = {}
    for i, lang in enumerate(LANGUAGES):
        lang_records = by_lang.get(lang, [])
        lang_seed = seed + i
        rng = random.Random(lang_seed)
        sampled = rng.sample(lang_records, min(num_samples, len(lang_records)))
        result[lang] = sampled

    return result
