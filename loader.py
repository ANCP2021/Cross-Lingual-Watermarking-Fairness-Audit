import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MarkLLM"))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig


MODEL_REGISTRY = {
    "mistral": {
        "base":      "mistralai/Mistral-Nemo-Base-2407",
        "instruct":  "mistralai/Mistral-Nemo-Instruct-2407",
        "xsir_config": "watermark_configs/XSIR_mistral.json",
        "dtype":     torch.float16,
    },
    "gemma": {
        "base":      "google/gemma-3-4b-pt",
        "instruct":  "google/gemma-3-4b-it",
        "xsir_config": "watermark_configs/XSIR_gemma.json",
        "dtype":     torch.bfloat16,
    },
    "qwen": {
        "base":      "Qwen/Qwen2.5-7B",
        "instruct":  "Qwen/Qwen2.5-7B-Instruct",
        "xsir_config": "watermark_configs/XSIR_qwen.json",
        "dtype":     torch.float16,
    },
}

GEN_CONFIG = {
    "temperature": 0.7,
    "max_new_tokens": 200,
    "min_new_tokens": 100,
    "do_sample": True,
}


def load_model_and_tokenizer(model_alias: str, variant: str):
    model_id = MODEL_REGISTRY[model_alias][variant]
    print(f"  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model:     {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=MODEL_REGISTRY[model_alias]["dtype"],
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _get_vocab_size(model) -> int:
    if hasattr(model.config, "vocab_size"):
        return model.config.vocab_size
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "vocab_size"):
        return model.config.text_config.vocab_size
    return model.get_output_embeddings().weight.shape[0]


def build_transformers_config(model, tokenizer, device: str = "cuda") -> TransformersConfig:
    return TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=_get_vocab_size(model),
        device=device,
        **GEN_CONFIG,
    )


def get_watermark_config_path(watermark: str, model_alias: str) -> str:
    if watermark == "XSIR":
        return MODEL_REGISTRY[model_alias]["xsir_config"]
    return os.path.join("watermark_configs", f"{watermark}.json")


def load_tokenizer_only(model_alias: str, variant: str):
    model_id = MODEL_REGISTRY[model_alias][variant]
    print(f"  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _get_vocab_size_from_config(model_alias: str, variant: str) -> int:
    model_id = MODEL_REGISTRY[model_alias][variant]
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, "vocab_size"):
        return config.vocab_size
    if hasattr(config, "text_config") and hasattr(config.text_config, "vocab_size"):
        return config.text_config.vocab_size
    raise ValueError(f"Cannot determine vocab_size from config for {model_id}")


def build_transformers_config_for_detection(tokenizer, model_alias: str, variant: str, device: str = "cpu") -> TransformersConfig:
    vocab_size = _get_vocab_size_from_config(model_alias, variant)
    print(f"  Detection vocab_size: {vocab_size}  (tokenizer reports {len(tokenizer)})")
    return TransformersConfig(
        model=None,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        device=device,
    )


def load_watermark(watermark_name: str, transformers_config: TransformersConfig, model_alias: str):
    config_path = get_watermark_config_path(watermark_name, model_alias)
    return AutoWatermark.load(
        watermark_name,
        algorithm_config=config_path,
        transformers_config=transformers_config,
    )
