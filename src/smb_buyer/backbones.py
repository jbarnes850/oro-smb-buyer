"""Supported Tinker backbones, the context partition and per-backbone list prices.

Pure data: the engine child imports this module too.
"""

CONTEXT = 262144
MAX_OUTPUT = 65536

# key -> (Tinker model, tokenizer model, official renderer by reasoning mode)
BACKBONES = {
    "qwen3.8": (
        "Qwen/Qwen3.8-27B:peft:262144",
        "Qwen/Qwen3.8-27B",
        {"on": "qwen3_8_medium_reasoning", "off": "qwen3_8_disable_thinking"},
    ),
    # Needed only by the economically selected SFT warm start and its evaluation.
    "qwen3.6": (
        "Qwen/Qwen3.6-35B-A3B:peft:262144",
        "Qwen/Qwen3.6-35B-A3B",
        {"on": "qwen3_5", "off": "qwen3_5_disable_thinking"},
    ),
}
DEFAULT_BACKBONE = "qwen3.8"

# Official Tinker list prices, USD per million tokens.
PRICES = {
    "qwen3.8": {"prefill": 0.6975, "cached_prefill": 0.1395, "sample": 5.595, "train": 7.46},
    "qwen3.6": {"prefill": 0.45, "cached_prefill": 0.09, "sample": 1.335, "train": 1.177},
}

# What the original adapter replays from earlier turns, by reasoning mode.
REASONING_REPLAY = {"on": "details_and_text", "off": None}


def backbone_of_model(model: str) -> str | None:
    """Backbone key for a Tinker model name (with or without the ``:peft:`` suffix)."""
    base = model.split(":", 1)[0]
    for key, (tinker_model, tokenizer_model, _) in BACKBONES.items():
        if model == tinker_model or base == tokenizer_model:
            return key
    return None

