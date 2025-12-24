# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Configuration des modèles Mistral AI pour LLMLingstral."""

MISTRAL_MODELS = {
    "default": "mistralai/Mistral-7B-v0.3",
    "small": "mistralai/Ministral-3-3B-Instruct-2512",
    "medium": "mistralai/Ministral-3-8B-Instruct-2512",
    "large": "mistralai/Mistral-Large-3",
    "quantized": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    "embedding": "intfloat/e5-mistral-7b-instruct",
}

# Modèles LLMLingua-2 (à entraîner - Phase 3)
MISTRAL_LINGUA2_MODELS = {
    "large": None,  # À créer: mistralai/mistral-lingua-2-large
    "small": None,  # À créer: mistralai/mistral-lingua-2-small
}

# Alias pour migration progressive
DEFAULT_MODEL = MISTRAL_MODELS["default"]
SMALL_MODEL = MISTRAL_MODELS["small"]
QUANTIZED_MODEL = MISTRAL_MODELS["quantized"]
EMBEDDING_MODEL = MISTRAL_MODELS["embedding"]
