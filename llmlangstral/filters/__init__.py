# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Multi-level filtering for LLMLangstral prompt compression.

This module provides pluggable filtering strategies that operate at
three granularities for progressive prompt compression:

Filtering levels:
    - Context: Select most relevant documents/passages
    - Sentence: Select most relevant sentences within documents
    - Token: Select tokens based on perplexity thresholds

Each filter receives a FilterContext with shared resources (tokenizer,
device) and callbacks to the main compressor (PPL computation, ranking).

Example:
    >>> from llmlangstral.filters import FilterContext, ContextLevelFilter
    >>> ctx = FilterContext(
    ...     tokenizer=tokenizer,
    ...     device="cuda",
    ...     max_position_embeddings=4096,
    ...     get_rank_results_fn=compressor.get_rank_results,
    ... )
    >>> filter = ContextLevelFilter(ctx)
    >>> contexts, ratios, indices, _ = filter.filter(
    ...     context=["doc1", "doc2", "doc3"],
    ...     context_tokens_length=[100, 200, 150],
    ...     target_token=300,
    ...     question="What is the answer?",
    ... )
"""

from .base import FilterBase, FilterContext
from .context import ContextLevelFilter
from .sentence import SentenceLevelFilter
from .token import TokenLevelFilter

__all__ = [
    "FilterBase",
    "FilterContext",
    "ContextLevelFilter",
    "SentenceLevelFilter",
    "TokenLevelFilter",
]
