# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""API-based ranking strategies (OpenAI, VoyageAI, Cohere)."""

from typing import List, Tuple

import numpy as np

from .base import APIBasedRanker
from .registry import RankingRegistry


@RankingRegistry.register("openai")
class OpenAIRanker(APIBasedRanker):
    """
    OpenAI embeddings API ranker.

    Uses OpenAI's text-embedding-ada-002 (or configured engine)
    for dense embeddings and dot product similarity.

    Requires: openai library
    Config keys: api_key, api_base, api_type, api_version, engine
    """

    name = "openai"

    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        import openai
        from sentence_transformers import util

        # Configure OpenAI client
        openai.api_key = self.api_config.get("api_key", "")
        openai.api_base = self.api_config.get("api_base", "https://api.openai.com/v1")
        openai.api_type = self.api_config.get("api_type", "open_ai")
        openai.api_version = self.api_config.get("api_version", "2023-05-15")
        engine = self.api_config.get("engine", "text-embedding-ada-002")

        def get_embed(text: str):
            return openai.Embedding.create(
                input=[text.replace("\n", " ")], engine=engine
            )["data"][0]["embedding"]

        doc_embeds = [get_embed(doc) for doc in corpus]
        query_embed = get_embed(query)

        doc_scores = -util.dot_score(doc_embeds, query_embed).cpu().numpy().reshape(-1)
        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx


@RankingRegistry.register("voyageai")
class VoyageAIRanker(APIBasedRanker):
    """
    VoyageAI embeddings API ranker.

    Uses VoyageAI's voyage-01 model for embeddings.

    Requires: voyageai library
    Config keys: voyageai_api_key
    """

    name = "voyageai"

    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        import voyageai
        from sentence_transformers import util

        voyageai.api_key = self.api_config.get("voyageai_api_key", "")

        def get_embed(text: str):
            return voyageai.get_embedding(text, model="voyage-01")

        doc_embeds = [get_embed(doc) for doc in corpus]
        query_embed = get_embed(query)

        doc_scores = -util.dot_score(doc_embeds, query_embed).cpu().numpy().reshape(-1)
        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx


@RankingRegistry.register("cohere")
class CohereRanker(APIBasedRanker):
    """
    Cohere rerank API ranker.

    Uses Cohere's rerank-english-v2.0 for document reranking.
    Returns top 20 documents by default.

    Requires: cohere library
    Config keys: cohere_api_key
    """

    name = "cohere"

    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        import cohere

        api_key = self.api_config.get("cohere_api_key", "")
        co = cohere.Client(api_key)

        results = co.rerank(
            model="rerank-english-v2.0",
            query=query,
            documents=corpus,
            top_n=20,
        )

        # Map ranked documents back to original indices
        c_map = {doc: idx for idx, doc in enumerate(corpus)}
        doc_rank = [c_map[result.document["text"]] for result in results]

        idx = [(ii, 0) for ii in doc_rank]
        return idx
