# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Neural embedding-based ranking strategies."""

from typing import List, Tuple

import numpy as np
import torch

from .base import ModelBasedRanker
from .registry import RankingRegistry


@RankingRegistry.register("sentbert")
class SentBertRanker(ModelBasedRanker):
    """
    SentenceBERT ranking using multi-qa-mpnet-base-dot-v1.

    Uses dense embeddings and dot product similarity for ranking.
    Good general-purpose semantic similarity.

    Requires: sentence_transformers library
    """

    name = "sentbert"
    model_name = "multi-qa-mpnet-base-dot-v1"

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        from sentence_transformers import util

        doc_embeds = self.model.encode(corpus)
        query_embed = self.model.encode(query)
        doc_scores = -util.dot_score(doc_embeds, query_embed).cpu().numpy().reshape(-1)

        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx


@RankingRegistry.register("bge")
class BGERanker(ModelBasedRanker):
    """
    BAAI BGE (bge-large-en-v1.5) embedding ranker.

    High-quality English embeddings with normalized dot product.

    Requires: sentence_transformers library
    """

    name = "bge"
    model_name = "BAAI/bge-large-en-v1.5"

    def _load_model(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        from sentence_transformers import util

        doc_embeds = self.model.encode(corpus, normalize_embeddings=True)
        query_embed = self.model.encode(query, normalize_embeddings=True)
        doc_scores = -util.dot_score(doc_embeds, query_embed).cpu().numpy().reshape(-1)

        idx = [(ii, 0) for ii in np.argsort(doc_scores)]
        return idx


@RankingRegistry.register("bge_reranker")
class BGEReranker(ModelBasedRanker):
    """
    BAAI BGE Reranker (bge-reranker-large).

    Cross-encoder reranker that scores query-document pairs directly.
    Higher accuracy but slower than bi-encoder approaches.

    Requires: transformers library
    """

    name = "bge_reranker"
    model_name = "BAAI/bge-reranker-large"

    def _load_model(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_name)
            .eval()
            .to(self.device)
        )

    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        pairs = [[doc, query] for doc in corpus]

        with torch.no_grad():
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            scores = (
                self.model(**inputs, return_dict=True)
                .logits.view(-1)
                .float()
            )

        idx = [(ii, 0) for ii in np.argsort(-scores.cpu().numpy())]
        return idx


@RankingRegistry.register("bge_llmembedder")
class BGELLMEmbedderRanker(ModelBasedRanker):
    """
    BAAI LLM-Embedder ranking.

    Uses instruction-prefixed embeddings with CLS pooling.
    Optimized for retrieval tasks.

    Requires: transformers library
    """

    name = "bge_llmembedder"
    model_name = "BAAI/llm-embedder"

    def _load_model(self):
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModel.from_pretrained(self.model_name)
            .eval()
            .to(self.device)
        )

    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        instruction_qa_query = "Represent this query for retrieving relevant documents: "
        instruction_qa_key = "Represent this document for retrieval: "

        queries = [instruction_qa_query + query for _ in corpus]
        keys = [instruction_qa_key + key for key in corpus]

        with torch.no_grad():
            query_inputs = self._tokenizer(
                queries,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            key_inputs = self._tokenizer(
                keys,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            query_outputs = self.model(**query_inputs)
            key_outputs = self.model(**key_inputs)

            # CLS pooling
            query_embeddings = query_outputs.last_hidden_state[:, 0]
            key_embeddings = key_outputs.last_hidden_state[:, 0]

            # Normalize
            query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
            key_embeddings = torch.nn.functional.normalize(key_embeddings, p=2, dim=1)

            similarity = query_embeddings @ key_embeddings.T

        idx = [(ii, 0) for ii in np.argsort(-similarity[0].cpu().numpy())]
        return idx


@RankingRegistry.register("jinza")
class JinzaRanker(ModelBasedRanker):
    """
    Jina AI embeddings (jina-embeddings-v2-base-en).

    Compact and efficient embeddings with good performance.

    Requires: transformers library
    """

    name = "jinza"
    model_name = "jinaai/jina-embeddings-v2-base-en"

    def _load_model(self):
        from transformers import AutoModel

        self._model = (
            AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            .eval()
            .to(self.device)
        )

    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        from numpy.linalg import norm

        def cos_sim(a, b):
            return (a @ b.T) / (norm(a) * norm(b))

        doc_embeds = self.model.encode(corpus)
        query_embed = self.model.encode(query)
        doc_scores = cos_sim(doc_embeds, query_embed)

        idx = [(ii, 0) for ii in np.argsort(-doc_scores)]
        return idx
