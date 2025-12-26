# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Registry for ranking strategies using the plugin pattern."""

from typing import Dict, List, Type

from .base import RankingStrategy


class RankingRegistry:
    """
    Plugin registry for ranking strategies.

    Allows registration of new ranking algorithms via decorator
    and retrieval by name at runtime. This enables extensibility
    without modifying the core compression code.

    Example:
        >>> @RankingRegistry.register("my_ranker")
        ... class MyRanker(RankingStrategy):
        ...     def rank(self, corpus, query, **kwargs):
        ...         return [(i, 0) for i in range(len(corpus))]

        >>> ranker = RankingRegistry.get("my_ranker", device="cpu")
        >>> results = ranker.rank(["doc1", "doc2"], "query")
    """

    _strategies: Dict[str, Type[RankingStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a ranking strategy.

        Args:
            name: Unique identifier for the strategy.

        Returns:
            Decorator function that registers the class.

        Example:
            >>> @RankingRegistry.register("bm25")
            ... class BM25Ranker(RankingStrategy):
            ...     pass
        """

        def decorator(strategy_cls: Type[RankingStrategy]):
            if name in cls._strategies:
                raise ValueError(f"Strategy '{name}' is already registered")
            cls._strategies[name] = strategy_cls
            strategy_cls.name = name
            return strategy_cls

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> RankingStrategy:
        """
        Get an instance of a registered strategy.

        Args:
            name: Name of the registered strategy.
            **kwargs: Arguments passed to the strategy constructor.

        Returns:
            An instance of the requested strategy.

        Raises:
            ValueError: If the strategy name is not registered.
        """
        if name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown ranking strategy: '{name}'. "
                f"Available strategies: {available}"
            )
        return cls._strategies[name](**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all registered strategy names.

        Returns:
            List of registered strategy names.
        """
        return list(cls._strategies.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a strategy is registered.

        Args:
            name: Name to check.

        Returns:
            True if the strategy is registered, False otherwise.
        """
        return name in cls._strategies
