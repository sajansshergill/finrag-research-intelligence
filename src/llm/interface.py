"""
Compatibility wrapper.

Core LLM interface code currently lives in the top-level `interface.py`.
Tests and other modules import it from `src.llm.interface`.
"""

from __future__ import annotations

from interface import (  # noqa: F401
    AnswerResult,
    Citation,
    FinRAGInterface,
    MockLLMClient,
    OpenAIClient,
    AnthropicClient,
    build_prompt,
    parse_citations,
)

__all__ = [
    "AnswerResult",
    "Citation",
    "FinRAGInterface",
    "MockLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "build_prompt",
    "parse_citations",
]

