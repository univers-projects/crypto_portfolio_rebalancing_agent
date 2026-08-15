"""Фабрика LLM-клієнтів поверх OpenRouter (OpenAI-сумісний API).

Ключ читається з .env через Settings і ніколи не хардкодиться.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import get_settings

Role = Literal["planner", "executor", "replanner", "explainer"]


class MissingApiKeyError(RuntimeError):
    """OPENROUTER_API_KEY не заданий — жоден LLM-виклик неможливий."""


@lru_cache(maxsize=8)
def get_llm(role: Role = "executor") -> ChatOpenAI:
    """Повернути LLM-клієнта для ролі planner / executor / replanner / explainer.

    Моделі й температури налаштовуються окремо для кожної ролі:
    executor працює з temperature=0 заради відтворюваності tool calls.
    """
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise MissingApiKeyError(
            "OPENROUTER_API_KEY не задано. Додайте його у .env перед запуском агента."
        )

    model = getattr(settings, f"{role}_model")
    temperature = getattr(settings, f"{role}_temperature")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=SecretStr(settings.openrouter_api_key),
        base_url=settings.openrouter_base_url,
        timeout=90,
        max_retries=2,
    )
