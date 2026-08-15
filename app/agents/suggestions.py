"""Підказки наступних питань у чаті.

Чат без підказок — глухий кут: користувач отримав відповідь і не знає, що ще
можна спитати. Тут генерується кілька доречних наступних кроків.

Як і скрізь у проєкті, LLM не є обовʼязковим: якщо він недоступний або віддав
щось невалідне, повертається детермінований набір, підібраний за тим, які
інструменти було використано.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings
from app.llm import get_llm
from app.prompts import load_prompt

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 90


class FollowUpQuestions(BaseModel):
    """Структурований вивід підказок."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: tuple[str, ...] = Field(min_length=1, max_length=5)

    @field_validator("questions", mode="after")
    @classmethod
    def _clean(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = []
        for question in value:
            text = question.strip().lstrip("-•0123456789. ").strip()
            if text and len(text) <= MAX_QUESTION_LENGTH:
                cleaned.append(text)
        if not cleaned:
            raise ValueError("Порожній набір підказок")
        return tuple(dict.fromkeys(cleaned))


# Детерміновані запасні набори за використаним інструментом
_FALLBACKS: dict[str, tuple[str, ...]] = {
    "knowledge_search": (
        "Як це виглядає в моєму портфелі?",
        "Який актив у мене найгірший за цим показником?",
        "Чому це впливає на рішення про ребаланс?",
    ),
    "get_top_liquid_assets": (
        "Порахуй метрики для трьох найліквідніших активів",
        "Чому стейблкоїни виключені з universe?",
        "Що означає ліквідність для ребалансу?",
    ),
    "calculate_asset_metrics": (
        "Що означає ця волатильність простими словами?",
        "Як ці метрики впливають на склад портфеля?",
        "Порівняй це з моїм поточним портфелем",
    ),
    "evaluate_portfolio": (
        "Що таке диверсифікація і чому вона важлива?",
        "Який актив найбільше псує цю оцінку?",
        "Що буде, якщо продати найслабший актив?",
    ),
}

_GENERIC: tuple[str, ...] = (
    "Що зараз у моєму портфелі?",
    "Що таке максимальна просадка?",
    "Чому оборот портфеля вважається витратами?",
)


def suggest_followups(
    question: str,
    answer: str,
    tools_used: list[str],
    llm: Any = None,
) -> tuple[str, ...]:
    """Запропонувати наступні питання. Ніколи не кидає виняток назовні."""
    settings = get_settings()
    if not settings.chat_suggestions_enabled:
        return ()

    limit = settings.chat_suggestions_count
    fallback = _fallback_for(tools_used)[:limit]

    try:
        client = llm if llm is not None else get_llm("explainer")
        structured = client.with_structured_output(FollowUpQuestions)
        result = structured.invoke(
            [
                SystemMessage(content=load_prompt("suggestions")),
                HumanMessage(content=_payload(question, answer, tools_used, limit)),
            ]
        )
    except Exception as error:  # noqa: BLE001 — підказки не мають ламати чат
        logger.debug("Підказки недоступні, використано запасні: %s", error)
        return fallback

    if not isinstance(result, FollowUpQuestions):
        return fallback
    return result.questions[:limit]


def _payload(question: str, answer: str, tools_used: list[str], limit: int) -> str:
    """Контекст для LLM."""
    tools = ", ".join(tools_used) if tools_used else "жодного"
    return (
        f"Питання користувача: {question}\n"
        f"Використані інструменти: {tools}\n"
        f"Відповідь асистента:\n{answer[:1500]}\n\n"
        f"Запропонуй рівно {limit} наступних питань."
    )


def _fallback_for(tools_used: list[str]) -> tuple[str, ...]:
    """Детермінований набір під використаний інструмент."""
    for tool in tools_used:
        if tool in _FALLBACKS:
            return _FALLBACKS[tool]
    return _GENERIC
