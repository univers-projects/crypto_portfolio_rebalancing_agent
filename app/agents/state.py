"""GuardedState — стан workflow, що зберігається у SqliteSaver.

Усі значення тримаються у JSON-сумісному вигляді (dict/list/примітиви),
а не як Pydantic-обʼєкти. Це гарантує коректну серіалізацію чекпойнтів
і читабельний вивід `get_state()`.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def _replace(_: Any, new: Any) -> Any:
    """Reducer: нове значення повністю заміщує попереднє."""
    return new


class GuardedState(TypedDict, total=False):
    """Стан щоденного циклу аналізу портфеля."""

    # Діалог і трасування
    messages: Annotated[list[AnyMessage], add_messages]
    trajectory: Annotated[list[dict[str, Any]], operator.add]
    tool_history: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[dict[str, str]], operator.add]

    # Вхідні дані аналізу
    current_portfolio: Annotated[dict[str, float], _replace]
    candidate_assets: Annotated[list[str], _replace]
    dropped_assets: Annotated[list[str], operator.add]
    lookback_days: Annotated[int, _replace]

    # Проміжні результати
    asset_metrics: Annotated[dict[str, Any], _replace]
    portfolio_metrics: Annotated[dict[str, Any], _replace]

    # Plan-and-Execute
    plan: Annotated[dict[str, Any], _replace]
    completed_steps: Annotated[list[dict[str, Any]], operator.add]
    replan_count: Annotated[int, _replace]

    # Рішення
    policy_evidence: Annotated[dict[str, Any], _replace]
    decision: Annotated[str, _replace]
    final_decision: Annotated[dict[str, Any], _replace]
    rebalance_proposal: Annotated[dict[str, Any] | None, _replace]

    # Human-in-the-Loop
    approval: Annotated[str, _replace]
    modified_positions: Annotated[list[dict[str, Any]] | None, _replace]
    execution_result: Annotated[dict[str, Any] | None, _replace]
