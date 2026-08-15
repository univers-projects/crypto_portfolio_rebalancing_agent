"""Стрічка ухвалених рішень.

Окремого журналу немає навмисно: усе потрібне вже пишеться у
`trajectory.jsonl`. Цей модуль лише читає його і збирає події в записи виду
«коли, яке рішення, чим закінчилось».

Джерела подій:
    decide / decision_made           -> сам вердикт і числа
    execute_rebalance / rejected     -> людина відмовила
    execute_rebalance / blocked      -> спроба виконати без підтвердження
    execute_rebalance / autonomy_*   -> дії, схвалені автоматично
    tool mock_execute_rebalance      -> ребаланс виконано (mock)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# Підсумок виконання, який може настати після рішення
_OUTCOMES = {
    "rejected": "відхилено людиною",
    "blocked": "заблоковано без підтвердження",
    "modification_requested": "план змінено",
}


@dataclass(frozen=True)
class DecisionRecord:
    """Один запис стрічки."""

    timestamp: str
    decision: str
    net_improvement: float | None
    threshold: float | None
    turnover: float | None
    outcome: str

    @property
    def when(self) -> str:
        """Коротка дата для показу."""
        try:
            return datetime.fromisoformat(self.timestamp).strftime("%d.%m %H:%M")
        except (TypeError, ValueError):
            return "—"


def read_history(limit: int = 20, path: Path | None = None) -> list[DecisionRecord]:
    """Прочитати останні рішення з trajectory-логу, найновіші останніми."""
    entries = _read_entries(path or get_settings().trajectory_log_path)
    records: list[DecisionRecord] = []

    for index, entry in enumerate(entries):
        if entry.get("event") != "decision_made":
            continue
        records.append(
            DecisionRecord(
                timestamp=str(entry.get("timestamp", "")),
                decision=str(entry.get("decision", "?")),
                net_improvement=_maybe_float(entry.get("net_improvement")),
                threshold=_maybe_float(entry.get("threshold")),
                turnover=_maybe_float(entry.get("turnover")),
                outcome=_resolve_outcome(entries, index),
            )
        )

    return records[-limit:]


def _resolve_outcome(entries: list[dict[str, Any]], start: int) -> str:
    """Знайти, чим завершилось рішення, у подіях після нього.

    Пошук зупиняється на наступному `decision_made`: усе, що далі, стосується
    вже іншого циклу.
    """
    for entry in entries[start + 1 :]:
        if entry.get("event") == "decision_made":
            break
        if entry.get("tool") == "mock_execute_rebalance" and entry.get("status") == "success":
            return "виконано (mock)"
        event = str(entry.get("event", ""))
        if event in _OUTCOMES:
            return _OUTCOMES[event]
    return "без змін"


def _read_entries(path: Path) -> list[dict[str, Any]]:
    """Усі рядки .jsonl; пошкоджені рядки пропускаються без падіння."""
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as error:
        logger.warning("Не вдалося прочитати історію рішень: %s", error)
    return entries


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
