"""Зароблена автономія: агент набирає довіру, а не отримує її на старті.

Ідея (progressive delegation): система не просить самостійності при першому
запуску. Вона веде облік того, як людина реагувала на однотипні дії, і лише
коли статистика стає переконливою — пропонує брати такі дії на себе.

Що делегується
--------------
Тільки **дрібні зміни ваги** (до `autonomy_max_delta`, за замовчуванням 5 п.п.).
Купівля нового активу, повний продаж і великі перекладання не делегуються
ніколи, скільки б схвалень не назбиралось.

Що делегування дає
------------------
Якщо весь план складається з дрібних змін, замість порядкового чек-листа
показується один рядок підсумку і одне підтвердження. Якщо дрібних дій лише
частина — вони підписані як «авто», решта переглядається як завжди.

Чого делегування НЕ робить
--------------------------
Воно не прибирає людину з контуру. Підтвердження плану лишається за нею, є
вихід у повний перегляд, а `mock_execute_rebalance` і далі вимагає
`approval_token == "approve"`. Це прискорення рутини, а не автоматичне
виконання угод.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# Типи дій, які взагалі можуть бути делеговані
DELEGABLE_ACTIONS = frozenset({"REDUCE", "INCREASE"})


@dataclass(frozen=True)
class AutonomyState:
    """Накопичена історія рішень і поточний рівень делегування."""

    accepted: int = 0
    rejected: int = 0
    granted: bool = False
    granted_at: str | None = None
    max_delta: float = 0.0

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "granted": self.granted,
            "granted_at": self.granted_at,
            "max_delta": self.max_delta,
        }


def is_delegable(action: dict[str, Any], max_delta: float | None = None) -> bool:
    """Чи належить дія до типу, який взагалі можна делегувати."""
    limit = max_delta if max_delta is not None else get_settings().autonomy_max_delta
    if str(action.get("action")) not in DELEGABLE_ACTIONS:
        return False
    delta = abs(
        float(action.get("to_weight") or 0.0) - float(action.get("from_weight") or 0.0)
    )
    return delta <= limit


def load_state(path: Path | None = None) -> AutonomyState:
    """Прочитати стан автономії; за відсутності файлу — чистий стан."""
    target = path or get_settings().autonomy_state_path
    if not target.exists():
        return AutonomyState()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        return AutonomyState(
            accepted=int(payload.get("accepted", 0)),
            rejected=int(payload.get("rejected", 0)),
            granted=bool(payload.get("granted", False)),
            granted_at=payload.get("granted_at"),
            max_delta=float(payload.get("max_delta", 0.0)),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        logger.warning("Стан автономії пошкоджено, починаю з нуля: %s", error)
        return AutonomyState()


def save_state(state: AutonomyState, path: Path | None = None) -> None:
    """Зберегти стан автономії."""
    target = path or get_settings().autonomy_state_path
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(
            json.dumps(state.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as error:
        logger.warning("Не вдалося зберегти стан автономії: %s", error)


def record_decisions(
    actions: list[dict[str, Any]],
    accepted: list[bool],
    state: AutonomyState,
) -> AutonomyState:
    """Додати до статистики рішення по делегованих за типом діях.

    Великі дії в облік не йдуть: вони ніколи не делегуються, тож їхні схвалення
    не мають наближати автономію.
    """
    plus = minus = 0
    for action, is_accepted in zip(actions, accepted, strict=True):
        if not is_delegable(action):
            continue
        if is_accepted:
            plus += 1
        else:
            minus += 1

    if not plus and not minus:
        return state

    return AutonomyState(
        accepted=state.accepted + plus,
        rejected=state.rejected + minus,
        granted=state.granted,
        granted_at=state.granted_at,
        max_delta=state.max_delta,
    )


def can_offer(state: AutonomyState) -> bool:
    """Чи вже назбиралось достатньо історії, щоб пропонувати делегування."""
    settings = get_settings()
    return (
        not state.granted
        and state.total >= settings.autonomy_min_decisions
        and state.acceptance_rate >= settings.autonomy_min_acceptance
    )


def grant(state: AutonomyState) -> AutonomyState:
    """Надати делегування дрібних змін ваги."""
    return AutonomyState(
        accepted=state.accepted,
        rejected=state.rejected,
        granted=True,
        granted_at=datetime.now(UTC).isoformat(),
        max_delta=get_settings().autonomy_max_delta,
    )


def revoke(state: AutonomyState) -> AutonomyState:
    """Відкликати делегування, зберігши накопичену статистику."""
    return AutonomyState(
        accepted=state.accepted,
        rejected=state.rejected,
        granted=False,
        granted_at=None,
        max_delta=0.0,
    )


def covers_all(actions: list[dict[str, Any]], state: AutonomyState) -> bool:
    """Чи весь план складається з делегованих дій.

    Тоді замість повного чек-листа показується один рядок і одне підтвердження:
    людина лишається в контурі, але не переглядає рутину порядково.
    """
    return (
        state.granted
        and bool(actions)
        and all(is_delegable(action, state.max_delta) for action in actions)
    )


def describe(state: AutonomyState) -> str:
    """Людиночитаний опис поточного рівня делегування."""
    settings = get_settings()
    lines = [
        f"Рішень по дрібних змінах ваги: {state.total} "
        f"(схвалено {state.accepted}, відхилено {state.rejected})",
    ]
    if state.total:
        lines.append(f"Частка схвалень: {state.acceptance_rate * 100:.0f}%")

    if state.granted:
        lines.append(
            f"Делеговано: зміни ваги до {state.max_delta * 100:.0f} п.п. "
            "позначаються автоматично."
        )
        lines.append("Фінальне підтвердження плану залишається за вами.")
        lines.append("Відкликати: /autonomy off")
    else:
        need = max(0, settings.autonomy_min_decisions - state.total)
        lines.append("Делегування не надано — кожна дія потребує вашої галочки.")
        if need:
            lines.append(
                f"Ще {need} рішень по дрібних діях — і агент запропонує взяти їх на себе."
            )
        elif state.acceptance_rate < settings.autonomy_min_acceptance:
            lines.append(
                f"Частка схвалень нижча за потрібні "
                f"{settings.autonomy_min_acceptance * 100:.0f}% — пропозиції не буде."
            )
    return "\n".join(lines)
