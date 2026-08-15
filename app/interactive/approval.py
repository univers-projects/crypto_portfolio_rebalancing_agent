"""Покрокове підтвердження ребалансу по кожному активу окремо.

Граф не змінюється: `interrupt_before=["execute_rebalance"]` лишається єдиною
точкою HITL. Цей модуль працює поверх переривання — проходить дії по одній,
збирає рішення людини і мапить їх у наявні гілки графа:

    усі "так"    -> approval="approve"
    усі "ні"     -> approval="reject"
    частково     -> approval="modify" з перерахованими вагами

Правило перерахунку ваг при частковому схваленні
------------------------------------------------
Схвалені дії задають цільову вагу своїх активів (locked). Активи, дій по яких
не схвалено, лишаються на поточній вазі (free). Сума при цьому майже завжди
не дорівнює 1.0 — розрив закривається пропорційним масштабуванням free-активів.

Приклад: схвалили BUY BNB 33.6%, відхилили REDUCE BTC 40%->26.4%. BNB
зафіксовано на 33.6%, решта активів пропорційно стискається так, щоб сума
знову стала 1.0.

Якщо результат порушує ліміти позиції (`min_position_weight`,
`max_position_weight`, `max_portfolio_assets`) або розрив закрити нема ким —
комбінація відхиляється з поясненням, і користувача просять обрати інакше.
Мовчазної підгонки не відбувається.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.data.analytics import compute_turnover
from app.domain.errors import InvalidPortfolioError
from app.domain.schemas import AssetMetrics, Portfolio, Position

# Похибка, у межах якої сума ваг вважається одиницею
WEIGHT_TOLERANCE = 1e-9


def merge_partial_approval(
    current: Portfolio,
    actions: Sequence[dict[str, Any]],
    accepted: Sequence[bool],
) -> Portfolio:
    """Побудувати цільовий портфель зі схваленої підмножини дій.

    Кидає InvalidPortfolioError із читабельним повідомленням, якщо комбінація
    схвалених дій не зводиться до валідного портфеля.
    """
    if len(actions) != len(accepted):
        raise InvalidPortfolioError("Кількість рішень не збігається з кількістю дій")

    settings = get_settings()
    target: dict[str, float] = dict(current.as_mapping())
    locked: set[str] = set()

    for action, is_accepted in zip(actions, accepted, strict=True):
        if not is_accepted:
            continue
        symbol = str(action["symbol"])
        target[symbol] = float(action["to_weight"])
        locked.add(symbol)

    free = [symbol for symbol in target if symbol not in locked and target[symbol] > 0]
    gap = 1.0 - sum(target.values())

    if abs(gap) > WEIGHT_TOLERANCE:
        target = _close_gap(target, free, gap)

    target = {symbol: weight for symbol, weight in target.items() if weight > WEIGHT_TOLERANCE}
    if not target:
        raise InvalidPortfolioError("Схвалені дії залишають портфель порожнім")

    target = _normalize(target)
    _check_limits(target, settings.min_position_weight, settings.max_position_weight)

    if len(target) > settings.max_portfolio_assets:
        raise InvalidPortfolioError(
            f"Результат містить {len(target)} активів при ліміті "
            f"{settings.max_portfolio_assets}"
        )

    positions = tuple(
        Position(symbol=symbol, weight=weight) for symbol, weight in sorted(target.items())
    )
    return Portfolio(positions=positions)


def _close_gap(
    target: dict[str, float], free: list[str], gap: float
) -> dict[str, float]:
    """Пропорційно масштабувати незафіксовані активи, щоб сума стала 1.0."""
    if not free:
        raise InvalidPortfolioError(
            f"Схвалені дії дають суму ваг {1.0 - gap:.3f}, а вільних активів для "
            "перерозподілу не лишилось. Схваліть додаткову дію або відхиліть одну "
            "з поточних."
        )

    free_total = sum(target[symbol] for symbol in free)
    if free_total <= WEIGHT_TOLERANCE:
        raise InvalidPortfolioError("Немає ваги, яку можна перерозподілити")

    factor = (free_total + gap) / free_total
    if factor < 0:
        raise InvalidPortfolioError(
            "Схвалені покупки перевищують доступну вагу портфеля. Схваліть також "
            "хоча б одне скорочення або продаж, щоб вивільнити капітал."
        )

    return {
        symbol: (weight * factor if symbol in free else weight)
        for symbol, weight in target.items()
    }


def _normalize(target: dict[str, float]) -> dict[str, float]:
    """Округлити ваги і прибрати залишок округлення з найбільшої позиції."""
    rounded = {symbol: round(weight, 6) for symbol, weight in target.items()}
    residual = round(1.0 - sum(rounded.values()), 6)
    if residual:
        largest = max(rounded, key=lambda symbol: rounded[symbol])
        rounded[largest] = round(rounded[largest] + residual, 6)
    return rounded


def _check_limits(target: dict[str, float], minimum: float, maximum: float) -> None:
    """Перевірити ліміти розміру позиції з поясненням, що саме порушено."""
    for symbol, weight in sorted(target.items()):
        if weight < minimum:
            raise InvalidPortfolioError(
                f"{symbol} отримує {weight * 100:.1f}% — менше мінімального розміру "
                f"позиції {minimum * 100:.0f}%"
            )
        if weight > maximum:
            raise InvalidPortfolioError(
                f"{symbol} отримує {weight * 100:.1f}% — більше максимального розміру "
                f"позиції {maximum * 100:.0f}%"
            )


@dataclass(frozen=True)
class SelectionSummary:
    """Наслідки поточного набору галочок — для живого підсумку в чек-листі."""

    accepted_count: int
    total_count: int
    portfolio: Portfolio | None
    turnover: float
    error: str | None

    @property
    def is_valid(self) -> bool:
        return self.error is None


def summarize_selection(
    current: Portfolio,
    actions: Sequence[dict[str, Any]],
    accepted: Sequence[bool],
) -> SelectionSummary:
    """Порахувати результат вибору, не кидаючи винятків.

    Викликається після кожного перемикання галочки, тому помилка тут — це
    нормальний стан UI, а не збій: користувач бачить, що комбінація неможлива,
    ще до підтвердження.
    """
    total = len(actions)
    count = sum(1 for flag in accepted if flag)

    if count == 0:
        return SelectionSummary(0, total, current, 0.0, None)

    try:
        portfolio = merge_partial_approval(current, actions, accepted)
    except InvalidPortfolioError as error:
        return SelectionSummary(count, total, None, 0.0, str(error))

    return SelectionSummary(
        count, total, portfolio, compute_turnover(current, portfolio), None
    )


def resolve_approval(
    current: Portfolio,
    actions: Sequence[dict[str, Any]],
    accepted: Sequence[bool],
) -> tuple[str, list[dict[str, Any]] | None]:
    """Перетворити покрокові рішення у пару (approval, modified_positions).

    Повертає одну з наявних гілок графа. Для часткового схвалення портфель
    перераховується і подається як HITL Modify.
    """
    if not any(accepted):
        return "reject", None
    if all(accepted):
        return "approve", None

    proposed = merge_partial_approval(current, actions, accepted)
    positions: list[dict[str, Any]] = [
        {"symbol": position.symbol, "weight": position.weight}
        for position in proposed.positions
    ]
    return "modify", positions


def preview_partial(
    current: Portfolio,
    actions: Sequence[dict[str, Any]],
    accepted: Sequence[bool],
    metrics: dict[str, AssetMetrics] | None = None,
) -> str:
    """Текстовий прев'ю результату часткового схвалення (для показу людині)."""
    del metrics  # зарезервовано для розширеного прев'ю
    try:
        proposed = merge_partial_approval(current, actions, accepted)
    except InvalidPortfolioError as error:
        return f"Комбінація неможлива: {error}"

    before = current.as_mapping()
    lines = ["Портфель після схвалених дій:"]
    for symbol, weight in sorted(proposed.as_mapping().items()):
        was = before.get(symbol, 0.0)
        lines.append(f"  {symbol:<6} {was * 100:5.1f}% -> {weight * 100:5.1f}%")
    for symbol, weight in sorted(before.items()):
        if symbol not in proposed.as_mapping():
            lines.append(f"  {symbol:<6} {weight * 100:5.1f}% -> продано")
    return "\n".join(lines)
