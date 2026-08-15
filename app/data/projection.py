"""Прогноз вартості портфеля на горизонт уперед.

Навіщо окремий шар
------------------
`decision_engine` ранжує кандидатів за історичною risk-adjusted якістю. Це
коректний спосіб *порівняти* набори активів, але сам по собі він нічого не
каже користувачу про майбутнє. Питання людини звучить «що буде далі».

Як оцінюється очікувана дохідність
----------------------------------
Не простим усередненням історії. Помилка такої оцінки спадає лише як sqrt(T),
тому навіть двох років бракує, щоб відрізнити активи один від одного за
дохідністю. Натомість очікувана дохідність збирається з двох джерел:

    очікувана = w * власна історія + (1 - w) * плата за ризик
    плата за ризик = risk_free_rate + risk_premium_per_vol * волатильність
    w = T / (T + estimation_prior_days)

Вага власної історії зростає з довжиною вибірки: на 180 днях w = 0.14,
на 720 днях w = 0.40.

Це не обережність заради обережності — така оцінка **точніша**. На mock-universe,
де істинний дрейф кожного активу відомий, RMSE оцінки:

    180 днів, сира історія            95.6%
    720 днів, сира історія            54.8%
    720 днів + усадка (w = 0.40)      29.9%

Волатильність, на відміну від дохідності, оцінюється з історії надійно
(відносна похибка 3.2% на 720 днях), тому вона береться без жодних поправок.

Модель розподілу — геометричний броунівський рух. Результат навмисно подається
діапазоном: медіана, песимістичний і оптимістичний перцентилі та ймовірність
завершити період у мінусі.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.data.analytics import compute_asset_metrics, evaluate_portfolio_metrics
from app.domain.errors import DomainError
from app.domain.schemas import Portfolio

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.0


@dataclass(frozen=True)
class Projection:
    """Прогноз вартості на заданий горизонт."""

    horizon_days: int
    start_value: float
    median_value: float
    downside_value: float
    upside_value: float
    loss_probability: float
    expected_annual_return: float
    volatility: float
    estimation_weight: float
    sample_days: int
    confidence: float

    @property
    def median_gain(self) -> float:
        return self.median_value - self.start_value

    def as_dict(self) -> dict[str, Any]:
        """Компактне подання для передачі в LLM."""
        return {
            "horizon_days": self.horizon_days,
            "start_value": round(self.start_value, 2),
            "median_value": round(self.median_value, 2),
            "median_gain": round(self.median_gain, 2),
            "downside_value": round(self.downside_value, 2),
            "upside_value": round(self.upside_value, 2),
            "loss_probability": round(self.loss_probability, 4),
            "expected_annual_return": round(self.expected_annual_return, 4),
            "volatility": round(self.volatility, 4),
            "estimated_on_days": self.sample_days,
            "own_history_weight": round(self.estimation_weight, 3),
        }


def estimation_weight(sample_days: int | None = None) -> float:
    """Яку вагу має власна історія активу проти базової плати за ризик."""
    settings = get_settings()
    days = sample_days or settings.estimation_lookback_days
    total = days + settings.estimation_prior_days
    return days / total if total else 1.0


def baseline_return(volatility: float) -> float:
    """Скільки актив «має» приносити за свій рівень ризику."""
    settings = get_settings()
    return settings.risk_free_rate + settings.risk_premium_per_vol * volatility


def expected_annual_return(
    historical_annual: float, volatility: float, sample_days: int | None = None
) -> float:
    """Очікувана річна дохідність: власна історія, підтягнута до плати за ризик."""
    weight = estimation_weight(sample_days)
    return weight * historical_annual + (1 - weight) * baseline_return(volatility)


def project(
    expected_return: float,
    volatility: float,
    start_value: float,
    horizon_days: int | None = None,
    sample_days: int | None = None,
) -> Projection:
    """Розподіл вартості на горизонт за вже оціненими параметрами.

    `expected_return` має бути *очікуваною* річною дохідністю, а не сирою
    історичною: усадка виконується в `expected_annual_return`.
    """
    settings = get_settings()
    horizon = horizon_days or settings.projection_horizon_days
    days = sample_days or settings.estimation_lookback_days

    years = horizon / DAYS_PER_YEAR
    drift = expected_return * years
    sigma = volatility * math.sqrt(years)
    # Логарифмічний дрейф: віднімаємо половину дисперсії, щоб математичне
    # сподівання простої дохідності дорівнювало drift
    log_drift = drift - 0.5 * sigma**2

    common = {
        "horizon_days": horizon,
        "start_value": start_value,
        "expected_annual_return": expected_return,
        "volatility": volatility,
        "estimation_weight": estimation_weight(days),
        "sample_days": days,
        "confidence": settings.projection_confidence,
    }

    if sigma <= 0:
        median = start_value * math.exp(log_drift)
        return Projection(
            median_value=median,
            downside_value=median,
            upside_value=median,
            loss_probability=0.0 if log_drift >= 0 else 1.0,
            **common,  # type: ignore[arg-type]
        )

    z = _z_score(settings.projection_confidence)
    return Projection(
        median_value=start_value * math.exp(log_drift),
        downside_value=start_value * math.exp(log_drift - z * sigma),
        upside_value=start_value * math.exp(log_drift + z * sigma),
        loss_probability=_normal_cdf(-log_drift / sigma),
        **common,  # type: ignore[arg-type]
    )


def project_asset(symbol: str, start_value: float) -> Projection | None:
    """Прогноз для одного активу; None, якщо даних бракує."""
    days = get_settings().estimation_lookback_days
    try:
        metrics = compute_asset_metrics(symbol, days)
    except DomainError as error:
        logger.debug("Прогноз для %s недоступний: %s", symbol, error)
        return None
    expected = expected_annual_return(metrics.annualized_return, metrics.volatility, days)
    return project(expected, metrics.volatility, start_value, sample_days=days)


def project_portfolio(
    weights: Mapping[str, float], start_value: float
) -> Projection | None:
    """Прогноз для набору ваг.

    Очікувана дохідність збирається поактивно і зважується; волатильність
    береться з портфельних метрик, тому враховує кореляції — саме тут
    проявляється виграш від диверсифікації.
    """
    if not weights:
        return None

    days = get_settings().estimation_lookback_days
    try:
        portfolio_metrics = evaluate_portfolio_metrics(
            Portfolio(positions=dict(weights)),  # type: ignore[arg-type]
            days,
        )
        expected = 0.0
        for symbol, weight in weights.items():
            metrics = compute_asset_metrics(symbol, days)
            expected += float(weight) * expected_annual_return(
                metrics.annualized_return, metrics.volatility, days
            )
    except (DomainError, ValueError) as error:
        logger.debug("Прогноз портфеля недоступний: %s", error)
        return None

    return project(
        expected, portfolio_metrics.volatility, start_value, sample_days=days
    )


def _normal_cdf(value: float) -> float:
    """Функція розподілу стандартної нормальної величини."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _z_score(confidence: float) -> float:
    """Квантиль нормального розподілу для двостороннього інтервалу.

    Значення табличні: обчислювати обернену функцію розподілу заради кількох
    можливих рівнів довіри було б надлишковим.
    """
    table = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    if confidence in table:
        return table[confidence]
    closest = min(table, key=lambda level: abs(level - confidence))
    return table[closest]
