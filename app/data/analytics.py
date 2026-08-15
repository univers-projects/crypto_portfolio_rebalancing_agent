"""Чисті аналітичні функції: метрики активів, оцінка портфеля, побудова кандидатів.

Це детермінований числовий шар. LLM тут не бере участі — він лише інтерпретує
результати. Завдяки цьому рішення HOLD/REBALANCE відтворюване і тестоване.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

from app.config import get_settings
from app.data.market_data import (
    TRADING_DAYS_PER_YEAR,
    daily_returns,
    get_asset_spec,
    get_price_history,
)
from app.domain.errors import InsufficientHistoryError
from app.domain.schemas import AssetMetrics, Portfolio, PortfolioMetrics, Position

# Мінімум спостережень, щоб волатильність мала сенс
MIN_OBSERVATIONS = 30


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: Sequence[float]) -> float:
    """Вибіркове стандартне відхилення."""
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _max_drawdown(prices: tuple[float, ...]) -> float:
    """Максимальна просадка як додатне число (0.35 означає -35%)."""
    peak = prices[0]
    worst = 0.0
    for price in prices:
        peak = max(peak, price)
        drawdown = (peak - price) / peak if peak > 0 else 0.0
        worst = max(worst, drawdown)
    return worst


def _trend_strength(prices: tuple[float, ...]) -> float:
    """Сила тренду в діапазоні [-1, 1].

    Рахується як t-подібна статистика нахилу лінійної регресії log-ціни
    за часом, стиснута через tanh. Стійкіша за просте порівняння SMA.
    """
    count = len(prices)
    if count < MIN_OBSERVATIONS:
        return 0.0

    xs = list(range(count))
    ys = [math.log(price) for price in prices]
    mean_x, mean_y = _mean(xs), _mean(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    if variance_x == 0:
        return 0.0

    slope = covariance / variance_x
    residuals = [y - (mean_y + slope * (x - mean_x)) for x, y in zip(xs, ys, strict=True)]
    residual_std = _stdev(residuals)
    if residual_std == 0:
        return 1.0 if slope > 0 else -1.0

    # Нормуємо нахил на шум і масштабуємо на горизонт
    signal = slope * count / residual_std
    return max(-1.0, min(1.0, math.tanh(signal / 3.0)))


def compute_asset_metrics(symbol: str, lookback_days: int) -> AssetMetrics:
    """Порахувати return / volatility / max drawdown / trend strength для активу."""
    prices = get_price_history(symbol, lookback_days)
    if len(prices) < MIN_OBSERVATIONS:
        raise InsufficientHistoryError(
            f"Для '{symbol}' потрібно щонайменше {MIN_OBSERVATIONS} спостережень, "
            f"отримано {len(prices)}"
        )

    spec = get_asset_spec(symbol)
    returns = daily_returns(prices)

    total_return = (prices[-1] / prices[0]) - 1.0
    # Аннуалізація арифметична (mean * 365), а не компаундована: компаундування
    # короткого вікна дає вибухові значення і робить sharpe непорівнюваним.
    annualized_return = _mean(returns) * TRADING_DAYS_PER_YEAR
    volatility = _stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)
    max_drawdown = _max_drawdown(prices)
    sharpe_like = annualized_return / volatility if volatility > 1e-9 else 0.0

    return AssetMetrics(
        symbol=spec.symbol,
        lookback_days=lookback_days,
        total_return=round(total_return, 6),
        annualized_return=round(annualized_return, 6),
        volatility=round(volatility, 6),
        max_drawdown=round(max_drawdown, 6),
        trend_strength=round(_trend_strength(prices), 6),
        sharpe_like=round(sharpe_like, 6),
        avg_daily_volume_usd=spec.daily_volume_usd,
    )


def _correlation(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Коефіцієнт кореляції Пірсона між двома рядами дохідностей."""
    size = min(len(left), len(right))
    if size < 2:
        return 0.0
    left, right = left[-size:], right[-size:]
    mean_left, mean_right = _mean(left), _mean(right)
    covariance = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right)
    )
    return covariance / denominator if denominator > 0 else 0.0


def _portfolio_return_series(
    portfolio: Portfolio, lookback_days: int
) -> tuple[tuple[float, ...], dict[str, tuple[float, ...]]]:
    """Ряд дохідностей портфеля при щоденному ребалансі до цільових ваг."""
    per_asset: dict[str, tuple[float, ...]] = {}
    for position in portfolio.positions:
        prices = get_price_history(position.symbol, lookback_days)
        per_asset[position.symbol] = daily_returns(prices)

    length = min(len(series) for series in per_asset.values())
    combined = []
    for index in range(length):
        combined.append(
            sum(
                position.weight * per_asset[position.symbol][index]
                for position in portfolio.positions
            )
        )
    return tuple(combined), per_asset


def _diversification_score(portfolio: Portfolio, per_asset: dict[str, tuple[float, ...]]) -> float:
    """Диверсифікація враховує і рівномірність ваг, і кореляцію активів."""
    weights = [position.weight for position in portfolio.positions]
    hhi = sum(weight**2 for weight in weights)
    count = len(weights)
    # Нормалізований HHI: 1.0 при рівних вагах, 0.0 при повній концентрації
    weight_evenness = (1.0 - hhi) / (1.0 - 1.0 / count) if count > 1 else 0.0

    if count < 2:
        return round(weight_evenness * 0.5, 6)

    correlations = [
        _correlation(per_asset[left], per_asset[right])
        for left, right in combinations(portfolio.symbols, 2)
    ]
    average_correlation = _mean(correlations)
    # Низька середня кореляція -> вища диверсифікація
    correlation_component = max(0.0, min(1.0, (1.0 - average_correlation) / 1.5))

    score = 0.5 * weight_evenness + 0.5 * correlation_component
    return round(max(0.0, min(1.0, score)), 6)


def evaluate_portfolio_metrics(portfolio: Portfolio, lookback_days: int) -> PortfolioMetrics:
    """Порахувати агреговані метрики та підсумковий quality_score портфеля."""
    returns, per_asset = _portfolio_return_series(portfolio, lookback_days)
    if len(returns) < MIN_OBSERVATIONS:
        raise InsufficientHistoryError(
            f"Недостатньо спільної історії для оцінки портфеля: {len(returns)} днів"
        )

    # Відновлюємо криву капіталу, щоб коректно порахувати просадку портфеля
    equity = [1.0]
    for daily_return in returns:
        equity.append(equity[-1] * (1.0 + daily_return))
    equity_curve = tuple(equity)

    portfolio_return = equity_curve[-1] - 1.0
    # Та сама арифметична аннуалізація, що й для окремих активів
    annualized_return = _mean(returns) * TRADING_DAYS_PER_YEAR
    volatility = _stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)
    max_drawdown = _max_drawdown(equity_curve)
    sharpe_like = annualized_return / volatility if volatility > 1e-9 else 0.0

    diversification = _diversification_score(portfolio, per_asset)
    hhi = sum(position.weight**2 for position in portfolio.positions)

    # risk_score: суміш волатильності, просадки та концентрації, стиснута в [0, 1]
    risk_score = max(
        0.0,
        min(1.0, 0.45 * min(volatility / 1.2, 1.0) + 0.35 * max_drawdown + 0.20 * hhi),
    )

    return PortfolioMetrics(
        portfolio_return=round(portfolio_return, 6),
        annualized_return=round(annualized_return, 6),
        volatility=round(volatility, 6),
        max_drawdown=round(max_drawdown, 6),
        sharpe_like=round(sharpe_like, 6),
        diversification_score=diversification,
        concentration_hhi=round(hhi, 6),
        risk_score=round(risk_score, 6),
        quality_score=round(_quality_score(sharpe_like, max_drawdown, diversification), 6),
    )


def _quality_score(sharpe_like: float, max_drawdown: float, diversification: float) -> float:
    """Єдина скалярна оцінка портфеля для порівняння варіантів.

    Основа — risk-adjusted дохідність, з бонусом за диверсифікацію
    та штрафом за глибину просадки.
    """
    return sharpe_like + 0.30 * diversification - 0.60 * max_drawdown


def compute_turnover(current: Portfolio, proposed: Portfolio) -> float:
    """Turnover як сума абсолютних змін ваг по всіх активах обох портфелів."""
    symbols = set(current.symbols) | set(proposed.symbols)
    return round(
        sum(abs(proposed.weight_of(symbol) - current.weight_of(symbol)) for symbol in symbols),
        6,
    )


def improvement_score(
    current: PortfolioMetrics, candidate: PortfolioMetrics, turnover: float
) -> float:
    """Чисте покращення після врахування вартості turnover.

    Саме це значення порівнюється з `minimum_improvement_score`.
    """
    settings = get_settings()
    raw_gain = candidate.quality_score - current.quality_score
    return round(raw_gain - settings.turnover_cost_per_unit * turnover, 6)


def _allocate_weights(metrics: list[AssetMetrics]) -> Portfolio | None:
    """Побудувати ваги для набору активів: inverse-vol із нахилом на якість.

    Ваги обмежені `min_position_weight` / `max_position_weight` і нормалізовані
    до 100%. Повертає None, якщо коректний розподіл побудувати неможливо.
    """
    settings = get_settings()
    scores = []
    for metric in metrics:
        if metric.volatility <= 1e-9:
            return None
        # Нахил у бік активів із кращим risk-adjusted профілем і позитивним трендом
        tilt = 1.0 + max(-0.5, min(1.0, 0.5 * metric.sharpe_like + 0.5 * metric.trend_strength))
        scores.append(max(tilt, 0.1) / metric.volatility)

    total = sum(scores)
    if total <= 0:
        return None

    weights = [score / total for score in scores]
    # Ітеративно застосовуємо межі ваг і перенормовуємо
    for _ in range(10):
        capped = [
            max(settings.min_position_weight, min(settings.max_position_weight, weight))
            for weight in weights
        ]
        total_capped = sum(capped)
        weights = [weight / total_capped for weight in capped]

    rounded = [round(weight, 4) for weight in weights]
    # Похибку округлення додаємо до найбільшої позиції
    drift = round(1.0 - sum(rounded), 4)
    largest = rounded.index(max(rounded))
    rounded[largest] = round(rounded[largest] + drift, 4)

    try:
        return Portfolio(
            positions=tuple(
                Position(symbol=metric.symbol, weight=weight)
                for metric, weight in zip(metrics, rounded, strict=True)
            )
        )
    except ValueError:
        return None


def asset_rank_score(metric: AssetMetrics) -> float:
    """Композитний бал активу для ранжування кандидатів."""
    return metric.sharpe_like + 0.4 * metric.trend_strength - 0.8 * metric.max_drawdown


def build_candidate_portfolios(
    metrics_by_symbol: dict[str, AssetMetrics],
    current: Portfolio | None = None,
    max_candidates: int = 8,
) -> list[Portfolio]:
    """Згенерувати портфелі-кандидати для порівняння з поточним.

    Кандидати навмисно будуються переважно як інкрементальні зміни поточного
    портфеля (переваження, заміна найслабшого активу, додавання нового). Це
    відповідає реальній практиці ребалансу і тримає turnover у розумних межах.
    Додатково генерується кілька "з нуля" портфелів різного розміру — вони
    зазвичай відсіюються лімітом turnover, але дають базу для порівняння.

    Розмір портфеля змінюється в межах 1..max_portfolio_assets: агент не
    зобов'язаний заповнювати всі 5 слотів.
    """
    settings = get_settings()
    ranked = sorted(metrics_by_symbol.values(), key=asset_rank_score, reverse=True)
    if not ranked:
        return []

    candidates: list[Portfolio] = []
    seen: set[tuple[tuple[str, float], ...]] = set()

    def add(portfolio: Portfolio | None) -> None:
        if portfolio is None:
            return
        key = tuple(sorted(portfolio.as_mapping().items()))
        if key in seen:
            return
        seen.add(key)
        candidates.append(portfolio)

    if current is not None:
        held = [
            metrics_by_symbol[symbol]
            for symbol in current.symbols
            if symbol in metrics_by_symbol
        ]
        if held:
            held_ranked = sorted(held, key=asset_rank_score, reverse=True)
            outsiders = [
                metric for metric in ranked if metric.symbol not in set(current.symbols)
            ]

            # 1. Ті самі активи, переважені під поточний risk-профіль
            add(_allocate_weights(held_ranked))

            # 2. Прибрати найслабший актив (якщо залишиться щонайменше один)
            if len(held_ranked) > 1:
                add(_allocate_weights(held_ranked[:-1]))

            # 3. Додати найкращого кандидата ззовні, якщо є вільний слот
            if outsiders and len(held_ranked) < settings.max_portfolio_assets:
                add(_allocate_weights([*held_ranked, outsiders[0]]))

            # 4. Замінити найслабший актив найкращим кандидатом ззовні
            if outsiders and len(held_ranked) > 1:
                add(_allocate_weights([*held_ranked[:-1], outsiders[0]]))

            # 5. Замінити два найслабші активи двома найкращими кандидатами
            if len(outsiders) > 1 and len(held_ranked) > 2:
                add(_allocate_weights([*held_ranked[:-2], outsiders[0], outsiders[1]]))

    # 6. Портфелі "з нуля" різного розміру — база для порівняння
    for size in range(1, min(settings.max_portfolio_assets, len(ranked)) + 1):
        if len(candidates) >= max_candidates:
            break
        add(_allocate_weights(ranked[:size]))

    return candidates[:max_candidates]
