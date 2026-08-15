"""Доменні tools для роботи з ринковими даними та метриками активів.

Tools 1-3 з рекомендованого набору:
  * get_top_liquid_assets
  * get_market_data
  * calculate_asset_metrics
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from app.data.analytics import compute_asset_metrics
from app.data.market_data import all_specs, get_asset_spec, get_price_history
from app.domain.errors import DomainError
from app.tools.base import ToolPayload, dump, failure, success, tool_contract
from app.tools.schemas import AssetMetricsInput, MarketDataInput, TopLiquidAssetsInput

# Скільки останніх цін повертати LLM: повний ряд роздуває контекст без користі
PRICE_PREVIEW_POINTS = 10


@tool_contract("get_top_liquid_assets")
def _get_top_liquid_assets(**kwargs: Any) -> ToolPayload:
    params = TopLiquidAssetsInput(**kwargs)

    eligible = []
    excluded: list[dict[str, str]] = []
    for spec in all_specs():
        if params.exclude_stablecoins and spec.is_stablecoin:
            excluded.append({"symbol": spec.symbol, "reason": "STABLECOIN"})
            continue
        if spec.history_days < params.min_history_days:
            excluded.append({"symbol": spec.symbol, "reason": "INSUFFICIENT_HISTORY"})
            continue
        if not spec.data_quality_ok:
            excluded.append({"symbol": spec.symbol, "reason": "BAD_DATA_QUALITY"})
            continue
        eligible.append(spec)

    eligible.sort(key=lambda spec: spec.daily_volume_usd, reverse=True)
    selected = eligible[: params.limit]

    return success(
        {
            "assets": [
                {
                    "symbol": spec.symbol,
                    "name": spec.name,
                    "liquidity_rank": index + 1,
                    "avg_daily_volume_usd": spec.daily_volume_usd,
                    "history_days": spec.history_days,
                }
                for index, spec in enumerate(selected)
            ],
            "count": len(selected),
            "excluded": excluded,
            "filters": params.model_dump(),
        }
    )


@tool_contract("get_market_data")
def _get_market_data(**kwargs: Any) -> ToolPayload:
    params = MarketDataInput(**kwargs)
    spec = get_asset_spec(params.symbol)
    prices = get_price_history(params.symbol, params.lookback_days)

    return success(
        {
            "symbol": spec.symbol,
            "name": spec.name,
            "lookback_days": params.lookback_days,
            "data_points": len(prices),
            "first_price": round(prices[0], 6),
            "last_price": round(prices[-1], 6),
            "period_return": round((prices[-1] / prices[0]) - 1.0, 6),
            "avg_daily_volume_usd": spec.daily_volume_usd,
            "recent_prices": [round(price, 6) for price in prices[-PRICE_PREVIEW_POINTS:]],
        }
    )


@tool_contract("calculate_asset_metrics")
def _calculate_asset_metrics(**kwargs: Any) -> ToolPayload:
    params = AssetMetricsInput(**kwargs)

    metrics: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for symbol in params.symbols:
        try:
            metrics.append(dump(compute_asset_metrics(symbol, params.lookback_days)))
        except DomainError as error:
            # Часткова помилка не валить увесь виклик: replanner побачить failed
            # і зможе виключити проблемний актив із кандидатів.
            failed.append({"symbol": symbol, "error_code": error.code, "message": error.message})

    if not metrics:
        codes = {item["error_code"] for item in failed}
        code = codes.pop() if len(codes) == 1 else "NO_METRICS_AVAILABLE"
        return failure(code, f"Не вдалося порахувати метрики для жодного активу: {failed}")

    return success(
        {
            "lookback_days": params.lookback_days,
            "metrics": metrics,
            "failed": failed,
            "computed_count": len(metrics),
        }
    )


get_top_liquid_assets = StructuredTool.from_function(
    func=_get_top_liquid_assets,
    name="get_top_liquid_assets",
    description=(
        "Повертає список найбільш ліквідних криптоактивів (universe) за середнім денним "
        "обсягом торгів. Автоматично відсіює stablecoins, активи з короткою історією та "
        "активи з некоректними даними. Використовуй ЦЕЙ tool, щоб дізнатися, які активи "
        "взагалі дозволено розглядати як кандидатів."
    ),
    args_schema=TopLiquidAssetsInput,
)

get_market_data = StructuredTool.from_function(
    func=_get_market_data,
    name="get_market_data",
    description=(
        "Повертає історичні (mock) ринкові дані для ОДНОГО активу: діапазон цін, "
        "дохідність за період та обсяг. Використовуй, коли потрібні сирі цінові дані. "
        "Може повернути помилку INSUFFICIENT_HISTORY, якщо історії замало — тоді актив "
        "слід виключити з кандидатів."
    ),
    args_schema=MarketDataInput,
)

calculate_asset_metrics = StructuredTool.from_function(
    func=_calculate_asset_metrics,
    name="calculate_asset_metrics",
    description=(
        "Рахує risk/performance метрики для СПИСКУ активів одразу: total return, "
        "annualized return, volatility, max drawdown, trend strength, sharpe-like. "
        "Це основний tool для порівняння активів між собою. Передавай усі цікаві "
        "тікери одним викликом, а не по одному."
    ),
    args_schema=AssetMetricsInput,
)
