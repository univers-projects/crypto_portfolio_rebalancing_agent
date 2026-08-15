"""Тести валідації Pydantic v2 схем (вимога: щонайменше 5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.schemas import (
    Plan,
    PlanStep,
    Portfolio,
    PortfolioDecision,
    Position,
    RebalanceAction,
    ReplanDecision,
)
from app.tools.schemas import (
    AssetMetricsInput,
    EvaluatePortfolioInput,
    KnowledgeSearchInput,
    MarketDataInput,
    MockExecuteRebalanceInput,
    TopLiquidAssetsInput,
)

# --- symbol ---


@pytest.mark.parametrize("bad_symbol", ["", "b", "BTC-USD", "toolongsymbolname", "B TC", "$$$"])
def test_rejects_invalid_symbol(bad_symbol: str) -> None:
    with pytest.raises(ValidationError):
        Position(symbol=bad_symbol, weight=0.5)


def test_normalizes_symbol_to_uppercase() -> None:
    assert Position(symbol="  btc ", weight=0.5).symbol == "BTC"


# --- limit / lookback ---


@pytest.mark.parametrize("bad_limit", [0, -1, -100, 101])
def test_rejects_non_positive_limit(bad_limit: int) -> None:
    with pytest.raises(ValidationError):
        TopLiquidAssetsInput(limit=bad_limit)


@pytest.mark.parametrize("bad_lookback", [0, -5, 10, 5000])
def test_rejects_invalid_lookback(bad_lookback: int) -> None:
    with pytest.raises(ValidationError):
        MarketDataInput(symbol="BTC", lookback_days=bad_lookback)


def test_market_data_input_accepts_valid_values() -> None:
    params = MarketDataInput(symbol="eth", lookback_days=180)
    assert params.symbol == "ETH"
    assert params.lookback_days == 180


# --- ваги позицій ---


@pytest.mark.parametrize("bad_weight", [-0.01, -1.0])
def test_rejects_negative_weight(bad_weight: float) -> None:
    with pytest.raises(ValidationError):
        Position(symbol="BTC", weight=bad_weight)


@pytest.mark.parametrize("bad_weight", [1.01, 1.5, 100.0])
def test_rejects_weight_above_one_hundred_percent(bad_weight: float) -> None:
    with pytest.raises(ValidationError):
        Position(symbol="BTC", weight=bad_weight)


# --- склад портфеля ---


def test_rejects_portfolio_with_more_than_five_assets() -> None:
    with pytest.raises(ValidationError):
        Portfolio(
            positions=(
                Position(symbol="BTC", weight=0.2),
                Position(symbol="ETH", weight=0.2),
                Position(symbol="SOL", weight=0.2),
                Position(symbol="BNB", weight=0.2),
                Position(symbol="ADA", weight=0.1),
                Position(symbol="LINK", weight=0.1),
            )
        )


def test_rejects_portfolio_whose_weights_do_not_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="Сума ваг"):
        Portfolio(positions={"BTC": 0.5, "ETH": 0.3})  # type: ignore[arg-type]


def test_rejects_portfolio_weights_above_one_hundred_percent() -> None:
    with pytest.raises(ValidationError, match="Сума ваг"):
        Portfolio(positions={"BTC": 0.7, "ETH": 0.6})  # type: ignore[arg-type]


def test_rejects_duplicate_assets() -> None:
    with pytest.raises(ValidationError, match="Дубльовані"):
        Portfolio(
            positions=(
                Position(symbol="BTC", weight=0.5),
                Position(symbol="btc", weight=0.5),
            )
        )


def test_rejects_empty_portfolio() -> None:
    with pytest.raises(ValidationError):
        Portfolio(positions=())


def test_accepts_three_asset_portfolio() -> None:
    """Портфель не зобов'язаний заповнювати всі 5 слотів."""
    portfolio = Portfolio(positions={"BTC": 0.5, "ETH": 0.35, "SOL": 0.15})  # type: ignore[arg-type]
    assert len(portfolio.positions) == 3
    assert portfolio.weight_of("BTC") == 0.5
    assert portfolio.weight_of("DOGE") == 0.0


# --- допустима allocation у tool-схемах ---


def test_evaluate_portfolio_rejects_invalid_allocation() -> None:
    with pytest.raises(ValidationError, match="Сума ваг"):
        EvaluatePortfolioInput(
            positions=[{"symbol": "BTC", "weight": 0.4}, {"symbol": "ETH", "weight": 0.4}]
        )


def test_evaluate_portfolio_rejects_more_than_five_positions() -> None:
    with pytest.raises(ValidationError):
        EvaluatePortfolioInput(
            positions=[
                {"symbol": symbol, "weight": weight}
                for symbol, weight in zip(
                    ["BTC", "ETH", "SOL", "BNB", "ADA", "LINK"],
                    [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
                    strict=True,
                )
            ]
        )


def test_evaluate_portfolio_rejects_zero_weight() -> None:
    with pytest.raises(ValidationError):
        EvaluatePortfolioInput(positions=[{"symbol": "BTC", "weight": 0.0}])


# --- дії ребалансу ---


def test_rebalance_action_rejects_inconsistent_increase() -> None:
    with pytest.raises(ValidationError, match="INCREASE"):
        RebalanceAction(action="INCREASE", symbol="BTC", from_weight=0.4, to_weight=0.3)


def test_rebalance_action_rejects_sell_with_nonzero_target() -> None:
    with pytest.raises(ValidationError, match="SELL"):
        RebalanceAction(action="SELL", symbol="AVAX", from_weight=0.15, to_weight=0.05)


# --- фінальне рішення ---


def test_hold_decision_cannot_contain_actions() -> None:
    with pytest.raises(ValidationError, match="HOLD"):
        PortfolioDecision(
            decision="HOLD",
            reasoning="Портфель уже достатньо якісний",
            current_portfolio=[Position(symbol="BTC", weight=1.0)],
            actions=[RebalanceAction(action="SELL", symbol="BTC", from_weight=1.0, to_weight=0.0)],
        )


def test_rebalance_decision_requires_actions() -> None:
    with pytest.raises(ValidationError, match="REBALANCE"):
        PortfolioDecision(
            decision="REBALANCE",
            reasoning="Знайдено кращу allocation",
            current_portfolio=[Position(symbol="BTC", weight=1.0)],
            proposed_portfolio=[Position(symbol="ETH", weight=1.0)],
            actions=[],
        )


# --- structured outputs планувальника ---


def test_plan_rejects_duplicate_step_ids() -> None:
    with pytest.raises(ValidationError, match="step_id"):
        Plan(
            goal="Проаналізувати портфель",
            steps=[
                PlanStep(step_id=1, description="Отримати universe"),
                PlanStep(step_id=1, description="Порахувати метрики"),
            ],
        )


def test_replan_revise_requires_steps() -> None:
    with pytest.raises(ValidationError, match="revise"):
        ReplanDecision(action="revise", reasoning="Дані недоступні", revised_steps=[])


def test_replan_normalizes_dropped_assets() -> None:
    decision = ReplanDecision(
        action="continue", reasoning="План валідний", dropped_assets=["badq", " newx "]
    )
    assert decision.dropped_assets == ["BADQ", "NEWX"]


# --- інші tool-схеми ---


def test_knowledge_search_rejects_short_query() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSearchInput(query="ab")


def test_asset_metrics_input_deduplicates_symbols() -> None:
    params = AssetMetricsInput(symbols=["btc", "BTC", "eth"])
    assert params.symbols == ["BTC", "ETH"]


def test_mock_execute_rejects_target_not_summing_to_one() -> None:
    with pytest.raises(ValidationError, match="цільових ваг"):
        MockExecuteRebalanceInput(
            operations=[{"action": "BUY", "symbol": "BTC", "to_weight": 0.5}],
            target_positions=[{"symbol": "BTC", "weight": 0.5}],
            approval_token="approve",
        )


def test_schemas_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MarketDataInput(symbol="BTC", lookback_days=180, unexpected="x")  # type: ignore[call-arg]
