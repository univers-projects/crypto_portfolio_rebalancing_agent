"""Тести доменних tools: success/error відповіді та стандартний JSON-контракт."""

from __future__ import annotations

from typing import Any

import pytest

from app.data.portfolio_store import load_portfolio
from app.rag.knowledge_base import knowledge_search
from app.tools.market_tools import (
    calculate_asset_metrics,
    get_market_data,
    get_top_liquid_assets,
)
from app.tools.portfolio_tools import evaluate_portfolio, mock_execute_rebalance

ALL_TOOL_OBJECTS = [
    get_top_liquid_assets,
    get_market_data,
    calculate_asset_metrics,
    evaluate_portfolio,
    mock_execute_rebalance,
    knowledge_search,
]


def assert_contract(response: Any) -> None:
    """Перевірити відповідність стандартному JSON-контракту."""
    assert isinstance(response, dict)
    assert response["status"] in {"success", "error"}
    if response["status"] == "success":
        assert isinstance(response["data"], dict)
        assert "error" not in response
    else:
        error = response["error"]
        assert set(error) == {"code", "message"}
        assert isinstance(error["code"], str) and error["code"]
        assert isinstance(error["message"], str) and error["message"]
        assert "data" not in response


# --- контракт і описи ---


@pytest.mark.parametrize("tool", ALL_TOOL_OBJECTS, ids=lambda tool: tool.name)
def test_every_tool_has_description_and_schema(tool: Any) -> None:
    assert tool.description and len(tool.description) > 40
    assert tool.args_schema is not None


# --- get_top_liquid_assets ---


def test_top_liquid_assets_success() -> None:
    response = get_top_liquid_assets.invoke(
        {"limit": 10, "min_history_days": 180, "exclude_stablecoins": True}
    )
    assert_contract(response)
    assert response["status"] == "success"
    data = response["data"]
    assert data["count"] == 10
    symbols = [asset["symbol"] for asset in data["assets"]]
    assert "BTC" in symbols
    # Stablecoins, молоді активи та зіпсовані дані відсіяні
    assert not {"USDT", "USDC", "DAI", "NEWX", "BADQ"} & set(symbols)


def test_top_liquid_assets_can_include_stablecoins() -> None:
    response = get_top_liquid_assets.invoke({"limit": 30, "exclude_stablecoins": False})
    symbols = [asset["symbol"] for asset in response["data"]["assets"]]
    assert "USDT" in symbols


def test_top_liquid_assets_rejects_invalid_limit() -> None:
    response = get_top_liquid_assets.func(limit=0)
    assert_contract(response)
    assert response["error"]["code"] == "VALIDATION_ERROR"


# --- get_market_data ---


def test_market_data_success() -> None:
    response = get_market_data.invoke({"symbol": "BTC", "lookback_days": 180})
    assert_contract(response)
    data = response["data"]
    assert data["symbol"] == "BTC"
    assert data["data_points"] == 180
    assert len(data["recent_prices"]) == 10


def test_market_data_unknown_symbol_returns_error() -> None:
    response = get_market_data.invoke({"symbol": "NOPE", "lookback_days": 180})
    assert_contract(response)
    assert response["error"]["code"] == "UNKNOWN_SYMBOL"


def test_market_data_insufficient_history_returns_error() -> None:
    """NEWX має лише 45 днів історії — має повернути INSUFFICIENT_HISTORY."""
    response = get_market_data.invoke({"symbol": "NEWX", "lookback_days": 180})
    assert_contract(response)
    assert response["error"]["code"] == "INSUFFICIENT_HISTORY"


def test_market_data_bad_quality_returns_error() -> None:
    response = get_market_data.invoke({"symbol": "BADQ", "lookback_days": 180})
    assert_contract(response)
    assert response["error"]["code"] == "INSUFFICIENT_HISTORY"


def test_market_data_is_deterministic() -> None:
    first = get_market_data.invoke({"symbol": "ETH", "lookback_days": 180})
    second = get_market_data.invoke({"symbol": "ETH", "lookback_days": 180})
    assert first["data"]["last_price"] == second["data"]["last_price"]


# --- calculate_asset_metrics ---


def test_asset_metrics_success() -> None:
    response = calculate_asset_metrics.invoke({"symbols": ["BTC", "ETH"], "lookback_days": 180})
    assert_contract(response)
    data = response["data"]
    assert data["computed_count"] == 2
    metric = data["metrics"][0]
    for field in ("total_return", "volatility", "max_drawdown", "trend_strength"):
        assert field in metric
    assert 0.0 <= metric["max_drawdown"] <= 1.0
    assert -1.0 <= metric["trend_strength"] <= 1.0


def test_asset_metrics_partial_failure_is_reported_not_fatal() -> None:
    """Один зіпсований актив не має валити весь виклик."""
    response = calculate_asset_metrics.invoke({"symbols": ["BTC", "BADQ"], "lookback_days": 180})
    assert response["status"] == "success"
    assert response["data"]["computed_count"] == 1
    assert response["data"]["failed"][0]["error_code"] == "INSUFFICIENT_HISTORY"


def test_asset_metrics_all_failed_returns_error() -> None:
    response = calculate_asset_metrics.invoke({"symbols": ["BADQ"], "lookback_days": 180})
    assert_contract(response)
    assert response["error"]["code"] == "INSUFFICIENT_HISTORY"


# --- evaluate_portfolio ---


def test_evaluate_portfolio_success() -> None:
    response = evaluate_portfolio.invoke(
        {
            "positions": [
                {"symbol": "BTC", "weight": 0.5},
                {"symbol": "ETH", "weight": 0.3},
                {"symbol": "SOL", "weight": 0.2},
            ],
            "lookback_days": 180,
        }
    )
    assert_contract(response)
    metrics = response["data"]["metrics"]
    assert 0.0 <= metrics["diversification_score"] <= 1.0
    assert 0.0 <= metrics["risk_score"] <= 1.0
    assert response["data"]["asset_count"] == 3


def test_evaluate_portfolio_rejects_bad_allocation() -> None:
    response = evaluate_portfolio.func(
        positions=[{"symbol": "BTC", "weight": 0.5}, {"symbol": "ETH", "weight": 0.2}]
    )
    assert_contract(response)
    assert response["error"]["code"] == "VALIDATION_ERROR"


# --- mock_execute_rebalance ---


def test_mock_execute_requires_approval() -> None:
    """Ризиковий tool відмовляє без явного підтвердження людини."""
    response = mock_execute_rebalance.invoke(
        {
            "operations": [{"action": "BUY", "symbol": "LINK", "to_weight": 1.0}],
            "target_positions": [{"symbol": "LINK", "weight": 1.0}],
            "approval_token": "reject",
        }
    )
    assert_contract(response)
    assert response["error"]["code"] == "APPROVAL_REQUIRED"


def test_mock_execute_does_not_change_portfolio_without_approval() -> None:
    before = load_portfolio().as_mapping()
    mock_execute_rebalance.invoke(
        {
            "operations": [{"action": "BUY", "symbol": "LINK", "to_weight": 1.0}],
            "target_positions": [{"symbol": "LINK", "weight": 1.0}],
            "approval_token": "modify",
        }
    )
    assert load_portfolio().as_mapping() == before


def test_mock_execute_success_persists_new_portfolio() -> None:
    response = mock_execute_rebalance.invoke(
        {
            "operations": [
                {"action": "SELL", "symbol": "AVAX", "to_weight": 0.0},
                {"action": "INCREASE", "symbol": "BTC", "to_weight": 0.55},
            ],
            "target_positions": [
                {"symbol": "BTC", "weight": 0.55},
                {"symbol": "ETH", "weight": 0.25},
                {"symbol": "SOL", "weight": 0.20},
            ],
            "approval_token": "approve",
        }
    )
    assert_contract(response)
    data = response["data"]
    assert data["operations_count"] == 2
    assert all(operation["simulated"] for operation in data["executed_operations"])
    assert load_portfolio().as_mapping() == {"BTC": 0.55, "ETH": 0.25, "SOL": 0.20}


def test_mock_execute_dry_run_does_not_persist() -> None:
    before = load_portfolio().as_mapping()
    response = mock_execute_rebalance.invoke(
        {
            "operations": [{"action": "INCREASE", "symbol": "BTC", "to_weight": 0.6}],
            "target_positions": [
                {"symbol": "BTC", "weight": 0.6},
                {"symbol": "ETH", "weight": 0.4},
            ],
            "approval_token": "approve",
            "dry_run": True,
        }
    )
    assert response["data"]["persisted"] is False
    assert load_portfolio().as_mapping() == before


# --- knowledge_search (Agentic RAG) ---


def test_knowledge_search_returns_relevant_document() -> None:
    response = knowledge_search.invoke({"query": "What is max drawdown?", "top_k": 3})
    assert_contract(response)
    titles = [item["title"] for item in response["data"]["results"]]
    assert "Maximum Drawdown" in titles


def test_knowledge_search_finds_turnover_document() -> None:
    response = knowledge_search.invoke({"query": "turnover and transaction costs", "top_k": 2})
    assert response["data"]["results"][0]["topic"] == "turnover"


def test_knowledge_search_rejects_empty_query() -> None:
    response = knowledge_search.func(query="  ")
    assert_contract(response)
    assert response["error"]["code"] == "VALIDATION_ERROR"


def test_knowledge_base_has_at_least_eight_documents() -> None:
    from app.rag.documents import document_count

    assert document_count() >= 8
