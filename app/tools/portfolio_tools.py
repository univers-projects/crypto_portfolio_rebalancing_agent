"""Доменні tools для оцінки портфеля та mock-виконання ребалансу.

Tools 4-5 з рекомендованого набору:
  * evaluate_portfolio
  * mock_execute_rebalance  (ризиковий — тільки після HITL approval)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import StructuredTool

from app.config import get_settings
from app.data.analytics import evaluate_portfolio_metrics
from app.data.portfolio_store import load_portfolio, save_portfolio
from app.domain.errors import ApprovalRequiredError
from app.domain.schemas import Portfolio, Position
from app.tools.base import ToolPayload, dump, success, tool_contract
from app.tools.schemas import EvaluatePortfolioInput, MockExecuteRebalanceInput


def _to_portfolio(positions: list[Any]) -> Portfolio:
    """Зібрати доменний Portfolio зі списку позицій tool-схеми."""
    return Portfolio(
        positions=tuple(
            Position(symbol=position.symbol, weight=position.weight) for position in positions
        )
    )


@tool_contract("evaluate_portfolio")
def _evaluate_portfolio(**kwargs: Any) -> ToolPayload:
    params = EvaluatePortfolioInput(**kwargs)
    portfolio = _to_portfolio(params.positions)
    metrics = evaluate_portfolio_metrics(portfolio, params.lookback_days)

    settings = get_settings()
    return success(
        {
            "portfolio": portfolio.as_mapping(),
            "asset_count": len(portfolio.positions),
            "lookback_days": params.lookback_days,
            "metrics": dump(metrics),
            "policy": {
                "max_portfolio_assets": settings.max_portfolio_assets,
                "minimum_improvement_score": settings.minimum_improvement_score,
            },
        }
    )


@tool_contract("mock_execute_rebalance")
def _mock_execute_rebalance(**kwargs: Any) -> ToolPayload:
    params = MockExecuteRebalanceInput(**kwargs)

    # Друга лінія захисту: навіть якщо LLM викличе tool самостійно,
    # без явного approve він нічого не виконає.
    if params.approval_token != "approve":
        raise ApprovalRequiredError(
            "Ребаланс не виконано: потрібне явне підтвердження людини "
            f"(отримано approval_token='{params.approval_token}')"
        )

    before = load_portfolio()
    after = _to_portfolio(params.target_positions)

    executed = [
        {
            "action": operation.action,
            "symbol": operation.symbol,
            "from_weight": before.weight_of(operation.symbol),
            "to_weight": operation.to_weight,
            "status": "FILLED",
            # Явно позначаємо, що угода фіктивна
            "order_id": f"MOCK-{operation.symbol}-{index + 1}",
            "simulated": True,
        }
        for index, operation in enumerate(params.operations)
    ]

    if not params.dry_run:
        save_portfolio(after)

    return success(
        {
            "executed_operations": executed,
            "operations_count": len(executed),
            "portfolio_before": before.as_mapping(),
            "portfolio_after": after.as_mapping(),
            "persisted": not params.dry_run,
            "executed_at": datetime.now(UTC).isoformat(),
            "note": "Це mock-виконання. Жодних реальних угод або коштів не задіяно.",
        }
    )


evaluate_portfolio = StructuredTool.from_function(
    func=_evaluate_portfolio,
    name="evaluate_portfolio",
    description=(
        "Оцінює портфель (поточний або запропонований) як єдине ціле: portfolio return, "
        "volatility, max drawdown, diversification score, concentration, risk score та "
        "підсумковий quality_score. Використовуй, щоб порівняти поточний портфель із "
        "альтернативним. Сума ваг має дорівнювати 1.0, максимум 5 активів."
    ),
    args_schema=EvaluatePortfolioInput,
)

mock_execute_rebalance = StructuredTool.from_function(
    func=_mock_execute_rebalance,
    name="mock_execute_rebalance",
    description=(
        "РИЗИКОВИЙ TOOL. Імітує виконання операцій ребалансу (BUY/SELL/INCREASE/REDUCE) "
        "і оновлює збережений портфель. Жодних реальних угод не виконується. "
        "Викликати ЛИШЕ після того, як людина явно підтвердила план: параметр "
        "approval_token має дорівнювати 'approve', інакше tool поверне APPROVAL_REQUIRED."
    ),
    args_schema=MockExecuteRebalanceInput,
)
