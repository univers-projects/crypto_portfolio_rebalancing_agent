"""Реєстр tools, доступних агентам.

Ризиковий `mock_execute_rebalance` навмисно НЕ входить у набір executor-а:
він викликається лише вузлом виконання після HITL-підтвердження.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.rag.knowledge_base import knowledge_search
from app.tools.market_tools import (
    calculate_asset_metrics,
    get_market_data,
    get_top_liquid_assets,
)
from app.tools.portfolio_tools import evaluate_portfolio, mock_execute_rebalance

# Аналітичні tools — безпечні, доступні ReAct-агенту без обмежень
ANALYSIS_TOOLS: tuple[BaseTool, ...] = (
    get_top_liquid_assets,
    get_market_data,
    calculate_asset_metrics,
    evaluate_portfolio,
    knowledge_search,
)

# Ризикові tools — тільки після human approval
RISKY_TOOLS: tuple[BaseTool, ...] = (mock_execute_rebalance,)

ALL_TOOLS: tuple[BaseTool, ...] = ANALYSIS_TOOLS + RISKY_TOOLS
