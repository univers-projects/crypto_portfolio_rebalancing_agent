"""Спільні фікстури pytest.

Кожен тест працює з ізольованим тимчасовим сховищем портфеля і чекпойнтів,
щоб запуски не впливали один на одного.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.config import get_settings
from app.data.portfolio_store import reset_portfolio, save_portfolio
from app.domain.schemas import Portfolio
from app.observability import trajectory


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path) -> Iterator[None]:
    """Перенаправити портфель, чекпойнти, trajectory і автономію у tmp_path."""
    settings = get_settings()
    original = (
        settings.portfolio_state_path,
        settings.sqlite_checkpoint_path,
        settings.trajectory_log_path,
        settings.autonomy_state_path,
    )
    settings.portfolio_state_path = tmp_path / "portfolio.json"
    settings.sqlite_checkpoint_path = tmp_path / "checkpoints.sqlite"
    settings.trajectory_log_path = tmp_path / "trajectory.jsonl"
    settings.autonomy_state_path = tmp_path / "autonomy.json"
    trajectory.reset()
    reset_portfolio()

    yield

    (
        settings.portfolio_state_path,
        settings.sqlite_checkpoint_path,
        settings.trajectory_log_path,
        settings.autonomy_state_path,
    ) = original
    trajectory.reset()


@pytest.fixture
def default_portfolio() -> Portfolio:
    """Стартовий портфель BTC/ETH/SOL/AVAX."""
    return reset_portfolio()


@pytest.fixture
def set_portfolio() -> Any:
    """Фабрика: записати довільний портфель у сховище."""

    def _set(mapping: dict[str, float]) -> Portfolio:
        portfolio = Portfolio(positions=mapping)  # type: ignore[arg-type]
        save_portfolio(portfolio)
        return portfolio

    return _set


@pytest.fixture
def candidate_symbols() -> list[str]:
    """Стабільний набір кандидатів для тестів рішення."""
    return ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "TRX"]
