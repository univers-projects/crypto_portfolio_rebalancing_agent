"""Локальне сховище портфеля (mock persistence через JSON-файл).

Реальні біржові рахунки не використовуються. Це імітація "поточного стану
користувача", який агент читає на початку циклу і оновлює після approve.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.domain.errors import InvalidPortfolioError
from app.domain.schemas import Portfolio

# Портфель за замовчуванням, якщо файл ще не створено
DEFAULT_PORTFOLIO: dict[str, float] = {
    "BTC": 0.40,
    "ETH": 0.25,
    "SOL": 0.20,
    "AVAX": 0.15,
}

# Стартовий портфель інтерактивного режиму.
#
# Свідомо неоптимальний: перевантажений BTC/ETH і тримає AVAX з відʼємним
# дрейфом, тому детермінований decision_engine на повному universe стабільно
# видає REBALANCE (net ≈ +2.05 при порозі 0.15, turnover ≈ 74%) із пʼятьма
# діями по пʼяти активах. Це дає передбачуваний матеріал для показу HITL.
# Вердикт не підроблюється: він рахується тією ж політикою, що й завжди.
DEMO_PORTFOLIO: dict[str, float] = dict(DEFAULT_PORTFOLIO)


def _resolve_path(path: Path | None = None) -> Path:
    resolved = path or get_settings().portfolio_state_path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_portfolio(path: Path | None = None) -> Portfolio:
    """Прочитати поточний портфель; за відсутності файлу — створити дефолтний."""
    target = _resolve_path(path)
    if not target.exists():
        portfolio = Portfolio(positions=DEFAULT_PORTFOLIO)  # type: ignore[arg-type]
        save_portfolio(portfolio, target)
        return portfolio

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        return Portfolio(positions=payload["positions"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
        raise InvalidPortfolioError(
            f"Не вдалося прочитати портфель з {target}: {error}"
        ) from error


def save_portfolio(portfolio: Portfolio, path: Path | None = None) -> None:
    """Зберегти портфель у JSON із міткою часу оновлення."""
    target = _resolve_path(path)
    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        "positions": [position.model_dump() for position in portfolio.positions],
    }
    try:
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as error:
        raise InvalidPortfolioError(f"Не вдалося зберегти портфель: {error}") from error


def reset_portfolio(path: Path | None = None) -> Portfolio:
    """Повернути сховище до стартового стану (використовується у тестах і demo)."""
    portfolio = Portfolio(positions=DEFAULT_PORTFOLIO)  # type: ignore[arg-type]
    save_portfolio(portfolio, path)
    return portfolio


def seed_demo_portfolio(path: Path | None = None) -> Portfolio:
    """Записати стартовий портфель інтерактивного режиму.

    Викликається на старті чат-сесії, щоб кожен запуск починався з однакового
    стану і давав відтворюваний REBALANCE для демонстрації HITL.
    """
    portfolio = Portfolio(positions=DEMO_PORTFOLIO)  # type: ignore[arg-type]
    save_portfolio(portfolio, path)
    return portfolio
