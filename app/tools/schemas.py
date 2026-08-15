"""Pydantic v2 схеми вхідних параметрів для доменних tools.

Кожна схема несе `Field(description=...)` — саме ці описи бачить LLM,
коли вирішує, який tool викликати і з якими аргументами.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.schemas import (
    MAX_PORTFOLIO_ASSETS,
    ActionType,
    ApprovalType,
    normalize_symbol,
)

MIN_LOOKBACK_DAYS = 30
MAX_LOOKBACK_DAYS = 720


class TopLiquidAssetsInput(BaseModel):
    """Параметри відбору найбільш ліквідних активів."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(
        default=25,
        gt=0,
        le=100,
        description="Скільки активів повернути (1-100). Типово 25.",
    )
    min_history_days: int = Field(
        default=180,
        ge=MIN_LOOKBACK_DAYS,
        le=MAX_LOOKBACK_DAYS,
        description="Мінімальна довжина цінової історії активу в днях.",
    )
    exclude_stablecoins: bool = Field(
        default=True,
        description="Виключити stablecoins (USDT, USDC, DAI) з universe.",
    )

    @field_validator("limit")
    @classmethod
    def _validate_limit(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("limit має бути додатним числом")
        return value


class MarketDataInput(BaseModel):
    """Параметри запиту історичних (mock) ринкових даних."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(description="Тікер активу, напр. 'BTC'.")
    lookback_days: int = Field(
        default=180,
        ge=MIN_LOOKBACK_DAYS,
        le=MAX_LOOKBACK_DAYS,
        description=f"Глибина історії в днях ({MIN_LOOKBACK_DAYS}-{MAX_LOOKBACK_DAYS}).",
    )

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        return normalize_symbol(value)

    @field_validator("lookback_days")
    @classmethod
    def _validate_lookback(cls, value: int) -> int:
        if value < MIN_LOOKBACK_DAYS:
            raise ValueError(
                f"lookback_days має бути щонайменше {MIN_LOOKBACK_DAYS} для стабільних метрик"
            )
        return value


class AssetMetricsInput(BaseModel):
    """Параметри розрахунку метрик активу."""

    model_config = ConfigDict(extra="forbid")

    symbols: list[str] = Field(
        min_length=1,
        max_length=30,
        description="Список тікерів для розрахунку метрик, напр. ['BTC', 'ETH'].",
    )
    lookback_days: int = Field(
        default=180,
        ge=MIN_LOOKBACK_DAYS,
        le=MAX_LOOKBACK_DAYS,
        description="Глибина історії для розрахунку метрик.",
    )

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, value: list[str]) -> list[str]:
        normalized = [normalize_symbol(symbol) for symbol in value]
        # Дедуплікація зі збереженням порядку
        return list(dict.fromkeys(normalized))


class PortfolioPositionInput(BaseModel):
    """Позиція портфеля у вигляді, зручному для передачі LLM."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(description="Тікер активу.")
    weight: float = Field(
        gt=0.0,
        le=1.0,
        description="Вага у частках одиниці: 0.4 означає 40%.",
    )

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        return normalize_symbol(value)


class EvaluatePortfolioInput(BaseModel):
    """Параметри оцінки портфеля."""

    model_config = ConfigDict(extra="forbid")

    positions: list[PortfolioPositionInput] = Field(
        min_length=1,
        max_length=MAX_PORTFOLIO_ASSETS,
        description=(
            f"Склад портфеля, від 1 до {MAX_PORTFOLIO_ASSETS} позицій. "
            "Сума ваг має дорівнювати 1.0."
        ),
    )
    lookback_days: int = Field(
        default=180,
        ge=MIN_LOOKBACK_DAYS,
        le=MAX_LOOKBACK_DAYS,
        description="Глибина історії для оцінки.",
    )

    @model_validator(mode="after")
    def _validate_positions(self) -> EvaluatePortfolioInput:
        symbols = [position.symbol for position in self.positions]
        if len(set(symbols)) != len(symbols):
            raise ValueError("Портфель містить дубльовані активи")
        total = sum(position.weight for position in self.positions)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Сума ваг має дорівнювати 1.0, отримано {total:.4f}")
        return self


class ExecuteOperationInput(BaseModel):
    """Одна mock-операція для виконання."""

    model_config = ConfigDict(extra="forbid")

    action: ActionType = Field(description="BUY, SELL, INCREASE, REDUCE або REPLACE.")
    symbol: str = Field(description="Тікер активу.")
    to_weight: float = Field(
        ge=0.0, le=1.0, description="Цільова вага після операції (0.0 для SELL)."
    )

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        return normalize_symbol(value)


class MockExecuteRebalanceInput(BaseModel):
    """Параметри mock-виконання ребалансу. Ризикова операція — вимагає HITL."""

    model_config = ConfigDict(extra="forbid")

    operations: list[ExecuteOperationInput] = Field(
        min_length=1,
        max_length=10,
        description="Список операцій до виконання.",
    )
    target_positions: list[PortfolioPositionInput] = Field(
        min_length=1,
        max_length=MAX_PORTFOLIO_ASSETS,
        description="Підсумковий склад портфеля після виконання операцій.",
    )
    approval_token: ApprovalType = Field(
        description=(
            "Має бути рівним 'approve'. Tool відмовляє у виконанні без явного "
            "підтвердження людини."
        )
    )
    dry_run: bool = Field(
        default=False,
        description="Якщо True — лише симуляція без збереження нового портфеля.",
    )

    @model_validator(mode="after")
    def _validate_target(self) -> MockExecuteRebalanceInput:
        total = sum(position.weight for position in self.target_positions)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Сума цільових ваг має дорівнювати 1.0, отримано {total:.4f}")
        return self


class KnowledgeSearchInput(BaseModel):
    """Параметри пошуку у базі знань з портфельного ризик-менеджменту."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=3,
        max_length=500,
        description="Питання природною мовою, напр. 'What is max drawdown?'.",
    )
    top_k: int = Field(default=3, ge=1, le=8, description="Скільки документів повернути.")

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query не може бути порожнім")
        return cleaned
