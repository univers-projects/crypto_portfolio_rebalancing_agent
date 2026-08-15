"""Pydantic v2 схеми домену: портфель, метрики, план, рішення.

Усі структури незмінні там, де це має сенс (`frozen=True`), щоб виключити
випадкову мутацію стану між вузлами графа.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Символ тікера: 2-10 великих латинських літер або цифр
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,10}$")

ActionType = Literal["BUY", "SELL", "INCREASE", "REDUCE", "REPLACE"]
DecisionType = Literal["HOLD", "REBALANCE"]
ApprovalType = Literal["approve", "reject", "modify"]

MAX_PORTFOLIO_ASSETS = 5
# Допуск на похибку округлення при перевірці суми ваг
WEIGHT_SUM_TOLERANCE = 1e-4


def normalize_symbol(value: str) -> str:
    """Привести тікер до канонічного вигляду та перевірити формат."""
    if not isinstance(value, str):
        raise ValueError("symbol має бути рядком")
    cleaned = value.strip().upper()
    if not SYMBOL_PATTERN.match(cleaned):
        raise ValueError(
            f"Некоректний symbol '{value}': очікується 2-10 великих літер/цифр (напр. 'BTC')"
        )
    return cleaned


class Position(BaseModel):
    """Одна позиція портфеля: актив і його вага у частках одиниці."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(description="Тікер активу, напр. 'BTC'")
    weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Вага позиції у частках одиниці (0.4 == 40%)",
    )

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        return normalize_symbol(value)

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        # ge/le вже покривають межі, але явна перевірка дає зрозуміліше повідомлення
        if value < 0:
            raise ValueError("Вага позиції не може бути відʼємною")
        if value > 1:
            raise ValueError("Вага позиції не може перевищувати 100%")
        return round(value, 6)

    @property
    def weight_pct(self) -> float:
        """Вага у відсотках для людиночитаного виводу."""
        return round(self.weight * 100, 2)


class Portfolio(BaseModel):
    """Портфель із 1-5 активів, ваги яких у сумі дорівнюють 100%."""

    model_config = ConfigDict(frozen=True)

    positions: tuple[Position, ...] = Field(
        min_length=1,
        max_length=MAX_PORTFOLIO_ASSETS,
        description="Від 1 до 5 позицій портфеля",
    )

    @field_validator("positions", mode="before")
    @classmethod
    def _coerce_positions(cls, value: object) -> object:
        """Дозволити передавати як список dict, так і dict {symbol: weight}."""
        if isinstance(value, dict):
            return tuple(Position(symbol=k, weight=v) for k, v in value.items())
        return value

    @model_validator(mode="after")
    def _validate_portfolio(self) -> Portfolio:
        if len(self.positions) > MAX_PORTFOLIO_ASSETS:
            raise ValueError(f"Портфель не може містити більше {MAX_PORTFOLIO_ASSETS} активів")

        symbols = [position.symbol for position in self.positions]
        duplicates = {symbol for symbol in symbols if symbols.count(symbol) > 1}
        if duplicates:
            raise ValueError(f"Дубльовані активи у портфелі: {sorted(duplicates)}")

        total = sum(position.weight for position in self.positions)
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"Сума ваг має дорівнювати 100%, отримано {total * 100:.2f}%"
            )

        if any(position.weight <= 0 for position in self.positions):
            raise ValueError("Позиція з нульовою вагою не повинна бути у портфелі")

        return self

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(position.symbol for position in self.positions)

    def weight_of(self, symbol: str) -> float:
        """Вага активу або 0.0, якщо його немає у портфелі."""
        target = normalize_symbol(symbol)
        for position in self.positions:
            if position.symbol == target:
                return position.weight
        return 0.0

    def as_mapping(self) -> dict[str, float]:
        return {position.symbol: position.weight for position in self.positions}

    def render(self) -> str:
        """Людиночитаний вигляд, який використовується у промптах і CLI."""
        lines = [
            f"{position.symbol} {position.weight_pct:.0f}%"
            for position in sorted(self.positions, key=lambda p: -p.weight)
        ]
        return "\n".join(lines)


class AssetMetrics(BaseModel):
    """Розраховані risk/performance характеристики одного активу."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    lookback_days: int = Field(gt=0)
    total_return: float = Field(description="Сукупна дохідність за період, у частках одиниці")
    annualized_return: float
    volatility: float = Field(ge=0.0, description="Річна волатильність")
    max_drawdown: float = Field(
        ge=0.0, le=1.0, description="Максимальна просадка як додатне число (0.35 == -35%)"
    )
    trend_strength: float = Field(
        ge=-1.0, le=1.0, description="Сила тренду: -1 сильний спад, +1 сильний ріст"
    )
    sharpe_like: float = Field(description="Відношення річної дохідності до волатильності")
    avg_daily_volume_usd: float = Field(ge=0.0)

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        return normalize_symbol(value)


class PortfolioMetrics(BaseModel):
    """Агреговані характеристики портфеля."""

    model_config = ConfigDict(frozen=True)

    portfolio_return: float
    annualized_return: float
    volatility: float = Field(ge=0.0)
    max_drawdown: float = Field(ge=0.0, le=1.0)
    sharpe_like: float
    diversification_score: float = Field(
        ge=0.0, le=1.0, description="1.0 — ідеально диверсифіковано, 0.0 — усе в одному активі"
    )
    concentration_hhi: float = Field(ge=0.0, le=1.0, description="Індекс Герфіндаля по вагах")
    risk_score: float = Field(ge=0.0, le=1.0, description="Сукупний ризик, більше — ризикованіше")
    quality_score: float = Field(description="Підсумкова risk-adjusted оцінка портфеля")


class RebalanceAction(BaseModel):
    """Одна запропонована зміна портфеля."""

    model_config = ConfigDict(frozen=True)

    action: ActionType
    symbol: str
    from_weight: float = Field(ge=0.0, le=1.0, default=0.0)
    to_weight: float = Field(ge=0.0, le=1.0, default=0.0)
    rationale: str = Field(default="", max_length=500)

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        return normalize_symbol(value)

    @model_validator(mode="after")
    def _validate_consistency(self) -> RebalanceAction:
        """Тип дії має відповідати напрямку зміни ваги."""
        if self.action == "BUY" and self.to_weight <= 0:
            raise ValueError("BUY має мати додатну цільову вагу")
        if self.action == "SELL" and self.to_weight != 0:
            raise ValueError("SELL має призводити до нульової цільової ваги")
        if self.action == "INCREASE" and self.to_weight <= self.from_weight:
            raise ValueError("INCREASE має збільшувати вагу")
        if self.action == "REDUCE" and self.to_weight >= self.from_weight:
            raise ValueError("REDUCE має зменшувати вагу")
        return self

    def render(self) -> str:
        match self.action:
            case "BUY":
                return f"BUY {self.symbol} with {self.to_weight * 100:.0f}% allocation"
            case "SELL":
                return f"SELL {self.symbol}"
            case "INCREASE" | "REDUCE":
                return (
                    f"{self.action} {self.symbol} from {self.from_weight * 100:.0f}%"
                    f" to {self.to_weight * 100:.0f}%"
                )
            case _:
                return (
                    f"REPLACE {self.symbol} ({self.from_weight * 100:.0f}%"
                    f" -> {self.to_weight * 100:.0f}%)"
                )


class RebalanceProposal(BaseModel):
    """Повна пропозиція ребалансу, яка йде на HITL-підтвердження."""

    model_config = ConfigDict(frozen=True)

    current_portfolio: Portfolio
    proposed_portfolio: Portfolio
    actions: tuple[RebalanceAction, ...] = Field(min_length=1)
    turnover: float = Field(ge=0.0, le=2.0, description="Сума абсолютних змін ваг")
    improvement_score: float
    rationale: str = Field(default="")

    @model_validator(mode="after")
    def _validate_proposal(self) -> RebalanceProposal:
        if self.proposed_portfolio.as_mapping() == self.current_portfolio.as_mapping():
            raise ValueError("Пропозиція не змінює портфель — це має бути HOLD")
        return self

    def render(self) -> str:
        actions = "\n".join(action.render() for action in self.actions)
        return (
            f"Current portfolio:\n{self.current_portfolio.render()}\n\n"
            f"Proposed portfolio:\n{self.proposed_portfolio.render()}\n\n"
            f"Actions:\n{actions}\n\n"
            f"Turnover: {self.turnover * 100:.1f}%\n"
            f"Improvement score: {self.improvement_score:+.4f}"
        )


class PortfolioDecision(BaseModel):
    """Фінальне структуроване рішення агента (structured output LLM)."""

    decision: DecisionType = Field(description="HOLD або REBALANCE")
    reasoning: str = Field(
        min_length=10,
        max_length=2000,
        description="Стисле обґрунтування рішення на основі порахованих метрик",
    )
    current_portfolio: list[Position] = Field(description="Поточний склад портфеля")
    proposed_portfolio: list[Position] | None = Field(
        default=None, description="Запропонований склад; None для HOLD"
    )
    actions: list[RebalanceAction] = Field(
        default_factory=list, description="Список змін; порожній для HOLD"
    )

    @model_validator(mode="after")
    def _validate_decision(self) -> PortfolioDecision:
        if self.decision == "HOLD":
            if self.actions:
                raise ValueError("Рішення HOLD не може містити дій")
            if self.proposed_portfolio:
                raise ValueError("Рішення HOLD не може містити proposed_portfolio")
        else:
            if not self.actions:
                raise ValueError("Рішення REBALANCE має містити хоча б одну дію")
            if not self.proposed_portfolio:
                raise ValueError("Рішення REBALANCE має містити proposed_portfolio")
            if len(self.proposed_portfolio) > MAX_PORTFOLIO_ASSETS:
                raise ValueError(
                    f"Запропонований портфель не може містити більше "
                    f"{MAX_PORTFOLIO_ASSETS} активів"
                )
        return self

    def render(self) -> str:
        header = f"Decision: {self.decision}"
        current = "\n".join(f"{p.symbol} {p.weight_pct:.0f}%" for p in self.current_portfolio)
        if self.decision == "HOLD":
            return (
                f"{header}\n\nCurrent portfolio:\n{current}\n\n"
                f"Reason:\n{self.reasoning}\n\nAction required:\nNone"
            )
        proposed = "\n".join(
            f"{p.symbol} {p.weight_pct:.0f}%" for p in (self.proposed_portfolio or [])
        )
        actions = "\n".join(action.render() for action in self.actions)
        return (
            f"{header}\n\nCurrent portfolio:\n{current}\n\n"
            f"Proposed portfolio:\n{proposed}\n\n"
            f"Actions:\n{actions}\n\nReason:\n{self.reasoning}"
        )


# --- Structured outputs для Plan-and-Execute ---


class PlanStep(BaseModel):
    """Один крок плану, сформульований як задача для ReAct-executor."""

    step_id: int = Field(ge=1, description="Порядковий номер кроку, починаючи з 1")
    description: str = Field(
        min_length=5,
        max_length=400,
        description="Що саме треба зробити. Формулювати як задачу, а не як виклик tool.",
    )


class Plan(BaseModel):
    """План аналізу портфеля, згенерований planner-ом."""

    goal: str = Field(min_length=5, max_length=400, description="Мета аналізу одним реченням")
    steps: list[PlanStep] = Field(
        min_length=1,
        max_length=10,
        description="Впорядкований список кроків аналізу (від 1 до 10)",
    )

    @model_validator(mode="after")
    def _validate_steps(self) -> Plan:
        ids = [step.step_id for step in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("step_id має бути унікальним у межах плану")
        return self


class ReplanDecision(BaseModel):
    """Рішення replanner-а після виконання чергового кроку."""

    action: Literal["continue", "revise", "finish"] = Field(
        description=(
            "continue — план валідний, йти далі; "
            "revise — замінити решту кроків новими; "
            "finish — даних достатньо для фінального рішення"
        )
    )
    reasoning: str = Field(min_length=5, max_length=1000)
    revised_steps: list[PlanStep] = Field(
        default_factory=list,
        description="Нові кроки замість решти плану. Обовʼязково для action='revise'.",
    )
    dropped_assets: list[str] = Field(
        default_factory=list,
        description="Активи, які треба виключити з кандидатів (напр. через INSUFFICIENT_HISTORY)",
    )

    @field_validator("dropped_assets")
    @classmethod
    def _validate_dropped(cls, value: list[str]) -> list[str]:
        return [normalize_symbol(symbol) for symbol in value]

    @model_validator(mode="after")
    def _validate_action(self) -> ReplanDecision:
        if self.action == "revise" and not self.revised_steps:
            raise ValueError("action='revise' вимагає непорожнього revised_steps")
        return self
