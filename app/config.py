"""Конфігурація застосунку через pydantic-settings.

Усі значення читаються з .env або змінних середовища.
Жодних хардкоджених секретів у коді.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Корінь проєкту — використовується для шляхів до sqlite/chroma/mock-даних
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Типобезпечні налаштування системи."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (OpenRouter, OpenAI-сумісний API) ---
    openrouter_api_key: str = Field(default="", description="Ключ доступу до OpenRouter")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")

    planner_model: str = Field(default="openai/gpt-5.6-luna")
    executor_model: str = Field(default="openai/gpt-5.6-luna")
    replanner_model: str = Field(default="openai/gpt-5.6-luna")
    explainer_model: str = Field(default="openai/gpt-5.6-luna")

    planner_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    executor_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    replanner_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # Explainer лише переказує готовий вердикт, тому вища температура безпечна
    # і дає живіший текст, ніж температура 0.
    explainer_temperature: float = Field(default=0.4, ge=0.0, le=2.0)

    # Умовна сума портфеля для пояснень «уявіть, що у вас $X».
    # Відсотки погано сприймаються нефахівцем, гроші — добре.
    explanation_reference_amount: float = Field(default=10_000.0, gt=0)

    # --- Прогноз уперед ---
    projection_horizon_days: int = Field(default=90, ge=7, le=365)
    projection_confidence: float = Field(default=0.90, gt=0.5, lt=1.0)

    # Вікно для оцінки параметрів прогнозу. Довше за вікно показу метрик:
    # похибка оцінки спадає як sqrt(T), тому вся доступна історія тут доречна.
    estimation_lookback_days: int = Field(default=720, ge=180, le=720)

    # Очікувана дохідність = w * власна історія + (1 - w) * плата за ризик,
    # де w = T / (T + estimation_prior_days). Довша вибірка -> більша довіра
    # до власного результату активу.
    #
    # Значення 1080 підібрано емпірично на mock-universe: воно дає RMSE оцінки
    # дрейфу 29.9% проти 54.8% у сирої 720-денної історії (див. README).
    estimation_prior_days: int = Field(default=1080, ge=0)
    risk_free_rate: float = Field(default=0.04, ge=0.0, le=0.5)
    # Скільки річної дохідності "належить" активу за одиницю його волатильності
    risk_premium_per_vol: float = Field(default=0.35, ge=0.0, le=2.0)

    # Вікно, на якому decision_engine ранжує портфелі. Дорівнює вікну оцінки
    # прогнозу свідомо: на коротшому вікні рушій ганявся за імпульсом і міг
    # купити актив, який прогноз вважає збитковим (ADA: +167% за 180 днів,
    # -50% за 720). Вердикт і прогноз мають дивитись на ті самі дані.
    decision_lookback_days: int = Field(default=720, ge=180, le=720)

    # --- Обмеження ReAct-агента ---
    react_max_steps: int = Field(default=10, ge=1, le=50)
    react_timeout_seconds: float = Field(default=120.0, gt=0)

    # --- Політика портфеля ---
    max_portfolio_assets: int = Field(default=5, ge=1, le=5)
    min_position_weight: float = Field(default=0.05, gt=0, lt=1)
    max_position_weight: float = Field(default=0.50, gt=0, le=1)

    # --- Turnover control ---
    minimum_improvement_score: float = Field(
        default=0.15,
        ge=0.0,
        description="Мінімальне покращення risk-adjusted score, щоб дозволити REBALANCE",
    )
    turnover_cost_per_unit: float = Field(
        default=0.35,
        ge=0.0,
        description="Штраф за одиницю turnover (сума абсолютних змін ваг)",
    )
    max_turnover: float = Field(default=0.80, gt=0, le=2.0)

    # --- Universe ---
    universe_limit: int = Field(default=25, ge=1, le=100)
    min_history_days: int = Field(default=180, ge=1)

    # --- Інфраструктура ---
    sqlite_checkpoint_path: Path = Field(default=PROJECT_ROOT / "data" / "checkpoints.sqlite")
    chroma_path: Path = Field(default=PROJECT_ROOT / "data" / "chroma")
    trajectory_log_path: Path = Field(default=PROJECT_ROOT / "data" / "trajectory.jsonl")
    portfolio_state_path: Path = Field(default=PROJECT_ROOT / "data" / "portfolio.json")
    autonomy_state_path: Path = Field(default=PROJECT_ROOT / "data" / "autonomy.json")

    # --- Чат ---
    chat_suggestions_enabled: bool = Field(default=True)
    chat_suggestions_count: int = Field(default=3, ge=1, le=5)

    # --- Зароблена автономія (progressive delegation) ---
    # Агент не отримує самостійності на старті: він набирає історію схвалень і
    # лише потім пропонує брати дрібні дії на себе.
    autonomy_min_decisions: int = Field(default=6, ge=1)
    autonomy_min_acceptance: float = Field(default=0.8, gt=0.0, le=1.0)
    # Максимальна зміна ваги, яку взагалі можна делегувати
    autonomy_max_delta: float = Field(default=0.05, gt=0.0, lt=0.5)

    market_data_seed: int = Field(default=20240101, description="Seed для відтворюваних mock-даних")

    @field_validator("openrouter_base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("openrouter_base_url має починатися з http:// або https://")
        return value.rstrip("/")

    @field_validator("max_position_weight")
    @classmethod
    def _validate_weight_bounds(cls, value: float) -> float:
        if value <= 0.1:
            raise ValueError("max_position_weight має бути більшим за 10%")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Кешований singleton налаштувань."""
    settings = Settings()
    settings.sqlite_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return settings
