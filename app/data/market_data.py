"""Детермінований генератор mock market data.

Реальні біржові API навмисно не використовуються. Дані генеруються з фіксованим
seed, тому будь-який запуск (у т.ч. у тестах) дає ідентичні ціни та метрики.

Модель ціни: геометричний броунівський рух із детермінованим псевдо-нормальним
шумом + слабка режимна компонента (синусоїда), щоб отримати реалістичні тренди
та просадки без залежності від numpy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings
from app.domain.errors import InsufficientHistoryError, UnknownSymbolError

# Кількість торгових днів у році для аннуалізації (крипто торгується 365 днів)
TRADING_DAYS_PER_YEAR = 365
# Максимальна довжина згенерованої історії
HISTORY_DAYS = 720


@dataclass(frozen=True)
class AssetSpec:
    """Статичний опис активу в universe."""

    symbol: str
    name: str
    rank: int
    daily_volume_usd: float
    history_days: int
    is_stablecoin: bool
    base_price: float
    annual_drift: float
    annual_volatility: float
    data_quality_ok: bool = True


# Universe: 26 активів, включно зі stablecoins, "молодими" активами та одним
# активом із навмисно зіпсованими даними — щоб replanner мав що відсіювати.
_UNIVERSE: tuple[AssetSpec, ...] = (
    AssetSpec("BTC", "Bitcoin", 1, 28_000_000_000, 720, False, 62_000, 0.45, 0.42),
    AssetSpec("ETH", "Ethereum", 2, 14_000_000_000, 720, False, 3_100, 0.38, 0.52),
    AssetSpec("USDT", "Tether", 3, 45_000_000_000, 720, True, 1.0, 0.0, 0.004),
    AssetSpec("BNB", "BNB", 4, 1_900_000_000, 720, False, 580, 0.30, 0.55),
    AssetSpec("SOL", "Solana", 5, 3_400_000_000, 720, False, 145, 0.62, 0.78),
    AssetSpec("USDC", "USD Coin", 6, 8_100_000_000, 720, True, 1.0, 0.0, 0.003),
    AssetSpec("XRP", "XRP", 7, 1_500_000_000, 720, False, 0.52, 0.10, 0.61),
    AssetSpec("DOGE", "Dogecoin", 8, 1_100_000_000, 720, False, 0.13, 0.05, 0.88),
    AssetSpec("ADA", "Cardano", 9, 620_000_000, 720, False, 0.45, -0.05, 0.64),
    AssetSpec("AVAX", "Avalanche", 10, 540_000_000, 720, False, 34.0, -0.12, 0.74),
    AssetSpec("LINK", "Chainlink", 11, 480_000_000, 720, False, 17.5, 0.41, 0.58),
    AssetSpec("TRX", "TRON", 12, 430_000_000, 720, False, 0.12, 0.22, 0.47),
    AssetSpec("DOT", "Polkadot", 13, 310_000_000, 720, False, 6.8, -0.18, 0.66),
    AssetSpec("MATIC", "Polygon", 14, 380_000_000, 720, False, 0.72, -0.25, 0.71),
    AssetSpec("LTC", "Litecoin", 15, 420_000_000, 720, False, 82.0, 0.08, 0.49),
    AssetSpec("DAI", "Dai", 16, 220_000_000, 720, True, 1.0, 0.0, 0.005),
    AssetSpec("UNI", "Uniswap", 17, 210_000_000, 720, False, 8.4, 0.19, 0.69),
    AssetSpec("ATOM", "Cosmos", 18, 190_000_000, 720, False, 8.1, -0.22, 0.68),
    AssetSpec("XLM", "Stellar", 19, 160_000_000, 720, False, 0.11, 0.03, 0.59),
    AssetSpec("NEAR", "NEAR Protocol", 20, 250_000_000, 720, False, 5.6, 0.35, 0.80),
    AssetSpec("ICP", "Internet Computer", 21, 140_000_000, 720, False, 11.2, -0.30, 0.85),
    AssetSpec("APT", "Aptos", 22, 175_000_000, 720, False, 8.9, 0.12, 0.83),
    AssetSpec("FIL", "Filecoin", 23, 130_000_000, 720, False, 4.7, -0.28, 0.77),
    AssetSpec("ARB", "Arbitrum", 24, 165_000_000, 720, False, 0.98, -0.15, 0.79),
    # Молодий актив: історії менше за min_history_days -> має відсіюватись universe-фільтром
    AssetSpec("NEWX", "NewChain", 25, 120_000_000, 45, False, 2.3, 0.90, 1.20),
    # Актив із некоректними даними -> get_market_data поверне помилку
    AssetSpec("BADQ", "BadQuality", 26, 110_000_000, 720, False, 3.1, 0.20, 0.60, False),
)

_BY_SYMBOL: dict[str, AssetSpec] = {spec.symbol: spec for spec in _UNIVERSE}


def get_asset_spec(symbol: str) -> AssetSpec:
    """Знайти опис активу або кинути UNKNOWN_SYMBOL."""
    spec = _BY_SYMBOL.get(symbol.strip().upper())
    if spec is None:
        raise UnknownSymbolError(f"Актив '{symbol}' відсутній у universe")
    return spec


def all_specs() -> tuple[AssetSpec, ...]:
    """Повний universe без фільтрів."""
    return _UNIVERSE


@lru_cache(maxsize=64)
def _price_series(symbol: str) -> tuple[float, ...]:
    """Згенерувати детермінований ряд цін для активу.

    Кешується, тому в межах процесу ряд стабільний, а seed прив'язаний
    до символу — різні активи не корелюють штучно на 100%.
    """
    spec = get_asset_spec(symbol)
    settings = get_settings()
    # Seed складається з глобального seed та хешу символу -> відтворюваність
    rng = random.Random(f"{settings.market_data_seed}:{spec.symbol}")

    days = min(spec.history_days, HISTORY_DAYS)
    daily_drift = spec.annual_drift / TRADING_DAYS_PER_YEAR
    daily_vol = spec.annual_volatility / math.sqrt(TRADING_DAYS_PER_YEAR)

    prices = [spec.base_price]
    for day in range(1, days):
        shock = rng.gauss(0.0, 1.0)
        # Режимна компонента створює чергування бичачих і ведмежих фаз.
        # Період ~126 днів: у вікні 180 днів проходить понад один повний цикл,
        # тому вона не вироджується у постійний дрейф і не роздуває дохідність.
        regime = 0.20 * daily_vol * math.sin(day / 20.0 + spec.rank)
        log_return = daily_drift - 0.5 * daily_vol**2 + daily_vol * shock + regime
        prices.append(max(prices[-1] * math.exp(log_return), 1e-9))
    return tuple(prices)


def get_price_history(symbol: str, lookback_days: int) -> tuple[float, ...]:
    """Останні `lookback_days` цін активу.

    Кидає INSUFFICIENT_HISTORY, якщо доступної історії менше, ніж запитано,
    та повертає помилку якості для активів із пошкодженими даними.
    """
    spec = get_asset_spec(symbol)
    if not spec.data_quality_ok:
        raise InsufficientHistoryError(
            f"Дані для '{spec.symbol}' неповні або некоректні і не придатні для аналізу"
        )

    series = _price_series(spec.symbol)
    if len(series) < lookback_days:
        raise InsufficientHistoryError(
            f"Для '{spec.symbol}' доступно лише {len(series)} днів історії, "
            f"запитано {lookback_days}"
        )
    return series[-lookback_days:]


def daily_returns(prices: tuple[float, ...]) -> tuple[float, ...]:
    """Прості денні дохідності з ряду цін."""
    if len(prices) < 2:
        return ()
    return tuple(
        (prices[index] / prices[index - 1]) - 1.0 for index in range(1, len(prices))
    )
