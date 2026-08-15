"""Уточнювальні питання до рішення — детерміновані, з готовими відповідями.

Чому питання не генерує LLM. Вони будуються з фактичного плану, тому:
зʼявляються миттєво (нуль латентності на старті), гарантовано мають відповідь
у даних і не можуть послатися на те, чого агент не рахував (ціна зараз, прогноз
наступної свічки). LLM лишається там, де контекст справді непередбачуваний —
у вільному чаті (`agents/suggestions.py`).

Розгорнутий наратив від LLM нікуди не зникає: він доступний на вимогу і
будується лениво, тільки якщо людина його попросила.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.explainer import money, odds_phrase, weakest_asset
from app.config import get_settings
from app.data.projection import project_asset, project_portfolio
from app.domain.schemas import AssetMetrics

# Оборот, вище якого його вартість треба проговорити окремо
HIGH_TURNOVER = 0.40

# Актив із такою ймовірністю збитку і нижче не є слабким сам по собі:
# якщо його ріжуть, причина в частці, а не в якості
DECENT_LOSS_PROBABILITY = 0.45

MAX_QUESTIONS = 4


@dataclass(frozen=True)
class DecisionQuestion:
    """Питання і готова відповідь на нього."""

    text: str
    answer: str


def build_questions(
    evidence: dict[str, Any],
    decision: str,
    metrics: dict[str, AssetMetrics] | None = None,
) -> tuple[DecisionQuestion, ...]:
    """Скласти набір питань, доречних саме для цього рішення."""
    asset_metrics = metrics or {}
    if decision == "HOLD":
        questions = _hold_questions(evidence)
    else:
        questions = _rebalance_questions(evidence, asset_metrics)

    questions.append(_trust_question(evidence))
    return tuple(questions[:MAX_QUESTIONS])


# --- REBALANCE --------------------------------------------------------------


def _rebalance_questions(
    evidence: dict[str, Any], metrics: dict[str, AssetMetrics]
) -> list[DecisionQuestion]:
    actions = evidence.get("proposed_actions") or []
    questions: list[DecisionQuestion] = []

    purchase = _first_action(actions, {"BUY"})
    if purchase is not None:
        questions.append(_purchase_question(purchase))

    trimmed = _trimmed_but_healthy(actions)
    if trimmed is not None:
        questions.append(_trimmed_question(trimmed))

    turnover = float(evidence.get("turnover") or 0.0)
    if turnover >= HIGH_TURNOVER:
        questions.append(_turnover_question(turnover))

    weakest = weakest_asset(evidence, metrics)
    if weakest is not None and len(questions) < MAX_QUESTIONS - 1:
        questions.append(_weakness_question(*weakest))

    return questions


def _purchase_question(action: dict[str, Any]) -> DecisionQuestion:
    symbol = str(action.get("symbol", "?"))
    weight = float(action.get("to_weight") or 0.0)
    amount = get_settings().explanation_reference_amount
    outlook = project_asset(symbol, 1000.0)

    answer = (
        f"{symbol} займає {weight * 100:.1f}% плану — це {money(amount * weight)} "
        f"із {money(amount)}. "
    )
    if outlook is not None:
        answer += (
            f"Очікувана дохідність {outlook.expected_annual_return * 100:+.0f}% річних "
            f"при коливаннях {outlook.volatility * 100:.0f}%: кожна вкладена тисяча "
            f"за {outlook.horizon_days} днів серединно перетворюється на "
            f"{money(outlook.median_value)} за ризику мінусу "
            f"{outlook.loss_probability * 100:.0f}%. "
        )
    answer += (
        "Вибір робиться не за самою дохідністю, а за співвідношенням дохідності до "
        "коливань і за тим, наскільки актив доповнює решту портфеля. Актив, який "
        "росте разом з усіма іншими, не додає стійкості, навіть якщо росте сильно."
    )
    return DecisionQuestion(f"Чому саме {symbol}, а не інший актив?", answer)


def _trimmed_question(action: dict[str, Any]) -> DecisionQuestion:
    symbol = str(action.get("symbol", "?"))
    from_weight = float(action.get("from_weight") or 0.0)
    to_weight = float(action.get("to_weight") or 0.0)
    amount = get_settings().explanation_reference_amount
    outlook = project_asset(symbol, 1000.0)

    forward = ""
    if outlook is not None:
        forward = (
            f"Його власний прогноз один із кращих у портфелі — "
            f"{outlook.expected_annual_return * 100:+.0f}% річних за ризику мінусу "
            f"{outlook.loss_probability * 100:.0f}%. "
        )
    answer = (
        f"{symbol} ріжеться не тому, що поганий. {forward}"
        f"Проблема в частці: {from_weight * 100:.1f}% портфеля — це "
        f"{money(amount * from_weight)} у одному імені, і результат усього портфеля "
        f"починає залежати від нього одного. Зменшення до {to_weight * 100:.1f}% "
        f"лишає позицію значною, але прибирає цю залежність."
    )
    return DecisionQuestion(
        f"Чим поганий {symbol}, якщо його теж зменшують?", answer
    )


def _turnover_question(turnover: float) -> DecisionQuestion:
    amount = get_settings().explanation_reference_amount
    answer = (
        f"Через ринок пройде {money(amount * turnover)} із {money(amount)}. Ці гроші "
        f"не зникають — вони переїжджають з одного активу в інший. Але за кожну "
        f"операцію біржа бере комісію, і ціна виконання ніколи не ідеальна. Тому "
        f"розрахунок спершу віднімає вартість переходу і лише потім дивиться, чи "
        f"лишилося щось варте зусиль. Саме цей віднімач і не дає перебудовувати "
        f"портфель заради дрібного виграшу."
    )
    return DecisionQuestion(
        f"Оборот {turnover * 100:.1f}% — це не забагато?", answer
    )


def _weakness_question(symbol: str, metric: AssetMetrics) -> DecisionQuestion:
    amount = get_settings().explanation_reference_amount
    answer = (
        f"Найслабша ланка — {symbol}. За {metric.lookback_days} днів дохідність "
        f"{metric.annualized_return * 100:+.1f}% річних при коливаннях "
        f"{metric.volatility * 100:.0f}% і просадці до "
        f"{metric.max_drawdown * 100:.0f}%. На умовних {money(amount)} це означає, "
        f"що в найгіршій точці позиція коштувала б помітно менше, ніж на неї "
        f"витратили. Тримати такий актив має сенс лише якщо він чимось "
        f"компенсує — тут не компенсує."
    )
    return DecisionQuestion("Що не так із поточним складом?", answer)


# --- HOLD -------------------------------------------------------------------


def _hold_questions(evidence: dict[str, Any]) -> list[DecisionQuestion]:
    net = float(evidence.get("net_improvement_after_turnover") or 0.0)
    threshold = float(evidence.get("minimum_improvement_score") or 0.0)
    turnover = float(evidence.get("turnover") or 0.0)
    amount = get_settings().explanation_reference_amount

    verdict = (
        "кращий варіант існує, але його перевага менша за вартість переходу"
        if net > 0
        else "жоден розглянутий варіант не виявився кращим за поточний склад"
    )
    first = DecisionQuestion(
        "Чому агент нічого не робить?",
        (
            f"Тому що {verdict}. Перебудова коштувала б обороту "
            f"{turnover * 100:.1f}% — це {money(amount * turnover)} через ринок, "
            f"із комісіями і неідеальною ціною виконання. Найдешевша дія тут — "
            f"не діяти."
        ),
    )
    second = DecisionQuestion(
        "Що має змінитись, щоб з'явилась пропозиція?",
        (
            f"Потрібно, щоб перевага кандидата після вирахування вартості переходу "
            f"перевищила поріг {threshold:.2f}. Зараз запас складає {net:+.4f}. "
            f"Це станеться, якщо якийсь актив помітно зміцниться відносно решти або "
            f"поточний склад просяде — тобто коли різниця стане достатньою, щоб "
            f"оплатити перебудову, а не з'їстися нею."
        ),
    )
    return [first, second]


# --- спільне ----------------------------------------------------------------


def _trust_question(evidence: dict[str, Any]) -> DecisionQuestion:
    amount = get_settings().explanation_reference_amount
    projection = project_portfolio(evidence.get("current_weights") or {}, amount)

    if projection is None:
        answer = (
            "Прогноз будується на історичних спостереженнях і подається діапазоном, "
            "а не точкою. Це оцінка ймовірностей, а не обіцянка."
        )
        return DecisionQuestion("Наскільки можна вірити цьому прогнозу?", answer)

    answer = (
        f"Рівно настільки, наскільки можна вірити діапазону, а не точці. Оцінка "
        f"спирається на {projection.sample_days} днів спостережень, і власний "
        f"результат активу враховується лише на "
        f"{projection.estimation_weight * 100:.0f}% — решта це те, скільки актив "
        f"взагалі має приносити за свій рівень ризику. Причина проста: дохідність "
        f"оцінюється з історії погано навіть на довгих вікнах, а от коливання — "
        f"добре. Тому результат завжди подається як коридор "
        f"{money(projection.downside_value)}–{money(projection.upside_value)} із "
        f"ймовірністю мінусу {projection.loss_probability * 100:.0f}%. "
        f"{odds_phrase(projection.loss_probability)}"
    )
    return DecisionQuestion("Наскільки можна вірити цьому прогнозу?", answer)


def _first_action(
    actions: list[dict[str, Any]], kinds: set[str]
) -> dict[str, Any] | None:
    """Найбільша за модулем дія одного з типів."""
    matching = [action for action in actions if action.get("action") in kinds]
    if not matching:
        return None
    return max(matching, key=lambda action: abs(_delta(action)))


def _trimmed_but_healthy(actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Найбільше скорочення активу, який сам по собі не є слабким.

    Саме цей випадок читач розуміє найгірше: актив ріжуть, хоча його прогноз
    непоганий. Причина — концентрація, і про неї треба сказати прямо.
    """
    best: dict[str, Any] | None = None
    for action in actions:
        if action.get("action") not in {"REDUCE", "SELL"}:
            continue
        outlook = project_asset(str(action.get("symbol", "")), 1000.0)
        if outlook is None or outlook.loss_probability > DECENT_LOSS_PROBABILITY:
            continue
        if best is None or abs(_delta(action)) > abs(_delta(best)):
            best = action
    return best


def _delta(action: dict[str, Any]) -> float:
    return float(action.get("to_weight") or 0.0) - float(action.get("from_weight") or 0.0)
