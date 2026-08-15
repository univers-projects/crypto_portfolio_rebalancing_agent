"""Розгорнуте пояснення рішення людською мовою.

Explainer нічого не вирішує. Числовий вердикт уже порахований у
`decision_engine`, сюди він приходить готовим, а завдання цього модуля —
переказати його зрозуміло: що зараз у портфелі, що саме не так, чому
запропонована алокація краща і скільки коштує перехід.

Два запобіжники, без яких модуль не можна вмикати у старт застосунку:

1. Якщо LLM недоступний (rate limit, таймаут, немає ключа) — повертається
   детермінований шаблон. Старт апки не має падати через чужий 429.
2. Якщо текст LLM суперечить вердикту політики — він відкидається на користь
   шаблону. LLM не може «передумати» щодо HOLD/REBALANCE.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.data.projection import Projection, project_asset, project_portfolio
from app.domain.schemas import AssetMetrics
from app.llm import get_llm
from app.observability import trajectory
from app.prompts import load_prompt

logger = logging.getLogger(__name__)

# Вердикт у тексті шукаємо як окреме слово, щоб не чіплятися за "HOLDING" тощо
_VERDICT_PATTERN = re.compile(r"\b(HOLD|REBALANCE)\b")

FORECAST_DISCLAIMER = (
    "Прогноз — це діапазон імовірних результатів, а не обіцянка: ринок не "
    "зобов'язаний повторювати минуле. Усі операції є симуляцією — реальні кошти, "
    "біржові рахунки та торгові ключі не задіяні."
)


def render_summary(
    evidence: dict[str, Any],
    decision: str,
    asset_metrics: dict[str, AssetMetrics] | None = None,
) -> str:
    """Короткий висновок першими рядками — без LLM і без затримки.

    Розгорнутий наратив лишається доступним на вимогу. Стіна тексту на старті
    не читається: висновок має бути видно одразу, а подробиці — за запитом.
    """
    metrics = asset_metrics or {}
    amount = get_settings().explanation_reference_amount
    current = project_portfolio(evidence.get("current_weights") or {}, amount)
    proposed = project_portfolio(evidence.get("best_candidate_weights") or {}, amount)

    lines: list[str] = []
    if decision == "REBALANCE" and current is not None and proposed is not None:
        drop = (current.loss_probability - proposed.loss_probability) * 100
        direction = "падає" if drop >= 0 else "зростає"
        lines.append(
            f"Головне: ризик збитку за {proposed.horizon_days} днів {direction} з "
            f"{current.loss_probability * 100:.0f}% до "
            f"{proposed.loss_probability * 100:.0f}%, тобто на {abs(drop):.0f} п.п."
        )
        lines.append(
            f"Серединний сценарій рухається з {money(current.median_value)} до "
            f"{money(proposed.median_value)} на умовних {money(amount)}."
        )
        turnover = float(evidence.get("turnover") or 0.0)
        lines.append(
            f"Ціна переходу — оборот {turnover * 100:.1f}%, тобто "
            f"{money(amount * turnover)} через ринок."
        )
        weakest = weakest_asset(evidence, metrics)
        if weakest is not None:
            symbol, metric = weakest
            lines.append(
                f"Слабка ланка — {symbol}: {metric.annualized_return * 100:+.0f}% "
                f"річних за {metric.lookback_days} днів при коливаннях "
                f"{metric.volatility * 100:.0f}%."
            )
    elif current is not None:
        lines.append(
            "Поточний склад лишається без змін: жоден варіант не виграв достатньо, "
            "щоб окупити перехід."
        )
        lines.append(
            f"Прогноз на {current.horizon_days} днів для {money(amount)}: серединно "
            f"{money(current.median_value)}, коридор "
            f"{money(current.downside_value)}–{money(current.upside_value)}, "
            f"ризик мінусу {current.loss_probability * 100:.0f}%."
        )
    else:
        lines.append("Даних для прогнозу недостатньо.")

    return "\n".join(lines)


def weakest_asset(
    evidence: dict[str, Any], metrics: dict[str, AssetMetrics]
) -> tuple[str, AssetMetrics] | None:
    """Актив поточного складу з найгіршим співвідношенням дохідності до ризику."""
    weights = evidence.get("current_weights") or {}
    present = [symbol for symbol in weights if symbol in metrics]
    if not present:
        return None
    symbol = min(present, key=lambda item: metrics[item].sharpe_like)
    return symbol, metrics[symbol]


def build_explanation(
    evidence: dict[str, Any],
    decision: str,
    asset_metrics: dict[str, AssetMetrics] | None = None,
    llm: Any = None,
) -> str:
    """Скласти розгорнуте пояснення рішення.

    Повертає готовий текст. Ніколи не кидає виняток назовні: якщо LLM недоступний,
    використовується детермінований шаблон.
    """
    metrics = asset_metrics or {}
    fallback = render_fallback_explanation(evidence, decision, metrics)

    try:
        client = llm if llm is not None else get_llm("explainer")
        payload = _build_payload(evidence, decision, metrics)
        response = client.invoke(
            [
                SystemMessage(content=load_prompt("explainer")),
                HumanMessage(content=payload),
            ]
        )
        text = str(getattr(response, "content", "") or "").strip()
    except Exception as error:  # noqa: BLE001 — будь-який збій LLM веде у шаблон
        logger.warning("Explainer недоступний, використано шаблон: %s", error)
        trajectory.record_event(
            node="explainer", event="explanation_fallback", reason=type(error).__name__
        )
        return fallback

    if not text:
        trajectory.record_event(node="explainer", event="explanation_fallback", reason="empty")
        return fallback

    if _contradicts_verdict(text, decision):
        logger.warning("Пояснення суперечить вердикту політики, відкинуто")
        trajectory.record_event(
            node="explainer", event="explanation_rejected", reason="verdict_mismatch"
        )
        return fallback

    trajectory.record_event(node="explainer", event="explanation_generated", chars=len(text))
    return text


def _contradicts_verdict(text: str, decision: str) -> bool:
    """Чи згадує текст протилежний вердикт як власний висновок."""
    mentioned = set(_VERDICT_PATTERN.findall(text.upper()))
    opposite = "HOLD" if decision == "REBALANCE" else "REBALANCE"
    return opposite in mentioned and decision not in mentioned


def _build_payload(
    evidence: dict[str, Any], decision: str, metrics: dict[str, AssetMetrics]
) -> str:
    """Зібрати evidence у текст для LLM."""
    settings = get_settings()
    amount = settings.explanation_reference_amount
    current_metrics = evidence.get("current_portfolio") or {}

    lines = [
        f"policy_verdict: {decision}",
        f"reference_amount: {amount}",
        f"projection_horizon_days: {settings.projection_horizon_days}",
        f"estimation_lookback_days: {settings.estimation_lookback_days}",
        f"net_improvement_after_turnover: {evidence.get('net_improvement_after_turnover')}",
        f"minimum_improvement_score: {evidence.get('minimum_improvement_score')}",
        f"turnover: {evidence.get('turnover')}",
        f"current_weights: {evidence.get('current_weights')}",
        f"current_portfolio_metrics_HISTORICAL: {current_metrics}",
    ]

    current_projection = project_portfolio(evidence.get('current_weights') or {}, amount)
    if current_projection is not None:
        lines.append(f"current_projection_FORWARD: {current_projection.as_dict()}")

    if evidence.get("best_candidate_weights"):
        lines.append(f"proposed_weights: {evidence['best_candidate_weights']}")
        lines.append(
            f"proposed_portfolio_metrics_HISTORICAL: {evidence.get('best_candidate_metrics')}"
        )
        proposed_projection = project_portfolio(
            evidence.get("best_candidate_weights") or {}, amount
        )
        if proposed_projection is not None:
            lines.append(f"proposed_projection_FORWARD: {proposed_projection.as_dict()}")
    if evidence.get("proposed_actions"):
        lines.append(f"proposed_actions: {evidence['proposed_actions']}")
    if metrics:
        rendered = {
            symbol: metric.model_dump(mode="json") for symbol, metric in sorted(metrics.items())
        }
        lines.append(f"asset_metrics: {rendered}")
    return "\n".join(lines)


def render_fallback_explanation(
    evidence: dict[str, Any], decision: str, metrics: dict[str, AssetMetrics]
) -> str:
    """Детермінований текст пояснення без участі LLM.

    Свідомо тримає той самий стиль, що й промпт: кожне важливе число
    перекладається у гроші на умовну суму портфеля.
    """
    amount = get_settings().explanation_reference_amount
    current = evidence.get("current_portfolio") or {}
    weights = evidence.get("current_weights") or {}
    current_projection = project_portfolio(weights, amount)

    parts = [_render_current_block(current, weights, amount)]
    if current_projection is not None:
        parts.append(_render_current_outlook(current_projection))

    if decision == "HOLD":
        parts.append(_render_hold_block(evidence, amount))
        parts.append(FORECAST_DISCLAIMER)
        return "\n\n".join(parts)

    weakness = _render_weakness_block(weights, metrics, amount)
    if weakness:
        parts.append(weakness)

    proposed = evidence.get("best_candidate_metrics") or {}
    proposed_weights = evidence.get("best_candidate_weights") or {}
    parts.append(_render_proposed_block(proposed, proposed_weights, amount))

    proposed_projection = project_portfolio(proposed_weights, amount)
    if current_projection is not None and proposed_projection is not None:
        parts.append(_render_comparison_block(current_projection, proposed_projection))

    parts.append(_render_turnover_block(evidence, amount))
    parts.append(_render_actions_block(evidence.get("proposed_actions") or [], metrics))
    parts.append(FORECAST_DISCLAIMER)
    return "\n\n".join(parts)


def _render_current_outlook(projection: Projection) -> str:
    """Чого чекати від поточного складу — прогноз, а не історія."""
    years = projection.sample_days / 365.0
    return (
        f"Тепер головне — чого чекати далі. Оцінка спирається на "
        f"{projection.sample_days} днів спостережень, тобто майже {years:.1f} роки. "
        f"Очікувана дохідність кожного активу складається з двох частин: його "
        f"власного результату за цей період і того, скільки актив узагалі має "
        f"приносити за свій рівень ризику. Чим довша історія, тим більша вага "
        f"власного результату — зараз вона становить "
        f"{projection.estimation_weight * 100:.0f}%.\n\n"
        f"За наступні {projection.horizon_days} днів із нинішнім складом серединний "
        f"сценарій — {money(projection.median_value)}, тобто "
        f"{money(abs(projection.median_gain))} "
        f"{'прибутку' if projection.median_gain >= 0 else 'збитку'}. "
        f"Реалістичний коридор широкий: від {money(projection.downside_value)} у "
        f"поганому сценарії до {money(projection.upside_value)} у хорошому. "
        f"Імовірність завершити період у мінусі — "
        f"{projection.loss_probability * 100:.0f}%. "
        f"{odds_phrase(projection.loss_probability)}"
    )


def _render_comparison_block(current: Projection, proposed: Projection) -> str:
    """Порівняння двох прогнозів — на цьому і будується рішення."""
    delta = proposed.median_value - current.median_value
    risk_drop = (current.loss_probability - proposed.loss_probability) * 100
    lines = [
        f"Що це змінює на тому ж горизонті в {proposed.horizon_days} днів. "
        f"Серединний сценарій зсувається з {money(current.median_value)} до "
        f"{money(proposed.median_value)} — це "
        f"{money(abs(delta))} різниці. "
        f"Поганий сценарій підтягується з {money(current.downside_value)} до "
        f"{money(proposed.downside_value)}."
    ]
    if risk_drop > 1:
        lines.append(
            f"Але важливіша не сама цифра прибутку, а ризик: імовірність завершити "
            f"період у мінусі падає з {current.loss_probability * 100:.0f}% до "
            f"{proposed.loss_probability * 100:.0f}%, тобто на {risk_drop:.0f} "
            f"процентних пункти. Саме за це і платимо витратами на перехід."
        )
    else:
        lines.append(
            "Ризик опинитись у мінусі при цьому майже не змінюється, тож основний "
            "аргумент тут — очікуваний результат, а не зниження ризику."
        )
    return " ".join(lines)


def _render_current_block(
    current: dict[str, Any], weights: dict[str, Any], amount: float
) -> str:
    """Поточний стан портфеля з перекладом у гроші."""
    composition = ", ".join(
        f"{symbol} {float(weight) * 100:.1f}%" for symbol, weight in sorted(weights.items())
    )
    period_return = float(current.get("portfolio_return") or 0.0)
    annual = float(current.get("annualized_return") or 0.0)
    drawdown = float(current.get("max_drawdown") or 0.0)
    sharpe = float(current.get("sharpe_like") or 0.0)

    return (
        f"Зараз портфель складається з: {composition}. "
        f"Уявіть, що в ньому {money(amount)} — далі всі суми будуть від цієї цифри.\n\n"
        f"За період спостереження такий склад дав {period_return * 100:+.1f}%, "
        f"тобто приблизно {money(abs(amount * period_return))} "
        f"{'прибутку' if period_return >= 0 else 'збитку'}; у перерахунку на рік це "
        f"{annual * 100:+.1f}%. "
        f"Найглибша просадка становила {drawdown * 100:.1f}%: у найгіршу точку періоду "
        f"портфель коштував би близько {money(amount * (1 - drawdown))}, тобто на "
        f"{money(amount * drawdown)} менше, ніж на піку до того. "
        f"Співвідношення винагороди до ризику — {sharpe:.2f}. Простими словами: "
        f"{_sharpe_phrase(sharpe)}."
    )


def _render_weakness_block(
    weights: dict[str, Any], metrics: dict[str, AssetMetrics], amount: float
) -> str:
    """Найслабший актив портфеля з ціною його присутності у грошах."""
    held = {
        symbol: metric
        for symbol, metric in metrics.items()
        if symbol in weights and weights[symbol] > 0
    }
    if not held:
        return ""

    symbol, metric = min(held.items(), key=lambda item: item[1].sharpe_like)
    if metric.sharpe_like >= 0:
        return (
            f"Явного аутсайдера немає: навіть найслабший актив {symbol} лишався в плюсі. "
            "Проблема радше в тому, як розподілені ваги, ніж у конкретному активі."
        )

    invested = amount * float(weights[symbol])
    left = invested * (1 + metric.total_return)
    return (
        f"Найбільше тягне вниз {symbol}. Він займає {float(weights[symbol]) * 100:.1f}% "
        f"портфеля — це {money(invested)} із ваших {money(amount)}. "
        f"За період ці гроші перетворилися б приблизно на {money(left)}, тобто "
        f"{money(invested - left)} просто зникли. "
        f"При цьому {symbol} коливався сильніше за портфель загалом "
        f"(волатильність {metric.volatility * 100:.1f}%) і провалювався на "
        f"{metric.max_drawdown * 100:.1f}%. Тобто ризику багато, а віддачі за нього немає — "
        f"саме це і намагається виправити пропозиція."
    )


def _render_proposed_block(
    proposed: dict[str, Any], weights: dict[str, Any], amount: float
) -> str:
    """Запропонована алокація у грошах."""
    lines = [
        "Пропонується перейти до складу: "
        + ", ".join(
            f"{symbol} {float(weight) * 100:.1f}% ({money(amount * float(weight))})"
            for symbol, weight in sorted(weights.items(), key=lambda item: -item[1])
        )
        + "."
    ]
    lines.append(
        "Підстава — не те, що ці активи росли раніше, а те, що такий набір дає "
        "кращий очікуваний профіль на майбутнє: вищий імовірний результат за "
        f"приблизно того самого розмаху коливань ({_pct(proposed.get('volatility'))} "
        "у річному вимірі)."
    )
    return " ".join(lines)


def _render_turnover_block(evidence: dict[str, Any], amount: float) -> str:
    """Чесна ціна переходу."""
    turnover = float(evidence.get("turnover") or 0.0)
    return (
        f"Тепер про ціну. Перебудова зачіпає {turnover * 100:.1f}% портфеля: з "
        f"{money(amount)} через ринок пройде близько {money(amount * turnover)}. "
        f"Ці гроші не зникають — вони просто переїжджають з одного активу в інший. "
        f"Але за кожну таку операцію біржа бере комісію, і ціна виконання ніколи не "
        f"буває ідеальною. Саме тому склад не міняють заради дрібного виграшу: "
        f"розрахунок спершу віднімає вартість переходу і лише потім дивиться, чи "
        f"лишилося щось варте зусиль. Тут запас над потрібним мінімумом дуже великий, "
        f"тож перебудова себе виправдовує навіть з урахуванням витрат."
    )


def _render_hold_block(evidence: dict[str, Any], amount: float) -> str:
    """Пояснення, чому нічого не змінюємо."""
    return (
        "Змінювати нічого не потрібно. Жоден із розглянутих варіантів не виявився "
        "настільки кращим, щоб окупити перехід. Уявіть, що заради ледь помітного "
        f"покращення ви переганяєте через біржу тисячі доларів із {money(amount)} і "
        "платите комісію за кожну операцію: витрати з'їдять весь виграш. "
        "Тому нічого не робити — це теж повноцінне рішення, а не бездіяльність."
    )


def _render_actions_block(
    actions: list[dict[str, Any]], metrics: dict[str, AssetMetrics]
) -> str:
    """Порядкове пояснення кожної дії."""
    if not actions:
        return "Конкретних дій не запропоновано."
    lines = ["Що конкретно змінюється:"]
    lines += [f"  - {explain_action(action, metrics)}" for action in actions]
    return "\n".join(lines)


def explain_action(action: dict[str, Any], metrics: dict[str, AssetMetrics]) -> str:
    """Пояснення однієї дії: скільки це грошей і чому саме так."""
    amount = get_settings().explanation_reference_amount
    symbol = action.get("symbol", "?")
    kind = action.get("action", "?")
    from_weight = float(action.get("from_weight") or 0.0)
    to_weight = float(action.get("to_weight") or 0.0)
    delta_money = abs(amount * (to_weight - from_weight))

    headline = {
        "BUY": (
            f"Купуємо {symbol} на {to_weight * 100:.1f}% портфеля — це "
            f"{money(amount * to_weight)} із {money(amount)}"
        ),
        "SELL": (
            f"Повністю продаємо {symbol} і вивільняємо "
            f"{money(amount * from_weight)}"
        ),
        "INCREASE": (
            f"Докуповуємо {symbol} на {money(delta_money)}: частка зростає з "
            f"{from_weight * 100:.1f}% до {to_weight * 100:.1f}%"
        ),
        "REDUCE": (
            f"Продаємо {symbol} на {money(delta_money)}: частка падає з "
            f"{from_weight * 100:.1f}% до {to_weight * 100:.1f}%"
        ),
    }.get(kind, f"{kind} {symbol} ({from_weight * 100:.1f}% -> {to_weight * 100:.1f}%)")

    # Прогноз рахуємо на тисячу вкладених — так позиції різного розміру
    # порівнюються між собою напряму
    outlook = project_asset(symbol, 1000.0)
    if outlook is None:
        return f"{headline}."

    forward = (
        f"очікувана дохідність {outlook.expected_annual_return * 100:+.0f}% річних "
        f"при коливаннях {outlook.volatility * 100:.0f}%; за "
        f"{outlook.horizon_days} днів кожна вкладена тисяча — це серединно "
        f"{money(outlook.median_value)} за ризику мінусу "
        f"{outlook.loss_probability * 100:.0f}%"
    )
    reason = f"{forward}, {_action_verdict(kind, outlook.loss_probability)}"

    # Нещодавня просадка береться з короткого вікна: це «як воно почувалось
    # останнім часом», а не підстава прогнозу
    metric = metrics.get(symbol)
    if metric is not None:
        reason += f". За останні {metric.lookback_days} днів просадка сягала "
        reason += f"{metric.max_drawdown * 100:.1f}%"
    return f"{headline}. Чому: {reason}."


# Нижче цього ризику мінусу актив вважається пристойним, і скорочення пояснюється
# розміром позиції, а не якістю активу
DECENT_LOSS_PROBABILITY = 0.45


def _action_verdict(kind: str, loss_probability: float) -> str:
    """Причина дії, узгоджена з прогнозом.

    Важливо не приписувати активу слабкість, коли його ріжуть через завелику
    частку: BTC із пристойним прогнозом скорочується заради зниження
    концентрації, а не тому, що він поганий.
    """
    if kind in {"BUY", "INCREASE"}:
        return "очікувана віддача виправдовує коливання"
    if loss_probability <= DECENT_LOSS_PROBABILITY:
        return (
            "сам актив непоганий — його частка просто завелика, і зменшення знижує "
            "залежність портфеля від одного імені"
        )
    return "забагато коливань за таку очікувану віддачу"


def odds_phrase(loss_probability: float) -> str:
    """Побутове формулювання ймовірності опинитись у мінусі."""
    if loss_probability >= 0.48:
        return "Простими словами: це майже підкидання монети."
    if loss_probability >= 0.40:
        return "Тобто шанси на плюс є, але перевага невелика."
    if loss_probability >= 0.30:
        return "Тобто перевага на боці плюса, хоча поганий сценарій цілком реальний."
    return "Тобто плюс значно ймовірніший за мінус, але не гарантований."



def _sharpe_phrase(value: float) -> str:
    """Побутове формулювання для коефіцієнта винагороди до ризику."""
    if value < 0:
        return "ризикували і втратили гроші"
    if value < 0.5:
        return "ризикували багато, а отримали за це мало"
    if value < 1.0:
        return "винагорода за ризик була помірною"
    if value < 2.0:
        return "ризик окупався добре"
    return "ризик окупався дуже добре"


def money(value: Any) -> str:
    """Сума у доларах із пробілом як роздільником тисяч."""
    if value is None:
        return "н/д"
    return "$" + f"{float(value):,.0f}".replace(",", " ")



def _pct(value: Any) -> str:
    """Відсотковий формат із захистом від None."""
    if value is None:
        return "н/д"
    return f"{float(value) * 100:.1f}%"

