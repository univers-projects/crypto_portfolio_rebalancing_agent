"""Тести інтерактивного режиму: часткове схвалення, explainer, роутинг чату.

Мережа не використовується: LLM підмінюється фейками, дані ринку детерміновані.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from app.agents.decision_engine import collect_metrics, evaluate_decision
from app.agents.explainer import (
    build_explanation,
    explain_action,
    money,
    render_fallback_explanation,
)
from app.config import get_settings
from app.data.analytics import compute_asset_metrics
from app.data.market_data import all_specs
from app.data.portfolio_store import DEMO_PORTFOLIO, seed_demo_portfolio
from app.data.projection import (
    baseline_return,
    estimation_weight,
    expected_annual_return,
    project,
    project_portfolio,
)
from app.domain.errors import InvalidPortfolioError
from app.domain.schemas import Portfolio
from app.interactive import visuals
from app.interactive.approval import merge_partial_approval, preview_partial, resolve_approval
from app.interactive.session import InteractiveSession
from tests.fakes import FailingLLM

ACTIONS = [
    {"action": "REDUCE", "symbol": "AVAX", "from_weight": 0.15, "to_weight": 0.078},
    {"action": "REDUCE", "symbol": "BTC", "from_weight": 0.40, "to_weight": 0.264},
    {"action": "REDUCE", "symbol": "ETH", "from_weight": 0.25, "to_weight": 0.087},
    {"action": "INCREASE", "symbol": "SOL", "from_weight": 0.20, "to_weight": 0.235},
    {"action": "BUY", "symbol": "BNB", "from_weight": 0.0, "to_weight": 0.336},
]


@pytest.fixture
def current() -> Portfolio:
    return Portfolio(positions=DEMO_PORTFOLIO)  # type: ignore[arg-type]


# --- Злиття часткового схвалення -----------------------------------------


def test_full_approval_reproduces_proposed_portfolio(current: Portfolio) -> None:
    merged = merge_partial_approval(current, ACTIONS, [True] * len(ACTIONS))
    assert merged.as_mapping() == pytest.approx(
        {"AVAX": 0.078, "BTC": 0.264, "ETH": 0.087, "SOL": 0.235, "BNB": 0.336}, abs=1e-3
    )


def test_partial_approval_weights_sum_to_one(current: Portfolio) -> None:
    accepted = [action["symbol"] == "BNB" for action in ACTIONS]
    merged = merge_partial_approval(current, ACTIONS, accepted)
    assert sum(position.weight for position in merged.positions) == pytest.approx(1.0, abs=1e-6)


def test_partial_approval_locks_accepted_symbol(current: Portfolio) -> None:
    """Схвалена дія задає точну вагу; решта стискається пропорційно."""
    accepted = [action["symbol"] == "BNB" for action in ACTIONS]
    merged = merge_partial_approval(current, ACTIONS, accepted)
    assert merged.weight_of("BNB") == pytest.approx(0.336, abs=1e-3)
    # Пропорції між незачепленими активами зберігаються
    assert merged.weight_of("BTC") / merged.weight_of("ETH") == pytest.approx(0.40 / 0.25, abs=1e-3)


def test_approving_only_reductions_raises_readable_error(current: Portfolio) -> None:
    """Схвалити лише скорочення не можна: вага нікуди не перерозподіляється."""
    accepted = [action["symbol"] != "BNB" for action in ACTIONS]
    with pytest.raises(InvalidPortfolioError) as info:
        merge_partial_approval(current, accepted=accepted, actions=ACTIONS)
    assert "вільних активів" in str(info.value)


def test_partial_approval_respects_max_position_weight(current: Portfolio) -> None:
    """Комбінація, що виводить позицію за ліміт, відхиляється, а не підганяється."""
    actions = [{"action": "SELL", "symbol": symbol, "from_weight": weight, "to_weight": 0.0}
               for symbol, weight in (("ETH", 0.25), ("SOL", 0.20), ("AVAX", 0.15))]
    with pytest.raises(InvalidPortfolioError) as info:
        merge_partial_approval(current, actions, [True, True, True])
    assert "максимального розміру" in str(info.value)


def test_mismatched_decision_count_is_rejected(current: Portfolio) -> None:
    with pytest.raises(InvalidPortfolioError):
        merge_partial_approval(current, ACTIONS, [True])


# --- Мапінг у гілки графа -------------------------------------------------


def test_resolve_approval_all_yes_is_approve(current: Portfolio) -> None:
    approval, positions = resolve_approval(current, ACTIONS, [True] * len(ACTIONS))
    assert approval == "approve"
    assert positions is None


def test_resolve_approval_all_no_is_reject(current: Portfolio) -> None:
    approval, positions = resolve_approval(current, ACTIONS, [False] * len(ACTIONS))
    assert approval == "reject"
    assert positions is None


def test_resolve_approval_partial_is_modify(current: Portfolio) -> None:
    accepted = [action["symbol"] == "BNB" for action in ACTIONS]
    approval, positions = resolve_approval(current, ACTIONS, accepted)
    assert approval == "modify"
    assert positions is not None
    assert sum(item["weight"] for item in positions) == pytest.approx(1.0, abs=1e-6)


def test_preview_reports_impossible_combination(current: Portfolio) -> None:
    accepted = [action["symbol"] != "BNB" for action in ACTIONS]
    assert "Комбінація неможлива" in preview_partial(current, ACTIONS, accepted)


# --- Explainer ------------------------------------------------------------


@pytest.fixture
def evidence(current: Portfolio, candidate_symbols: list[str]) -> dict[str, Any]:
    verdict, _ = evaluate_decision(current, candidate_symbols, 180)
    payload = verdict.as_evidence()
    payload["current_weights"] = current.as_mapping()
    return payload


def test_explanation_falls_back_when_llm_fails(evidence: dict[str, Any]) -> None:
    """Недоступний LLM не має ламати старт застосунку."""
    text = build_explanation(evidence, "REBALANCE", {}, llm=FailingLLM())
    assert "Зараз портфель складається з" in text


def test_explanation_rejected_when_it_contradicts_verdict(evidence: dict[str, Any]) -> None:
    """LLM не може перекрити вердикт політики."""
    from tests.fakes import FakeLLM

    llm = FakeLLM([AIMessage(content="Рекомендую HOLD, змінювати нічого не варто.")])
    text = build_explanation(evidence, "REBALANCE", {}, llm=llm)
    assert "Рекомендую HOLD" not in text


def test_explanation_uses_llm_text_when_consistent(evidence: dict[str, Any]) -> None:
    from tests.fakes import FakeLLM

    narrative = "Портфель варто перебудувати, бо BNB дає кращий профіль ризику."
    llm = FakeLLM([AIMessage(content=narrative)])
    assert build_explanation(evidence, "REBALANCE", {}, llm=llm) == narrative


def test_empty_llm_response_falls_back(evidence: dict[str, Any]) -> None:
    from tests.fakes import FakeLLM

    llm = FakeLLM([AIMessage(content="   ")])
    assert "Зараз портфель складається з" in build_explanation(evidence, "REBALANCE", {}, llm=llm)


def test_fallback_hold_explains_why_nothing_changes(evidence: dict[str, Any]) -> None:
    text = render_fallback_explanation(evidence, "HOLD", {})
    assert "Змінювати нічого не потрібно" in text
    assert "витрати з'їдять весь виграш" in text


def test_explain_action_without_metrics_still_readable() -> None:
    text = explain_action(ACTIONS[-1], {})
    assert "Купуємо BNB" in text


# --- Переклад цифр у гроші ------------------------------------------------


def test_explanation_translates_percentages_intomoney(evidence: dict[str, Any]) -> None:
    """Кожен важливий відсоток має супроводжуватись сумою в грошах."""
    text = render_fallback_explanation(evidence, "REBALANCE", {})
    amount = get_settings().explanation_reference_amount
    assert "Уявіть" in text
    assert f"${amount:,.0f}".replace(",", " ") in text
    # Просадка перекладена у вартість портфеля в найгіршій точці
    drawdown = float(evidence["current_portfolio"]["max_drawdown"])
    assert money(amount * (1 - drawdown)) in text


def test_turnover_is_stated_inmoney(evidence: dict[str, Any]) -> None:
    """Оборот подається як сума, що проходить через ринок, а не лише як відсоток."""
    text = render_fallback_explanation(evidence, "REBALANCE", {})
    amount = get_settings().explanation_reference_amount
    assert money(amount * float(evidence["turnover"])) in text


def test_action_explanation_reports_amount_inmoney() -> None:
    """Дія показує, скільки саме грошей рухається."""
    amount = get_settings().explanation_reference_amount
    text = explain_action(ACTIONS[-1], {})
    assert money(amount * 0.336) in text


def test_explanation_never_promises_future_returns(evidence: dict[str, Any]) -> None:
    """Прогноз має подаватись як діапазон, а не як обіцянка."""
    text = render_fallback_explanation(evidence, "REBALANCE", {})
    assert "не обіцянка" in text
    assert "ринок не зобов'язаний повторювати минуле" in text


# --- Прогноз уперед -------------------------------------------------------


def test_explanation_leads_with_forward_projection(evidence: dict[str, Any]) -> None:
    """Пояснення показує очікування на горизонт, а не лише минуле."""
    horizon = get_settings().projection_horizon_days
    text = render_fallback_explanation(evidence, "REBALANCE", {})
    assert f"За наступні {horizon} днів" in text
    assert "Імовірність завершити період у мінусі" in text


def test_projection_is_reported_as_a_range(evidence: dict[str, Any]) -> None:
    """Медіана завжди супроводжується поганим і хорошим сценаріями."""
    amount = get_settings().explanation_reference_amount
    projection = project_portfolio(evidence["current_weights"], amount)
    assert projection is not None
    text = render_fallback_explanation(evidence, "REBALANCE", {})
    for value in (
        projection.median_value,
        projection.downside_value,
        projection.upside_value,
    ):
        assert money(value) in text


def test_confidence_grows_with_sample_length() -> None:
    """Довша історія дає більшу вагу власному результату активу."""
    assert estimation_weight(180) < estimation_weight(720)
    assert 0.0 < estimation_weight(180) < 1.0
    assert estimation_weight(720) == pytest.approx(0.40, abs=0.01)


def test_expected_return_blends_history_with_risk_premium() -> None:
    """Очікувана дохідність — суміш власного результату і плати за ризик."""
    historical, volatility = 2.00, 0.50
    weight = estimation_weight(720)
    expected = expected_annual_return(historical, volatility, 720)

    assert expected == pytest.approx(
        weight * historical + (1 - weight) * baseline_return(volatility), abs=1e-9
    )
    # Екстремальна історія підтягується до розумного рівня
    assert expected < historical
    # Але власний результат усе ще піднімає оцінку над базовою
    assert expected > baseline_return(volatility)


def test_blended_estimate_beats_raw_history_on_known_drifts() -> None:
    """Головне виправдання методу: він точніший, а не просто обережніший.

    У mock-universe істинний дрейф кожного активу відомий, тому точність оцінки
    можна виміряти напряму. Якщо ця перевірка почне падати — метод втратив сенс.
    """
    days = get_settings().estimation_lookback_days
    raw_errors: list[float] = []
    blended_errors: list[float] = []

    for spec in all_specs():
        if spec.is_stablecoin or not spec.data_quality_ok or spec.history_days < days:
            continue
        metrics = compute_asset_metrics(spec.symbol, days)
        raw_errors.append(metrics.annualized_return - spec.annual_drift)
        blended_errors.append(
            expected_annual_return(metrics.annualized_return, metrics.volatility, days)
            - spec.annual_drift
        )

    assert len(raw_errors) > 10

    def rmse(errors: list[float]) -> float:
        return math.sqrt(sum(error**2 for error in errors) / len(errors))

    # Змішана оцінка має бути щонайменше в півтора раза точнішою
    assert rmse(blended_errors) * 1.5 < rmse(raw_errors)


def test_volatility_is_used_without_adjustment() -> None:
    """Розмах коливань береться повністю: він оцінюється надійно."""
    low = project(0.20, 0.10, 10_000)
    high = project(0.20, 0.90, 10_000)
    assert low.volatility == 0.10
    assert high.volatility == 0.90
    low_band = low.upside_value - low.downside_value
    high_band = high.upside_value - high.downside_value
    assert high_band > low_band * 5


def test_projection_loss_probability_is_sane() -> None:
    """Нульовий дрейф дає шанс мінусу трохи вище 50% (через дисперсію)."""
    flat = project(0.0, 0.40, 10_000)
    assert 0.5 < flat.loss_probability < 0.6
    strong = project(2.0, 0.20, 10_000)
    assert strong.loss_probability < flat.loss_probability


def test_portfolio_projection_benefits_from_diversification() -> None:
    """Кореляції враховані: портфель коливається менше за окремі активи."""
    projection = project_portfolio({"BTC": 0.4, "ETH": 0.25, "SOL": 0.2, "AVAX": 0.15}, 10_000)
    assert projection is not None
    days = get_settings().estimation_lookback_days
    worst_asset_vol = max(
        compute_asset_metrics(symbol, days).volatility
        for symbol in ("BTC", "ETH", "SOL", "AVAX")
    )
    assert projection.volatility < worst_asset_vol


def test_reduce_of_healthy_asset_is_framed_as_concentration() -> None:
    """Пристойний актив не можна називати слабким лише тому, що його ріжуть."""
    metrics, _ = collect_metrics(["BTC"], 180)
    text = explain_action(
        {"action": "REDUCE", "symbol": "BTC", "from_weight": 0.40, "to_weight": 0.264},
        metrics,
    )
    assert "частка просто завелика" in text
    assert "забагато коливань" not in text


def test_headline_reports_risk_change(evidence: dict[str, Any]) -> None:
    """Картка називає зниження ризику, а не лише очікуваний прибуток."""
    amount = get_settings().explanation_reference_amount
    before = project_portfolio(evidence["current_weights"], amount)
    after = project_portfolio(evidence["best_candidate_weights"], amount)
    _, details = _session([], _Recorder())._headline_text(evidence, "REBALANCE", before, after)
    assert any("ризик збитку падає" in line for line in details)


# --- Візуальний шар -------------------------------------------------------


def test_forecast_bar_places_markers_in_order() -> None:
    """Смуга діапазону: ├ песимістично, ● серединно, ┤ оптимістично."""
    bar = visuals.range_bar(8_000, 10_000, 13_000, 7_500, 13_500)
    assert bar.index("├") < bar.index("●") < bar.index("┤")


def test_forecast_shares_one_scale_between_rows() -> None:
    """Два сценарії малюються на спільній шкалі, інакше їх не порівняти."""
    rows = [
        visuals.RangeRow("зараз", 8_000, 10_000, 12_000, 0.40),
        visuals.RangeRow("план", 9_000, 11_000, 13_000, 0.30),
    ]
    text = visuals.render_forecast(rows, 90, 10_000)
    assert "зараз" in text and "план" in text
    lines = [line for line in text.splitlines() if "├" in line]
    assert len(lines) == 2
    # Кращий сценарій зсунутий праворуч на тій самій шкалі
    assert lines[1].index("●") > lines[0].index("●")


def test_drift_marks_assets_outside_the_band() -> None:
    """Актив за межами смуги ±5 п.п. позначається окремо."""
    text = visuals.render_drift({"BTC": 0.40, "SOL": 0.20}, {"BTC": 0.26, "SOL": 0.22})
    assert "поза" in text
    assert "ок" in text
    assert "+14.0 п.п." in text


def test_checklist_renders_toggles_and_summary() -> None:
    text = visuals.render_checklist(
        list(ACTIONS),
        [True, False, True, True, True],
        {"BNB": (1085.0, 0.38)},
        [False] * len(ACTIONS),
        "Обрано 4 з 5 · оборот 61.4%",
        "Вийде: BNB 33.6",
        "[a] усі · [n] жодної",
    )
    assert "[x]" in text and "[ ]" in text
    assert "Обрано 4 з 5" in text
    # Дужки підказки не мають бути зʼїдені розміткою rich
    assert "[a] усі" in text


def test_checklist_marks_delegated_actions() -> None:
    """Делеговані дії підписані, щоб було видно, за що відповідає агент."""
    text = visuals.render_checklist(
        list(ACTIONS),
        [True] * len(ACTIONS),
        {},
        [False, False, False, True, False],
        "Обрано 5 з 5",
        "",
        "[↵] далі",
    )
    assert "авто" in text


def test_action_explanation_is_forward_looking() -> None:
    """Пояснення дії веде очікуваннями, а недавня історія лишається контекстом."""
    metrics, _ = collect_metrics(["BNB"], 180)
    text = explain_action(ACTIONS[-1], metrics)
    horizon = get_settings().projection_horizon_days
    assert "очікувана дохідність" in text
    assert f"за {horizon} днів" in text
    assert "За останні 180 днів просадка" in text


def test_headline_leads_with_plain_language(evidence: dict[str, Any]) -> None:
    """Картка рішення починається зі зрозумілого формулювання, а не з балів."""
    out = _Recorder()
    session = _session([], out)
    amount = get_settings().explanation_reference_amount
    before = project_portfolio(evidence["current_weights"], amount)
    after = project_portfolio(evidence["best_candidate_weights"], amount)

    subtitle, details = session._headline_text(evidence, "REBALANCE", before, after)
    card = visuals.render_headline("REBALANCE", subtitle, details)

    assert "ПЕРЕБАЛАНСУВАТИ портфель" in card
    assert "оборот" in card
    # Сирі числа лишаються, але службовим рядком, а не заголовком
    assert any("[розрахунок:" in line for line in details)


def test_headline_for_hold_explains_inaction(evidence: dict[str, Any]) -> None:
    out = _Recorder()
    subtitle, _ = _session([], out)._headline_text(evidence, "HOLD", None, None)
    card = visuals.render_headline("HOLD", subtitle, [])
    assert "НІЧОГО НЕ ЗМІНЮВАТИ" in card


def test_fallback_avoids_internal_jargon(evidence: dict[str, Any]) -> None:
    """Пояснення не має протікати внутрішніми термінами системи."""
    text = render_fallback_explanation(evidence, "REBALANCE", {})
    for jargon in ("policy", "вердикт", "improvement score", "quality score", "HHI"):
        assert jargon.lower() not in text.lower()


def test_weakness_block_names_the_worst_asset(
    evidence: dict[str, Any], candidate_symbols: list[str]
) -> None:
    """Слабке місце названо конкретним активом, а не абстрактно."""
    metrics, _ = collect_metrics(sorted(evidence["current_weights"]), 180)
    text = render_fallback_explanation(evidence, "REBALANCE", metrics)
    worst = min(metrics.items(), key=lambda item: item[1].sharpe_like)[0]
    assert f"Найбільше тягне вниз {worst}" in text


# --- Роутинг чату ---------------------------------------------------------


class _Recorder:
    """Збирач виводу сесії."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str) -> None:
        self.lines.append(str(text))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _session(inputs: list[str], out: _Recorder) -> InteractiveSession:
    queue = list(inputs)

    def reader(_: str) -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    return InteractiveSession("test-thread", reader=reader, writer=out)


def test_exit_command_stops_session() -> None:
    out = _Recorder()
    assert _session([], out)._handle_command("/exit", workflow=None) is False


def test_help_command_lists_commands() -> None:
    out = _Recorder()
    session = _session([], out)
    assert session._handle_command("/help", workflow=None) is True
    assert "/portfolio" in out.text


def test_portfolio_command_shows_weights() -> None:
    out = _Recorder()
    seed_demo_portfolio()
    _session([], out)._handle_command("/portfolio", workflow=None)
    assert "BTC" in out.text


def test_unknown_command_is_reported() -> None:
    out = _Recorder()
    assert _session([], out)._handle_command("/nope", workflow=None) is True
    assert "Невідома команда" in out.text


def test_checklist_starts_with_everything_selected(current: Portfolio) -> None:
    """За замовчуванням обрано весь план — Enter підтверджує його цілком."""
    out = _Recorder()
    session = _session([""], out)
    assert session._select_actions(current, list(ACTIONS), {}) == [True] * len(ACTIONS)


def test_checklist_toggles_by_number(current: Portfolio) -> None:
    out = _Recorder()
    session = _session(["2", "3", ""], out)
    assert session._select_actions(current, list(ACTIONS), {}) == [
        True,
        False,
        False,
        True,
        True,
    ]


def test_checklist_toggle_is_reversible(current: Portfolio) -> None:
    out = _Recorder()
    session = _session(["2", "2", ""], out)
    assert session._select_actions(current, list(ACTIONS), {}) == [True] * len(ACTIONS)


def test_checklist_none_then_all(current: Portfolio) -> None:
    out = _Recorder()
    session = _session(["n", "a", ""], out)
    assert session._select_actions(current, list(ACTIONS), {}) == [True] * len(ACTIONS)


def test_checklist_allows_confirming_empty_selection(current: Portfolio) -> None:
    """Жодної галочки — валідний стан: це відмова."""
    out = _Recorder()
    session = _session(["n", ""], out)
    assert session._select_actions(current, list(ACTIONS), {}) == [False] * len(ACTIONS)


def test_checklist_blocks_confirming_impossible_combination(current: Portfolio) -> None:
    """Неможливу комбінацію видно одразу і підтвердити її не можна."""
    out = _Recorder()
    # Знімаємо купівлю BNB -> вивільнену вагу нема куди подіти
    session = _session(["5", "", "5", ""], out)
    assert session._select_actions(current, list(ACTIONS), {}) == [True] * len(ACTIONS)
    assert "неможливо" in out.text
    assert "Спершу зробіть комбінацію можливою" in out.text


def test_checklist_shows_live_summary(current: Portfolio) -> None:
    """Підсумок перераховується до підтвердження, а не після."""
    out = _Recorder()
    _session(["2", ""], out)._select_actions(current, list(ACTIONS), {})
    assert "Обрано 4 з 5" in out.text
    assert "оборот" in out.text
    assert "ризик збитку" in out.text
    assert "Вийде:" in out.text


def test_checklist_reprompts_on_garbage(current: Portfolio) -> None:
    out = _Recorder()
    session = _session(["ага", "99", ""], out)
    assert session._select_actions(current, list(ACTIONS), {}) == [True] * len(ACTIONS)
    assert "Не зрозумів" in out.text


def test_checklist_returns_none_on_interrupt(current: Portfolio) -> None:
    out = _Recorder()
    assert _session([], out)._select_actions(current, list(ACTIONS), {}) is None


def test_checklist_warns_when_toggle_changes_nothing(current: Portfolio) -> None:
    """Зняти галочку зі скорочення часто не змінює ваг — це треба сказати."""
    out = _Recorder()
    _session(["2", ""], out)._select_actions(current, list(ACTIONS), {})
    assert "Склад той самий, що й з усіма діями" in out.text


def test_revised_plan_identical_to_original_is_flagged() -> None:
    """Повторне підтвердження ідентичного плану пояснюється, а не мовчить."""
    out = _Recorder()
    _session(["n"], out)._confirm_revised(list(ACTIONS), list(ACTIONS))
    assert "збігається з початковим" in out.text


def test_revised_plan_with_changes_is_not_flagged() -> None:
    out = _Recorder()
    changed = [*ACTIONS[:-1], {**ACTIONS[-1], "to_weight": 0.20}]
    _session(["n"], out)._confirm_revised(changed, list(ACTIONS))
    assert "збігається з початковим" not in out.text


def test_answer_renders_used_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Підсумок відповіді читає ключ 'tool' зі ReActResult.tool_calls."""
    from app.agents.react_agent import ReActResult

    result = ReActResult(
        output="Максимальна просадка — це найглибше падіння від піку.",
        steps=2,
        stop_reason="completed",
        tool_calls=[{"tool": "knowledge_search", "response": {"status": "success"}}],
    )
    monkeypatch.setattr("app.interactive.session.run_react_task", lambda *a, **k: result)

    out = _Recorder()
    _session([], out)._answer("what is max drawdown?")
    assert "knowledge_search" in out.text
    assert "найглибше падіння" in out.text


def test_answer_survives_agent_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Збій агента не має завершувати чат."""

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("LLM недоступний")

    monkeypatch.setattr("app.interactive.session.run_react_task", _boom)
    out = _Recorder()
    _session([], out)._answer("будь-що")
    assert "Не вдалося обробити питання" in out.text


# --- Демо-старт -----------------------------------------------------------


def test_demo_portfolio_yields_rebalance(candidate_symbols: list[str]) -> None:
    """Стартовий стан інтерактивного режиму детерміновано вимагає ребалансу."""
    seeded = seed_demo_portfolio()
    verdict, _ = evaluate_decision(seeded, candidate_symbols, 180)
    assert verdict.decision == "REBALANCE"
    assert verdict.net_improvement >= verdict.threshold
    assert verdict.proposal is not None
    assert len(verdict.proposal.actions) >= 3
