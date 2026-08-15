"""Тести короткого висновку і детермінованих питань до рішення.

Мережа не використовується: питання і відповіді будуються з даних, LLM тут
не бере участі взагалі.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.decision_engine import collect_metrics, evaluate_decision
from app.agents.decision_questions import MAX_QUESTIONS, build_questions
from app.agents.explainer import render_summary
from app.config import get_settings
from app.domain.schemas import AssetMetrics, Portfolio
from app.interactive.session import InteractiveSession

DEMO = {"BTC": 0.40, "ETH": 0.25, "SOL": 0.20, "AVAX": 0.15}
# Склад, який на вікні рішення вже достатньо якісний
GOOD = {"BTC": 0.30, "BNB": 0.22, "TRX": 0.26, "XRP": 0.22}


def _evidence(positions: dict[str, float], candidates: list[str]) -> dict[str, Any]:
    portfolio = Portfolio(positions=positions)  # type: ignore[arg-type]
    verdict, _ = evaluate_decision(
        portfolio, candidates, get_settings().decision_lookback_days
    )
    payload = verdict.as_evidence()
    payload["current_weights"] = portfolio.as_mapping()
    return payload


def _metrics(positions: dict[str, float]) -> dict[str, AssetMetrics]:
    collected, _ = collect_metrics(
        sorted(positions), get_settings().decision_lookback_days
    )
    return collected


@pytest.fixture
def rebalance_evidence(candidate_symbols: list[str]) -> dict[str, Any]:
    return _evidence(DEMO, candidate_symbols)


@pytest.fixture
def hold_evidence(candidate_symbols: list[str]) -> dict[str, Any]:
    return _evidence(GOOD, candidate_symbols)


# --- Питання --------------------------------------------------------------


def test_rebalance_questions_cover_purchase_and_turnover(
    rebalance_evidence: dict[str, Any]
) -> None:
    """Питання будуються з фактичного плану, а не з загального шаблону."""
    questions = build_questions(rebalance_evidence, "REBALANCE", _metrics(DEMO))
    joined = " ".join(item.text for item in questions)

    bought = [
        action["symbol"]
        for action in rebalance_evidence["proposed_actions"]
        if action["action"] == "BUY"
    ]
    assert bought, "демо-план має містити покупку"
    assert any(symbol in joined for symbol in bought)
    assert "Оборот" in joined
    assert "прогноз" in joined.lower()


def test_questions_are_capped(rebalance_evidence: dict[str, Any]) -> None:
    """Список входів має лишатись коротким, інакше він сам стає стіною."""
    questions = build_questions(rebalance_evidence, "REBALANCE", _metrics(DEMO))
    assert 0 < len(questions) <= MAX_QUESTIONS


def test_every_question_has_a_ready_answer(rebalance_evidence: dict[str, Any]) -> None:
    """Питання без відповіді не має існувати — це головна перевага детермінізму."""
    for question in build_questions(rebalance_evidence, "REBALANCE", _metrics(DEMO)):
        assert question.answer.strip()
        assert len(question.answer) > 80


def test_hold_questions_explain_inaction(hold_evidence: dict[str, Any]) -> None:
    """Для HOLD питання інші: людину цікавить саме бездіяльність."""
    questions = build_questions(hold_evidence, "HOLD", _metrics(GOOD))
    joined = " ".join(item.text for item in questions)
    assert "нічого не робить" in joined
    assert "з'явилась пропозиція" in joined


def test_questions_never_promise_prices(rebalance_evidence: dict[str, Any]) -> None:
    """Питання обмежені тим, що агент справді вміє рахувати."""
    questions = build_questions(rebalance_evidence, "REBALANCE", _metrics(DEMO))
    joined = " ".join(item.text for item in questions).lower()
    for forbidden in ("ціна зараз", "завтра", "наступн", "коли купувати"):
        assert forbidden not in joined


# --- Короткий висновок ----------------------------------------------------


def test_summary_leads_with_risk_and_stays_short(
    rebalance_evidence: dict[str, Any]
) -> None:
    """Висновок має бути видно одразу, без прокрутки."""
    summary = render_summary(rebalance_evidence, "REBALANCE", _metrics(DEMO))
    lines = summary.splitlines()
    assert len(lines) <= 4
    assert len(summary) < 500
    assert all(len(line) <= 78 for line in lines), "рядок не має переноситись у панелі"
    assert "ризик" in lines[0].lower()


def test_summary_works_for_hold(hold_evidence: dict[str, Any]) -> None:
    summary = render_summary(hold_evidence, "HOLD", _metrics(GOOD))
    assert "без змін" in summary
    assert "ризик мінусу" in summary


def test_summary_needs_no_llm(rebalance_evidence: dict[str, Any]) -> None:
    """Висновок будується з даних: жодного мережевого виклику тут бути не може."""
    first = render_summary(rebalance_evidence, "REBALANCE", _metrics(DEMO))
    second = render_summary(rebalance_evidence, "REBALANCE", _metrics(DEMO))
    assert first == second


# --- Взаємодія в сесії ----------------------------------------------------


class _Recorder:
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


def test_enter_skips_straight_to_confirmation(
    rebalance_evidence: dict[str, Any]
) -> None:
    """Порожній ввід не показує довгий текст — це шлях за замовчуванням."""
    out = _Recorder()
    _session([""], out)._explore_decision(rebalance_evidence, "REBALANCE", _metrics(DEMO))
    assert "Спитайте, якщо цікаво" in out.text
    assert "Прогноз — це діапазон імовірних результатів" not in out.text


def test_number_shows_answer_without_llm(rebalance_evidence: dict[str, Any]) -> None:
    out = _Recorder()
    questions = build_questions(rebalance_evidence, "REBALANCE", _metrics(DEMO))
    _session(["1", ""], out)._explore_decision(
        rebalance_evidence, "REBALANCE", _metrics(DEMO)
    )
    assert questions[0].answer in out.text


def test_e_expands_full_explanation(
    rebalance_evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Повний наратив будується лениво — лише на явний запит."""
    from app.interactive import session as session_module

    calls: list[str] = []

    def fake_explanation(*args: Any, **kwargs: Any) -> str:
        calls.append("built")
        return "ПОВНИЙ ТЕКСТ"

    monkeypatch.setattr(session_module, "build_explanation", fake_explanation)

    out = _Recorder()
    _session([""], out)._explore_decision(rebalance_evidence, "REBALANCE", _metrics(DEMO))
    assert not calls, "без запиту довгий текст не має будуватись"

    out = _Recorder()
    _session(["e", ""], out)._explore_decision(
        rebalance_evidence, "REBALANCE", _metrics(DEMO)
    )
    assert calls == ["built"]
    assert "ПОВНИЙ ТЕКСТ" in out.text


def test_unknown_input_is_explained(rebalance_evidence: dict[str, Any]) -> None:
    out = _Recorder()
    _session(["zzz", ""], out)._explore_decision(
        rebalance_evidence, "REBALANCE", _metrics(DEMO)
    )
    assert "Не зрозумів" in out.text


def test_panel_is_printed_once(rebalance_evidence: dict[str, Any]) -> None:
    """Панель не передруковується після кожної відповіді.

    Інакше кожна відповідь виштовхує попередню за межі екрана.
    """
    out = _Recorder()
    _session(["1", "2", ""], out)._explore_decision(
        rebalance_evidence, "REBALANCE", _metrics(DEMO)
    )
    assert out.text.count("Спитайте, якщо цікаво") == 1
    assert "ще питання" in out.text


def test_summary_names_risk_direction(rebalance_evidence: dict[str, Any]) -> None:
    """Зниження ризику не має читатись як зростання."""
    summary = render_summary(rebalance_evidence, "REBALANCE", _metrics(DEMO))
    assert "падає" in summary
    assert "тобто на" in summary
