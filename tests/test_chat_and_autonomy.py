"""Тести чату (підказки, /whatif, /history) і зароблених повноважень.

Мережа не використовується: LLM підмінюється фейками.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.agents.suggestions import FollowUpQuestions, suggest_followups
from app.config import get_settings
from app.data.portfolio_store import seed_demo_portfolio
from app.interactive import autonomy, visuals
from app.interactive.session import InteractiveSession
from app.observability import history, trajectory
from tests.fakes import FailingLLM

REDUCE_SMALL = {"action": "REDUCE", "symbol": "BTC", "from_weight": 0.40, "to_weight": 0.37}
INCREASE_SMALL = {"action": "INCREASE", "symbol": "SOL", "from_weight": 0.20, "to_weight": 0.23}
REDUCE_BIG = {"action": "REDUCE", "symbol": "ETH", "from_weight": 0.25, "to_weight": 0.08}
BUY_NEW = {"action": "BUY", "symbol": "BNB", "from_weight": 0.0, "to_weight": 0.33}


class _Recorder:
    """Збирає і вивід, і тексти запитів — підказки часто живуть у промпті."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str) -> None:
        self.lines.append(str(text))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@pytest.fixture
def current_demo() -> Any:
    """Поточний портфель для перевірок чек-листа."""
    return seed_demo_portfolio()


def _session(inputs: list[str], out: _Recorder) -> InteractiveSession:
    queue = list(inputs)

    def reader(prompt: str) -> str:
        out(prompt)
        if not queue:
            raise EOFError
        return queue.pop(0)

    return InteractiveSession("test-thread", reader=reader, writer=out)


# --- Що взагалі можна делегувати ------------------------------------------


def test_only_small_weight_changes_are_delegable() -> None:
    assert autonomy.is_delegable(REDUCE_SMALL)
    assert autonomy.is_delegable(INCREASE_SMALL)


def test_large_changes_are_never_delegable() -> None:
    """Скільки б схвалень не назбиралось, велике перекладання лишається за людиною."""
    assert not autonomy.is_delegable(REDUCE_BIG)


def test_buying_a_new_asset_is_never_delegable() -> None:
    assert not autonomy.is_delegable(BUY_NEW)


def test_full_sale_is_never_delegable() -> None:
    sell = {"action": "SELL", "symbol": "ETH", "from_weight": 0.02, "to_weight": 0.0}
    assert not autonomy.is_delegable(sell)


# --- Накопичення довіри ---------------------------------------------------


def test_only_delegable_actions_count_toward_trust() -> None:
    """Схвалення великих дій не наближає автономію."""
    state = autonomy.record_decisions(
        [REDUCE_BIG, BUY_NEW, REDUCE_SMALL], [True, True, True], autonomy.AutonomyState()
    )
    assert state.total == 1
    assert state.accepted == 1


def test_rejections_are_counted_too() -> None:
    state = autonomy.record_decisions(
        [REDUCE_SMALL, INCREASE_SMALL], [True, False], autonomy.AutonomyState()
    )
    assert (state.accepted, state.rejected) == (1, 1)
    assert state.acceptance_rate == pytest.approx(0.5)


def test_offer_requires_enough_history() -> None:
    """Кількох схвалень замало: потрібен поріг за обсягом."""
    state = autonomy.AutonomyState(accepted=2, rejected=0)
    assert not autonomy.can_offer(state)


def test_offer_requires_high_acceptance_rate() -> None:
    """Історія є, але людина часто відмовляє — пропозиції не буде."""
    state = autonomy.AutonomyState(accepted=5, rejected=5)
    assert state.total >= get_settings().autonomy_min_decisions
    assert not autonomy.can_offer(state)


def test_offer_appears_when_trust_is_earned() -> None:
    assert autonomy.can_offer(autonomy.AutonomyState(accepted=8, rejected=1))


def test_granted_state_is_not_offered_again() -> None:
    state = autonomy.grant(autonomy.AutonomyState(accepted=8, rejected=1))
    assert state.granted
    assert not autonomy.can_offer(state)


def test_revoke_keeps_accumulated_history() -> None:
    """Відкликання не стирає статистику — інакше довіру довелось би заробляти з нуля."""
    granted = autonomy.grant(autonomy.AutonomyState(accepted=8, rejected=1))
    revoked = autonomy.revoke(granted)
    assert not revoked.granted
    assert (revoked.accepted, revoked.rejected) == (8, 1)


def test_state_survives_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "autonomy.json"
    state = autonomy.grant(autonomy.AutonomyState(accepted=9, rejected=1))
    autonomy.save_state(state, path)
    loaded = autonomy.load_state(path)
    assert loaded.granted
    assert loaded.accepted == 9
    assert loaded.max_delta == pytest.approx(get_settings().autonomy_max_delta)


def test_corrupted_state_file_does_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "autonomy.json"
    path.write_text("{ не json", encoding="utf-8")
    assert autonomy.load_state(path) == autonomy.AutonomyState()


def test_describe_explains_what_is_missing() -> None:
    text = autonomy.describe(autonomy.AutonomyState(accepted=2, rejected=0))
    assert "Делегування не надано" in text
    assert "Ще" in text


def test_describe_states_human_keeps_final_say() -> None:
    """Делегування не має читатись як автоматичне виконання угод."""
    text = autonomy.describe(autonomy.grant(autonomy.AutonomyState(accepted=8, rejected=1)))
    assert "Фінальне підтвердження плану залишається за вами" in text


# --- Команда /autonomy ----------------------------------------------------


def test_autonomy_command_shows_status() -> None:
    out = _Recorder()
    _session([], out)._handle_autonomy("/autonomy")
    assert "Делегування не надано" in out.text


def test_autonomy_on_is_refused_without_earned_trust() -> None:
    out = _Recorder()
    _session([], out)._handle_autonomy("/autonomy on")
    assert "Ще рано" in out.text
    assert not autonomy.load_state().granted


def test_autonomy_on_then_off() -> None:
    autonomy.save_state(autonomy.AutonomyState(accepted=8, rejected=1))
    out = _Recorder()
    session = _session([], out)

    session._handle_autonomy("/autonomy on")
    assert autonomy.load_state().granted

    session._handle_autonomy("/autonomy off")
    assert not autonomy.load_state().granted
    assert "відкликано" in out.text


def test_autonomy_rejects_unknown_argument() -> None:
    out = _Recorder()
    _session([], out)._handle_autonomy("/autonomy maybe")
    assert "Використання" in out.text


def test_offer_is_declined_without_granting() -> None:
    """Відповідь «ні» не має надавати повноважень, але має зберегти статистику."""
    autonomy.save_state(autonomy.AutonomyState(accepted=7, rejected=0))
    out = _Recorder()
    _session(["n"], out)._update_autonomy([REDUCE_SMALL], [True])
    state = autonomy.load_state()
    assert not state.granted
    assert state.accepted == 8


def test_offer_accepted_grants_delegation() -> None:
    autonomy.save_state(autonomy.AutonomyState(accepted=7, rejected=0))
    out = _Recorder()
    _session(["y"], out)._update_autonomy([REDUCE_SMALL], [True])
    assert autonomy.load_state().granted
    assert "Делеговано" in out.text


# --- Скорочений шлях для повністю делегованого плану ----------------------


def test_shortcut_applies_only_when_every_action_is_delegable() -> None:
    granted = autonomy.grant(autonomy.AutonomyState(accepted=8, rejected=1))
    assert autonomy.covers_all([REDUCE_SMALL, INCREASE_SMALL], granted)
    assert not autonomy.covers_all([REDUCE_SMALL, BUY_NEW], granted)


def test_shortcut_never_applies_without_delegation() -> None:
    plain = autonomy.AutonomyState(accepted=8, rejected=1)
    assert not autonomy.covers_all([REDUCE_SMALL], plain)


def test_shortcut_still_asks_the_human(current_demo: Any) -> None:
    """Скорочений шлях економить перегляд, але не прибирає підтвердження."""
    autonomy.save_state(autonomy.grant(autonomy.AutonomyState(accepted=8, rejected=1)))
    out = _Recorder()
    decisions = _session(["y"], out)._select_actions(current_demo, [REDUCE_SMALL], {})
    assert decisions == [True]
    assert "детальний перегляд пропущено" in out.text
    assert "Виконати?" in out.text


def test_shortcut_can_be_declined(current_demo: Any) -> None:
    autonomy.save_state(autonomy.grant(autonomy.AutonomyState(accepted=8, rejected=1)))
    out = _Recorder()
    assert _session(["n"], out)._select_actions(current_demo, [REDUCE_SMALL], {}) == [False]


def test_shortcut_can_expand_into_full_checklist(current_demo: Any) -> None:
    """Людина завжди може вимагати подробиць."""
    autonomy.save_state(autonomy.grant(autonomy.AutonomyState(accepted=8, rejected=1)))
    out = _Recorder()
    decisions = _session(["d", ""], out)._select_actions(current_demo, [REDUCE_SMALL], {})
    assert decisions == [True]
    assert "Підтвердження" in out.text


# --- /whatif --------------------------------------------------------------


def test_whatif_without_arguments_shows_usage() -> None:
    out = _Recorder()
    _session([], out)._handle_whatif("/whatif")
    assert "Використання" in out.text


def test_whatif_reports_invalid_weights() -> None:
    out = _Recorder()
    _session([], out)._handle_whatif("/whatif BTC=0.5,ETH=0.2")
    assert "Не вдалося розібрати склад" in out.text


def test_whatif_projects_hypothetical_portfolio() -> None:
    out = _Recorder()
    seed_demo_portfolio()
    _session([], out)._handle_whatif("/whatif BTC=0.4,ETH=0.3,SOL=0.3")
    assert "Гіпотетичний склад" in out.text
    assert "Прогноз на" in out.text
    assert "зачепив би" in out.text


def test_whatif_does_not_touch_the_portfolio() -> None:
    """Розрахунок «а що якщо» не має змінювати реальний стан."""
    before = seed_demo_portfolio().as_mapping()
    _session([], _Recorder())._handle_whatif("/whatif BTC=0.4,ETH=0.3,SOL=0.3")
    from app.data.portfolio_store import load_portfolio

    assert load_portfolio().as_mapping() == before


# --- /history -------------------------------------------------------------


def _write_trajectory(entries: list[dict[str, Any]]) -> Path:
    path = get_settings().trajectory_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def test_history_is_empty_without_completed_cycles() -> None:
    assert history.read_history() == []


def test_history_reads_decisions_from_trajectory() -> None:
    _write_trajectory(
        [
            {
                "timestamp": "2026-08-13T09:00:00+00:00",
                "node": "decide",
                "event": "decision_made",
                "decision": "REBALANCE",
                "net_improvement": 2.05,
                "threshold": 0.15,
                "turnover": 0.742,
            },
            {"node": "execute_rebalance", "tool": "mock_execute_rebalance", "status": "success"},
        ]
    )
    records = history.read_history()
    assert len(records) == 1
    assert records[0].decision == "REBALANCE"
    assert records[0].outcome == "виконано (mock)"
    assert records[0].when == "13.08 09:00"


def test_history_marks_rejected_decision() -> None:
    _write_trajectory(
        [
            {
                "timestamp": "2026-08-13T09:00:00+00:00",
                "event": "decision_made",
                "decision": "REBALANCE",
            },
            {"node": "execute_rebalance", "event": "rejected"},
        ]
    )
    assert history.read_history()[0].outcome == "відхилено людиною"


def test_history_does_not_leak_outcome_between_cycles() -> None:
    """Виконання другого циклу не має приписуватись першому."""
    _write_trajectory(
        [
            {
                "timestamp": "2026-08-12T09:00:00+00:00",
                "event": "decision_made",
                "decision": "HOLD",
            },
            {
                "timestamp": "2026-08-13T09:00:00+00:00",
                "event": "decision_made",
                "decision": "REBALANCE",
            },
            {"node": "execute_rebalance", "tool": "mock_execute_rebalance", "status": "success"},
        ]
    )
    records = history.read_history()
    assert records[0].outcome == "без змін"
    assert records[1].outcome == "виконано (mock)"


def test_history_survives_broken_lines() -> None:
    path = get_settings().trajectory_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"event": "decision_made", "decision": "HOLD"}\nне json\n', encoding="utf-8"
    )
    assert len(history.read_history()) == 1


def test_history_command_renders_timeline() -> None:
    _write_trajectory(
        [
            {
                "timestamp": "2026-08-13T09:00:00+00:00",
                "event": "decision_made",
                "decision": "REBALANCE",
                "turnover": 0.74,
            },
            {"node": "execute_rebalance", "event": "rejected"},
        ]
    )
    trajectory.reset()
    text = visuals.render_history(history.read_history())
    assert "Стрічка рішень" in text
    assert "REBALANCE" in text
    assert "оборот 74%" in text


# --- Підказки наступних питань --------------------------------------------


def test_suggestions_fall_back_when_llm_fails() -> None:
    """Чат не має ламатись через недоступний LLM."""
    questions = suggest_followups("що таке просадка?", "…", ["knowledge_search"], llm=FailingLLM())
    assert questions
    assert all(question.strip() for question in questions)


def test_fallback_depends_on_the_tool_that_was_used() -> None:
    concept = suggest_followups("що таке просадка?", "…", ["knowledge_search"], llm=FailingLLM())
    universe = suggest_followups("які активи?", "…", ["get_top_liquid_assets"], llm=FailingLLM())
    assert concept != universe


def test_suggestions_respect_configured_count() -> None:
    settings = get_settings()
    original = settings.chat_suggestions_count
    settings.chat_suggestions_count = 2
    try:
        assert len(suggest_followups("q", "a", [], llm=FailingLLM())) == 2
    finally:
        settings.chat_suggestions_count = original


def test_suggestions_can_be_disabled() -> None:
    settings = get_settings()
    settings.chat_suggestions_enabled = False
    try:
        assert suggest_followups("q", "a", [], llm=FailingLLM()) == ()
    finally:
        settings.chat_suggestions_enabled = True


def test_suggestion_model_strips_numbering_and_duplicates() -> None:
    model = FollowUpQuestions(questions=("1. Що таке просадка?", "- Що таке просадка?", "Далі?"))
    assert model.questions == ("Що таке просадка?", "Далі?")


def test_suggestion_model_rejects_overlong_questions() -> None:
    with pytest.raises(ValueError, match="Порожній набір"):
        FollowUpQuestions(questions=("х" * 200,))


def test_suggestions_render_as_numbered_list() -> None:
    text = visuals.render_suggestions(("Перше?", "Друге?"))
    assert "Далі можна спитати:" in text
    assert "1  Перше?" in text


def test_empty_suggestions_render_as_nothing() -> None:
    assert visuals.render_suggestions(()) == ""
