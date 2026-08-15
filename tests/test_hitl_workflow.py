"""Тести Human-in-the-Loop, чекпойнтера та відновлення стану.

LLM повністю замінений: планування і рішення йдуть детермінованим fallback-шляхом,
а ReAct-executor підмінений заглушкою. Це робить тести швидкими і стабільними.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agents import plan_execute
from app.agents.plan_execute import build_workflow
from app.agents.react_agent import ReActResult
from app.data.portfolio_store import load_portfolio
from tests.fakes import FailingLLM

# Портфель, для якого політика впевнено дає REBALANCE
REBALANCE_PORTFOLIO = {"BTC": 0.40, "ETH": 0.25, "SOL": 0.20, "AVAX": 0.15}
# Портфель, який уже достатньо якісний -> HOLD на вікні рішення (720 днів).
# ADA сюди не входить: за 180 днів вона виглядає найкращим активом, за 720 —
# одним із найгірших, тому портфель із нею на вікні рішення дає REBALANCE.
HOLD_PORTFOLIO = {"BTC": 0.32, "BNB": 0.23, "TRX": 0.26, "XRP": 0.19}


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Прибрати всі мережеві виклики: planner/replanner/decider ідуть у fallback."""
    monkeypatch.setattr(plan_execute, "get_llm", lambda role="executor": FailingLLM())

    def fake_react(task: str, **kwargs: Any) -> ReActResult:
        return ReActResult(
            output=f"Крок виконано: {task[:60]}",
            steps=1,
            stop_reason="completed",
            tool_calls=[
                {
                    "tool": "calculate_asset_metrics",
                    "args": {},
                    "response": {"status": "success", "data": {}},
                }
            ],
        )

    monkeypatch.setattr(plan_execute, "run_react_task", fake_react)


@pytest.fixture
def workflow_with_checkpointer(tmp_path: Any) -> Iterator[tuple[Any, dict[str, Any]]]:
    """Скомпільований граф із SqliteSaver і сталим thread_id."""
    path = tmp_path / "test_checkpoints.sqlite"
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        workflow = build_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-thread"}}
        yield workflow, config


# --- HOLD ---


def test_hold_decision_skips_hitl(
    workflow_with_checkpointer: tuple[Any, dict[str, Any]], set_portfolio: Any
) -> None:
    """Для HOLD переривання не потрібне — workflow завершується одразу."""
    set_portfolio(HOLD_PORTFOLIO)
    workflow, config = workflow_with_checkpointer

    workflow.invoke({"messages": []}, config=config)
    state = workflow.get_state(config)

    assert state.next == ()
    assert state.values["decision"] == "HOLD"
    assert state.values["final_decision"]["proposed_portfolio"] is None
    assert state.values["final_decision"]["actions"] == []
    assert state.values.get("execution_result") is None
    # Портфель незмінний
    assert load_portfolio().as_mapping() == HOLD_PORTFOLIO


# --- REBALANCE -> interrupt ---


def test_rebalance_interrupts_before_execution(
    workflow_with_checkpointer: tuple[Any, dict[str, Any]], set_portfolio: Any
) -> None:
    """Workflow зупиняється перед execute_rebalance і нічого не виконує."""
    set_portfolio(REBALANCE_PORTFOLIO)
    workflow, config = workflow_with_checkpointer

    workflow.invoke({"messages": []}, config=config)
    state = workflow.get_state(config)

    assert state.next == ("execute_rebalance",)
    assert state.values["decision"] == "REBALANCE"
    assert state.values["rebalance_proposal"]["actions"]
    # До підтвердження портфель не змінюється
    assert load_portfolio().as_mapping() == REBALANCE_PORTFOLIO
    assert state.values.get("execution_result") is None


# --- REBALANCE -> interrupt -> approve -> mock execution ---


def test_approve_executes_mock_rebalance(
    workflow_with_checkpointer: tuple[Any, dict[str, Any]], set_portfolio: Any
) -> None:
    set_portfolio(REBALANCE_PORTFOLIO)
    workflow, config = workflow_with_checkpointer

    workflow.invoke({"messages": []}, config=config)
    proposal = workflow.get_state(config).values["rebalance_proposal"]
    expected = {
        position["symbol"]: position["weight"]
        for position in proposal["proposed_portfolio"]["positions"]
    }

    # Рішення людини записується у стан, після чого workflow продовжується
    workflow.update_state(config, {"approval": "approve"})
    workflow.invoke(None, config=config)
    state = workflow.get_state(config)

    assert state.next == ()
    result = state.values["execution_result"]
    assert result["status"] == "executed"
    assert result["operations_count"] == len(proposal["actions"])
    assert all(operation["simulated"] for operation in result["executed_operations"])
    # Портфель оновлено саме на запропонований
    assert load_portfolio().as_mapping() == expected
    assert state.values["current_portfolio"] == expected


# --- REBALANCE -> interrupt -> reject -> no change ---


def test_reject_leaves_portfolio_unchanged(
    workflow_with_checkpointer: tuple[Any, dict[str, Any]], set_portfolio: Any
) -> None:
    set_portfolio(REBALANCE_PORTFOLIO)
    workflow, config = workflow_with_checkpointer

    workflow.invoke({"messages": []}, config=config)
    workflow.update_state(config, {"approval": "reject"})
    workflow.invoke(None, config=config)
    state = workflow.get_state(config)

    assert state.next == ()
    assert state.values["execution_result"]["status"] == "rejected"
    assert load_portfolio().as_mapping() == REBALANCE_PORTFOLIO
    assert state.values["current_portfolio"] == REBALANCE_PORTFOLIO


# --- REBALANCE -> interrupt -> modify -> повторний interrupt -> approve ---


def test_modify_revalidates_and_interrupts_again(
    workflow_with_checkpointer: tuple[Any, dict[str, Any]], set_portfolio: Any
) -> None:
    set_portfolio(REBALANCE_PORTFOLIO)
    workflow, config = workflow_with_checkpointer
    workflow.invoke({"messages": []}, config=config)

    modified = [
        {"symbol": "BTC", "weight": 0.50},
        {"symbol": "ETH", "weight": 0.30},
        {"symbol": "SOL", "weight": 0.20},
    ]
    workflow.update_state(config, {"approval": "modify", "modified_positions": modified})
    workflow.invoke(None, config=config)
    state = workflow.get_state(config)

    # Знову зупинилися перед виконанням, тепер уже зі зміненим планом
    assert state.next == ("execute_rebalance",)
    proposal = state.values["rebalance_proposal"]
    assert {
        position["symbol"]: position["weight"]
        for position in proposal["proposed_portfolio"]["positions"]
    } == {"BTC": 0.50, "ETH": 0.30, "SOL": 0.20}
    assert load_portfolio().as_mapping() == REBALANCE_PORTFOLIO

    # Підтверджуємо змінений план
    workflow.update_state(config, {"approval": "approve"})
    workflow.invoke(None, config=config)

    assert load_portfolio().as_mapping() == {"BTC": 0.50, "ETH": 0.30, "SOL": 0.20}


def test_invalid_modification_is_rejected(
    workflow_with_checkpointer: tuple[Any, dict[str, Any]], set_portfolio: Any
) -> None:
    """Змінений план із неправильною сумою ваг не проходить валідацію."""
    set_portfolio(REBALANCE_PORTFOLIO)
    workflow, config = workflow_with_checkpointer
    workflow.invoke({"messages": []}, config=config)
    original = workflow.get_state(config).values["rebalance_proposal"]

    workflow.update_state(
        config,
        {
            "approval": "modify",
            "modified_positions": [
                {"symbol": "BTC", "weight": 0.9},
                {"symbol": "ETH", "weight": 0.9},
            ],
        },
    )
    workflow.invoke(None, config=config)
    state = workflow.get_state(config)

    assert {"tool_name": "validate_proposal", "error_code": "INVALID_PORTFOLIO"} in state.values[
        "errors"
    ]
    # Початкова пропозиція збережена, портфель не змінено
    assert state.values["rebalance_proposal"] == original
    assert load_portfolio().as_mapping() == REBALANCE_PORTFOLIO


def test_execution_blocked_without_approval(
    workflow_with_checkpointer: tuple[Any, dict[str, Any]], set_portfolio: Any
) -> None:
    """Продовження без запису рішення людини не виконує операцій."""
    set_portfolio(REBALANCE_PORTFOLIO)
    workflow, config = workflow_with_checkpointer
    workflow.invoke({"messages": []}, config=config)

    workflow.invoke(None, config=config)
    state = workflow.get_state(config)

    assert state.values["execution_result"]["status"] == "blocked"
    assert {
        "tool_name": "execute_rebalance",
        "error_code": "APPROVAL_REQUIRED",
    } in state.values["errors"]
    assert load_portfolio().as_mapping() == REBALANCE_PORTFOLIO


# --- Checkpointer ---


def test_state_persists_between_separate_invocations(
    tmp_path: Any, set_portfolio: Any
) -> None:
    """Стан переживає повне закриття і повторне відкриття SqliteSaver."""
    set_portfolio(REBALANCE_PORTFOLIO)
    path = tmp_path / "persist.sqlite"
    config = {"configurable": {"thread_id": "persisted-thread"}}

    # Перша сесія: доходимо до interrupt і закриваємо все
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        workflow = build_workflow(checkpointer=checkpointer)
        workflow.invoke({"messages": []}, config=config)
        first_proposal = workflow.get_state(config).values["rebalance_proposal"]

    # Друга сесія: новий об'єкт графа і нове з'єднання з тим самим файлом
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        workflow = build_workflow(checkpointer=checkpointer)
        state = workflow.get_state(config)

        assert state.next == ("execute_rebalance",)
        assert state.values["rebalance_proposal"] == first_proposal
        assert state.values["decision"] == "REBALANCE"

        # І workflow можна довести до кінця
        workflow.update_state(config, {"approval": "approve"})
        workflow.invoke(None, config=config)
        assert workflow.get_state(config).values["execution_result"]["status"] == "executed"


def test_threads_are_isolated(tmp_path: Any, set_portfolio: Any) -> None:
    """Різні thread_id не бачать стану один одного."""
    set_portfolio(REBALANCE_PORTFOLIO)
    path = tmp_path / "threads.sqlite"

    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        workflow = build_workflow(checkpointer=checkpointer)
        config_a = {"configurable": {"thread_id": "thread-a"}}
        workflow.invoke({"messages": []}, config=config_a)

        state_b = workflow.get_state({"configurable": {"thread_id": "thread-b"}})
        assert state_b.values == {}
        assert workflow.get_state(config_a).values["decision"] == "REBALANCE"


def test_state_history_is_available(
    workflow_with_checkpointer: tuple[Any, dict[str, Any]], set_portfolio: Any
) -> None:
    """Історія чекпойнтів дозволяє відтворити хід виконання."""
    set_portfolio(REBALANCE_PORTFOLIO)
    workflow, config = workflow_with_checkpointer
    workflow.invoke({"messages": []}, config=config)

    history = list(workflow.get_state_history(config))
    assert len(history) > 3
    visited = {node for snapshot in history for node in snapshot.next}
    assert "planner" in visited or "executor" in visited
