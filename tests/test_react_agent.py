"""Тести ReAct-агента: базовий цикл LLM -> tool -> observation та запобіжники."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.react_agent import build_react_graph, run_react_task
from app.config import get_settings
from app.observability import trajectory
from tests.fakes import FailingLLM, FakeLLM, tool_call_message


def test_basic_llm_tool_observation_cycle() -> None:
    """LLM просить tool -> tool виконується -> observation повертається в LLM."""
    llm = FakeLLM(
        [
            tool_call_message(
                "calculate_asset_metrics", {"symbols": ["BTC"], "lookback_days": 180}
            ),
            AIMessage(content="BTC має Sharpe-like 0.17 за 180 днів."),
        ]
    )

    result = run_react_task("Порахуй метрики BTC", llm=llm)

    assert result.stop_reason == "completed"
    assert result.succeeded
    # Рівно один виклик tool, і він успішний
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool"] == "calculate_asset_metrics"
    assert result.tool_calls[0]["response"]["status"] == "success"
    assert "Sharpe" in result.output

    # LLM отримав observation як ToolMessage із валідним JSON
    second_call_messages = llm.invocations[1]
    tool_messages = [m for m in second_call_messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    observation = json.loads(tool_messages[0].content)
    assert observation["status"] == "success"
    assert observation["data"]["metrics"][0]["symbol"] == "BTC"


def test_agent_answers_without_calling_tools() -> None:
    """Агент не зобов'язаний викликати tools на кожен запит."""
    llm = FakeLLM([AIMessage(content="Портфель — це набір позицій.")])
    result = run_react_task("Що таке портфель?", llm=llm)

    assert result.tool_calls == []
    assert result.stop_reason == "completed"


def test_duplicate_tool_call_is_blocked() -> None:
    """Повторний ідентичний виклик не виконується вдруге."""
    duplicate_args = {"symbols": ["BTC"], "lookback_days": 180}
    llm = FakeLLM(
        [
            tool_call_message("calculate_asset_metrics", duplicate_args, call_id="a"),
            tool_call_message("calculate_asset_metrics", duplicate_args, call_id="b"),
            AIMessage(content="Використовую попередню observation."),
        ]
    )

    result = run_react_task("Порахуй метрики BTC двічі", llm=llm)

    # Реально виконано лише перший виклик
    assert len(result.tool_calls) == 1
    duplicate_events = [
        event
        for event in trajectory.snapshot()
        if event.get("event") == "duplicate_tool_call"
    ]
    assert len(duplicate_events) == 1


def test_max_steps_guard_stops_the_loop() -> None:
    """Нескінченний цикл зупиняється лічильником кроків."""
    settings = get_settings()
    original = settings.react_max_steps
    settings.react_max_steps = 3
    try:
        # LLM нескінченно просить tool із новими аргументами
        responses = [
            tool_call_message(
                "get_market_data",
                {"symbol": "BTC", "lookback_days": 100 + index},
                call_id=f"c{index}",
            )
            for index in range(10)
        ]
        result = run_react_task("Зациклись", llm=FakeLLM(responses))

        assert result.stop_reason == "max_steps"
        assert result.steps <= settings.react_max_steps
    finally:
        settings.react_max_steps = original


def test_timeout_guard_stops_the_loop() -> None:
    settings = get_settings()
    original = settings.react_timeout_seconds
    settings.react_timeout_seconds = 0.0001
    try:
        result = run_react_task("Щось довге", llm=FakeLLM([AIMessage(content="ок")]))
        assert result.stop_reason == "timeout"
    finally:
        settings.react_timeout_seconds = original


def test_tool_error_is_returned_as_observation_not_exception() -> None:
    """Помилка tool стає observation, а не винятком."""
    llm = FakeLLM(
        [
            tool_call_message("get_market_data", {"symbol": "NEWX", "lookback_days": 180}),
            AIMessage(content="NEWX має недостатньо історії, виключаю його."),
        ]
    )
    result = run_react_task("Візьми дані NEWX", llm=llm)

    assert result.stop_reason == "completed"
    assert result.errors == [
        {"tool_name": "get_market_data", "error_code": "INSUFFICIENT_HISTORY"}
    ]
    observation = json.loads(
        [m for m in llm.invocations[1] if isinstance(m, ToolMessage)][0].content
    )
    assert observation["status"] == "error"
    assert observation["error"]["code"] == "INSUFFICIENT_HISTORY"


def test_unknown_tool_is_handled_gracefully() -> None:
    llm = FakeLLM(
        [
            tool_call_message("delete_everything", {}),
            AIMessage(content="Такого інструмента немає."),
        ]
    )
    result = run_react_task("Виклич неіснуючий tool", llm=llm)

    assert result.stop_reason == "completed"
    assert result.tool_calls[0]["response"]["error"]["code"] == "UNKNOWN_TOOL"


def test_llm_failure_does_not_crash_the_graph() -> None:
    result = run_react_task("Що завгодно", llm=FailingLLM())
    assert result.stop_reason == "llm_error"


def test_trajectory_is_logged_as_json_events() -> None:
    """Кожен крок пишеться у JSON-траєкторію з полями step/node/tool/status."""
    trajectory.reset()
    llm = FakeLLM(
        [
            tool_call_message("get_market_data", {"symbol": "BTC", "lookback_days": 180}),
            AIMessage(content="Готово."),
        ]
    )
    run_react_task("Дані BTC", llm=llm)

    events = trajectory.snapshot()
    assert events, "Траєкторія має бути непорожньою"
    assert all("step" in event and "node" in event for event in events)

    tool_events = [event for event in events if event.get("tool") == "get_market_data"]
    assert len(tool_events) == 1
    assert tool_events[0]["status"] == "success"
    assert tool_events[0]["symbol"] == "BTC"

    # Записи мають бути серіалізовними у JSON
    assert json.loads(json.dumps(events, default=str))

    # І продубльовані у .jsonl файл
    log_path = get_settings().trajectory_log_path
    assert log_path.exists()
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(line.get("tool") == "get_market_data" for line in lines)


def test_error_events_expose_error_code_in_trajectory() -> None:
    trajectory.reset()
    llm = FakeLLM(
        [
            tool_call_message("get_market_data", {"symbol": "BADQ", "lookback_days": 180}),
            AIMessage(content="Дані непридатні."),
        ]
    )
    run_react_task("Дані BADQ", llm=llm)

    assert trajectory.errors_only() == [
        {"tool_name": "get_market_data", "error_code": "INSUFFICIENT_HISTORY"}
    ]


def test_graph_can_be_reused_across_tasks() -> None:
    llm = FakeLLM([AIMessage(content="перше"), AIMessage(content="друге")])
    graph = build_react_graph(llm=llm)

    first = run_react_task("завдання 1", graph=graph)
    second = run_react_task("завдання 2", graph=graph)

    assert first.output == "перше"
    assert second.output == "друге"
