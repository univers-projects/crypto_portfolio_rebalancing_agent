"""ReAct-агент на LangGraph з жорсткими запобіжниками.

Цикл: LLM -> рішення про tool -> виконання tool -> observation -> LLM.

Реалізовані обмеження:
  * max_steps = 10 (конфігуровано) — жорсткий стоп ітерацій;
  * timeout = 120 сек — перевіряється перед кожним кроком;
  * детекція повторних однакових tool calls (tool + аргументи);
  * JSON-логування траєкторії кожного кроку;
  * graceful handling помилок tools — вони повертаються в LLM як observation.
"""

from __future__ import annotations

import json
import logging
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import get_settings
from app.llm import get_llm
from app.observability import trajectory
from app.prompts import load_prompt
from app.tools.base import failure, to_json
from app.tools.registry import ANALYSIS_TOOLS

logger = logging.getLogger(__name__)

# Повідомлення, яке отримує LLM, коли повторює той самий виклик
REPEAT_NOTICE = (
    "Цей самий tool з тими самими аргументами вже викликався. Повторний виклик "
    "заблоковано. Використай попередню observation або зміни підхід."
)


class ReActState(TypedDict, total=False):
    """Стан ReAct-циклу."""

    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    started_at: float
    stop_reason: str
    tool_calls_seen: list[str]
    # Накопичувальний reducer: інакше кожен прохід tools-вузла затирав би попередні записи
    trajectory: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class ReActResult:
    """Результат роботи ReAct-агента для одного завдання."""

    output: str
    steps: int
    stop_reason: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.stop_reason == "completed"


def _call_signature(tool_call: dict[str, Any]) -> str:
    """Стабільний підпис виклику tool для детекції повторів."""
    args = json.dumps(tool_call.get("args", {}), sort_keys=True, ensure_ascii=False, default=str)
    return f"{tool_call.get('name')}::{args}"


def build_react_graph(
    tools: tuple[BaseTool, ...] = ANALYSIS_TOOLS,
    *,
    system_prompt: str | None = None,
    llm: Any = None,
) -> Any:
    """Скомпілювати граф ReAct-агента.

    `llm` можна підмінити у тестах, щоб не звертатися до реальної моделі.
    """
    settings = get_settings()
    tools_by_name = {tool.name: tool for tool in tools}
    prompt = system_prompt or load_prompt("react_executor")
    model = (llm if llm is not None else get_llm("executor")).bind_tools(list(tools))

    def agent_node(state: ReActState) -> dict[str, Any]:
        """Крок міркування: LLM вирішує — викликати tool чи дати відповідь."""
        steps = state.get("steps", 0)
        started_at = state.get("started_at") or time.monotonic()
        elapsed = time.monotonic() - started_at

        # Запобіжник 1: таймаут
        if elapsed > settings.react_timeout_seconds:
            trajectory.record_event(
                node="react_agent", event="timeout", elapsed_seconds=round(elapsed, 2)
            )
            return {
                "stop_reason": "timeout",
                "messages": [
                    AIMessage(
                        content=(
                            f"Ліміт часу {settings.react_timeout_seconds:.0f}с вичерпано. "
                            "Повертаю проміжні результати."
                        )
                    )
                ],
            }

        # Запобіжник 2: максимальна кількість кроків
        if steps >= settings.react_max_steps:
            trajectory.record_event(node="react_agent", event="max_steps_reached", steps=steps)
            return {
                "stop_reason": "max_steps",
                "messages": [
                    AIMessage(
                        content=(
                            f"Досягнуто ліміт у {settings.react_max_steps} кроків. "
                            "Повертаю проміжні результати."
                        )
                    )
                ],
            }

        messages: list[BaseMessage] = [SystemMessage(content=prompt), *state["messages"]]
        try:
            response = model.invoke(messages)
        except Exception as error:  # noqa: BLE001 — збій LLM не має валити граф
            # Без traceback: помилка вже потрапляє у стан і trajectory-лог
            logger.warning("Помилка виклику LLM у ReAct-агенті: %s", error)
            trajectory.record_event(
                node="react_agent", event="llm_error", error=str(error)[:300]
            )
            return {
                "stop_reason": "llm_error",
                "messages": [AIMessage(content=f"Помилка LLM: {error}")],
                "steps": steps + 1,
            }

        tool_calls = getattr(response, "tool_calls", None) or []
        trajectory.record_event(
            node="react_agent",
            event="reasoning",
            step_index=steps + 1,
            requested_tools=[call["name"] for call in tool_calls],
            started_at=started_at,
        )
        return {
            "messages": [response],
            "steps": steps + 1,
            "started_at": started_at,
            "stop_reason": "" if tool_calls else "completed",
        }

    def tools_node(state: ReActState) -> dict[str, Any]:
        """Виконати запитані tools, залогувати результат, повернути observations."""
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []
        seen = list(state.get("tool_calls_seen", []))
        observations: list[AnyMessage] = []
        recorded: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            signature = _call_signature(tool_call)
            name = tool_call["name"]

            # Запобіжник 3: детекція повторних однакових викликів
            if signature in seen:
                trajectory.record_event(
                    node="react_agent",
                    event="duplicate_tool_call",
                    tool=name,
                    status="skipped",
                )
                observations.append(
                    ToolMessage(
                        content=to_json(failure("DUPLICATE_TOOL_CALL", REPEAT_NOTICE)),
                        tool_call_id=tool_call["id"],
                        name=name,
                    )
                )
                continue

            seen.append(signature)
            tool = tools_by_name.get(name)
            if tool is None:
                payload = failure("UNKNOWN_TOOL", f"Tool '{name}' недоступний")
            else:
                # tool_contract усередині вже гарантує JSON-контракт і логування
                payload = tool.invoke(tool_call["args"])

            recorded.append({"tool": name, "args": tool_call["args"], "response": payload})
            observations.append(
                ToolMessage(
                    content=to_json(payload), tool_call_id=tool_call["id"], name=name
                )
            )

        return {
            "messages": observations,
            "tool_calls_seen": seen,
            "trajectory": recorded,
        }

    def route(state: ReActState) -> str:
        """Продовжувати цикл лише якщо LLM попросив tools і запобіжники не спрацювали."""
        if state.get("stop_reason"):
            return END
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(ReActState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    # recursion_limit із запасом: власні лічильники зупинять цикл раніше
    return graph.compile()


def run_react_task(
    task: str,
    *,
    context: str = "",
    tools: tuple[BaseTool, ...] = ANALYSIS_TOOLS,
    llm: Any = None,
    graph: Any = None,
) -> ReActResult:
    """Виконати одну задачу ReAct-агентом і повернути структурований результат."""
    settings = get_settings()
    compiled = graph if graph is not None else build_react_graph(tools, llm=llm)

    user_content = f"{task}\n\nКонтекст:\n{context}" if context else task
    initial: ReActState = {
        "messages": [HumanMessage(content=user_content)],
        "steps": 0,
        "started_at": time.monotonic(),
        "tool_calls_seen": [],
        "trajectory": [],
    }

    with trajectory.TrajectoryRecorder("executor"):
        final_state = compiled.invoke(
            initial,
            config={"recursion_limit": settings.react_max_steps * 2 + 5},
        )

    output = _extract_output(final_state["messages"])
    recorded = final_state.get("trajectory", [])
    errors = [
        {
            "tool_name": item["tool"],
            "error_code": item["response"].get("error", {}).get("code", "UNKNOWN_ERROR"),
        }
        for item in recorded
        if item["response"].get("status") == "error"
    ]

    return ReActResult(
        output=output,
        steps=final_state.get("steps", 0),
        stop_reason=final_state.get("stop_reason") or "completed",
        tool_calls=recorded,
        trajectory=trajectory.snapshot(),
        errors=errors,
    )


def _extract_output(messages: list[AnyMessage]) -> str:
    """Дістати останню текстову відповідь LLM."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            content = message.content
            if isinstance(content, list):
                # Мультимодальний контент: збираємо лише текстові частини
                parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = " ".join(part for part in parts if part)
            if content:
                return str(content).strip()
    return "Агент не повернув текстової відповіді."
