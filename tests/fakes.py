"""Підроблені LLM для тестів: жодних мережевих викликів.

FakeLLM відтворює лише той контракт, який реально використовує код:
`bind_tools()`, `with_structured_output()` та `invoke()`.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


class FakeLLM:
    """LLM, що повертає наперед задану послідовність відповідей."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.invocations: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> FakeLLM:
        self.bound_tools = tools
        return self

    def with_structured_output(self, schema: Any) -> FakeLLM:
        self.structured_schema = schema
        return self

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        self.invocations.append(messages)
        if not self._responses:
            return AIMessage(content="Готово.")
        return self._responses.pop(0)


class FailingLLM:
    """LLM, що завжди падає — для перевірки fallback-шляхів."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or RuntimeError("LLM недоступний")

    def bind_tools(self, tools: list[Any]) -> FailingLLM:
        return self

    def with_structured_output(self, schema: Any) -> FailingLLM:
        return self

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        raise self._error


def tool_call_message(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    """AIMessage із запитом на виклик tool."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )
