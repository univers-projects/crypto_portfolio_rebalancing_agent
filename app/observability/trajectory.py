"""JSON-логування траєкторії агента.

Кожен важливий крок (виклик tool, перехід між вузлами графа, рішення)
пишеться як окремий JSON-рядок у .jsonl та накопичується в памʼяті поточного
запуску, щоб потрапити у GuardedState.trajectory.

Приклад запису:
    {"step": 4, "node": "executor", "tool": "get_market_data",
     "status": "error", "error_code": "INSUFFICIENT_HISTORY"}
"""

from __future__ import annotations

import json
import logging
import threading
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# Поточний вузол графа — щоб tool-події автоматично отримували правильний `node`.
# ContextVar доречний саме тут: вузли графа виконуються у власних контекстах.
_current_node: ContextVar[str] = ContextVar("current_node", default="unknown")

_lock = threading.Lock()
_buffer: list[dict[str, Any]] = []
# Лічильник кроків навмисно глобальний, а не ContextVar: LangGraph виконує вузли
# в окремих контекстах, і ContextVar скидався б у 0 на кожному вузлі.
_step_counter = 0


class TrajectoryRecorder:
    """Контекстний менеджер, що позначає всі вкладені події певним вузлом графа."""

    def __init__(self, node: str) -> None:
        self._node = node
        self._token: Any = None

    def __enter__(self) -> TrajectoryRecorder:
        self._token = _current_node.set(self._node)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._token is not None:
            _current_node.reset(self._token)


def _next_step() -> int:
    """Монотонний номер кроку в межах поточного запуску."""
    global _step_counter
    with _lock:
        _step_counter += 1
        return _step_counter


def _write(entry: dict[str, Any]) -> None:
    """Додати запис у буфер і дописати у .jsonl файл."""
    with _lock:
        _buffer.append(entry)

    try:
        path: Path = get_settings().trajectory_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError as error:
        # Логування не має ламати основний workflow
        logger.warning("Не вдалося записати trajectory-лог: %s", error)


def record_event(node: str | None = None, **fields: Any) -> dict[str, Any]:
    """Записати довільну подію траєкторії."""
    entry = {
        "step": _next_step(),
        "timestamp": datetime.now(UTC).isoformat(),
        "node": node or _current_node.get(),
        **fields,
    }
    _write(entry)
    return entry


def record_tool_event(
    tool_name: str, arguments: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
    """Записати результат виклику tool у стандартному форматі."""
    status = response.get("status", "unknown")
    entry: dict[str, Any] = {
        "tool": tool_name,
        "status": status,
        "arguments": _compact_arguments(arguments),
    }
    # Символ виносимо окремим полем — так лог читабельніший
    if "symbol" in arguments:
        entry["symbol"] = arguments["symbol"]
    if status == "error":
        entry["error_code"] = response.get("error", {}).get("code", "UNKNOWN_ERROR")
    return record_event(**entry)


def _compact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Обрізати великі аргументи, щоб лог залишався читабельним."""
    compact: dict[str, Any] = {}
    for key, value in arguments.items():
        text = str(value)
        compact[key] = value if len(text) <= 200 else f"{text[:200]}..."
    return compact


def snapshot() -> list[dict[str, Any]]:
    """Копія накопичених подій поточного процесу."""
    with _lock:
        return list(_buffer)


def reset() -> None:
    """Очистити буфер і лічильник кроків (використовується між запусками і в тестах)."""
    global _step_counter
    with _lock:
        _buffer.clear()
        _step_counter = 0


def errors_only() -> list[dict[str, Any]]:
    """Лише помилкові події — зручно для GuardedState.errors."""
    return [
        {"tool_name": entry.get("tool"), "error_code": entry.get("error_code")}
        for entry in snapshot()
        if entry.get("status") == "error"
    ]
