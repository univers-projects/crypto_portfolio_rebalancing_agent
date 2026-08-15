"""Стандартизований JSON-контракт для всіх доменних tools.

Успіх:
    {"status": "success", "data": {...}}

Помилка:
    {"status": "error", "error": {"code": "ERROR_CODE", "message": "..."}}

Tools ніколи не піднімають винятки назовні: LLM має отримати структуровану
помилку як observation і самостійно вирішити, що робити далі.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from app.domain.errors import DomainError
from app.observability.trajectory import record_tool_event

ToolPayload = dict[str, Any]
T = TypeVar("T", bound=Callable[..., ToolPayload])


def success(data: ToolPayload) -> ToolPayload:
    """Побудувати успішну відповідь tool."""
    return {"status": "success", "data": data}


def failure(code: str, message: str) -> ToolPayload:
    """Побудувати помилкову відповідь tool."""
    return {"status": "error", "error": {"code": code, "message": message}}


def dump(model: BaseModel) -> ToolPayload:
    """Серіалізувати Pydantic-модель у примітиви, придатні для JSON."""
    return model.model_dump(mode="json")


def tool_contract(tool_name: str) -> Callable[[T], T]:
    """Декоратор: гарантує JSON-контракт, логування та безпечну обробку помилок.

    Перехоплює доменні винятки (з кодом), помилки валідації Pydantic
    і будь-які неочікувані винятки, перетворюючи їх на error-відповідь.
    """

    def decorator(func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> ToolPayload:
            try:
                response = func(*args, **kwargs)
            except PydanticValidationError as error:
                response = failure("VALIDATION_ERROR", _format_pydantic_error(error))
            except DomainError as error:
                response = failure(error.code, error.message)
            except Exception as error:  # noqa: BLE001 — tool не має падати назовні
                response = failure("UNEXPECTED_ERROR", f"{type(error).__name__}: {error}")

            record_tool_event(tool_name=tool_name, arguments=kwargs, response=response)
            return response

        return wrapper  # type: ignore[return-value]

    return decorator


def _format_pydantic_error(error: PydanticValidationError) -> str:
    """Стиснути помилки валідації у зрозумілий для LLM рядок."""
    parts = [
        f"{'.'.join(str(item) for item in issue['loc']) or 'input'}: {issue['msg']}"
        for issue in error.errors()
    ]
    return "; ".join(parts)


def to_json(payload: ToolPayload) -> str:
    """Компактний JSON-рядок для передачі в LLM як observation."""
    return json.dumps(payload, ensure_ascii=False, default=str)
