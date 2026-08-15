"""Доменні винятки, згруповані за областю відповідальності.

Кожен виняток несе стабільний `code`, який потрапляє у JSON-контракт tools
та у trajectory-лог (поле `error_code`).
"""

from __future__ import annotations


class DomainError(Exception):
    """Базовий доменний виняток із машинозчитуваним кодом."""

    code: str = "DOMAIN_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class UnknownSymbolError(DomainError):
    """Символ відсутній у дозволеному universe."""

    code = "UNKNOWN_SYMBOL"


class InsufficientHistoryError(DomainError):
    """Недостатньо історичних даних для запитаного lookback."""

    code = "INSUFFICIENT_HISTORY"


class InvalidPortfolioError(DomainError):
    """Портфель порушує правила складу або ваг."""

    code = "INVALID_PORTFOLIO"


class ValidationError(DomainError):
    """Некоректні вхідні параметри tool."""

    code = "VALIDATION_ERROR"


class KnowledgeBaseError(DomainError):
    """Помилка доступу до бази знань RAG."""

    code = "KNOWLEDGE_BASE_ERROR"


class ExecutionError(DomainError):
    """Помилка під час mock-виконання ребалансу."""

    code = "EXECUTION_ERROR"


class ApprovalRequiredError(DomainError):
    """Спроба виконати ризикову операцію без HITL-підтвердження."""

    code = "APPROVAL_REQUIRED"
