"""Прості текстові блоки інтерактивного режиму.

Складна візуалізація (шкали, смуги, чек-лист) живе у `visuals.py` і йде через
`rich`. Тут лишилося те, що не потребує оформлення: банер, довідка, підсумок
виконання і хвіст траєкторії.
"""

from __future__ import annotations

from typing import Any

SEPARATOR = "=" * 72
THIN = "-" * 72

BANNER = """
AI Crypto Portfolio Rebalancing Agent
Усі торгові операції — симуляція. Реальні кошти та біржові ключі не задіяні.
""".strip()

HELP_TEXT = """
Доступні команди:
  /portfolio            поточний склад портфеля
  /whatif BTC=0.4,...   прогноз для довільного складу (портфель не змінюється)
  /history              стрічка ухвалених рішень
  /autonomy [on|off]    рівень делегування дрібних дій
  /state                збережений стан LangGraph (get_state)
  /trajectory           кроки JSON-траєкторії за сесію
  /rerun                повторити цикл аналізу і, за потреби, ребаланс
  /help                 цей список
  /exit                 вийти

Будь-який інший текст — питання до агента. Після кожної відповіді
показуються підказки, що можна спитати далі.
""".strip()


def header(title: str) -> str:
    """Заголовок секції."""
    return f"\n{SEPARATOR}\n{title}\n{SEPARATOR}"


def render_execution_result(result: dict[str, Any]) -> str:
    """Підсумок виконання (mock)."""
    status = result.get("status", "unknown")
    if status != "executed":
        return f"Статус: {status}. {result.get('message', '')}".strip()

    lines = [
        f"Статус: виконано (mock), операцій: {result.get('operations_count', 0)}",
        f"Портфель до:    {result.get('portfolio_before')}",
        f"Портфель після: {result.get('portfolio_after')}",
    ]
    return "\n".join(lines)


def render_trajectory(entries: list[dict[str, Any]], limit: int = 15) -> str:
    """Останні записи траєкторії у компактному вигляді."""
    if not entries:
        return "Траєкторія порожня."
    lines = []
    for entry in entries[-limit:]:
        marker = entry.get("tool") or entry.get("event") or "-"
        status = entry.get("status", "")
        code = entry.get("error_code", "")
        suffix = f" [{status}{'/' + code if code else ''}]" if status else ""
        lines.append(f"  {entry.get('step'):>3}. {entry.get('node', '?'):<18} {marker}{suffix}")
    return "\n".join(lines)
