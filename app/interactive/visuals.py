"""Візуальна мова інтерактивного режиму.

Числа в терміналі погано сприймаються списком. Тут вони перетворюються на
малюнки: діапазон прогнозу як шкала, ваги як смуги, відхилення від цілі як
парне порівняння.

Рендеринг іде через `rich`, але назовні модуль віддає **рядки**, а не
renderable-обʼєкти. Так інтерактивна сесія лишається тестованою: писар
(`writer`) приймає текст, і в pytest, де stdout не є терміналом, кольори
автоматично зникають.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_console = Console(soft_wrap=False)

# Ширини підібрані так, щоб панелі вміщались у стандартні 80 колонок
RANGE_WIDTH = 26
WEIGHT_WIDTH = 22


def to_text(renderable: RenderableType) -> str:
    """Відрендерити renderable у рядок (з ANSI в терміналі, без нього в тестах)."""
    with _console.capture() as capture:
        _console.print(renderable, end="")
    return capture.get()


def money(value: float) -> str:
    """Сума у доларах із пробілом як роздільником тисяч."""
    return "$" + f"{value:,.0f}".replace(",", " ")


# --- Діапазон прогнозу ----------------------------------------------------


@dataclass(frozen=True)
class RangeRow:
    """Один рядок шкали прогнозу."""

    label: str
    low: float
    mid: float
    high: float
    loss_probability: float
    accent: str = "cyan"


def range_bar(
    low: float, mid: float, high: float, scale_min: float, scale_max: float
) -> str:
    """Смуга «песимістично — серединно — оптимістично» на спільній шкалі."""
    span = scale_max - scale_min
    if span <= 0:
        return "─" * RANGE_WIDTH

    def position(value: float) -> int:
        ratio = (value - scale_min) / span
        return max(0, min(RANGE_WIDTH - 1, round(ratio * (RANGE_WIDTH - 1))))

    start, middle, end = position(low), position(mid), position(high)
    cells = [" "] * RANGE_WIDTH
    for index in range(start, end + 1):
        cells[index] = "─"
    cells[start] = "├"
    cells[end] = "┤"
    cells[middle] = "●"
    return "".join(cells)


def render_forecast(rows: list[RangeRow], horizon_days: int, amount: float) -> str:
    """Шкала прогнозу для одного або двох сценаріїв."""
    if not rows:
        return ""

    scale_min = min(row.low for row in rows)
    scale_max = max(row.high for row in rows)
    # Невеликий відступ по краях, щоб маркери не липли до країв шкали
    padding = (scale_max - scale_min) * 0.06
    scale_min -= padding
    scale_max += padding

    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", style="dim", no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(justify="right", no_wrap=True)
    table.add_column(justify="right", no_wrap=True)

    table.add_row(
        "",
        Text("гірше ← серединно → краще".center(RANGE_WIDTH), style="dim"),
        Text("діапазон", style="dim"),
        Text("ризик", style="dim"),
    )

    for row in rows:
        values = (
            f"{row.low:,.0f} · {row.mid:,.0f} · {row.high:,.0f}".replace(",", " ")
        )
        table.add_row(
            row.label,
            Text(range_bar(row.low, row.mid, row.high, scale_min, scale_max), style=row.accent),
            values,
            Text(f"{row.loss_probability * 100:.0f}%", style=_risk_style(row.loss_probability)),
        )

    title = f"Прогноз на {horizon_days} днів · умовні {money(amount)}"
    return to_text(Panel(table, title=title, title_align="left", border_style="dim"))


def _risk_style(loss_probability: float) -> str:
    """Колір для ймовірності збитку."""
    if loss_probability >= 0.45:
        return "red"
    if loss_probability >= 0.35:
        return "yellow"
    return "green"


# --- Ваги і відхилення ----------------------------------------------------


def render_allocation(weights: dict[str, float], title: str = "Портфель") -> str:
    """Ваги портфеля як горизонтальні смуги."""
    if not weights:
        return "Портфель порожній."

    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold", min_width=6, no_wrap=True)
    table.add_column(justify="right", min_width=6, no_wrap=True)
    table.add_column(no_wrap=True)

    largest = max(weights.values())
    for symbol, weight in sorted(weights.items(), key=lambda item: -item[1]):
        filled = max(1, round(weight / largest * WEIGHT_WIDTH))
        table.add_row(
            symbol,
            f"{weight * 100:.1f}%",
            Text("█" * filled, style="cyan"),
        )
    return to_text(Panel(table, title=title, title_align="left", border_style="dim"))


def render_drift(
    current: dict[str, float], target: dict[str, float], band: float = 0.05
) -> str:
    """Відхилення поточних ваг від цільових.

    Смуга `band` — типова для robo-advisor практика: доки актив у межах
    ±3–5 п.п. від цілі, він вважається на місці.
    """
    symbols = sorted(set(current) | set(target))
    if not symbols:
        return ""

    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold", min_width=6, no_wrap=True)
    table.add_column(justify="right", min_width=6, no_wrap=True)
    table.add_column(justify="right", min_width=6, no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(justify="right", min_width=9, no_wrap=True)
    table.add_column(no_wrap=True)

    table.add_row(
        "",
        Text("ціль", style="dim"),
        Text("зараз", style="dim"),
        "",
        Text("різниця", style="dim"),
        "",
    )

    largest = max([*current.values(), *target.values()]) or 1.0
    for symbol in symbols:
        now = current.get(symbol, 0.0)
        goal = target.get(symbol, 0.0)
        delta = now - goal
        outside = abs(delta) > band
        filled = max(0, round(now / largest * WEIGHT_WIDTH))
        table.add_row(
            symbol,
            f"{goal * 100:.1f}%",
            f"{now * 100:.1f}%",
            Text("█" * filled if filled else "·", style="yellow" if outside else "cyan"),
            Text(f"{delta * 100:+.1f} п.п.", style="yellow" if outside else "dim"),
            Text("поза" if outside else "ок", style="yellow" if outside else "green"),
        )

    return to_text(
        Panel(
            table,
            title=f"Відхилення від цілі (смуга ±{band * 100:.0f} п.п.)",
            title_align="left",
            border_style="dim",
        )
    )


# --- Чек-лист підтвердження -----------------------------------------------


ACTION_LABELS = {
    "BUY": "купити",
    "SELL": "продати",
    "INCREASE": "додати",
    "REDUCE": "зменшити",
}


def render_checklist(
    actions: list[dict[str, Any]],
    accepted: list[bool],
    outlooks: dict[str, tuple[float, float]],
    delegated: list[bool],
    summary_line: str,
    result_line: str,
    hint: str,
) -> str:
    """Список дій із галочками і живим підсумком.

    `outlooks` — символ -> (серединна вартість тисячі, ймовірність збитку).
    `delegated` — які дії позначені автоматично за наданим делегуванням.
    """
    table = Table.grid(padding=(0, 1))
    table.add_column(no_wrap=True)
    table.add_column(justify="right", style="dim", no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(justify="right", no_wrap=True)
    table.add_column(justify="right", no_wrap=True)
    table.add_column(justify="right", no_wrap=True)

    table.add_column(no_wrap=True)

    table.add_row(
        "",
        "",
        Text("дія", style="dim"),
        Text("з", style="dim"),
        Text("у", style="dim"),
        Text("ризик", style="dim"),
        "",
    )

    for index, (action, is_on, is_auto) in enumerate(
        zip(actions, accepted, delegated, strict=True), start=1
    ):
        symbol = str(action.get("symbol", "?"))
        kind = str(action.get("action", "?"))
        median, loss = outlooks.get(symbol, (0.0, 0.0))
        label = f"{ACTION_LABELS.get(kind, kind)} {symbol}"
        table.add_row(
            Text("[x]" if is_on else "[ ]", style="green" if is_on else "dim"),
            str(index),
            Text(label, style="" if is_on else "dim"),
            f"{float(action.get('from_weight') or 0.0) * 100:.1f}%",
            Text(f"{float(action.get('to_weight') or 0.0) * 100:.1f}%", style="bold"),
            Text(
                f"{loss * 100:.0f}%" if median else "—",
                style=_risk_style(loss) if median else "dim",
            ),
            Text("авто" if is_auto else "", style="cyan"),
        )

    parts: list[RenderableType] = [table, Text("─" * 58, style="dim"), Text(summary_line)]
    if result_line:
        parts.append(Text(result_line, style="dim"))

    return to_text(
        Panel(
            Group(*parts),
            title="Підтвердження",
            title_align="left",
            # Text, а не рядок: інакше rich зʼїсть [a] і [n] як теги розмітки
            subtitle=Text(hint, style="dim"),
            subtitle_align="left",
            border_style="yellow",
        )
    )


# --- Стрічка рішень і підказки --------------------------------------------


def render_history(records: list[Any]) -> str:
    """Хронологія ухвалених рішень."""
    if not records:
        return to_text(
            Text("Історія порожня — жодного завершеного циклу ще не було.", style="dim")
        )

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(justify="right", no_wrap=True)
    table.add_column(no_wrap=True)

    for record in records:
        style = "yellow" if record.decision == "REBALANCE" else "green"
        turnover = (
            f"оборот {record.turnover * 100:.0f}%"
            if record.decision == "REBALANCE" and record.turnover
            else ""
        )
        table.add_row(
            record.when,
            Text(record.decision, style=style),
            turnover,
            Text(record.outcome, style="dim"),
        )

    return to_text(
        Panel(table, title="Стрічка рішень", title_align="left", border_style="dim")
    )


def render_suggestions(questions: tuple[str, ...]) -> str:
    """Підказки наступних питань."""
    if not questions:
        return ""
    body = Text()
    body.append("Далі можна спитати:", style="dim")
    for index, question in enumerate(questions, start=1):
        body.append(f"\n  {index}  ", style="dim")
        body.append(question)
    return to_text(body)


def render_decision_questions(summary: str, questions: tuple[str, ...]) -> str:
    """Короткий висновок і входи в подробиці замість стіни тексту."""
    body = Text(summary)
    if questions:
        body.append("\n\nСпитайте, якщо цікаво:", style="dim")
        for index, question in enumerate(questions, start=1):
            body.append(f"\n  [{index}] ", style="bold")
            body.append(question)
    body.append("\n  [e] ", style="bold")
    body.append("показати повне обґрунтування")
    body.append("\n  [↵] ", style="bold")
    body.append("далі")
    return to_text(Panel(body, title="Рішення", border_style="dim"))


# --- Заголовки ------------------------------------------------------------


def render_headline(decision: str, subtitle: str, details: list[str]) -> str:
    """Картка рішення: вердикт великим, деталі поруч."""
    verdict = (
        Text("ПЕРЕБАЛАНСУВАТИ портфель", style="bold yellow")
        if decision == "REBALANCE"
        else Text("НІЧОГО НЕ ЗМІНЮВАТИ", style="bold green")
    )
    body = Text()
    body.append_text(verdict)
    body.append("\n")
    body.append(subtitle)
    for line in details:
        body.append("\n")
        body.append(line, style="dim")
    return to_text(Panel(body, border_style="yellow" if decision == "REBALANCE" else "green"))


def section(title: str) -> str:
    """Роздільник секції."""
    return to_text(Text(f"\n{title}", style="bold"))
