"""Інтерактивна сесія: старт з ребалансом, покроковий HITL, далі чат.

Життєвий цикл:

    1. Портфель скидається до демо-стану (відтворюваний REBALANCE).
    2. Виконується щоденний цикл Plan-and-Execute до HITL-переривання.
    3. Показується короткий висновок і набір уточнювальних питань;
       розгорнутий наратив будується лише на вимогу.
    4. Кожна запропонована дія підтверджується окремо.
    5. Застосунок не завершується, а переходить у режим чату з ReAct-агентом.

Переривання виконує граф (`interrupt_before=["execute_rebalance"]`), а цей
модуль лише спілкується з людиною поверх нього.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from app.agents.decision_engine import collect_metrics
from app.agents.decision_questions import build_questions
from app.agents.explainer import build_explanation, explain_action, render_summary
from app.agents.plan_execute import build_workflow, open_checkpointer
from app.agents.react_agent import run_react_task
from app.agents.suggestions import suggest_followups
from app.config import get_settings
from app.data.analytics import compute_turnover
from app.data.portfolio_store import load_portfolio, seed_demo_portfolio
from app.data.projection import project_asset, project_portfolio
from app.domain.errors import DomainError
from app.domain.schemas import AssetMetrics, Portfolio
from app.interactive import autonomy, render, visuals
from app.interactive.approval import (
    SelectionSummary,
    resolve_approval,
    summarize_selection,
)
from app.observability import history, trajectory

logger = logging.getLogger(__name__)

# Скільки разів дозволяємо переграти план після Modify, щоб не зациклитись
MAX_APPROVAL_ROUNDS = 3

# Підказка керування чек-листом
CHECKLIST_HINT = "[1-9] перемкнути · [a] усі · [n] жодної · [?N] чому · [↵] далі"


def _parse_weights(raw: str) -> Portfolio:
    """Розібрати рядок виду 'BTC=0.4,ETH=0.3' у портфель."""
    positions: dict[str, float] = {}
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        symbol, _, weight = chunk.partition("=")
        if not weight:
            raise ValueError(f"очікується SYMBOL=вага, отримано '{chunk.strip()}'")
        positions[symbol.strip().upper()] = float(weight)
    return Portfolio(positions=positions)  # type: ignore[arg-type]


def _same_actions(
    left: list[dict[str, Any]], right: list[dict[str, Any]], tolerance: float = 1e-4
) -> bool:
    """Чи однакові два набори дій за складом і кінцевими вагами."""
    if len(left) != len(right):
        return False
    def by_symbol(action: dict[str, Any]) -> str:
        return str(action.get("symbol"))

    for first, second in zip(
        sorted(left, key=by_symbol), sorted(right, key=by_symbol), strict=True
    ):
        if first.get("symbol") != second.get("symbol"):
            return False
        delta = abs(float(first.get("to_weight") or 0.0) - float(second.get("to_weight") or 0.0))
        if delta > tolerance:
            return False
    return True


def _same_weights(
    left: dict[str, float], right: dict[str, float], tolerance: float = 1e-4
) -> bool:
    """Чи однакові два набори ваг у межах похибки округлення."""
    if set(left) != set(right):
        return False
    return all(abs(left[symbol] - right[symbol]) <= tolerance for symbol in left)


class InteractiveSession:
    """Одна сесія роботи з агентом у терміналі."""

    def __init__(
        self,
        thread_id: str,
        reader: Callable[[str], str] = input,
        writer: Callable[[str], None] = print,
        *,
        seed_demo: bool = True,
    ) -> None:
        self._thread_id = thread_id
        self._base_thread_id = thread_id
        self._run_index = 0
        self._read = reader
        self._write = writer
        self._seed_demo = seed_demo
        self._config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    # --- Публічний вхід ---------------------------------------------------

    def run(self) -> int:
        """Запустити сесію. Повертає код виходу процесу."""
        self._write(render.BANNER)
        self._write(f"\nThread: {self._thread_id}")

        with open_checkpointer() as checkpointer:
            workflow = build_workflow(checkpointer=checkpointer)
            try:
                self._startup_cycle(workflow)
                self._chat_loop(workflow)
            except (KeyboardInterrupt, EOFError):
                self._write("\nСесію завершено.")
        return 0

    # --- Стартовий цикл ---------------------------------------------------

    def _startup_cycle(self, workflow: Any) -> None:
        """Прогнати цикл аналізу і, за потреби, провести через HITL."""
        if self._seed_demo:
            seeded = seed_demo_portfolio()
            self._write(visuals.render_allocation(seeded.as_mapping(), "Стартовий портфель"))

        self._write(render.header("Аналіз ринку і портфеля"))
        self._write("Це займає до півтори хвилини: планування, збір метрик, рішення.\n")

        trajectory.reset()
        self._stream_cycle(workflow)
        state = workflow.get_state(self._config)

        evidence = state.values.get("policy_evidence") or {}
        decision = state.values.get("decision") or "HOLD"
        metrics = self._asset_metrics(state.values)

        self._render_decision(evidence, decision)
        self._explore_decision(evidence, decision, metrics)

        if not state.next:
            self._write(
                "\nЗмін не потрібно, підтвердження не запитується — HOLD є "
                "завершеним рішенням."
            )
            return

        self._run_approval(workflow, metrics)

    def _explore_decision(
        self,
        evidence: dict[str, Any],
        decision: str,
        metrics: dict[str, AssetMetrics],
    ) -> None:
        """Короткий висновок і входи в подробиці замість стіни тексту.

        Розгорнутий наратив від LLM будується лениво — тільки якщо його
        попросили. Так старт не чекає на генерацію тексту, який більшість
        не читає.
        """
        questions = build_questions(evidence, decision, metrics)
        summary = render_summary(evidence, decision, metrics)

        self._write("")
        self._write(
            visuals.render_decision_questions(
                summary, tuple(item.text for item in questions)
            )
        )
        # Панель друкується один раз; далі — короткий рядок, інакше кожна
        # відповідь виштовхує попередню за межі екрана
        hint = f"[1-{len(questions)}] ще питання · [e] повне обґрунтування · [↵] далі"

        while True:
            try:
                answer = self._read("> ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                return

            if answer == "":
                return
            if answer == "e":
                self._write("")
                self._write(build_explanation(evidence, decision, metrics))
            elif answer.isdigit() and 1 <= int(answer) <= len(questions):
                index = int(answer) - 1
                self._write("")
                self._write(f"{questions[index].text}\n\n{questions[index].answer}")
            else:
                self._write("Не зрозумів. Введіть номер питання, e або Enter.")
                continue
            self._write(f"\n{hint}")

    def _render_decision(self, evidence: dict[str, Any], decision: str) -> None:
        """Картка рішення: вердикт, шкала прогнозу, відхилення від цілі."""
        amount = get_settings().explanation_reference_amount
        current_weights = evidence.get("current_weights") or {}
        target_weights = evidence.get("best_candidate_weights") or {}

        before = project_portfolio(current_weights, amount)
        after = project_portfolio(target_weights, amount) if target_weights else None

        subtitle, details = self._headline_text(evidence, decision, before, after)
        self._write("")
        self._write(visuals.render_headline(decision, subtitle, details))

        rows = []
        if before is not None:
            rows.append(
                visuals.RangeRow(
                    "зараз",
                    before.downside_value,
                    before.median_value,
                    before.upside_value,
                    before.loss_probability,
                )
            )
        if after is not None:
            rows.append(
                visuals.RangeRow(
                    "план",
                    after.downside_value,
                    after.median_value,
                    after.upside_value,
                    after.loss_probability,
                    accent="green",
                )
            )
        if rows:
            horizon = (after or before).horizon_days  # type: ignore[union-attr]
            self._write(visuals.render_forecast(rows, horizon, amount))

        if target_weights:
            self._write(visuals.render_drift(current_weights, target_weights))

    def _headline_text(
        self,
        evidence: dict[str, Any],
        decision: str,
        before: Any,
        after: Any,
    ) -> tuple[str, list[str]]:
        """Підзаголовок і дрібні рядки картки рішення."""
        net = evidence.get("net_improvement_after_turnover")
        threshold = evidence.get("minimum_improvement_score")
        turnover = evidence.get("turnover")
        actions = evidence.get("proposed_actions") or []

        details: list[str] = []
        if decision == "REBALANCE" and before is not None and after is not None:
            moved = float(turnover or 0.0) * get_settings().explanation_reference_amount
            subtitle = (
                f"{len(actions)} дій · оборот {float(turnover or 0.0) * 100:.1f}% "
                f"({visuals.money(moved)})"
            )
            drop = (before.loss_probability - after.loss_probability) * 100
            if drop > 1:
                details.append(
                    f"Головне не більший прибуток, а менший шанс втратити: "
                    f"ризик збитку падає на {drop:.0f} п.п."
                )
        else:
            subtitle = "Жоден варіант не виграв достатньо, щоб окупити перехід."

        if net is not None and threshold is not None:
            comparison = ">=" if float(net) >= float(threshold) else "<"
            details.append(
                f"[розрахунок: чисте покращення {float(net):+.4f} {comparison} "
                f"поріг {float(threshold):.4f}]"
            )
        return subtitle, details

    def _stream_cycle(self, workflow: Any, resume: bool = False) -> None:
        """Виконати граф, показуючи прогрес по вузлах."""
        settings = get_settings()
        initial: dict[str, Any] | None = None
        if not resume:
            initial = {"messages": [], "lookback_days": settings.decision_lookback_days}

        for chunk in workflow.stream(initial, config=self._config):
            for node in chunk:
                self._write(f"  ... {node}")

    # --- HITL -------------------------------------------------------------

    def _run_approval(self, workflow: Any, metrics: dict[str, Any]) -> None:
        """Провести підтвердження, за потреби кілька раундів після Modify."""
        for round_index in range(MAX_APPROVAL_ROUNDS):
            state = workflow.get_state(self._config)
            if not state.next:
                return

            proposal = state.values.get("rebalance_proposal") or {}
            actions = list(proposal.get("actions") or [])
            if not actions:
                self._write("Пропозиція порожня — виконувати нема чого.")
                return

            current = Portfolio(positions=state.values["current_portfolio"])

            if round_index == 0:
                original_actions = list(actions)
                update = self._collect_per_action(current, actions, metrics)
            else:
                update = self._confirm_revised(actions, original_actions)

            if update is None:
                self._write("\nПідтвердження скасовано — трактую як відмову.")
                update = {"approval": "reject"}

            workflow.update_state(self._config, update)
            self._write("")
            self._stream_cycle(workflow, resume=True)

            final = workflow.get_state(self._config)
            if not final.next:
                self._report_execution(final)
                return

            self._write(visuals.section("Потрібне повторне підтвердження"))

        self._write("Досягнуто ліміту раундів підтвердження. Ребаланс не виконано.")

    def _collect_per_action(
        self,
        current: Portfolio,
        actions: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Показати чек-лист дій і дати людині перемикати їх до підтвердження."""
        self._write(visuals.section("Підтвердження"))
        self._write(
            "Дії позначені галочками. Перемикайте будь-які — підсумок унизу\n"
            "перераховується одразу, ще до підтвердження."
        )

        decisions = self._select_actions(current, actions, metrics)
        if decisions is None:
            return None

        self._update_autonomy(actions, decisions)

        try:
            approval, positions = resolve_approval(current, actions, decisions)
        except DomainError as error:
            self._write(f"\nЦя комбінація не зводиться до валідного портфеля: {error}")
            self._write("Трактую як відмову; можна повторити цикл командою /rerun.")
            return {"approval": "reject"}

        update: dict[str, Any] = {"approval": approval}
        if positions is not None:
            update["modified_positions"] = positions
        return update

    def _select_actions(
        self,
        current: Portfolio,
        actions: list[dict[str, Any]],
        metrics: dict[str, Any],
        force_checklist: bool = False,
    ) -> list[bool] | None:
        """Цикл чек-листа. None — ввід перервано.

        Виходить лише коли поточний набір галочок зводиться до валідного
        портфеля: неможливу комбінацію видно одразу, а не після підтвердження.
        """
        state = autonomy.load_state()
        accepted = [True] * len(actions)
        delegated = (
            [autonomy.is_delegable(action, state.max_delta) for action in actions]
            if state.granted
            else [False] * len(actions)
        )
        if not force_checklist and autonomy.covers_all(actions, state):
            return self._confirm_delegated(current, actions, state)

        if state.granted and any(delegated):
            self._write(
                f"{sum(delegated)} дрібних змін позначено автоматично — ви делегували "
                f"зміни ваги до {state.max_delta * 100:.0f} п.п. (/autonomy)"
            )
        outlooks = self._action_outlooks(actions)
        full_plan = summarize_selection(current, actions, accepted).portfolio

        while True:
            summary = summarize_selection(current, actions, accepted)
            self._write("")
            self._write(
                visuals.render_checklist(
                    actions,
                    accepted,
                    outlooks,
                    delegated,
                    self._summary_line(current, summary),
                    self._result_line(summary, full_plan),
                    CHECKLIST_HINT,
                )
            )

            try:
                answer = self._read("> ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                return None

            if answer == "":
                if summary.is_valid:
                    return accepted
                self._write("Спершу зробіть комбінацію можливою — див. підказку вище.")
            elif answer == "a":
                accepted = [True] * len(actions)
            elif answer == "n":
                accepted = [False] * len(actions)
            elif answer.startswith("?"):
                self._explain_one(answer[1:], actions, metrics)
            elif answer.isdigit() and 1 <= int(answer) <= len(actions):
                index = int(answer) - 1
                accepted[index] = not accepted[index]
            else:
                self._write("Не зрозумів. Введіть номер дії, a, n, ?номер або Enter.")

    def _confirm_delegated(
        self,
        current: Portfolio,
        actions: list[dict[str, Any]],
        state: autonomy.AutonomyState,
    ) -> list[bool] | None:
        """Скорочений шлях, коли весь план у межах делегованого.

        Людина не зникає з контуру — вона підтверджує план одним рядком замість
        порядкового перегляду. Відмова тут так само можлива.
        """
        summary = summarize_selection(current, actions, [True] * len(actions))
        self._write("")
        self._write(
            f"Весь план — дрібні зміни ваги в межах делегованих "
            f"{state.max_delta * 100:.0f} п.п., тож детальний перегляд пропущено."
        )
        self._write(self._summary_line(current, summary))
        self._write(self._result_line(summary, None))
        try:
            answer = self._read("Виконати? [y] так · [n] ні · [d] показати деталі\n> ")
        except (KeyboardInterrupt, EOFError):
            return None

        choice = answer.strip().lower()
        if choice == "d":
            # Людина захотіла подробиць — повертаємось до повного чек-листа
            return self._select_actions(current, actions, {}, force_checklist=True)
        if choice in {"y", "yes", "так", "т", ""}:
            return [True] * len(actions)
        return [False] * len(actions)

    def _update_autonomy(
        self, actions: list[dict[str, Any]], decisions: list[bool]
    ) -> None:
        """Оновити статистику довіри і, за потреби, запропонувати делегування."""
        state = autonomy.record_decisions(actions, decisions, autonomy.load_state())
        if not autonomy.can_offer(state):
            autonomy.save_state(state)
            return

        self._write("")
        self._write(
            f"Ви схвалили {state.accepted} із {state.total} дрібних змін ваги "
            f"({state.acceptance_rate * 100:.0f}%)."
        )
        self._write(
            f"Дозволити позначати зміни до "
            f"{get_settings().autonomy_max_delta * 100:.0f} п.п. автоматично? "
            "Фінальне підтвердження плану залишиться за вами."
        )
        try:
            answer = self._read("[y] так · [n] ні · відкликати будь-коли через /autonomy\n> ")
        except (KeyboardInterrupt, EOFError):
            answer = "n"

        if answer.strip().lower() in {"y", "yes", "так", "т"}:
            state = autonomy.grant(state)
            self._write("Делеговано. Перегляд і скасування — командою /autonomy.")
            trajectory.record_event(node="autonomy", event="granted", accepted=state.accepted)
        else:
            self._write("Гаразд, кожна дія й далі потребуватиме вашої галочки.")
        autonomy.save_state(state)

    def _action_outlooks(
        self, actions: list[dict[str, Any]]
    ) -> dict[str, tuple[float, float]]:
        """Прогноз на тисячу вкладених для кожного активу з плану."""
        outlooks: dict[str, tuple[float, float]] = {}
        for action in actions:
            symbol = str(action.get("symbol", ""))
            if symbol in outlooks:
                continue
            projection = project_asset(symbol, 1000.0)
            if projection is not None:
                outlooks[symbol] = (projection.median_value, projection.loss_probability)
        return outlooks

    def _summary_line(self, current: Portfolio, summary: SelectionSummary) -> str:
        """Рядок живого підсумку під чек-листом."""
        head = f"Обрано {summary.accepted_count} з {summary.total_count}"
        if summary.error:
            return f"{head} · неможливо: {summary.error}"
        if summary.accepted_count == 0 or summary.portfolio is None:
            return f"{head} · портфель лишається без змін"

        line = f"{head} · оборот {summary.turnover * 100:.1f}%"

        amount = get_settings().explanation_reference_amount
        before = project_portfolio(current.as_mapping(), amount)
        after = project_portfolio(summary.portfolio.as_mapping(), amount)
        if before is not None and after is not None:
            line += (
                f" · ризик збитку {before.loss_probability * 100:.0f}%"
                f" -> {after.loss_probability * 100:.0f}%"
            )
        return line

    def _result_line(self, summary: SelectionSummary, full_plan: Portfolio | None) -> str:
        """Підсумковий склад портфеля після обраних дій."""
        if summary.portfolio is None or summary.accepted_count == 0:
            return ""

        result = summary.portfolio.as_mapping()
        parts = [
            f"{symbol} {weight * 100:.1f}"
            for symbol, weight in sorted(result.items(), key=lambda item: -item[1])
        ]
        line = "Вийде: " + " · ".join(parts)

        # Зняти одну галочку зі скорочення часто не змінює нічого: решта ваг
        # зафіксована, тому вивільнене повертається тому ж активу. Краще сказати
        # це прямо, ніж лишати враження, що інтерфейс не реагує.
        if (
            full_plan is not None
            and summary.accepted_count < summary.total_count
            and _same_weights(result, full_plan.as_mapping())
        ):
            line += "\nСклад той самий, що й з усіма діями: решта ваг зафіксована,"
            line += " тож вивільнене повертається тому ж активу."
        return line

    def _explain_one(
        self, raw_index: str, actions: list[dict[str, Any]], metrics: dict[str, Any]
    ) -> None:
        """Розгорнуте пояснення однієї дії за номером."""
        if not raw_index.strip().isdigit():
            self._write("Вкажіть номер дії, наприклад ?3")
            return
        index = int(raw_index.strip()) - 1
        if not 0 <= index < len(actions):
            self._write(f"Немає дії з номером {index + 1}.")
            return
        action = actions[index]
        self._write("")
        self._write(explain_action(action, metrics))
        self._write(self._detailed_reason(action, metrics))

    def _detailed_reason(self, action: dict[str, Any], metrics: dict[str, Any]) -> str:
        """Розширене пояснення однієї дії з усіма метриками активу."""
        symbol = action.get("symbol", "")
        metric = metrics.get(symbol)
        if metric is None:
            return f"Детальних метрик для {symbol} немає."
        data = metric.model_dump(mode="json")
        lines = [f"Метрики {symbol} за {data['lookback_days']} днів:"]
        lines.append(f"  дохідність (річна):  {data['annualized_return'] * 100:+.1f}%")
        lines.append(f"  волатильність:       {data['volatility'] * 100:.1f}%")
        lines.append(f"  дохідність/ризик:    {data['sharpe_like']:+.2f}")
        lines.append(f"  макс. просадка:      {data['max_drawdown'] * 100:.1f}%")
        lines.append(f"  сила тренду:         {data['trend_strength']:+.2f}")
        lines.append(f"  сер. обсяг за добу:  ${data['avg_daily_volume_usd']:,.0f}")
        return "\n".join(lines)

    def _confirm_revised(
        self, actions: list[dict[str, Any]], original: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Після Modify підтверджуємо переглянутий план цілком, а не поактивно."""
        if _same_actions(actions, original):
            self._write(
                "Переглянутий план збігається з початковим: знята галочка не змінила\n"
                "жодної кінцевої ваги, бо решта позицій зафіксована."
            )
        self._write("План до виконання:")
        for action in actions:
            self._write(
                f"  {action['action']} {action['symbol']}: "
                f"{action['from_weight'] * 100:.1f}% -> {action['to_weight'] * 100:.1f}%"
            )
        try:
            answer = self._read("\nВиконати цей план? [y/n]\n> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return None
        return {"approval": "approve" if answer in {"y", "yes", "так", "т"} else "reject"}

    def _report_execution(self, state: Any) -> None:
        """Показати результат виконання (mock) і новий склад портфеля."""
        result = state.values.get("execution_result") or {}
        self._write(visuals.section("Результат"))
        self._write(render.render_execution_result(result))
        self._write(visuals.render_allocation(load_portfolio().as_mapping(), "Портфель зараз"))

    # --- Чат --------------------------------------------------------------

    def _chat_loop(self, workflow: Any) -> None:
        """Режим очікування питань. Виходить лише за командою або EOF."""
        self._write(render.header("Режим чату"))
        self._write(render.HELP_TEXT)

        while True:
            try:
                text = self._read("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                self._write("\nДо побачення.")
                return

            if not text:
                continue
            if text.startswith("/"):
                if not self._handle_command(text, workflow):
                    return
                continue

            self._answer(text)

    def _handle_command(self, text: str, workflow: Any) -> bool:
        """Виконати службову команду. False — час завершувати сесію."""
        command = text.split()[0].lower()
        match command:
            case "/exit" | "/quit":
                self._write("До побачення.")
                return False
            case "/help":
                self._write(render.HELP_TEXT)
            case "/portfolio":
                self._write(visuals.render_allocation(load_portfolio().as_mapping()))
            case "/state":
                self._write(self._render_state(workflow))
            case "/trajectory":
                self._write(render.render_trajectory(self._session_trajectory()))
            case "/history":
                self._write(visuals.render_history(history.read_history()))
            case "/autonomy":
                self._handle_autonomy(text)
            case "/whatif":
                self._handle_whatif(text)
            case "/rerun":
                self._rerun(workflow)
            case _:
                self._write(f"Невідома команда '{command}'. /help — список команд.")
        return True

    def _rerun(self, workflow: Any) -> None:
        """Повторити цикл у новому треді.

        Новий thread_id, а не той самий: інакше повторний прогін нашаровується
        на завершений чекпойнт, і `/state` перестає відповідати конкретному
        циклу. Портфель при цьому не скидається — рахуємо від фактичного стану.
        """
        self._run_index += 1
        self._thread_id = f"{self._base_thread_id}-r{self._run_index}"
        self._config = {"configurable": {"thread_id": self._thread_id}}
        self._write(f"Новий thread: {self._thread_id}")
        self._seed_demo = False
        self._startup_cycle(workflow)

    def _session_trajectory(self) -> list[dict[str, Any]]:
        """Уся траєкторія сесії з .jsonl.

        Буфер у памʼяті скидається перед кожним питанням, щоб нумерація кроків
        не росла нескінченно, тому повну історію читаємо з файлу.
        """
        path = get_settings().trajectory_log_path
        if not path.exists():
            return trajectory.snapshot()
        entries: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Не вдалося прочитати trajectory-лог: %s", error)
            return trajectory.snapshot()
        return entries

    def _handle_autonomy(self, text: str) -> None:
        """Показати або змінити рівень делегування."""
        parts = text.split()
        argument = parts[1].lower() if len(parts) > 1 else ""
        state = autonomy.load_state()

        if argument == "off":
            if not state.granted:
                self._write("Делегування і так не надано.")
                return
            autonomy.save_state(autonomy.revoke(state))
            self._write("Делегування відкликано. Кожна дія знову потребує галочки.")
            trajectory.record_event(node="autonomy", event="revoked")
            return

        if argument == "on":
            if state.granted:
                self._write("Делегування вже надано.")
                return
            if not autonomy.can_offer(state):
                self._write(
                    "Ще рано: агент не набрав достатньо історії схвалень.\n"
                    + autonomy.describe(state)
                )
                return
            autonomy.save_state(autonomy.grant(state))
            self._write("Делеговано зміни ваги до порогу. Відкликати: /autonomy off")
            trajectory.record_event(node="autonomy", event="granted", manual=True)
            return

        if argument:
            self._write("Використання: /autonomy [on|off]")
            return

        self._write(autonomy.describe(state))

    def _handle_whatif(self, text: str) -> None:
        """Порахувати прогноз для довільного складу без зміни портфеля."""
        raw = text.partition(" ")[2].strip()
        if not raw:
            self._write(
                "Використання: /whatif BTC=0.4,ETH=0.3,SOL=0.3\n"
                "Портфель не змінюється — це лише розрахунок."
            )
            return

        try:
            hypothetical = _parse_weights(raw)
        except (ValueError, DomainError) as error:
            self._write(f"Не вдалося розібрати склад: {error}")
            return

        amount = get_settings().explanation_reference_amount
        current = load_portfolio()
        before = project_portfolio(current.as_mapping(), amount)
        after = project_portfolio(hypothetical.as_mapping(), amount)
        if before is None or after is None:
            self._write("Для цього складу бракує даних, щоб побудувати прогноз.")
            return

        self._write(visuals.render_allocation(hypothetical.as_mapping(), "Гіпотетичний склад"))
        self._write(
            visuals.render_forecast(
                [
                    visuals.RangeRow(
                        "зараз",
                        before.downside_value,
                        before.median_value,
                        before.upside_value,
                        before.loss_probability,
                    ),
                    visuals.RangeRow(
                        "варіант",
                        after.downside_value,
                        after.median_value,
                        after.upside_value,
                        after.loss_probability,
                        accent="green",
                    ),
                ],
                after.horizon_days,
                amount,
            )
        )
        turnover = compute_turnover(current, hypothetical)
        self._write(
            f"Перехід зачепив би {turnover * 100:.1f}% портфеля "
            f"({visuals.money(turnover * amount)}). Це лише розрахунок — "
            "портфель не змінено."
        )

    def _render_state(self, workflow: Any) -> str:
        """Демонстрація get_state() для поточного треду."""
        state = workflow.get_state(self._config)
        if not state.created_at:
            return f"Для thread '{self._thread_id}' немає збереженого стану."
        values = state.values
        return "\n".join(
            [
                f"next node:         {state.next}",
                f"created_at:        {state.created_at}",
                f"decision:          {values.get('decision')}",
                f"approval:          {values.get('approval') or '(немає)'}",
                f"current_portfolio: {values.get('current_portfolio')}",
                f"dropped_assets:    {values.get('dropped_assets')}",
                f"errors:            {values.get('errors')}",
            ]
        )

    def _answer(self, question: str) -> None:
        """Відповісти на довільне питання через ReAct-агента."""
        trajectory.reset()
        try:
            result = run_react_task(question)
        except Exception as error:  # noqa: BLE001 — чат не має падати через збій LLM
            logger.warning("Помилка ReAct-агента: %s", error)
            self._write(f"Не вдалося обробити питання: {error}")
            return

        self._write("")
        self._write(result.output or "(порожня відповідь)")
        # У ReActResult.tool_calls записи мають ключ "tool", а не "name"
        tools_used = [str(call.get("tool", "?")) for call in result.tool_calls]
        used = ", ".join(tools_used) or "жодного"
        self._write(f"\n[кроків: {result.steps}, tools: {used}, stop: {result.stop_reason}]")
        self._write(visuals.render_suggestions(
            suggest_followups(question, result.output, tools_used)
        ))

    # --- Допоміжне --------------------------------------------------------

    def _asset_metrics(self, values: dict[str, Any]) -> dict[str, Any]:
        """Порахувати метрики активів, що беруть участь у рішенні."""
        settings = get_settings()
        # У стані портфель зберігається як мапа symbol -> weight
        symbols = set(values.get("current_portfolio") or {})
        proposal = values.get("rebalance_proposal") or {}
        for action in proposal.get("actions") or []:
            symbols.add(action["symbol"])

        lookback = values.get("lookback_days") or settings.decision_lookback_days
        metrics, _ = collect_metrics(sorted(symbols), lookback)
        return metrics
