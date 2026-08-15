"""CLI для запуску агента: щоденний цикл, HITL-рішення та інспекція стану.

Приклади:
    python -m app                          # інтерактивний режим (старт + чат)
    python -m app.cli chat
    python -m app.cli run --thread daily-2026-08-12
    python -m app.cli state --thread daily-2026-08-12
    python -m app.cli approve --thread daily-2026-08-12
    python -m app.cli reject --thread daily-2026-08-12
    python -m app.cli modify --thread daily-2026-08-12 --positions "BTC=0.5,ETH=0.3,SOL=0.2"
    python -m app.cli ask "What is max drawdown?"
    python -m app.cli portfolio --set "BNB=0.35,SOL=0.30,ADA=0.35"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.agents.plan_execute import build_workflow, open_checkpointer
from app.agents.react_agent import run_react_task
from app.data.portfolio_store import load_portfolio, reset_portfolio, save_portfolio
from app.domain.schemas import Portfolio, Position
from app.observability import trajectory

logger = logging.getLogger(__name__)

SEPARATOR = "=" * 72


def _parse_positions(raw: str) -> Portfolio:
    """Розібрати рядок виду 'BTC=0.5,ETH=0.3,SOL=0.2' у Portfolio."""
    positions = []
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        symbol, _, weight = chunk.partition("=")
        if not weight:
            raise ValueError(f"Некоректний формат позиції: '{chunk}'. Очікується SYMBOL=вага")
        positions.append(Position(symbol=symbol.strip(), weight=float(weight)))
    return Portfolio(positions=tuple(positions))


def _print_interrupt(state: Any) -> None:
    """Показати пропозицію, що чекає на підтвердження людини."""
    proposal = state.values.get("rebalance_proposal")
    print(f"\n{SEPARATOR}\nHUMAN-IN-THE-LOOP: потрібне підтвердження\n{SEPARATOR}")
    if proposal:
        actions = "\n".join(
            f"  {action['action']} {action['symbol']}"
            f" ({action['from_weight'] * 100:.0f}% -> {action['to_weight'] * 100:.0f}%)"
            for action in proposal["actions"]
        )
        print(f"Decision: REBALANCE\n\nProposed actions:\n{actions}")
        print(f"\nTurnover: {proposal['turnover'] * 100:.1f}%")
        print(f"Improvement score: {proposal['improvement_score']:+.4f}")
    print(f"\nНаступний вузол: {state.next}")
    print("\nОберіть дію:  approve  |  reject  |  modify")


def _run_daily_cycle(thread_id: str, lookback: int | None) -> int:
    """Запустити щоденний цикл до кінця або до HITL-переривання."""
    trajectory.reset()
    with open_checkpointer() as checkpointer:
        workflow = build_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        initial: dict[str, Any] = {"messages": []}
        if lookback:
            initial["lookback_days"] = lookback

        workflow.invoke(initial, config=config)
        state = workflow.get_state(config)

        if state.next:
            _print_interrupt(state)
            return 0

        decision = state.values.get("final_decision", {})
        print(f"\n{SEPARATOR}")
        print(_render_decision(decision))
        print(SEPARATOR)
        return 0


def _render_decision(decision: dict[str, Any]) -> str:
    """Людиночитаний вивід фінального рішення."""
    if not decision:
        return "Рішення відсутнє."
    lines = [f"Decision: {decision['decision']}", "", "Current portfolio:"]
    lines += [f"{p['symbol']} {p['weight'] * 100:.0f}%" for p in decision["current_portfolio"]]
    if decision["decision"] == "REBALANCE" and decision.get("proposed_portfolio"):
        lines += ["", "Proposed portfolio:"]
        lines += [
            f"{p['symbol']} {p['weight'] * 100:.0f}%" for p in decision["proposed_portfolio"]
        ]
        lines += ["", "Actions:"]
        lines += [
            f"{a['action']} {a['symbol']}"
            f" ({a['from_weight'] * 100:.0f}% -> {a['to_weight'] * 100:.0f}%)"
            for a in decision["actions"]
        ]
    lines += ["", "Reason:", decision["reasoning"]]
    if decision["decision"] == "HOLD":
        lines += ["", "Action required:", "None"]
    return "\n".join(lines)


def _resume(thread_id: str, approval: str, positions: str | None = None) -> int:
    """Продовжити перерваний workflow рішенням людини."""
    with open_checkpointer() as checkpointer:
        workflow = build_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        state = workflow.get_state(config)
        if not state.next:
            print(f"Thread '{thread_id}' не очікує підтвердження (next={state.next}).")
            return 1

        update: dict[str, Any] = {"approval": approval}
        if approval == "modify":
            if not positions:
                print("Для modify потрібен параметр --positions, напр. 'BTC=0.5,ETH=0.5'")
                return 1
            portfolio = _parse_positions(positions)
            update["modified_positions"] = [
                {"symbol": p.symbol, "weight": p.weight} for p in portfolio.positions
            ]

        # Записуємо рішення людини у стан і відновлюємо виконання
        workflow.update_state(config, update)
        workflow.invoke(None, config=config)
        final = workflow.get_state(config)

        if final.next:
            # Modify -> повторний interrupt на оновленому плані
            _print_interrupt(final)
            return 0

        result = final.values.get("execution_result") or {}
        print(f"\n{SEPARATOR}")
        print(f"Статус виконання: {result.get('status', 'unknown')}")
        if result.get("status") == "executed":
            print(f"Операцій виконано: {result['operations_count']} (mock)")
            print(f"Портфель до:    {result['portfolio_before']}")
            print(f"Портфель після: {result['portfolio_after']}")
        else:
            print(result.get("message", ""))
            print(f"Портфель: {load_portfolio().as_mapping()}")
        print(SEPARATOR)
        return 0


def _show_state(thread_id: str) -> int:
    """Продемонструвати get_state(): що збережено у чекпойнті."""
    with open_checkpointer() as checkpointer:
        workflow = build_workflow(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        state = workflow.get_state(config)

        if not state.created_at:
            print(f"Для thread '{thread_id}' немає збереженого стану.")
            return 1

        values = state.values
        print(f"{SEPARATOR}\nget_state() для thread '{thread_id}'\n{SEPARATOR}")
        print(f"next node:        {state.next}")
        print(f"created_at:       {state.created_at}")
        print(f"checkpoint_id:    {state.config['configurable'].get('checkpoint_id')}")
        print(f"current_portfolio:{values.get('current_portfolio')}")
        print(f"candidate_assets: {values.get('candidate_assets')}")
        print(f"dropped_assets:   {values.get('dropped_assets')}")
        print(f"decision:         {values.get('decision')}")
        print(f"approval:         {values.get('approval') or '(немає)'}")
        print(f"completed_steps:  {len(values.get('completed_steps', []))}")
        print(f"tool_history:     {len(values.get('tool_history', []))} викликів")
        print(f"errors:           {values.get('errors')}")
        plan = values.get("plan") or {}
        if plan:
            print(f"\nplan.goal: {plan.get('goal')}")
            for step in plan.get("steps", []):
                print(f"  [{step['step_id']}] {step['description']}")
        print(SEPARATOR)
        return 0


def _ask(question: str) -> int:
    """Одноразовий запит до ReAct-агента (демонстрація agentic RAG)."""
    trajectory.reset()
    result = run_react_task(question)
    print(f"\n{SEPARATOR}")
    print(result.output)
    print(f"\n{SEPARATOR}")
    print(f"stop_reason: {result.stop_reason} | steps: {result.steps}")
    print("Викликані tools:", [call["tool"] for call in result.tool_calls] or "жодного")
    return 0


def _portfolio(set_value: str | None, do_reset: bool) -> int:
    """Показати або змінити збережений портфель."""
    if do_reset:
        print(f"Портфель скинуто до дефолтного: {reset_portfolio().as_mapping()}")
        return 0
    if set_value:
        portfolio = _parse_positions(set_value)
        save_portfolio(portfolio)
        print(f"Портфель збережено:\n{portfolio.render()}")
        return 0
    print(load_portfolio().render())
    return 0


def _trajectory_tail(limit: int) -> int:
    """Показати останні записи JSON-траєкторії."""
    from app.config import get_settings

    path = get_settings().trajectory_log_path
    if not path.exists():
        print("Traject-лог порожній.")
        return 1
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines[-limit:]:
        print(json.dumps(json.loads(line), ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crypto-rebalancer", description="AI Crypto Portfolio Rebalancing Agent"
    )
    parser.add_argument("--verbose", action="store_true", help="Детальне логування")
    sub = parser.add_subparsers(dest="command", required=False)

    chat_cmd = sub.add_parser("chat", help="Інтерактивний режим: старт з ребалансом і чат")
    chat_cmd.add_argument("--thread", default=None, help="thread_id для чекпойнта")
    chat_cmd.add_argument(
        "--no-seed",
        action="store_true",
        help="Не скидати портфель до демо-стану на старті",
    )

    run_cmd = sub.add_parser("run", help="Запустити щоденний цикл аналізу")
    run_cmd.add_argument("--thread", default="daily-default", help="thread_id для чекпойнта")
    run_cmd.add_argument("--lookback", type=int, default=None, help="Горизонт аналізу в днях")

    for name, help_text in (
        ("approve", "Підтвердити ребаланс і виконати mock-операції"),
        ("reject", "Відхилити ребаланс без змін портфеля"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--thread", default="daily-default")

    modify_cmd = sub.add_parser("modify", help="Змінити план і повторно пройти перевірку")
    modify_cmd.add_argument("--thread", default="daily-default")
    modify_cmd.add_argument(
        "--positions", required=True, help="Новий склад, напр. 'BTC=0.5,ETH=0.3,SOL=0.2'"
    )

    state_cmd = sub.add_parser("state", help="Показати збережений стан (get_state)")
    state_cmd.add_argument("--thread", default="daily-default")

    ask_cmd = sub.add_parser("ask", help="Питання до ReAct-агента")
    ask_cmd.add_argument("question", help="Текст питання")

    portfolio_cmd = sub.add_parser("portfolio", help="Переглянути або змінити портфель")
    portfolio_cmd.add_argument("--set", dest="set_value", default=None)
    portfolio_cmd.add_argument("--reset", action="store_true")

    trajectory_cmd = sub.add_parser("trajectory", help="Останні записи JSON-траєкторії")
    trajectory_cmd.add_argument("--limit", type=int, default=20)

    return parser


def _chat(thread_id: str | None, no_seed: bool) -> int:
    """Запустити інтерактивну сесію."""
    from app.interactive.session import InteractiveSession

    resolved = thread_id or f"chat-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    return InteractiveSession(resolved, seed_demo=not no_seed).run()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Без підкоманди — інтерактивний режим: `python -m app`
    if args.command is None:
        return _chat(None, no_seed=False)

    try:
        match args.command:
            case "chat":
                return _chat(args.thread, args.no_seed)
            case "run":
                return _run_daily_cycle(args.thread, args.lookback)
            case "approve" | "reject":
                return _resume(args.thread, args.command)
            case "modify":
                return _resume(args.thread, "modify", args.positions)
            case "state":
                return _show_state(args.thread)
            case "ask":
                return _ask(args.question)
            case "portfolio":
                return _portfolio(args.set_value, args.reset)
            case "trajectory":
                return _trajectory_tail(args.limit)
            case _:
                print(f"Невідома команда: {args.command}")
                return 1
    except KeyboardInterrupt:
        print("\nПерервано користувачем.")
        return 130
    except Exception as error:  # noqa: BLE001 — CLI має завершуватись охайно
        logger.exception("Команда завершилась помилкою")
        print(f"Помилка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
