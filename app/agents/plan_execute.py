"""Plan-and-Execute workflow на LangGraph з HITL та SqliteSaver.

Граф:

    load_portfolio -> planner -> executor -> replanner --(continue)--> executor
                                                       --(finish)----> decide
                                                       --(revise)----> executor

    decide --(HOLD)------> END
           --(REBALANCE)-> validate_proposal -> [interrupt_before] execute_rebalance
                                                       |
                              approve -> mock-виконання -> END
                              reject  -> без змін       -> END
                              modify  -> validate_proposal (повторний interrupt)

Executor — повноцінний ReAct-агент: planner каже, ЩО зробити, executor сам
вирішує, ЯК і якими tools це зробити.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.agents.decision_engine import PolicyVerdict, build_actions, evaluate_decision
from app.agents.react_agent import run_react_task
from app.agents.state import GuardedState
from app.config import get_settings
from app.data.portfolio_store import load_portfolio
from app.domain.errors import ApprovalRequiredError, DomainError, ExecutionError
from app.domain.schemas import (
    Plan,
    PlanStep,
    Portfolio,
    PortfolioDecision,
    Position,
    RebalanceProposal,
    ReplanDecision,
)
from app.llm import get_llm
from app.observability import trajectory
from app.prompts import load_prompt
from app.tools.market_tools import get_top_liquid_assets
from app.tools.portfolio_tools import mock_execute_rebalance

logger = logging.getLogger(__name__)

# Скільки разів replanner може переписати план, перш ніж граф піде до рішення
MAX_REPLANS = 3
# Скільки ліквідних активів беремо в короткий список кандидатів
CANDIDATE_SHORTLIST = 10

# План за замовчуванням, якщо LLM недоступний — граф лишається робочим
FALLBACK_PLAN = Plan(
    goal="Оцінити поточний портфель і вирішити HOLD або REBALANCE",
    steps=[
        PlanStep(step_id=1, description="Визначити universe найбільш ліквідних активів"),
        PlanStep(step_id=2, description="Порахувати risk/performance метрики кандидатів"),
        PlanStep(step_id=3, description="Оцінити поточний портфель як ціле"),
        PlanStep(step_id=4, description="Порівняти поточний портфель з альтернативами"),
    ],
)


# --- Вузли графа ---


def _load_portfolio_node(state: GuardedState) -> dict[str, Any]:
    """Крок 1-3: поточний портфель, ліквідний universe, короткий список кандидатів."""
    with trajectory.TrajectoryRecorder("load_portfolio"):
        settings = get_settings()
        portfolio = load_portfolio()

        response = get_top_liquid_assets.invoke(
            {
                "limit": settings.universe_limit,
                "min_history_days": settings.min_history_days,
                "exclude_stablecoins": True,
            }
        )
        if response["status"] == "error":
            # Без universe працюємо принаймні з поточними активами
            universe: list[str] = list(portfolio.symbols)
            errors = [
                {
                    "tool_name": "get_top_liquid_assets",
                    "error_code": response["error"]["code"],
                }
            ]
        else:
            universe = [asset["symbol"] for asset in response["data"]["assets"]]
            errors = []

        dropped = set(state.get("dropped_assets", []))
        shortlist = [symbol for symbol in universe if symbol not in dropped][
            :CANDIDATE_SHORTLIST
        ]
        # Поточні активи завжди залишаються в аналізі
        candidates = list(dict.fromkeys([*portfolio.symbols, *shortlist]))

        trajectory.record_event(
            node="load_portfolio",
            event="portfolio_loaded",
            portfolio=portfolio.as_mapping(),
            candidate_count=len(candidates),
        )

        return {
            "current_portfolio": portfolio.as_mapping(),
            "candidate_assets": candidates,
            "lookback_days": state.get("lookback_days") or settings.decision_lookback_days,
            "errors": errors,
            "messages": [
                HumanMessage(
                    content=(
                        "Проаналізуй портфель за сьогодні і виріши HOLD або REBALANCE.\n"
                        f"Поточний портфель:\n{portfolio.render()}\n"
                        f"Кандидати: {', '.join(candidates)}"
                    )
                )
            ],
        }


def _planner_node(state: GuardedState) -> dict[str, Any]:
    """Декомпозиція задачі у структурований Plan через with_structured_output."""
    with trajectory.TrajectoryRecorder("planner"):
        context = (
            f"Поточний портфель: {state.get('current_portfolio', {})}\n"
            f"Доступні кандидати: {', '.join(state.get('candidate_assets', []))}\n"
            f"Горизонт аналізу: {state.get('lookback_days')} днів"
        )
        try:
            planner = get_llm("planner").with_structured_output(Plan)
            plan = planner.invoke(
                [
                    SystemMessage(content=load_prompt("planner")),
                    HumanMessage(content=context),
                ]
            )
        except Exception as error:  # noqa: BLE001 — fallback тримає граф робочим
            logger.warning("Planner недоступний (%s), використовую резервний план", error)
            trajectory.record_event(node="planner", event="fallback_plan", error=str(error)[:200])
            plan = FALLBACK_PLAN

        assert isinstance(plan, Plan)
        trajectory.record_event(
            node="planner", event="plan_created", steps=len(plan.steps), goal=plan.goal
        )
        return {
            "plan": plan.model_dump(mode="json"),
            "replan_count": 0,
            "messages": [AIMessage(content=f"План ({len(plan.steps)} кроків): {plan.goal}")],
        }


def _executor_node(state: GuardedState) -> dict[str, Any]:
    """Виконати наступний невиконаний крок плану ReAct-агентом."""
    with trajectory.TrajectoryRecorder("executor"):
        plan = Plan.model_validate(state["plan"])
        done = {item["step_id"] for item in state.get("completed_steps", [])}
        pending = [step for step in plan.steps if step.step_id not in done]

        if not pending:
            return {"messages": [AIMessage(content="Усі кроки плану виконано.")]}

        step = pending[0]
        context = (
            f"Поточний портфель: {state.get('current_portfolio', {})}\n"
            f"Кандидати для аналізу: {', '.join(state.get('candidate_assets', []))}\n"
            f"Горизонт: {state.get('lookback_days')} днів\n"
            f"Вже виконано кроків: {len(done)}"
        )

        try:
            result = run_react_task(step.description, context=context)
            output, errors = result.output, result.errors
            tool_calls = [
                {
                    "step_id": step.step_id,
                    "tool": call["tool"],
                    "status": call["response"].get("status"),
                }
                for call in result.tool_calls
            ]
            stop_reason = result.stop_reason
        except Exception as error:  # noqa: BLE001 — крок не має валити весь workflow
            logger.exception("Крок %s провалився", step.step_id)
            output = f"Крок не виконано через помилку: {error}"
            errors = [{"tool_name": "react_executor", "error_code": "EXECUTOR_FAILURE"}]
            tool_calls, stop_reason = [], "error"

        trajectory.record_event(
            node="executor",
            event="step_completed",
            step_id=step.step_id,
            stop_reason=stop_reason,
            tool_calls=len(tool_calls),
        )

        return {
            "completed_steps": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "result": output,
                    "stop_reason": stop_reason,
                }
            ],
            "tool_history": tool_calls,
            "errors": errors,
            "messages": [AIMessage(content=f"[Крок {step.step_id}] {output}")],
        }


def _replanner_node(state: GuardedState) -> dict[str, Any]:
    """Переглянути план у світлі результатів і помилок попередніх кроків."""
    with trajectory.TrajectoryRecorder("replanner"):
        plan = Plan.model_validate(state["plan"])
        completed = state.get("completed_steps", [])
        done = {item["step_id"] for item in completed}
        remaining = [step for step in plan.steps if step.step_id not in done]
        replan_count = state.get("replan_count", 0)

        # Немає що виконувати або вичерпано ліміт переплануваннь -> до рішення
        if not remaining or replan_count >= MAX_REPLANS:
            trajectory.record_event(node="replanner", event="finish", reason="plan_exhausted")
            return {"messages": [AIMessage(content="План завершено, переходжу до рішення.")]}

        recent_errors = state.get("errors", [])
        context = (
            f"Мета: {plan.goal}\n\n"
            f"Виконані кроки:\n"
            + "\n".join(
                f"- [{item['step_id']}] {item['description']}\n  Результат: {item['result'][:600]}"
                for item in completed
            )
            + f"\n\nПомилки tools: {recent_errors or 'немає'}\n"
            + "\nКроки, що залишились:\n"
            + "\n".join(f"- [{step.step_id}] {step.description}" for step in remaining)
        )

        try:
            replanner = get_llm("replanner").with_structured_output(ReplanDecision)
            decision = replanner.invoke(
                [
                    SystemMessage(content=load_prompt("replanner")),
                    HumanMessage(content=context),
                ]
            )
        except Exception as error:  # noqa: BLE001
            logger.warning("Replanner недоступний (%s), продовжую за планом", error)
            trajectory.record_event(node="replanner", event="fallback", error=str(error)[:200])
            return {"messages": [AIMessage(content="Replanner недоступний, продовжую за планом.")]}

        assert isinstance(decision, ReplanDecision)
        trajectory.record_event(
            node="replanner",
            event="replan_decision",
            action=decision.action,
            dropped_assets=decision.dropped_assets,
        )

        update: dict[str, Any] = {
            "messages": [AIMessage(content=f"Replanner: {decision.action} — {decision.reasoning}")]
        }

        # Активи з проблемними даними виключаємо з подальшого аналізу
        if decision.dropped_assets:
            remaining_candidates = [
                symbol
                for symbol in state.get("candidate_assets", [])
                if symbol not in decision.dropped_assets
            ]
            update["dropped_assets"] = decision.dropped_assets
            update["candidate_assets"] = remaining_candidates

        if decision.action == "revise":
            # Замінюємо решту кроків новими, зберігаючи вже виконані
            executed_steps = [step for step in plan.steps if step.step_id in done]
            next_id = max([step.step_id for step in plan.steps], default=0)
            revised = []
            for offset, step in enumerate(decision.revised_steps, start=1):
                revised.append(PlanStep(step_id=next_id + offset, description=step.description))
            new_plan = Plan(goal=plan.goal, steps=[*executed_steps, *revised][:10])
            update["plan"] = new_plan.model_dump(mode="json")
            update["replan_count"] = replan_count + 1
        elif decision.action == "finish":
            # Позначаємо решту кроків як пропущені, щоб маршрутизація пішла до decide
            update["completed_steps"] = [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "result": "Пропущено: replanner визнав дані достатніми.",
                    "stop_reason": "skipped",
                }
                for step in remaining
            ]

        return update


def _decide_node(state: GuardedState) -> dict[str, Any]:
    """Порахувати вердикт політики і оформити його як PortfolioDecision."""
    with trajectory.TrajectoryRecorder("decide"):
        current = Portfolio(positions=state["current_portfolio"])  # type: ignore[arg-type]
        candidates = [
            symbol
            for symbol in state.get("candidate_assets", [])
            if symbol not in set(state.get("dropped_assets", []))
        ]
        lookback = state.get("lookback_days") or get_settings().decision_lookback_days

        verdict, errors = evaluate_decision(current, candidates, lookback)
        evidence = verdict.as_evidence()
        evidence["current_weights"] = current.as_mapping()

        decision_model = _compose_decision(verdict, current, state)

        trajectory.record_event(
            node="decide",
            event="decision_made",
            decision=decision_model.decision,
            net_improvement=verdict.net_improvement,
            threshold=verdict.threshold,
            turnover=verdict.turnover,
        )

        update: dict[str, Any] = {
            "policy_evidence": evidence,
            "portfolio_metrics": verdict.current_metrics.model_dump(mode="json"),
            "decision": decision_model.decision,
            "final_decision": decision_model.model_dump(mode="json"),
            "errors": errors,
            "messages": [AIMessage(content=decision_model.render())],
        }
        if verdict.proposal is not None:
            update["rebalance_proposal"] = verdict.proposal.model_dump(mode="json")
        return update


def _compose_decision(
    verdict: PolicyVerdict, current: Portfolio, state: GuardedState
) -> PortfolioDecision:
    """Отримати структуроване рішення від LLM; за збою — детермінований fallback.

    LLM формулює обґрунтування, але не може змінити вердикт політики: якщо
    його вивід суперечить розрахунку, він відкидається.
    """
    fallback = _fallback_decision(verdict, current)

    evidence = verdict.as_evidence()
    evidence["current_weights"] = current.as_mapping()
    summary = "\n".join(
        f"- [{item['step_id']}] {item['result'][:400]}"
        for item in state.get("completed_steps", [])
    )

    try:
        decider = get_llm("planner").with_structured_output(PortfolioDecision)
        decision = decider.invoke(
            [
                SystemMessage(content=load_prompt("decider")),
                HumanMessage(
                    content=(
                        f"Хід аналізу:\n{summary}\n\n"
                        f"Кількісні докази (JSON):\n{evidence}\n\n"
                        f"policy_verdict = {verdict.decision}. Дотримуйся його."
                    )
                ),
            ]
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("Decider недоступний (%s), використовую детермінований вивід", error)
        trajectory.record_event(node="decide", event="fallback_decision", error=str(error)[:200])
        return fallback

    if not isinstance(decision, PortfolioDecision) or decision.decision != verdict.decision:
        # LLM спробував перекрити політику — ігноруємо його вердикт
        trajectory.record_event(node="decide", event="llm_verdict_overridden")
        return fallback

    # Числові ваги завжди беремо з детермінованого розрахунку
    if verdict.decision == "REBALANCE" and verdict.proposal is not None:
        return PortfolioDecision(
            decision="REBALANCE",
            reasoning=decision.reasoning,
            current_portfolio=list(current.positions),
            proposed_portfolio=list(verdict.proposal.proposed_portfolio.positions),
            actions=list(verdict.proposal.actions),
        )
    return PortfolioDecision(
        decision="HOLD",
        reasoning=decision.reasoning,
        current_portfolio=list(current.positions),
        proposed_portfolio=None,
        actions=[],
    )


def _fallback_decision(verdict: PolicyVerdict, current: Portfolio) -> PortfolioDecision:
    """Детерміноване рішення без участі LLM."""
    if verdict.decision == "REBALANCE" and verdict.proposal is not None:
        return PortfolioDecision(
            decision="REBALANCE",
            reasoning=(
                f"Кандидат покращує risk-adjusted профіль на {verdict.net_improvement:+.4f} "
                f"після врахування turnover {verdict.turnover * 100:.1f}%, що перевищує поріг "
                f"{verdict.threshold:.4f}. Quality score зростає з "
                f"{verdict.current_metrics.quality_score:.4f} до "
                f"{verdict.best_candidate_metrics.quality_score:.4f}."  # type: ignore[union-attr]
            ),
            current_portfolio=list(current.positions),
            proposed_portfolio=list(verdict.proposal.proposed_portfolio.positions),
            actions=list(verdict.proposal.actions),
        )
    return PortfolioDecision(
        decision="HOLD",
        reasoning=(
            f"No candidate portfolio provides a sufficiently better risk-adjusted profile "
            f"to justify turnover. Найкраще чисте покращення {verdict.net_improvement:+.4f} "
            f"не досягає порогу {verdict.threshold:.4f}. Поточний quality score "
            f"{verdict.current_metrics.quality_score:.4f}, volatility "
            f"{verdict.current_metrics.volatility:.2f}, max drawdown "
            f"{verdict.current_metrics.max_drawdown:.2f}."
        ),
        current_portfolio=list(current.positions),
        proposed_portfolio=None,
        actions=[],
    )


def _validate_proposal_node(state: GuardedState) -> dict[str, Any]:
    """Перевірити пропозицію перед HITL (у т.ч. після Modify)."""
    with trajectory.TrajectoryRecorder("validate_proposal"):
        modified = state.get("modified_positions")
        if not modified:
            trajectory.record_event(node="validate_proposal", event="proposal_ready")
            return {"approval": ""}

        current = Portfolio(positions=state["current_portfolio"])  # type: ignore[arg-type]
        try:
            proposed = Portfolio(
                positions=tuple(
                    Position(symbol=item["symbol"], weight=float(item["weight"]))
                    for item in modified
                )
            )
            actions = build_actions(current, proposed)
            if not actions:
                raise ValueError("Змінений план не відрізняється від поточного портфеля")

            from app.data.analytics import compute_turnover  # локальний імпорт: уникаємо циклу

            proposal = RebalanceProposal(
                current_portfolio=current,
                proposed_portfolio=proposed,
                actions=actions,
                turnover=compute_turnover(current, proposed),
                improvement_score=0.0,
                rationale="План змінено користувачем (HITL Modify).",
            )
        except (ValueError, DomainError) as error:
            trajectory.record_event(
                node="validate_proposal",
                event="invalid_modification",
                status="error",
                error_code="INVALID_PORTFOLIO",
            )
            return {
                "approval": "",
                "modified_positions": None,
                "errors": [
                    {"tool_name": "validate_proposal", "error_code": "INVALID_PORTFOLIO"}
                ],
                "messages": [
                    AIMessage(content=f"Змінений план відхилено як невалідний: {error}")
                ],
            }

        trajectory.record_event(node="validate_proposal", event="modification_accepted")
        return {
            "rebalance_proposal": proposal.model_dump(mode="json"),
            "modified_positions": None,
            "approval": "",
            "messages": [AIMessage(content=f"Оновлений план:\n{proposal.render()}")],
        }


def _execute_rebalance_node(state: GuardedState) -> dict[str, Any]:
    """Виконати mock-ребаланс. Вузол досяжний лише через HITL-переривання."""
    with trajectory.TrajectoryRecorder("execute_rebalance"):
        approval = state.get("approval", "")
        proposal_data = state.get("rebalance_proposal")

        if approval == "reject":
            trajectory.record_event(
                node="execute_rebalance", event="rejected", decision="no_changes"
            )
            return {
                "execution_result": {
                    "status": "rejected",
                    "message": "Користувач відхилив пропозицію. Портфель без змін.",
                    "portfolio": state.get("current_portfolio", {}),
                },
                "messages": [AIMessage(content="Ребаланс відхилено. Портфель залишено без змін.")],
            }

        if approval == "modify":
            trajectory.record_event(node="execute_rebalance", event="modification_requested")
            return {"messages": [AIMessage(content="Отримано запит на зміну плану.")]}

        if approval != "approve":
            # Захист від виконання без явного підтвердження
            trajectory.record_event(
                node="execute_rebalance",
                event="blocked",
                status="error",
                error_code=ApprovalRequiredError.code,
            )
            return {
                "execution_result": {
                    "status": "blocked",
                    "message": "Виконання заблоковано: немає підтвердження людини.",
                },
                "errors": [
                    {"tool_name": "execute_rebalance", "error_code": ApprovalRequiredError.code}
                ],
            }

        if not proposal_data:
            return {
                "execution_result": {"status": "error", "message": "Пропозиція відсутня"},
                "errors": [{"tool_name": "execute_rebalance", "error_code": ExecutionError.code}],
            }

        proposal = RebalanceProposal.model_validate(proposal_data)
        response = mock_execute_rebalance.invoke(
            {
                "operations": [
                    {
                        "action": action.action,
                        "symbol": action.symbol,
                        "to_weight": action.to_weight,
                    }
                    for action in proposal.actions
                ],
                "target_positions": [
                    {"symbol": position.symbol, "weight": position.weight}
                    for position in proposal.proposed_portfolio.positions
                ],
                "approval_token": "approve",
            }
        )

        if response["status"] == "error":
            return {
                "execution_result": {"status": "error", "message": response["error"]["message"]},
                "errors": [
                    {
                        "tool_name": "mock_execute_rebalance",
                        "error_code": response["error"]["code"],
                    }
                ],
                "messages": [AIMessage(content=f"Виконання не вдалося: {response['error']}")],
            }

        data = response["data"]
        return {
            "execution_result": {"status": "executed", **data},
            "current_portfolio": data["portfolio_after"],
            "tool_history": [{"tool": "mock_execute_rebalance", "status": "success"}],
            "messages": [
                AIMessage(
                    content=(
                        f"Виконано {data['operations_count']} mock-операцій. "
                        f"Новий портфель: {data['portfolio_after']}"
                    )
                )
            ],
        }


# --- Маршрутизація ---


def _route_after_replan(state: GuardedState) -> str:
    """Чи є ще невиконані кроки плану."""
    plan = Plan.model_validate(state["plan"])
    done = {item["step_id"] for item in state.get("completed_steps", [])}
    if any(step.step_id not in done for step in plan.steps):
        return "executor"
    return "decide"


def _route_after_decide(state: GuardedState) -> str:
    """HOLD завершує workflow; REBALANCE веде до HITL."""
    if state.get("decision") == "REBALANCE" and state.get("rebalance_proposal"):
        return "validate_proposal"
    return END


def _route_after_execution(state: GuardedState) -> str:
    """Modify повертає план на повторну валідацію та повторний interrupt."""
    if state.get("approval") == "modify":
        return "validate_proposal"
    return END


# --- Побудова графа ---


def build_workflow(
    checkpointer: Any = None,
) -> Any:
    """Скомпілювати Plan-and-Execute граф із HITL-перериванням.

    `interrupt_before=["execute_rebalance"]` зупиняє workflow перед будь-якою
    mock buy/sell операцією і чекає на рішення людини.
    """
    graph = StateGraph(GuardedState)

    graph.add_node("load_portfolio", _load_portfolio_node)
    graph.add_node("planner", _planner_node)
    graph.add_node("executor", _executor_node)
    graph.add_node("replanner", _replanner_node)
    graph.add_node("decide", _decide_node)
    graph.add_node("validate_proposal", _validate_proposal_node)
    graph.add_node("execute_rebalance", _execute_rebalance_node)

    graph.add_edge(START, "load_portfolio")
    graph.add_edge("load_portfolio", "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")
    graph.add_conditional_edges(
        "replanner", _route_after_replan, {"executor": "executor", "decide": "decide"}
    )
    graph.add_conditional_edges(
        "decide", _route_after_decide, {"validate_proposal": "validate_proposal", END: END}
    )
    graph.add_edge("validate_proposal", "execute_rebalance")
    graph.add_conditional_edges(
        "execute_rebalance",
        _route_after_execution,
        {"validate_proposal": "validate_proposal", END: END},
    )

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute_rebalance"],
    )


def open_checkpointer(path: Path | None = None) -> Any:
    """Контекстний менеджер SqliteSaver, який зберігає стан між викликами."""
    target = path or get_settings().sqlite_checkpoint_path
    target.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver.from_conn_string(str(target))
