"""Детермінований шар прийняття рішення HOLD / REBALANCE.

Числовий вердикт рахується тут, а не в LLM. LLM відповідає за формулювання
обґрунтування та структурованого виводу, але не може перекрити політику
turnover control. Це робить поведінку агента відтворюваною і тестованою.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.data.analytics import (
    build_candidate_portfolios,
    compute_asset_metrics,
    compute_turnover,
    evaluate_portfolio_metrics,
    improvement_score,
)
from app.domain.errors import DomainError
from app.domain.schemas import (
    Portfolio,
    PortfolioMetrics,
    RebalanceAction,
    RebalanceProposal,
)

# Зміна ваги, менша за це значення, вважається шумом і не породжує окремої дії
WEIGHT_CHANGE_EPSILON = 0.005


@dataclass(frozen=True)
class PolicyVerdict:
    """Результат кількісної перевірки: чи виправданий ребаланс."""

    decision: str
    current_metrics: PortfolioMetrics
    best_candidate: Portfolio | None
    best_candidate_metrics: PortfolioMetrics | None
    turnover: float
    net_improvement: float
    threshold: float
    proposal: RebalanceProposal | None
    rejected_candidates: tuple[dict[str, Any], ...] = ()

    def as_evidence(self) -> dict[str, Any]:
        """Компактне подання для передачі в LLM як evidence."""
        evidence: dict[str, Any] = {
            "policy_verdict": self.decision,
            "minimum_improvement_score": self.threshold,
            "net_improvement_after_turnover": self.net_improvement,
            "turnover": self.turnover,
            "current_portfolio": self.current_metrics.model_dump(mode="json"),
            "current_weights": {},
        }
        if self.best_candidate and self.best_candidate_metrics:
            evidence["best_candidate_weights"] = self.best_candidate.as_mapping()
            evidence["best_candidate_metrics"] = self.best_candidate_metrics.model_dump(
                mode="json"
            )
        if self.proposal:
            evidence["proposed_actions"] = [
                action.model_dump(mode="json") for action in self.proposal.actions
            ]
        evidence["rejected_candidates"] = list(self.rejected_candidates)
        return evidence


def build_actions(current: Portfolio, proposed: Portfolio) -> tuple[RebalanceAction, ...]:
    """Побудувати мінімальний набір дій, що переводить current у proposed."""
    actions: list[RebalanceAction] = []
    symbols = sorted(set(current.symbols) | set(proposed.symbols))

    for symbol in symbols:
        from_weight = current.weight_of(symbol)
        to_weight = proposed.weight_of(symbol)
        delta = to_weight - from_weight

        if abs(delta) < WEIGHT_CHANGE_EPSILON:
            continue

        if from_weight == 0:
            action = "BUY"
        elif to_weight == 0:
            action = "SELL"
        elif delta > 0:
            action = "INCREASE"
        else:
            action = "REDUCE"

        actions.append(
            RebalanceAction(
                action=action,  # type: ignore[arg-type]
                symbol=symbol,
                from_weight=from_weight,
                to_weight=to_weight,
                rationale=f"Зміна ваги на {delta * 100:+.1f} в.п.",
            )
        )

    # Спочатку продажі — так план читається у порядку вивільнення капіталу
    order = {"SELL": 0, "REDUCE": 1, "INCREASE": 2, "BUY": 3, "REPLACE": 4}
    return tuple(sorted(actions, key=lambda item: order[item.action]))


def collect_metrics(
    symbols: list[str], lookback_days: int
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Порахувати метрики для списку активів, зібравши помилки окремо.

    Активи, для яких дані недоступні, не валять аналіз — вони просто
    виключаються, а їхні error codes повертаються для replanner-а.
    """
    metrics: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    for symbol in symbols:
        try:
            metrics[symbol] = compute_asset_metrics(symbol, lookback_days)
        except DomainError as error:
            errors.append({"tool_name": "calculate_asset_metrics", "error_code": error.code})
    return metrics, errors


def evaluate_decision(
    current: Portfolio,
    candidate_symbols: list[str],
    lookback_days: int,
) -> tuple[PolicyVerdict, list[dict[str, str]]]:
    """Порівняти поточний портфель із кандидатами і винести вердикт.

    REBALANCE дозволений лише якщо чисте покращення (після штрафу за turnover)
    перевищує `minimum_improvement_score` І turnover не перевищує ліміт.
    """
    settings = get_settings()

    # Метрики рахуємо і для кандидатів, і для поточних активів
    universe = list(dict.fromkeys([*candidate_symbols, *current.symbols]))
    metrics_by_symbol, errors = collect_metrics(universe, lookback_days)

    current_metrics = evaluate_portfolio_metrics(current, lookback_days)

    # Кандидатів будуємо лише з активів, для яких є валідні метрики
    eligible = {
        symbol: metric
        for symbol, metric in metrics_by_symbol.items()
        if symbol in candidate_symbols or symbol in current.symbols
    }
    candidates = build_candidate_portfolios(eligible, current=current)

    best: Portfolio | None = None
    best_metrics: PortfolioMetrics | None = None
    best_improvement = float("-inf")
    best_turnover = 0.0
    rejected: list[dict[str, Any]] = []

    for candidate in candidates:
        if candidate.as_mapping() == current.as_mapping():
            continue
        try:
            candidate_metrics = evaluate_portfolio_metrics(candidate, lookback_days)
        except DomainError as error:
            errors.append({"tool_name": "evaluate_portfolio", "error_code": error.code})
            continue

        turnover = compute_turnover(current, candidate)
        net = improvement_score(current_metrics, candidate_metrics, turnover)

        if turnover > settings.max_turnover:
            rejected.append(
                {
                    "weights": candidate.as_mapping(),
                    "reason": "TURNOVER_LIMIT_EXCEEDED",
                    "turnover": turnover,
                }
            )
            continue

        if net > best_improvement:
            best, best_metrics, best_improvement, best_turnover = (
                candidate,
                candidate_metrics,
                net,
                turnover,
            )

    # Жодного придатного кандидата -> HOLD
    if best is None or best_metrics is None:
        return (
            PolicyVerdict(
                decision="HOLD",
                current_metrics=current_metrics,
                best_candidate=None,
                best_candidate_metrics=None,
                turnover=0.0,
                net_improvement=0.0,
                threshold=settings.minimum_improvement_score,
                proposal=None,
                rejected_candidates=tuple(rejected),
            ),
            errors,
        )

    meets_threshold = best_improvement >= settings.minimum_improvement_score
    actions = build_actions(current, best) if meets_threshold else ()

    proposal: RebalanceProposal | None = None
    if meets_threshold and actions:
        proposal = RebalanceProposal(
            current_portfolio=current,
            proposed_portfolio=best,
            actions=actions,
            turnover=best_turnover,
            improvement_score=round(best_improvement, 6),
            rationale=(
                f"Чисте покращення {best_improvement:+.4f} перевищує поріг "
                f"{settings.minimum_improvement_score:.4f} при turnover "
                f"{best_turnover * 100:.1f}%."
            ),
        )

    return (
        PolicyVerdict(
            decision="REBALANCE" if proposal else "HOLD",
            current_metrics=current_metrics,
            best_candidate=best,
            best_candidate_metrics=best_metrics,
            turnover=best_turnover,
            net_improvement=round(best_improvement, 6),
            threshold=settings.minimum_improvement_score,
            proposal=proposal,
            rejected_candidates=tuple(rejected),
        ),
        errors,
    )
