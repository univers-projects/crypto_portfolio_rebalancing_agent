"""Тести аналітики та політики turnover control (HOLD / REBALANCE)."""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.decision_engine import build_actions, evaluate_decision
from app.config import get_settings
from app.data.analytics import (
    build_candidate_portfolios,
    compute_asset_metrics,
    compute_turnover,
    evaluate_portfolio_metrics,
    improvement_score,
)
from app.data.projection import project_portfolio
from app.domain.errors import InsufficientHistoryError
from app.domain.schemas import Portfolio

REBALANCE_PORTFOLIO = {"BTC": 0.40, "ETH": 0.25, "SOL": 0.20, "AVAX": 0.15}
HOLD_PORTFOLIO = {"BNB": 0.35, "SOL": 0.30, "ADA": 0.35}


# --- детермінованість ---


def test_metrics_are_reproducible() -> None:
    first = compute_asset_metrics("BTC", 180)
    second = compute_asset_metrics("BTC", 180)
    assert first.model_dump() == second.model_dump()


def test_metrics_are_within_expected_ranges() -> None:
    metrics = compute_asset_metrics("SOL", 180)
    assert 0.0 <= metrics.max_drawdown <= 1.0
    assert -1.0 <= metrics.trend_strength <= 1.0
    assert metrics.volatility > 0


def test_short_history_asset_raises() -> None:
    with pytest.raises(InsufficientHistoryError):
        compute_asset_metrics("NEWX", 180)


# --- turnover ---


def test_turnover_is_zero_for_identical_portfolios() -> None:
    portfolio = Portfolio(positions=REBALANCE_PORTFOLIO)  # type: ignore[arg-type]
    assert compute_turnover(portfolio, portfolio) == 0.0


def test_turnover_counts_both_sides_of_a_swap() -> None:
    current = Portfolio(positions={"BTC": 0.5, "ETH": 0.5})  # type: ignore[arg-type]
    proposed = Portfolio(positions={"BTC": 0.5, "SOL": 0.5})  # type: ignore[arg-type]
    # ETH 0.5 -> 0, SOL 0 -> 0.5
    assert compute_turnover(current, proposed) == pytest.approx(1.0)


def test_improvement_score_penalizes_turnover() -> None:
    current = evaluate_portfolio_metrics(
        Portfolio(positions=REBALANCE_PORTFOLIO), 180  # type: ignore[arg-type]
    )
    candidate = evaluate_portfolio_metrics(
        Portfolio(positions=HOLD_PORTFOLIO), 180  # type: ignore[arg-type]
    )
    cheap = improvement_score(current, candidate, turnover=0.1)
    expensive = improvement_score(current, candidate, turnover=0.9)
    assert cheap > expensive


# --- склад кандидатів ---


def test_candidates_never_exceed_five_assets() -> None:
    symbols = ["BTC", "ETH", "SOL", "BNB", "ADA", "LINK", "TRX", "DOGE"]
    metrics = {symbol: compute_asset_metrics(symbol, 180) for symbol in symbols}
    current = Portfolio(positions=REBALANCE_PORTFOLIO)  # type: ignore[arg-type]

    candidates = build_candidate_portfolios(metrics, current=current)

    assert candidates
    max_assets = get_settings().max_portfolio_assets
    for portfolio in candidates:
        assert 1 <= len(portfolio.positions) <= max_assets
        total = sum(position.weight for position in portfolio.positions)
        assert total == pytest.approx(1.0, abs=1e-4)


def test_candidates_vary_in_size() -> None:
    """Агент не зобов'язаний завжди пропонувати рівно 5 активів."""
    symbols = ["BTC", "ETH", "SOL", "BNB", "ADA", "LINK", "TRX"]
    metrics = {symbol: compute_asset_metrics(symbol, 180) for symbol in symbols}
    current = Portfolio(positions=REBALANCE_PORTFOLIO)  # type: ignore[arg-type]

    sizes = {len(p.positions) for p in build_candidate_portfolios(metrics, current=current)}
    assert len(sizes) > 1


def test_candidate_weights_respect_position_limits() -> None:
    settings = get_settings()
    symbols = ["BTC", "ETH", "SOL", "BNB", "ADA"]
    metrics = {symbol: compute_asset_metrics(symbol, 180) for symbol in symbols}

    for portfolio in build_candidate_portfolios(metrics):
        if len(portfolio.positions) == 1:
            continue
        for position in portfolio.positions:
            assert position.weight >= settings.min_position_weight - 1e-3
            assert position.weight <= settings.max_position_weight + 1e-3


# --- побудова дій ---


def test_build_actions_covers_all_change_types() -> None:
    current = Portfolio(positions={"BTC": 0.40, "ETH": 0.25, "SOL": 0.20, "AVAX": 0.15})  # type: ignore[arg-type]
    proposed = Portfolio(positions={"BTC": 0.45, "ETH": 0.30, "SOL": 0.15, "LINK": 0.10})  # type: ignore[arg-type]

    actions = build_actions(current, proposed)
    by_symbol = {action.symbol: action.action for action in actions}

    assert by_symbol == {
        "BTC": "INCREASE",
        "ETH": "INCREASE",
        "SOL": "REDUCE",
        "AVAX": "SELL",
        "LINK": "BUY",
    }
    # Продажі йдуть першими
    assert actions[0].action == "SELL"


def test_build_actions_ignores_negligible_changes() -> None:
    current = Portfolio(positions={"BTC": 0.50, "ETH": 0.50})  # type: ignore[arg-type]
    proposed = Portfolio(positions={"BTC": 0.501, "ETH": 0.499})  # type: ignore[arg-type]
    assert build_actions(current, proposed) == ()


# --- вердикт політики ---


def test_good_portfolio_yields_hold(candidate_symbols: list[str]) -> None:
    verdict, errors = evaluate_decision(
        Portfolio(positions=HOLD_PORTFOLIO), candidate_symbols, 180  # type: ignore[arg-type]
    )
    assert verdict.decision == "HOLD"
    assert verdict.proposal is None
    assert verdict.net_improvement < verdict.threshold
    assert errors == []


def test_weak_portfolio_yields_rebalance(candidate_symbols: list[str]) -> None:
    verdict, _ = evaluate_decision(
        Portfolio(positions=REBALANCE_PORTFOLIO), candidate_symbols, 180  # type: ignore[arg-type]
    )
    assert verdict.decision == "REBALANCE"
    assert verdict.proposal is not None
    assert verdict.net_improvement >= verdict.threshold
    assert len(verdict.proposal.proposed_portfolio.positions) <= 5


def test_high_threshold_forces_hold(
    candidate_symbols: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turnover control: підняття порогу перетворює REBALANCE на HOLD."""
    settings = get_settings()
    original = settings.minimum_improvement_score
    settings.minimum_improvement_score = 100.0
    try:
        verdict, _ = evaluate_decision(
            Portfolio(positions=REBALANCE_PORTFOLIO), candidate_symbols, 180  # type: ignore[arg-type]
        )
        assert verdict.decision == "HOLD"
        assert verdict.proposal is None
    finally:
        settings.minimum_improvement_score = original


def test_turnover_limit_rejects_expensive_candidates(candidate_symbols: list[str]) -> None:
    settings = get_settings()
    original = settings.max_turnover
    settings.max_turnover = 0.01
    try:
        verdict, _ = evaluate_decision(
            Portfolio(positions=REBALANCE_PORTFOLIO), candidate_symbols, 180  # type: ignore[arg-type]
        )
        assert verdict.decision == "HOLD"
        assert any(
            item["reason"] == "TURNOVER_LIMIT_EXCEEDED" for item in verdict.rejected_candidates
        )
    finally:
        settings.max_turnover = original


def test_unavailable_assets_are_reported_as_errors() -> None:
    """Активи з поганими даними не валять аналіз, а потрапляють у errors."""
    verdict, errors = evaluate_decision(
        Portfolio(positions=HOLD_PORTFOLIO),  # type: ignore[arg-type]
        ["BTC", "BADQ", "NEWX"],
        180,
    )
    codes = {item["error_code"] for item in errors}
    assert codes == {"INSUFFICIENT_HISTORY"}
    assert verdict.decision in {"HOLD", "REBALANCE"}


def test_evidence_payload_is_json_serializable(candidate_symbols: list[str]) -> None:
    import json

    verdict, _ = evaluate_decision(
        Portfolio(positions=REBALANCE_PORTFOLIO), candidate_symbols, 180  # type: ignore[arg-type]
    )
    evidence: dict[str, Any] = verdict.as_evidence()
    assert json.loads(json.dumps(evidence, default=str))
    assert evidence["policy_verdict"] == verdict.decision
    assert "minimum_improvement_score" in evidence


# --- узгодженість вікна рішення з вікном оцінки прогнозу ---


def test_decision_window_matches_estimation_window() -> None:
    """Рушій рішення і прогноз мають дивитись на ті самі дані.

    Розбіжність вікон дозволяла купити актив, який прогноз вважає збитковим.
    """
    settings = get_settings()
    assert settings.decision_lookback_days == settings.estimation_lookback_days


def test_plan_does_not_worsen_forward_outlook(candidate_symbols: list[str]) -> None:
    """План не має погіршувати прогноз, на підставі якого його ж пояснюють.

    Регресія на розбіжність вікон: на 180 днях рушій обирав ADA (+167% за
    180 днів, -50% за 720) і видавав план із гіршим прогнозом, ніж поточний
    портфель.
    """
    settings = get_settings()
    verdict, _ = evaluate_decision(
        Portfolio(positions=REBALANCE_PORTFOLIO),  # type: ignore[arg-type]
        candidate_symbols,
        settings.decision_lookback_days,
    )
    assert verdict.decision == "REBALANCE"
    assert verdict.best_candidate is not None

    current = project_portfolio(REBALANCE_PORTFOLIO, 10_000.0)
    planned = project_portfolio(verdict.best_candidate.as_mapping(), 10_000.0)

    assert planned.median_value > current.median_value
    assert planned.loss_probability < current.loss_probability
