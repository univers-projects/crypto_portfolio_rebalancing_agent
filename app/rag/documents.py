"""База знань з портфельного ризик-менеджменту (11 документів).

Документи навмисно тримаються у коді, а не у зовнішньому сервісі: це робить
індексацію відтворюваною і незалежною від мережі.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeDocument:
    """Один документ бази знань із метаданими для фільтрації та цитування."""

    doc_id: str
    title: str
    topic: str
    content: str


KNOWLEDGE_DOCUMENTS: tuple[KnowledgeDocument, ...] = (
    KnowledgeDocument(
        doc_id="kb-001",
        title="Risk Management Fundamentals",
        topic="risk_management",
        content=(
            "Risk management in portfolio construction is the practice of deciding how much "
            "loss a portfolio is allowed to experience before the strategy is considered "
            "broken. The core tools are position limits, diversification, and explicit risk "
            "budgets. A risk budget assigns each position a share of total portfolio risk "
            "rather than a share of capital, which matters because a 10% allocation to a "
            "very volatile asset can contribute far more risk than a 10% allocation to a "
            "stable one. Risk-adjusted return metrics such as the Sharpe ratio divide excess "
            "return by volatility, allowing portfolios with different risk levels to be "
            "compared fairly. A portfolio that earns more only by taking proportionally more "
            "risk has not actually improved. In crypto, where annualized volatility of 60 to "
            "90 percent is common, risk management dominates return forecasting: controlling "
            "drawdown and concentration usually adds more long-run value than trying to "
            "predict which asset outperforms next."
        ),
    ),
    KnowledgeDocument(
        doc_id="kb-002",
        title="Portfolio Diversification",
        topic="diversification",
        content=(
            "Diversification reduces portfolio risk by combining assets whose returns do not "
            "move together. The benefit comes from correlation, not from the number of "
            "holdings: five assets that all move with Bitcoin provide far less diversification "
            "than three genuinely independent assets. A useful diversification score combines "
            "two components. The first is weight evenness, often measured with the "
            "Herfindahl-Hirschman Index computed over portfolio weights; a lower HHI means "
            "capital is spread more evenly. The second is the average pairwise correlation of "
            "asset returns; lower average correlation means each additional holding genuinely "
            "reduces variance. Beyond roughly five to eight holdings the marginal "
            "diversification benefit in crypto becomes very small, because most large-cap "
            "tokens share a dominant market factor. This is why a cap of five assets is not a "
            "meaningful constraint on diversification quality, while it does keep turnover, "
            "monitoring cost, and operational complexity low."
        ),
    ),
    KnowledgeDocument(
        doc_id="kb-003",
        title="Volatility and Its Measurement",
        topic="volatility",
        content=(
            "Volatility measures the dispersion of returns around their mean and is the most "
            "widely used proxy for risk. It is normally computed as the standard deviation of "
            "daily returns, then annualized by multiplying by the square root of the number of "
            "periods per year. Crypto markets trade continuously, so 365 is used rather than "
            "the 252 trading days used for equities. Volatility is not constant: it clusters, "
            "meaning calm periods follow calm periods and turbulent periods follow turbulent "
            "ones, so a volatility estimate from a 180-day window is a description of the "
            "recent past rather than a forecast. Volatility is symmetric and therefore "
            "penalizes upside moves as well as downside ones, which is why it should be read "
            "alongside max drawdown. In portfolio construction, inverse-volatility weighting "
            "assigns smaller weights to more volatile assets so that each position contributes "
            "a comparable amount of risk."
        ),
    ),
    KnowledgeDocument(
        doc_id="kb-004",
        title="Maximum Drawdown",
        topic="max_drawdown",
        content=(
            "Maximum drawdown is the largest peak-to-trough decline in the value of a "
            "portfolio or asset over a given period, expressed as a percentage of the peak. It "
            "is computed by tracking the running maximum of the equity curve and recording the "
            "largest relative fall below that running maximum. Unlike volatility, max drawdown "
            "captures only downside outcomes and reflects the path of returns rather than "
            "their dispersion, which makes it a good proxy for the pain an investor actually "
            "experiences. Recovery from a drawdown is asymmetric: a 50 percent decline "
            "requires a 100 percent gain to return to the previous peak, and an 80 percent "
            "decline requires a 400 percent gain. Crypto assets routinely post drawdowns of 70 "
            "to 90 percent in bear markets. A portfolio-level max drawdown that is materially "
            "smaller than the drawdown of its individual components is evidence that "
            "diversification is working."
        ),
    ),
    KnowledgeDocument(
        doc_id="kb-005",
        title="Liquidity and the Investable Universe",
        topic="liquidity",
        content=(
            "Liquidity is the ability to enter or exit a position without materially moving "
            "the price. It is commonly proxied by average daily traded volume, order book "
            "depth, and bid-ask spread. Liquidity determines which assets belong in the "
            "investable universe at all: an asset that cannot absorb the intended position "
            "size without significant slippage is not investable regardless of how attractive "
            "its metrics look. Restricting the universe to the top 20 to 30 assets by volume "
            "is a practical filter that also screens out most manipulation-prone and "
            "thinly-traded tokens. Liquidity is regime-dependent and tends to evaporate "
            "precisely during market stress, when exit is most needed, so historical average "
            "volume overstates liquidity available in a crisis. Assets with very short price "
            "history should also be excluded, because risk metrics computed from a few weeks "
            "of data are statistically meaningless."
        ),
    ),
    KnowledgeDocument(
        doc_id="kb-006",
        title="Crypto Market Risk",
        topic="crypto_market_risk",
        content=(
            "Crypto markets carry risks that are absent or much smaller in traditional asset "
            "classes. Market risk is dominated by a single factor: most large-cap tokens are "
            "strongly correlated with Bitcoin, and that correlation rises toward one during "
            "sharp selloffs, exactly when diversification is most needed. Beyond price risk "
            "there is protocol and smart contract risk, exchange and custody risk, regulatory "
            "risk that can change an asset's legal status overnight, and concentration risk "
            "from token supply held by a small number of addresses. Markets trade 24 hours a "
            "day with no circuit breakers, so gaps and liquidation cascades can develop "
            "without pause. The practical implication for portfolio construction is that "
            "diversification within crypto reduces idiosyncratic risk but cannot remove the "
            "common market factor, and that position sizing must assume correlations will "
            "spike in stress."
        ),
    ),
    KnowledgeDocument(
        doc_id="kb-007",
        title="Portfolio Rebalancing Principles",
        topic="rebalancing",
        content=(
            "Rebalancing restores a portfolio to its target weights after price movements have "
            "caused them to drift. The two standard approaches are calendar rebalancing, which "
            "acts at fixed intervals, and threshold rebalancing, which acts only when a weight "
            "deviates from its target by more than a set tolerance. Threshold rebalancing "
            "generally produces fewer and more meaningful trades. Rebalancing is inherently "
            "contrarian: it sells assets that have appreciated and buys those that have "
            "lagged, which controls risk but can underperform during strong sustained trends. "
            "The decision to rebalance should always be a comparison, not a reflex: the "
            "expected improvement in the risk-adjusted profile must exceed the cost of "
            "trading. Because estimates of return and volatility are noisy, small differences "
            "between a current and a candidate portfolio are usually statistical noise rather "
            "than genuine improvement, and acting on them systematically destroys value. "
            "Holding the existing portfolio is a legitimate active decision."
        ),
    ),
    KnowledgeDocument(
        doc_id="kb-008",
        title="Turnover and Transaction Costs",
        topic="turnover",
        content=(
            "Turnover measures how much of a portfolio is traded, typically as the sum of the "
            "absolute changes in position weights. A turnover of 0.40 means 40 percent of the "
            "portfolio changed hands. Every unit of turnover incurs costs: exchange fees, "
            "bid-ask spread, slippage against the order book, and in many jurisdictions a "
            "taxable event. These costs are certain and immediate, while the expected benefit "
            "of the trade is uncertain and lies in the future. This asymmetry is the core "
            "argument for a minimum improvement threshold: a candidate portfolio should "
            "replace the current one only when its estimated improvement in risk-adjusted "
            "quality exceeds the modeled turnover cost by a clear margin. Implementing this as "
            "an explicit penalty term, where net improvement equals raw improvement minus a "
            "cost coefficient times turnover, prevents overtrading and makes the tradeoff "
            "auditable. Without such a rule an optimizer will rebalance on every run, because "
            "noise almost always makes some alternative look marginally better."
        ),
    ),
    KnowledgeDocument(
        doc_id="kb-009",
        title="Position Sizing",
        topic="position_sizing",
        content=(
            "Position sizing determines how much capital each holding receives and usually "
            "matters more than asset selection. Equal weighting is a robust baseline that "
            "requires no estimation and avoids concentration, but it ignores the fact that "
            "assets differ greatly in volatility. Inverse-volatility weighting scales each "
            "position by the reciprocal of its volatility so that each contributes a similar "
            "share of portfolio risk, and it is far more stable out of sample than "
            "mean-variance optimization, which is notoriously sensitive to noisy return "
            "estimates. Practical implementations add explicit caps and floors: a maximum "
            "weight limits single-asset blowup risk, and a minimum weight prevents positions "
            "so small they add complexity without affecting outcomes. Positions below roughly "
            "five percent rarely change portfolio behavior and are usually better dropped "
            "entirely than kept as residual holdings."
        ),
    ),
    KnowledgeDocument(
        doc_id="kb-010",
        title="Stablecoin Risk",
        topic="stablecoin_risk",
        content=(
            "Stablecoins are tokens designed to hold a fixed value, usually one US dollar, and "
            "are backed by fiat reserves, by overcollateralized crypto, or by algorithmic "
            "mechanisms. They are excluded from a growth-oriented investable universe for two "
            "reasons. First, they have essentially no expected return and near-zero "
            "volatility, so including them in a risk-metric ranking distorts every comparison: "
            "any Sharpe-like ratio computed against a near-zero denominator becomes "
            "meaningless. Second, they are not risk-free. Fiat-backed stablecoins carry "
            "issuer, reserve quality, and counterparty risk; crypto-backed ones can be "
            "liquidated in sharp downturns; algorithmic designs have repeatedly failed "
            "outright, with the collapse of UST in 2022 the most prominent example. A "
            "stablecoin holding is best treated as a cash-management decision made separately "
            "from asset allocation, not as one of the portfolio's risk assets."
        ),
    ),
    KnowledgeDocument(
        doc_id="kb-011",
        title="Asset Correlation",
        topic="correlation",
        content=(
            "Correlation measures the degree to which two assets' returns move together, "
            "ranging from -1 for perfect opposition through 0 for independence to +1 for "
            "perfect co-movement. It is the input that determines how much variance reduction "
            "diversification actually delivers: adding an asset with correlation near 1 to an "
            "existing holding barely reduces portfolio volatility, while adding one with low "
            "or negative correlation reduces it substantially even if the new asset is "
            "individually volatile. Correlation is unstable over time and in crypto is "
            "regime-dependent, typically sitting between 0.6 and 0.9 among large-cap tokens in "
            "calm conditions and converging toward 1 during liquidations. Correlation "
            "estimates from short windows are unreliable, so at least several months of daily "
            "data should be used. A correlation matrix should be read together with weights: "
            "low correlation cannot rescue a portfolio that is 80 percent concentrated in one "
            "position."
        ),
    ),
)


def document_count() -> int:
    """Кількість документів у базі знань (мінімальна вимога — 8)."""
    return len(KNOWLEDGE_DOCUMENTS)
