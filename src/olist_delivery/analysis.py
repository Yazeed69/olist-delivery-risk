"""Statistical analysis for fulfilment outcomes and dissatisfaction.

Functions return tables and serializable metrics. Console formatting belongs in
the pipeline entry point, while charts belong in :mod:`olist_delivery.visualization`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .features import OUTCOME_ORDER
from .validation import require_columns, require_non_empty


NUMERIC_CONTROLS = [
    "item_count",
    "log_order_price",
    "freight_ratio",
    "total_weight_g",
    "total_volume_cm3",
    "promised_delivery_window_days",
]

CATEGORICAL_CONTROLS = ["purchase_month", "purchase_year"]

ANALYSIS_COLUMNS = {
    "deadline_outcome",
    "dissatisfied",
    "review_score",
    *NUMERIC_CONTROLS,
    *CATEGORICAL_CONTROLS,
}


@dataclass(frozen=True)
class AdjustedModelResult:
    """Results from the adjusted dissatisfaction logistic regression."""

    estimates: pd.DataFrame
    pseudo_r_squared: float
    log_likelihood: float
    n_observations: int
    converged: bool


def validate_analysis_input(orders: pd.DataFrame) -> None:
    """Require the columns and rows used by the dissatisfaction analysis."""
    name = "analysis_orders"
    require_non_empty(orders, name)
    require_columns(orders, ANALYSIS_COLUMNS, name)


def dissatisfaction_rates_by_outcome(
    orders: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Calculate dissatisfaction rates and Wilson confidence intervals."""
    validate_analysis_input(orders)
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    grouped = (
        orders.groupby("deadline_outcome", observed=True)
        .agg(
            orders=("dissatisfied", "size"),
            dissatisfied=("dissatisfied", "sum"),
        )
        .reindex(OUTCOME_ORDER)
        .reset_index()
    )

    grouped["rate"] = grouped["dissatisfied"] / grouped["orders"]
    grouped["share_of_orders"] = grouped["orders"] / grouped["orders"].sum()

    baseline = grouped.loc[
        grouped["deadline_outcome"] == OUTCOME_ORDER[0], "rate"
    ].iloc[0]
    grouped["vs_baseline"] = grouped["rate"] / baseline

    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)
    n = grouped["orders"]
    proportion = grouped["rate"]
    denominator = 1 + z**2 / n
    centre = (proportion + z**2 / (2 * n)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / n
            + z**2 / (4 * n**2)
        )
        / denominator
    )
    grouped["ci_low"] = centre - margin
    grouped["ci_high"] = centre + margin
    grouped["confidence_level"] = confidence_level

    return grouped


def review_score_distribution(orders: pd.DataFrame) -> pd.DataFrame:
    """Return the full 1-5 review distribution for every deadline outcome."""
    validate_analysis_input(orders)

    counts = pd.crosstab(orders["deadline_outcome"], orders["review_score"])
    counts = counts.reindex(index=OUTCOME_ORDER, columns=range(1, 6), fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)

    result = pd.DataFrame(
        {
            "deadline_outcome": OUTCOME_ORDER,
            "orders": counts.sum(axis=1).to_numpy(),
        }
    )
    for score in range(1, 6):
        result[f"score_{score}_count"] = counts[score].to_numpy()
        result[f"score_{score}_share"] = shares[score].to_numpy()

    return result


def chi_square_association(orders: pd.DataFrame) -> dict[str, float | int]:
    """Test whether deadline outcome and dissatisfaction are independent."""
    validate_analysis_input(orders)

    contingency = pd.crosstab(
        orders["deadline_outcome"],
        orders["dissatisfied"],
    ).reindex(OUTCOME_ORDER)
    chi_square, p_value, degrees_of_freedom, _ = stats.chi2_contingency(contingency)

    denominator = len(orders) * (min(contingency.shape) - 1)
    cramers_v = np.sqrt(chi_square / denominator)

    return {
        "chi_square": float(chi_square),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees_of_freedom),
        "cramers_v": float(cramers_v),
        "n_observations": int(len(orders)),
    }


def build_dissatisfaction_design_matrix(
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """Build the adjusted-regression matrix with explicit reference groups."""
    validate_analysis_input(orders)
    matrix = pd.DataFrame(index=orders.index)

    for outcome in OUTCOME_ORDER[1:]:
        matrix[outcome] = (orders["deadline_outcome"] == outcome).astype(float)

    for control in NUMERIC_CONTROLS:
        matrix[control] = orders[control].astype(float)

    for control in CATEGORICAL_CONTROLS:
        dummies = pd.get_dummies(
            orders[control],
            prefix=control,
            drop_first=True,
            dtype=float,
        )
        matrix = pd.concat([matrix, dummies], axis=1)

    return sm.add_constant(matrix, has_constant="add")


def fit_adjusted_dissatisfaction_model(
    orders: pd.DataFrame,
) -> AdjustedModelResult:
    """Fit logistic regression for dissatisfaction with order controls."""
    matrix = build_dissatisfaction_design_matrix(orders)
    target = orders["dissatisfied"].astype(int)
    fitted = sm.Logit(target, matrix).fit(disp=False)

    confidence_interval = fitted.conf_int()
    estimates = pd.DataFrame(
        {
            "term": fitted.params.index,
            "coefficient": fitted.params.to_numpy(),
            "odds_ratio": np.exp(fitted.params.to_numpy()),
            "ci_low": np.exp(confidence_interval[0].to_numpy()),
            "ci_high": np.exp(confidence_interval[1].to_numpy()),
            "p_value": fitted.pvalues.to_numpy(),
        }
    )

    return AdjustedModelResult(
        estimates=estimates,
        pseudo_r_squared=float(fitted.prsquared),
        log_likelihood=float(fitted.llf),
        n_observations=int(fitted.nobs),
        converged=bool(fitted.mle_retvals["converged"]),
    )


def seller_count_summary(orders: pd.DataFrame) -> pd.DataFrame:
    """Summarize dissatisfaction and recorded lateness by seller count.

    This supports the investigation that motivated restricting the final study
    population to single-seller orders.
    """
    name = "seller_count_orders"
    required = {
        "order_id",
        "num_unique_sellers",
        "dissatisfied",
        "late_delivery",
    }
    require_non_empty(orders, name)
    require_columns(orders, required, name)

    summary = (
        orders.groupby("num_unique_sellers")
        .agg(
            orders=("order_id", "size"),
            dissatisfied=("dissatisfied", "sum"),
            late_deliveries=("late_delivery", "sum"),
        )
        .reset_index()
        .sort_values("num_unique_sellers")
    )
    summary["dissatisfaction_rate"] = summary["dissatisfied"] / summary["orders"]
    summary["late_delivery_rate"] = summary["late_deliveries"] / summary["orders"]
    return summary.reset_index(drop=True)
