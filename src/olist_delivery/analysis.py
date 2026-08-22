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

CATEGORICAL_CONTROLS = [
    "purchase_month",
    "purchase_year",
    "primary_product_category",
    "customer_state",
]

ANALYSIS_COLUMNS = {
    "deadline_outcome",
    "dissatisfied",
    "review_score",
    "seller_id",
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
    covariance_type: str
    n_clusters: int


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
    """Fit adjusted logistic regression with seller-clustered uncertainty."""
    matrix = build_dissatisfaction_design_matrix(orders)
    target = orders["dissatisfied"].astype(int)
    seller_groups = orders["seller_id"].to_numpy()
    fitted = sm.GLM(
        target,
        matrix,
        family=sm.families.Binomial(),
    ).fit(
        maxiter=200,
        cov_type="cluster",
        cov_kwds={"groups": seller_groups},
    )

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
        pseudo_r_squared=float(fitted.pseudo_rsquared(kind="mcf")),
        log_likelihood=float(fitted.llf),
        n_observations=int(fitted.nobs),
        converged=bool(fitted.converged),
        covariance_type="seller_clustered",
        n_clusters=int(orders["seller_id"].nunique()),
    )


def dissatisfaction_threshold_sensitivity(
    orders: pd.DataFrame,
    thresholds: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Recalculate outcome rates under alternative dissatisfaction cutoffs."""
    name = "sensitivity_orders"
    require_non_empty(orders, name)
    require_columns(orders, {"deadline_outcome", "review_score"}, name)

    rows = []
    for threshold in thresholds:
        if threshold < 1 or threshold > 4:
            raise ValueError("thresholds must be integers from 1 through 4")

        flagged = orders["review_score"].le(threshold)
        grouped = (
            orders.assign(flagged=flagged)
            .groupby("deadline_outcome", observed=True)
            .agg(
                orders=("flagged", "size"),
                flagged_orders=("flagged", "sum"),
            )
            .reindex(OUTCOME_ORDER)
        )
        grouped["rate"] = grouped["flagged_orders"] / grouped["orders"]
        baseline = float(grouped.loc[OUTCOME_ORDER[0], "rate"])
        grouped["vs_baseline"] = grouped["rate"] / baseline
        grouped = grouped.reset_index()
        grouped.insert(0, "maximum_review_score", threshold)
        rows.append(grouped)

    return pd.concat(rows, ignore_index=True)


def monthly_delivery_drift(orders: pd.DataFrame) -> pd.DataFrame:
    """Summarize target prevalence and promised windows by purchase month."""
    name = "drift_orders"
    required = {
        "order_id",
        "order_purchase_timestamp",
        "late_delivery",
        "promised_delivery_window_days",
    }
    require_non_empty(orders, name)
    require_columns(orders, required, name)

    working = orders.copy()
    working["purchase_month"] = (
        working["order_purchase_timestamp"].dt.to_period("M").astype(str)
    )
    summary = (
        working.groupby("purchase_month")
        .agg(
            orders=("order_id", "size"),
            late_deliveries=("late_delivery", "sum"),
            late_delivery_rate=("late_delivery", "mean"),
            median_promised_window_days=(
                "promised_delivery_window_days",
                "median",
            ),
            promised_window_q25=(
                "promised_delivery_window_days",
                lambda values: values.quantile(0.25),
            ),
            promised_window_q75=(
                "promised_delivery_window_days",
                lambda values: values.quantile(0.75),
            ),
        )
        .reset_index()
        .sort_values("purchase_month")
    )
    return summary.reset_index(drop=True)


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
        "promised_delivery_window_days",
        "review_creation_date",
        "order_delivered_customer_date",
    }
    require_non_empty(orders, name)
    require_columns(orders, required, name)

    working = orders.copy()
    review_date = working["review_creation_date"].dt.normalize()
    delivery_date = working["order_delivered_customer_date"].dt.normalize()
    working["review_created_before_delivery"] = review_date < delivery_date
    working["review_delay_days"] = (review_date - delivery_date).dt.days

    summary = (
        working.groupby("num_unique_sellers")
        .agg(
            orders=("order_id", "size"),
            dissatisfied=("dissatisfied", "sum"),
            late_deliveries=("late_delivery", "sum"),
            median_promised_window_days=(
                "promised_delivery_window_days",
                "median",
            ),
            reviews_created_before_delivery=(
                "review_created_before_delivery",
                "sum",
            ),
            median_review_delay_days=("review_delay_days", "median"),
        )
        .reset_index()
        .sort_values("num_unique_sellers")
    )
    summary["dissatisfaction_rate"] = summary["dissatisfied"] / summary["orders"]
    summary["late_delivery_rate"] = summary["late_deliveries"] / summary["orders"]
    summary["review_before_delivery_rate"] = (
        summary["reviews_created_before_delivery"] / summary["orders"]
    )
    return summary.reset_index(drop=True)
