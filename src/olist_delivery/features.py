"""Feature engineering for the Olist delivery analysis.

This module converts a validated, one-row-per-order table into analytical
variables. It does not load files, filter the study population, fit models, or
create plots.
"""

import numpy as np
import pandas as pd


DISSATISFIED_MAX_SCORE = 2

DEADLINE_OUTCOMES = {
    (True, True): "both_deadlines_met",
    (False, True): "late_handoff_recovered",
    (True, False): "on_time_handoff_late_delivery",
    (False, False): "both_deadlines_missed",
}

OUTCOME_ORDER = [
    "both_deadlines_met",
    "late_handoff_recovered",
    "on_time_handoff_late_delivery",
    "both_deadlines_missed",
]

LATE_DELIVERY_OUTCOMES = {
    "on_time_handoff_late_delivery",
    "both_deadlines_missed",
}


def add_dissatisfaction_target(orders: pd.DataFrame) -> pd.DataFrame:
    """Flag 1- and 2-star reviews as customer dissatisfaction."""
    featured = orders.copy()
    featured["dissatisfied"] = (
        featured["review_score"] <= DISSATISFIED_MAX_SCORE
    ).astype(int)
    return featured


def add_deadline_outcome(orders: pd.DataFrame) -> pd.DataFrame:
    """Classify seller handoff and customer delivery deadline performance.

    ``shipping_limit_date`` contains a full timestamp and is compared directly.
    ``order_estimated_delivery_date`` contains only a calendar date, so the
    actual delivery timestamp is normalized before comparison.
    """
    featured = orders.copy()

    handoff_on_time = (
        featured["order_delivered_carrier_date"]
        <= featured["shipping_limit_date"]
    )
    delivery_on_time = (
        featured["order_delivered_customer_date"].dt.normalize()
        <= featured["order_estimated_delivery_date"]
    )

    outcome = pd.Series(index=featured.index, dtype="object")
    for (handoff, delivery), label in DEADLINE_OUTCOMES.items():
        selected = (handoff_on_time == handoff) & (delivery_on_time == delivery)
        outcome.loc[selected] = label

    featured["deadline_outcome"] = pd.Categorical(
        outcome,
        categories=OUTCOME_ORDER,
        ordered=True,
    )
    return featured


def add_order_controls(orders: pd.DataFrame) -> pd.DataFrame:
    """Add order-value, freight, and promised-window variables."""
    featured = orders.copy()
    featured["log_order_price"] = np.log1p(featured["total_price"])
    featured["freight_ratio"] = (
        featured["total_freight_value"]
        / featured["total_price"].clip(lower=0.01)
    )
    featured["promised_delivery_window_days"] = (
        featured["order_estimated_delivery_date"]
        - featured["order_purchase_timestamp"].dt.normalize()
    ).dt.days
    return featured


def add_purchase_timing(orders: pd.DataFrame) -> pd.DataFrame:
    """Extract features known from the purchase timestamp at checkout."""
    featured = orders.copy()
    purchase = featured["order_purchase_timestamp"]

    featured["purchase_hour"] = purchase.dt.hour
    featured["purchase_day_of_week"] = purchase.dt.dayofweek
    featured["purchase_is_weekend"] = (purchase.dt.dayofweek >= 5).astype(int)
    featured["purchase_month"] = purchase.dt.month
    featured["purchase_year"] = purchase.dt.year

    featured["purchase_hour_sin"] = np.sin(
        2 * np.pi * featured["purchase_hour"] / 24
    )
    featured["purchase_hour_cos"] = np.cos(
        2 * np.pi * featured["purchase_hour"] / 24
    )
    featured["purchase_day_sin"] = np.sin(
        2 * np.pi * featured["purchase_day_of_week"] / 7
    )
    featured["purchase_day_cos"] = np.cos(
        2 * np.pi * featured["purchase_day_of_week"] / 7
    )
    featured["purchase_month_sin"] = np.sin(
        2 * np.pi * (featured["purchase_month"] - 1) / 12
    )
    featured["purchase_month_cos"] = np.cos(
        2 * np.pi * (featured["purchase_month"] - 1) / 12
    )
    return featured


def add_late_delivery_target(orders: pd.DataFrame) -> pd.DataFrame:
    """Flag orders that missed the customer-facing delivery promise."""
    featured = orders.copy()
    featured["late_delivery"] = featured["deadline_outcome"].isin(
        LATE_DELIVERY_OUTCOMES
    ).astype(int)
    return featured


def build_delivery_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Build features and the target needed for late-delivery modelling.

    This deliberately does not require review fields. Delivery-risk modelling
    should represent every eligible delivered order rather than only customers
    who submitted an unambiguous review.
    """
    featured = add_deadline_outcome(orders)
    featured = add_order_controls(featured)
    featured = add_purchase_timing(featured)
    featured = add_late_delivery_target(featured)
    return featured


def build_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Build the review-based analytical feature table."""
    featured = add_dissatisfaction_target(orders)
    return build_delivery_features(featured)
