import pandas as pd

from olist_delivery.analysis import (
    dissatisfaction_threshold_sensitivity,
    monthly_delivery_drift,
    seller_count_summary,
)
from olist_delivery.features import OUTCOME_ORDER


def test_dissatisfaction_sensitivity_reports_all_thresholds_and_outcomes() -> None:
    rows = []
    for outcome in OUTCOME_ORDER:
        for score in range(1, 6):
            rows.append(
                {
                    "deadline_outcome": outcome,
                    "review_score": score,
                }
            )
    orders = pd.DataFrame(rows)

    sensitivity = dissatisfaction_threshold_sensitivity(orders)

    assert len(sensitivity) == 12
    threshold_three = sensitivity.loc[
        sensitivity["maximum_review_score"] == 3
    ]
    assert (threshold_three["rate"] == 0.6).all()
    assert (threshold_three["vs_baseline"] == 1.0).all()


def test_monthly_delivery_drift_keeps_target_and_promise_together() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-01-01", "2018-01-15", "2018-02-01"]
            ),
            "late_delivery": [0, 1, 0],
            "promised_delivery_window_days": [10, 20, 30],
        }
    )

    drift = monthly_delivery_drift(orders)

    assert drift["purchase_month"].tolist() == ["2018-01", "2018-02"]
    assert drift["orders"].tolist() == [2, 1]
    assert drift["late_delivery_rate"].tolist() == [0.5, 0.0]
    assert drift["median_promised_window_days"].tolist() == [15.0, 30.0]


def test_seller_summary_diagnoses_promises_and_pre_delivery_reviews() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "num_unique_sellers": [1, 2, 2],
            "dissatisfied": [0, 1, 1],
            "late_delivery": [0, 0, 1],
            "promised_delivery_window_days": [10, 20, 30],
            "review_creation_date": pd.to_datetime(
                ["2018-01-11", "2018-01-19", "2018-01-31"]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2018-01-10", "2018-01-20", "2018-01-30"]
            ),
        }
    )

    summary = seller_count_summary(orders)
    two_sellers = summary.loc[summary["num_unique_sellers"] == 2].iloc[0]

    assert two_sellers["median_promised_window_days"] == 25
    assert two_sellers["review_before_delivery_rate"] == 0.5
