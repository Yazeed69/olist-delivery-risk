import pandas as pd

from olist_delivery.cleaning import keep_plausible_shipping_deadlines
from olist_delivery.features import OUTCOME_ORDER, add_deadline_outcome


def test_deadline_outcome_covers_all_handoff_and_delivery_combinations() -> None:
    orders = pd.DataFrame(
        {
            "order_delivered_carrier_date": pd.to_datetime(
                [
                    "2018-01-02 10:00",
                    "2018-01-03 10:00",
                    "2018-01-02 10:00",
                    "2018-01-03 10:00",
                ]
            ),
            "shipping_limit_date": pd.to_datetime(
                [
                    "2018-01-02 12:00",
                    "2018-01-02 12:00",
                    "2018-01-02 12:00",
                    "2018-01-02 12:00",
                ]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                [
                    "2018-01-05 23:59",
                    "2018-01-05 12:00",
                    "2018-01-06 00:01",
                    "2018-01-06 12:00",
                ]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2018-01-05"] * 4
            ),
        }
    )

    featured = add_deadline_outcome(orders)

    assert featured["deadline_outcome"].tolist() == OUTCOME_ORDER


def test_shipping_deadline_filter_keeps_inclusive_zero_to_sixty_days() -> None:
    purchase = pd.Timestamp("2018-01-01 12:00")
    orders = pd.DataFrame(
        {
            "order_id": ["before", "same", "sixty", "after"],
            "order_purchase_timestamp": [purchase] * 4,
            "shipping_limit_date": [
                purchase - pd.Timedelta(seconds=1),
                purchase,
                purchase + pd.Timedelta(days=60),
                purchase + pd.Timedelta(days=60, seconds=1),
            ],
        }
    )

    kept = keep_plausible_shipping_deadlines(orders)

    assert kept["order_id"].tolist() == ["same", "sixty"]
