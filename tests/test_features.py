import numpy as np
import pandas as pd

from olist_delivery.features import build_features


def test_build_features_creates_targets_controls_and_purchase_timing() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["order_1", "order_2"],
            "review_score": [2, 5],
            "total_price": [99.0, 199.0],
            "total_freight_value": [10.0, 20.0],
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-01-06 14:30", "2018-02-05 09:00"]
            ),
            "shipping_limit_date": pd.to_datetime(
                ["2018-01-08", "2018-02-07"]
            ),
            "order_delivered_carrier_date": pd.to_datetime(
                ["2018-01-07", "2018-02-06"]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2018-01-11", "2018-02-08"]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2018-01-10", "2018-02-10"]
            ),
        }
    )

    featured = build_features(orders)

    assert featured["dissatisfied"].tolist() == [1, 0]
    assert featured["deadline_outcome"].astype(str).tolist() == [
        "on_time_handoff_late_delivery",
        "both_deadlines_met",
    ]
    assert featured["late_delivery"].tolist() == [1, 0]
    assert featured["purchase_is_weekend"].tolist() == [1, 0]
    assert featured["purchase_hour"].tolist() == [14, 9]
    assert featured["promised_delivery_window_days"].tolist() == [4, 5]
    np.testing.assert_allclose(featured["freight_ratio"], [10 / 99, 20 / 199])


def test_build_features_does_not_mutate_the_input() -> None:
    orders = pd.DataFrame(
        {
            "review_score": [5],
            "total_price": [10.0],
            "total_freight_value": [1.0],
            "order_purchase_timestamp": pd.to_datetime(["2018-01-01"]),
            "shipping_limit_date": pd.to_datetime(["2018-01-02"]),
            "order_delivered_carrier_date": pd.to_datetime(["2018-01-02"]),
            "order_delivered_customer_date": pd.to_datetime(["2018-01-03"]),
            "order_estimated_delivery_date": pd.to_datetime(["2018-01-03"]),
        }
    )
    original = orders.copy(deep=True)

    build_features(orders)

    pd.testing.assert_frame_equal(orders, original)
