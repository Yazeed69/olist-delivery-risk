import numpy as np
import pandas as pd

from olist_delivery.modeling import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    POST_CHECKOUT_COLUMNS,
    temporal_train_test_split,
    top_fraction_metrics,
)


def _model_orders() -> pd.DataFrame:
    rows = 20
    data: dict[str, object] = {
        "order_id": [f"order_{index:02d}" for index in range(rows)],
        "order_purchase_timestamp": pd.date_range(
            "2018-01-01", periods=rows, freq="D"
        ),
        "late_delivery": [index % 2 for index in range(rows)],
    }
    for feature in NUMERIC_FEATURES:
        data[feature] = np.arange(rows, dtype=float)
    for feature in CATEGORICAL_FEATURES:
        data[feature] = ["group_a" if index % 2 else "group_b" for index in range(rows)]
    return pd.DataFrame(data)


def test_temporal_split_trains_only_on_earlier_orders() -> None:
    split = temporal_train_test_split(_model_orders(), test_fraction=0.20)

    assert split.train_timestamps.max() < split.test_timestamps.min()
    assert set(split.train_order_ids).isdisjoint(split.test_order_ids)
    assert len(split.y_train) == 16
    assert len(split.y_test) == 4


def test_top_fraction_metrics_use_an_exact_ranked_capacity() -> None:
    target = pd.Series([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    probabilities = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.4, 0.7, 0.5, 0.6, 0.0])

    metrics = top_fraction_metrics(target, probabilities, fraction=0.20)

    assert metrics["flagged_orders"] == 2
    assert metrics["captured_late_deliveries"] == 2
    assert metrics["capture_rate"] == 2 / 3
    assert metrics["precision"] == 1.0


def test_model_features_exclude_every_known_post_checkout_field() -> None:
    assert set(MODEL_FEATURES).isdisjoint(POST_CHECKOUT_COLUMNS)
