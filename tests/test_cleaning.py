import pandas as pd

from olist_delivery.cleaning import aggregate_order_items


def test_order_item_aggregation_uses_a_deterministic_primary_category() -> None:
    order_items = pd.DataFrame(
        {
            "order_id": ["order_1", "order_1", "order_1"],
            "product_id": ["a", "b", "c"],
            "seller_id": ["seller", "seller", "seller"],
            "price": [10.0, 20.0, 30.0],
            "freight_value": [1.0, 2.0, 3.0],
            "product_weight_g": [100.0, 200.0, 300.0],
            "product_volume_cm3": [10.0, 20.0, 30.0],
            "product_category_name": ["books", "toys", "books"],
            "shipping_limit_date": pd.to_datetime(
                ["2018-01-02", "2018-01-03", "2018-01-04"]
            ),
        }
    )

    aggregated = aggregate_order_items(order_items)

    assert aggregated.loc[0, "primary_product_category"] == "books"
    assert aggregated.loc[0, "total_price"] == 60.0
