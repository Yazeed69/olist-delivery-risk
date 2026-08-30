import pandas as pd

from olist_delivery.pipeline import build_analysis_population


def _raw_tables(rows: int = 40, reviewed_rows: int = 30) -> dict[str, pd.DataFrame]:
    purchases = pd.date_range("2017-01-01", periods=rows, freq="14D")
    order_ids = [f"order_{index}" for index in range(rows)]
    customer_ids = [f"customer_{index}" for index in range(rows)]
    product_ids = [f"product_{index}" for index in range(rows)]

    delivered_customer = [
        purchase + pd.Timedelta(days=12 if index % 3 == 0 else 8)
        for index, purchase in enumerate(purchases)
    ]
    orders = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
            "order_status": "delivered",
            "order_purchase_timestamp": purchases,
            "order_approved_at": purchases + pd.Timedelta(hours=1),
            "order_delivered_carrier_date": purchases + pd.Timedelta(days=2),
            "order_delivered_customer_date": delivered_customer,
            "order_estimated_delivery_date": purchases.normalize()
            + pd.Timedelta(days=10),
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_unique_id": [f"person_{index}" for index in range(rows)],
            "customer_state": ["SP" if index % 2 else "RJ" for index in range(rows)],
            "customer_zip_code_prefix": [1000 + index % 2 for index in range(rows)],
        }
    )
    order_items = pd.DataFrame(
        {
            "order_id": order_ids,
            "product_id": product_ids,
            "seller_id": ["seller_a"] * rows,
            "shipping_limit_date": purchases + pd.Timedelta(days=3),
            "price": [50.0 + index for index in range(rows)],
            "freight_value": [5.0] * rows,
        }
    )
    products = pd.DataFrame(
        {
            "product_id": product_ids,
            "product_category_name": ["books" if index % 2 else "toys" for index in range(rows)],
            "product_weight_g": [500.0] * rows,
            "product_length_cm": [10.0] * rows,
            "product_height_cm": [5.0] * rows,
            "product_width_cm": [8.0] * rows,
        }
    )
    reviews = pd.DataFrame(
        {
            "review_id": [f"review_{index}" for index in range(reviewed_rows)],
            "order_id": order_ids[:reviewed_rows],
            "review_score": [1 if index % 3 == 0 else 5 for index in range(reviewed_rows)],
            "review_creation_date": purchases[:reviewed_rows] + pd.Timedelta(days=13),
            "review_answer_timestamp": purchases[:reviewed_rows] + pd.Timedelta(days=14),
        }
    )
    sellers = pd.DataFrame(
        {
            "seller_id": ["seller_a"],
            "seller_state": ["SP"],
            "seller_zip_code_prefix": [1000],
        }
    )
    geolocation = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [1000, 1001],
            "geolocation_lat": [-23.55, -22.91],
            "geolocation_lng": [-46.63, -43.17],
        }
    )
    return {
        "orders": orders,
        "customers": customers,
        "order_items": order_items,
        "products": products,
        "reviews": reviews,
        "sellers": sellers,
        "geolocation": geolocation,
    }


def test_raw_tables_produce_separate_analysis_and_modeling_cohorts() -> None:
    analysis, modeling, flow, seller_summary = build_analysis_population(_raw_tables())

    assert len(analysis) == 30
    assert len(modeling) == 40
    assert "review_score" in analysis
    assert "review_score" not in modeling
    assert modeling["order_id"].is_unique
    assert set(flow["cohort"]) == {"delivery_model", "dissatisfaction_analysis"}
    assert seller_summary["orders"].sum() == 30
