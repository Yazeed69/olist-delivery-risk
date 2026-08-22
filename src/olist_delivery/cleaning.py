"""Cleaning transformations for the Olist delivery analysis.

Functions in this module accept DataFrames and return new DataFrames. File
access belongs in :mod:`olist_delivery.data`; analytical features and models
belong in their own modules.
"""

import numpy as np
import pandas as pd


PRODUCT_DIMENSION_COLUMNS = [
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]

ORDER_DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

REVIEW_DATE_COLUMNS = ["review_creation_date", "review_answer_timestamp"]

REVIEW_OUTPUT_COLUMNS = [
    "order_id",
    "review_id",
    "review_score",
    *REVIEW_DATE_COLUMNS,
]

UNKNOWN_CATEGORY = "unknown"
MAX_SHIPPING_LIMIT_DAYS = 60
EARTH_RADIUS_KM = 6_371.0


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    """Return products with usable physical measurements.

    Missing categories are retained as ``unknown`` because category is not
    required to calculate weight or volume.
    """
    cleaned = products.copy()
    cleaned = cleaned.dropna(subset=PRODUCT_DIMENSION_COLUMNS)

    measurable = (cleaned[PRODUCT_DIMENSION_COLUMNS] > 0).all(axis=1)
    cleaned = cleaned.loc[measurable].copy()

    cleaned["product_category_name"] = cleaned["product_category_name"].fillna(
        UNKNOWN_CATEGORY
    )
    cleaned["product_volume_cm3"] = (
        cleaned["product_length_cm"]
        * cleaned["product_height_cm"]
        * cleaned["product_width_cm"]
    )

    return cleaned[
        [
            "product_id",
            "product_category_name",
            "product_weight_g",
            "product_volume_cm3",
        ]
    ].copy()


def parse_review_dates(reviews: pd.DataFrame) -> pd.DataFrame:
    """Parse review timestamps, coercing malformed values to missing."""
    parsed = reviews.copy()
    for column in REVIEW_DATE_COLUMNS:
        parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
    return parsed


def drop_shared_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Remove survey responses linked to more than one order.

    A shared review describes a basket rather than an individual order. This
    must happen before any order-level filtering so a shared response cannot
    appear unique after one of its sibling orders disappears.
    """
    orders_per_review = reviews.groupby("review_id")["order_id"].transform("nunique")
    return reviews.loc[orders_per_review == 1].copy()


def drop_conflicting_review_orders(reviews: pd.DataFrame) -> pd.DataFrame:
    """Remove orders that received more than one distinct review score."""
    scores_per_order = reviews.groupby("order_id")["review_score"].transform("nunique")
    return reviews.loc[scores_per_order == 1].copy()


def keep_latest_review(reviews: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest response when duplicate reviews agree on score."""
    ordered = reviews.sort_values("review_answer_timestamp")
    return ordered.drop_duplicates(subset="order_id", keep="last").copy()


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Return one unambiguous review per order."""
    cleaned = parse_review_dates(reviews)
    cleaned = drop_shared_reviews(cleaned)
    cleaned = drop_conflicting_review_orders(cleaned)
    cleaned = keep_latest_review(cleaned)
    return cleaned[REVIEW_OUTPUT_COLUMNS].sort_values(
        "review_answer_timestamp"
    ).reset_index(drop=True)


def keep_delivered_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Keep orders recorded with delivered status."""
    return orders.loc[orders["order_status"] == "delivered"].copy()


def parse_order_dates(orders: pd.DataFrame) -> pd.DataFrame:
    """Parse required order timestamps and remove incomplete records."""
    parsed = orders.copy()
    for column in ORDER_DATE_COLUMNS:
        parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
    return parsed.dropna(subset=ORDER_DATE_COLUMNS).copy()


def keep_coherent_order_timelines(orders: pd.DataFrame) -> pd.DataFrame:
    """Keep orders with a coherent purchase-to-delivery timeline.

    Payment approval is checked against purchase time only. In the source data
    it can be recorded after carrier pickup, so enforcing approval before pickup
    would discard otherwise coherent deliveries.
    """
    coherent = (
        (orders["order_approved_at"] >= orders["order_purchase_timestamp"])
        & (
            orders["order_delivered_carrier_date"]
            >= orders["order_purchase_timestamp"]
        )
        & (
            orders["order_delivered_customer_date"]
            >= orders["order_delivered_carrier_date"]
        )
    )
    return orders.loc[coherent].copy()


def attach_customer_details(
    orders: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    """Attach stable customer identity and delivery-location fields."""
    customer_columns = [
        "customer_id",
        "customer_unique_id",
        "customer_state",
        "customer_zip_code_prefix",
    ]
    joined = orders.merge(
        customers[customer_columns],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    return joined.drop(columns=["customer_id", "order_status"])


def clean_orders(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Return delivered orders with valid timestamps and customer details."""
    cleaned = keep_delivered_orders(orders)
    cleaned = parse_order_dates(cleaned)
    cleaned = keep_coherent_order_timelines(cleaned)
    return attach_customer_details(cleaned, customers)


def parse_order_item_dates(order_items: pd.DataFrame) -> pd.DataFrame:
    """Parse item-level seller handoff deadlines."""
    parsed = order_items.copy()
    parsed["shipping_limit_date"] = pd.to_datetime(
        parsed["shipping_limit_date"], errors="coerce"
    )
    return parsed


def attach_product_details(
    order_items: pd.DataFrame, products: pd.DataFrame
) -> pd.DataFrame:
    """Keep order items whose products have usable physical measurements."""
    return order_items.merge(
        products,
        on="product_id",
        how="inner",
        validate="many_to_one",
    )


def aggregate_order_items(order_items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item-level records to one row per order."""
    def most_common_category(categories: pd.Series) -> str:
        """Return a deterministic modal category for an order."""
        counts = categories.value_counts()
        modes = counts.loc[counts == counts.max()].index
        return str(sorted(modes)[0])

    return (
        order_items.groupby("order_id")
        .agg(
            item_count=("product_id", "size"),
            num_unique_sellers=("seller_id", "nunique"),
            seller_id=("seller_id", "first"),
            total_price=("price", "sum"),
            total_freight_value=("freight_value", "sum"),
            total_weight_g=("product_weight_g", "sum"),
            total_volume_cm3=("product_volume_cm3", "sum"),
            primary_product_category=(
                "product_category_name",
                most_common_category,
            ),
            shipping_limit_date=("shipping_limit_date", "max"),
        )
        .reset_index()
    )


def clean_order_items(
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    eligible_order_ids: pd.Series,
) -> pd.DataFrame:
    """Return one item summary per eligible order."""
    cleaned = parse_order_item_dates(order_items)
    cleaned = attach_product_details(cleaned, products)
    cleaned = cleaned.loc[cleaned["order_id"].isin(eligible_order_ids)].copy()
    return aggregate_order_items(cleaned)


def attach_order_items(
    orders: pd.DataFrame, order_items: pd.DataFrame
) -> pd.DataFrame:
    """Join one-row-per-order item summaries to cleaned orders."""
    return order_items.merge(
        orders,
        on="order_id",
        how="inner",
        validate="one_to_one",
    )


def keep_single_seller_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Keep orders whose delivery timestamp represents one seller shipment."""
    single_seller = orders.loc[orders["num_unique_sellers"] == 1].copy()
    return single_seller.drop(columns="num_unique_sellers")


def build_zip_coordinates(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Return one mean latitude/longitude coordinate per postal-code prefix."""
    return (
        geolocation.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            geolocation_lat=("geolocation_lat", "mean"),
            geolocation_lng=("geolocation_lng", "mean"),
        )
    )


def haversine_km(
    lat1: pd.Series,
    lng1: pd.Series,
    lat2: pd.Series,
    lng2: pd.Series,
) -> pd.Series:
    """Calculate straight-line great-circle distance in kilometres."""
    lat1_rad, lng1_rad, lat2_rad, lng2_rad = map(
        np.radians, (lat1, lng1, lat2, lng2)
    )
    a = (
        np.sin((lat2_rad - lat1_rad) / 2) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin((lng2_rad - lng1_rad) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def attach_geography(
    orders: pd.DataFrame,
    sellers: pd.DataFrame,
    geolocation: pd.DataFrame,
) -> pd.DataFrame:
    """Attach seller geography and seller-to-customer distance."""
    seller_columns = ["seller_id", "seller_state", "seller_zip_code_prefix"]
    joined = orders.merge(
        sellers[seller_columns],
        on="seller_id",
        how="left",
        validate="many_to_one",
    )

    coordinates = build_zip_coordinates(geolocation).set_index(
        "geolocation_zip_code_prefix"
    )
    for side in ("customer", "seller"):
        prefix = joined[f"{side}_zip_code_prefix"]
        joined[f"{side}_lat"] = prefix.map(coordinates["geolocation_lat"])
        joined[f"{side}_lng"] = prefix.map(coordinates["geolocation_lng"])

    joined["distance_km"] = haversine_km(
        joined["seller_lat"],
        joined["seller_lng"],
        joined["customer_lat"],
        joined["customer_lng"],
    ).round(2)
    joined["same_state"] = (
        joined["customer_state"] == joined["seller_state"]
    ).astype(int)

    location_columns = [
        "customer_zip_code_prefix",
        "seller_zip_code_prefix",
        "customer_lat",
        "customer_lng",
        "seller_lat",
        "seller_lng",
    ]
    return joined.drop(columns=location_columns)


def attach_reviews(orders: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Keep orders that have one usable review."""
    return orders.merge(
        reviews,
        on="order_id",
        how="inner",
        validate="one_to_one",
    )


def keep_reviews_after_purchase(orders: pd.DataFrame) -> pd.DataFrame:
    """Remove records whose review response predates the purchase."""
    valid = orders["review_answer_timestamp"] >= orders["order_purchase_timestamp"]
    return orders.loc[valid].copy()


def keep_plausible_shipping_deadlines(
    orders: pd.DataFrame,
    max_days: int = MAX_SHIPPING_LIMIT_DAYS,
) -> pd.DataFrame:
    """Keep shipping deadlines from purchase time through ``max_days`` later."""
    gap = orders["shipping_limit_date"] - orders["order_purchase_timestamp"]
    gap_days = gap.dt.total_seconds() / 86_400
    plausible = gap_days.between(0, max_days, inclusive="both")
    return orders.loc[plausible].copy()
