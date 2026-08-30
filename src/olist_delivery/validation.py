"""Runtime data-quality checks for the Olist delivery pipeline.

Unlike unit tests, these validations run against the real dataset whenever the
pipeline executes. They turn assumptions about schema, grain, and valid values
into explicit failures rather than allowing bad data to reach the analysis.
"""

from collections.abc import Iterable, Mapping

import pandas as pd


RAW_REQUIRED_COLUMNS = {
    "orders": {
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    },
    "customers": {
        "customer_id",
        "customer_unique_id",
        "customer_state",
        "customer_zip_code_prefix",
    },
    "order_items": {
        "order_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    },
    "products": {
        "product_id",
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    },
    "reviews": {
        "review_id",
        "order_id",
        "review_score",
        "review_creation_date",
        "review_answer_timestamp",
    },
    "sellers": {
        "seller_id",
        "seller_state",
        "seller_zip_code_prefix",
    },
    "geolocation": {
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
    },
}

RAW_UNIQUE_KEYS = {
    "orders": "order_id",
    "customers": "customer_id",
    "products": "product_id",
    "sellers": "seller_id",
}

PRODUCT_COLUMNS = {
    "product_id",
    "product_category_name",
    "product_weight_g",
    "product_volume_cm3",
}

REVIEW_COLUMNS = {
    "order_id",
    "review_id",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
}

CLEAN_ORDER_REQUIRED_COLUMNS = {
    "order_id",
    "customer_unique_id",
    "customer_state",
    "seller_id",
    "seller_state",
    "order_purchase_timestamp",
    "order_approved_at",
    "shipping_limit_date",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "review_id",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
    "item_count",
    "total_price",
    "total_freight_value",
    "total_weight_g",
    "total_volume_cm3",
    "primary_product_category",
    "distance_km",
    "same_state",
}

DELIVERY_ORDER_REQUIRED_COLUMNS = CLEAN_ORDER_REQUIRED_COLUMNS - {
    "review_id",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
}

ORDER_DATE_COLUMNS = {
    "order_purchase_timestamp",
    "order_approved_at",
    "shipping_limit_date",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "review_creation_date",
    "review_answer_timestamp",
}


class DataValidationError(ValueError):
    """Raised when a pipeline dataset violates an expected invariant."""


def require_non_empty(dataframe: pd.DataFrame, name: str) -> None:
    """Require a dataset to contain at least one row."""
    if dataframe.empty:
        raise DataValidationError(f"{name} is empty")


def require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    """Require all named columns to be present."""
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {missing}")


def require_unique_key(
    dataframe: pd.DataFrame,
    column: str,
    name: str,
) -> None:
    """Require a key column to be non-missing and unique."""
    missing = int(dataframe[column].isna().sum())
    if missing:
        raise DataValidationError(
            f"{name}.{column} contains {missing:,} missing values"
        )

    duplicates = int(dataframe[column].duplicated().sum())
    if duplicates:
        raise DataValidationError(
            f"{name}.{column} contains {duplicates:,} duplicate values"
        )


def require_no_missing(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    """Require selected columns to contain no missing values."""
    missing = dataframe[list(columns)].isna().sum()
    missing = missing.loc[missing > 0]
    if not missing.empty:
        details = ", ".join(
            f"{column}={int(count):,}" for column, count in missing.items()
        )
        raise DataValidationError(f"{name} contains missing values: {details}")


def require_datetime_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    """Require selected columns to use pandas datetime dtypes."""
    invalid = [
        column
        for column in columns
        if not isinstance(dataframe[column].dtype, pd.DatetimeTZDtype)
        and not pd.api.types.is_datetime64_any_dtype(dataframe[column])
    ]
    if invalid:
        raise DataValidationError(f"{name} has non-datetime columns: {sorted(invalid)}")


def require_values_between(
    dataframe: pd.DataFrame,
    column: str,
    lower: float,
    upper: float,
    name: str,
) -> None:
    """Require non-missing numeric values to fall within an inclusive range."""
    values = dataframe[column].dropna()
    invalid = int((~values.between(lower, upper, inclusive="both")).sum())
    if invalid:
        raise DataValidationError(
            f"{name}.{column} contains {invalid:,} values outside "
            f"[{lower}, {upper}]"
        )


def require_condition(condition: pd.Series, message: str) -> None:
    """Require a row-level Boolean condition to hold everywhere."""
    invalid = int((~condition.fillna(False)).sum())
    if invalid:
        raise DataValidationError(f"{message}: {invalid:,} invalid rows")


def validate_raw_tables(raw_tables: Mapping[str, pd.DataFrame]) -> None:
    """Validate the presence, schema, and source grain of all raw tables."""
    missing_tables = sorted(set(RAW_REQUIRED_COLUMNS) - set(raw_tables))
    if missing_tables:
        raise DataValidationError(f"Missing raw tables: {missing_tables}")

    for name, required_columns in RAW_REQUIRED_COLUMNS.items():
        dataframe = raw_tables[name]
        require_non_empty(dataframe, name)
        require_columns(dataframe, required_columns, name)

    for name, key in RAW_UNIQUE_KEYS.items():
        require_unique_key(raw_tables[name], key, name)


def validate_clean_products(products: pd.DataFrame) -> None:
    """Validate the one-row-per-product cleaned product table."""
    name = "clean_products"
    require_non_empty(products, name)
    require_columns(products, PRODUCT_COLUMNS, name)
    require_unique_key(products, "product_id", name)
    require_no_missing(products, PRODUCT_COLUMNS, name)
    require_condition(
        products["product_weight_g"] > 0,
        f"{name}.product_weight_g must be positive",
    )
    require_condition(
        products["product_volume_cm3"] > 0,
        f"{name}.product_volume_cm3 must be positive",
    )


def validate_clean_reviews(reviews: pd.DataFrame) -> None:
    """Validate the one-unambiguous-review-per-order table."""
    name = "clean_reviews"
    require_non_empty(reviews, name)
    require_columns(reviews, REVIEW_COLUMNS, name)
    require_unique_key(reviews, "order_id", name)
    require_unique_key(reviews, "review_id", name)
    require_no_missing(reviews, REVIEW_COLUMNS, name)
    require_datetime_columns(
        reviews,
        {"review_creation_date", "review_answer_timestamp"},
        name,
    )
    require_values_between(reviews, "review_score", 1, 5, name)


def validate_clean_orders(
    orders: pd.DataFrame,
    max_shipping_limit_days: int = 60,
) -> None:
    """Validate the final one-row-per-order cleaned analysis population."""
    name = "clean_orders"
    require_non_empty(orders, name)
    require_columns(orders, CLEAN_ORDER_REQUIRED_COLUMNS, name)
    require_unique_key(orders, "order_id", name)

    required_non_missing = CLEAN_ORDER_REQUIRED_COLUMNS - {"distance_km"}
    require_no_missing(orders, required_non_missing, name)
    require_datetime_columns(orders, ORDER_DATE_COLUMNS, name)

    require_values_between(orders, "review_score", 1, 5, name)
    require_values_between(orders, "same_state", 0, 1, name)

    require_condition(
        orders["item_count"] > 0,
        f"{name}.item_count must be positive",
    )
    require_condition(
        orders["total_price"] > 0,
        f"{name}.total_price must be positive",
    )
    require_condition(
        orders["total_freight_value"] >= 0,
        f"{name}.total_freight_value must be nonnegative",
    )
    require_condition(
        orders["total_weight_g"] > 0,
        f"{name}.total_weight_g must be positive",
    )
    require_condition(
        orders["total_volume_cm3"] > 0,
        f"{name}.total_volume_cm3 must be positive",
    )

    known_distance = orders["distance_km"].dropna()
    if (known_distance < 0).any():
        invalid = int((known_distance < 0).sum())
        raise DataValidationError(
            f"{name}.distance_km contains {invalid:,} negative values"
        )

    require_condition(
        orders["order_approved_at"] >= orders["order_purchase_timestamp"],
        f"{name} has approval before purchase",
    )
    require_condition(
        orders["order_delivered_carrier_date"]
        >= orders["order_purchase_timestamp"],
        f"{name} has carrier pickup before purchase",
    )
    require_condition(
        orders["order_delivered_customer_date"]
        >= orders["order_delivered_carrier_date"],
        f"{name} has customer delivery before carrier pickup",
    )
    require_condition(
        orders["review_answer_timestamp"] >= orders["order_purchase_timestamp"],
        f"{name} has review response before purchase",
    )

    shipping_gap_days = (
        orders["shipping_limit_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    require_condition(
        shipping_gap_days.between(
            0,
            max_shipping_limit_days,
            inclusive="both",
        ),
        f"{name} has an implausible shipping deadline",
    )


def validate_delivery_orders(
    orders: pd.DataFrame,
    max_shipping_limit_days: int = 60,
) -> None:
    """Validate the review-independent cohort used for delivery modelling."""
    name = "delivery_orders"
    require_non_empty(orders, name)
    require_columns(orders, DELIVERY_ORDER_REQUIRED_COLUMNS, name)
    require_unique_key(orders, "order_id", name)

    required_non_missing = DELIVERY_ORDER_REQUIRED_COLUMNS - {"distance_km"}
    require_no_missing(orders, required_non_missing, name)
    require_datetime_columns(
        orders,
        ORDER_DATE_COLUMNS
        - {"review_creation_date", "review_answer_timestamp"},
        name,
    )
    require_values_between(orders, "same_state", 0, 1, name)

    for column in ("item_count", "total_price", "total_weight_g", "total_volume_cm3"):
        require_condition(orders[column] > 0, f"{name}.{column} must be positive")
    require_condition(
        orders["total_freight_value"] >= 0,
        f"{name}.total_freight_value must be nonnegative",
    )

    known_distance = orders["distance_km"].dropna()
    if (known_distance < 0).any():
        invalid = int((known_distance < 0).sum())
        raise DataValidationError(
            f"{name}.distance_km contains {invalid:,} negative values"
        )

    require_condition(
        orders["order_approved_at"] >= orders["order_purchase_timestamp"],
        f"{name} has approval before purchase",
    )
    require_condition(
        orders["order_delivered_carrier_date"]
        >= orders["order_purchase_timestamp"],
        f"{name} has carrier pickup before purchase",
    )
    require_condition(
        orders["order_delivered_customer_date"]
        >= orders["order_delivered_carrier_date"],
        f"{name} has customer delivery before carrier pickup",
    )

    shipping_gap_days = (
        orders["shipping_limit_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    require_condition(
        shipping_gap_days.between(0, max_shipping_limit_days, inclusive="both"),
        f"{name} has an implausible shipping deadline",
    )
