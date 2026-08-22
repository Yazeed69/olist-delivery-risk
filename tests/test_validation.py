import pandas as pd
import pytest

from olist_delivery.validation import (
    DataValidationError,
    require_unique_key,
    validate_clean_products,
    validate_raw_tables,
)


def test_raw_validation_reports_all_missing_tables() -> None:
    with pytest.raises(DataValidationError, match="Missing raw tables"):
        validate_raw_tables({})


@pytest.mark.parametrize(
    "values, expected_message",
    [
        (["a", "a"], "duplicate"),
        (["a", None], "missing"),
    ],
)
def test_unique_key_rejects_duplicates_and_missing_values(
    values: list[str | None],
    expected_message: str,
) -> None:
    table = pd.DataFrame({"order_id": values})

    with pytest.raises(DataValidationError, match=expected_message):
        require_unique_key(table, "order_id", "orders")


def test_clean_product_validation_rejects_nonpositive_measurements() -> None:
    products = pd.DataFrame(
        {
            "product_id": ["product_1"],
            "product_category_name": ["books"],
            "product_weight_g": [0.0],
            "product_volume_cm3": [100.0],
        }
    )

    with pytest.raises(DataValidationError, match="must be positive"):
        validate_clean_products(products)
