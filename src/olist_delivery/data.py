from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


RAW_FILES = {
    "orders": RAW_DIR / "olist_orders_dataset.csv",
    "customers": RAW_DIR / "olist_customers_dataset.csv",
    "order_items": RAW_DIR / "olist_order_items_dataset.csv",
    "products": RAW_DIR / "olist_products_dataset.csv",
    "reviews": RAW_DIR / "olist_order_reviews_dataset.csv",
    "sellers": RAW_DIR / "olist_sellers_dataset.csv",
    "geolocation": RAW_DIR / "olist_geolocation_dataset.csv",
}


def load_raw_table(name: str) -> pd.DataFrame:
    if name not in RAW_FILES:
        available = ", ".join(sorted(RAW_FILES))
        raise KeyError(f"Unknown raw table '{name}'. Available tables: {available}")

    path = RAW_FILES[name]

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing raw data file: {path}\n"
            "Download the Olist dataset and place the CSV files in data/raw/."
        )

    return pd.read_csv(path)