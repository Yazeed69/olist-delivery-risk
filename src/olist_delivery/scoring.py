"""Batch scoring for the persisted late-delivery model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .data import OUTPUTS_DIR
from .modeling import MODEL_FEATURES, ORDER_ID_COLUMN


DEFAULT_MODEL_PATH = OUTPUTS_DIR / "late_delivery_model.joblib"
DEFAULT_METADATA_PATH = OUTPUTS_DIR / "late_delivery_model_metadata.json"


def load_model_bundle(
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> tuple[Any, dict[str, Any]]:
    """Load a serialized estimator and verify its declared feature contract."""
    model = joblib.load(model_path)
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("features") != MODEL_FEATURES:
        raise ValueError("Model metadata does not match the current feature contract")
    return model, metadata


def score_dataframe(data: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Return order identifiers and predicted late-delivery probabilities."""
    missing = sorted(set(MODEL_FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Scoring input is missing required features: {missing}")

    result = pd.DataFrame(index=data.index)
    if ORDER_ID_COLUMN in data:
        result[ORDER_ID_COLUMN] = data[ORDER_ID_COLUMN].astype(str)
    else:
        result[ORDER_ID_COLUMN] = [f"row_{index}" for index in data.index]
    result["late_delivery_risk"] = model.predict_proba(data[MODEL_FEATURES])[:, 1]
    return result.reset_index(drop=True)


def score_csv(
    input_path: Path,
    output_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
    """Score one CSV against the persisted model and write a compact result."""
    model, _ = load_model_bundle(model_path, metadata_path)
    scored = score_dataframe(pd.read_csv(input_path), model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    return scored


def main() -> None:
    """Command-line entry point for batch scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV containing model features")
    parser.add_argument("output", type=Path, help="destination CSV for risk scores")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()

    scored = score_csv(args.input, args.output, args.model, args.metadata)
    print(f"Scored {len(scored):,} orders to {args.output}")


if __name__ == "__main__":
    main()
