"""Checkout-time late-delivery modelling for the Olist project.

The evaluation uses a chronological holdout: models train on earlier orders and
are evaluated on later orders. All imputation, scaling, and categorical encoding
are fitted inside scikit-learn pipelines using training data only.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .validation import (
    DataValidationError,
    require_columns,
    require_datetime_columns,
    require_non_empty,
    require_values_between,
)


NUMERIC_FEATURES = [
    "item_count",
    "log_order_price",
    "total_price",
    "total_freight_value",
    "freight_ratio",
    "total_weight_g",
    "total_volume_cm3",
    "promised_delivery_window_days",
    "distance_km",
    "same_state",
    "purchase_hour",
    "purchase_day_of_week",
    "purchase_is_weekend",
]

CATEGORICAL_FEATURES = [
    "purchase_month",
    "purchase_year",
    "customer_state",
]

MODEL_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]

POST_CHECKOUT_COLUMNS = {
    "deadline_outcome",
    "dissatisfied",
    "review_score",
    "review_id",
    "review_creation_date",
    "review_answer_timestamp",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "shipping_limit_date",
}

TARGET_COLUMN = "late_delivery"
TIME_COLUMN = "order_purchase_timestamp"
ORDER_ID_COLUMN = "order_id"

DEFAULT_TEST_FRACTION = 0.20
DEFAULT_RISK_BANDS = 10
RANDOM_STATE = 42


@dataclass(frozen=True)
class TemporalSplit:
    """Earlier training orders and later held-out orders."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_order_ids: pd.Series
    test_order_ids: pd.Series
    train_timestamps: pd.Series
    test_timestamps: pd.Series
    cutoff: pd.Timestamp


@dataclass
class ModelingResult:
    """Fitted candidates and their held-out evaluation artifacts."""

    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    risk_scores: pd.DataFrame
    risk_bands: pd.DataFrame
    permutation_importance: pd.DataFrame
    logistic_coefficients: pd.DataFrame
    split_metadata: dict[str, Any]


def validate_modeling_input(orders: pd.DataFrame) -> None:
    """Validate the model population, target, timestamp, and feature policy."""
    name = "model_orders"
    required = {
        ORDER_ID_COLUMN,
        TIME_COLUMN,
        TARGET_COLUMN,
        *MODEL_FEATURES,
    }
    require_non_empty(orders, name)
    require_columns(orders, required, name)
    require_datetime_columns(orders, {TIME_COLUMN}, name)
    require_values_between(orders, TARGET_COLUMN, 0, 1, name)

    leaked = sorted(set(MODEL_FEATURES) & POST_CHECKOUT_COLUMNS)
    if leaked:
        raise DataValidationError(
            f"Model features include post-checkout information: {leaked}"
        )

    if orders[ORDER_ID_COLUMN].isna().any():
        raise DataValidationError(f"{name}.{ORDER_ID_COLUMN} contains missing values")
    if orders[ORDER_ID_COLUMN].duplicated().any():
        duplicates = int(orders[ORDER_ID_COLUMN].duplicated().sum())
        raise DataValidationError(
            f"{name}.{ORDER_ID_COLUMN} contains {duplicates:,} duplicate values"
        )
    if orders[TARGET_COLUMN].nunique() != 2:
        raise DataValidationError(f"{name}.{TARGET_COLUMN} must contain both classes")


def temporal_train_test_split(
    orders: pd.DataFrame,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> TemporalSplit:
    """Train on earlier orders and reserve the latest fraction for testing."""
    validate_modeling_input(orders)
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    ordered = orders.sort_values([TIME_COLUMN, ORDER_ID_COLUMN]).reset_index(drop=True)
    split_position = int(np.floor(len(ordered) * (1 - test_fraction)))
    if split_position <= 0 or split_position >= len(ordered):
        raise ValueError("test_fraction leaves an empty train or test population")

    cutoff = ordered.loc[split_position, TIME_COLUMN]
    train = ordered.loc[ordered[TIME_COLUMN] < cutoff].copy()
    test = ordered.loc[ordered[TIME_COLUMN] >= cutoff].copy()

    if train.empty or test.empty:
        raise DataValidationError("Temporal split produced an empty train or test set")
    if train[TARGET_COLUMN].nunique() != 2 or test[TARGET_COLUMN].nunique() != 2:
        raise DataValidationError(
            "Both temporal partitions must contain both target classes"
        )

    return TemporalSplit(
        X_train=train[MODEL_FEATURES].copy(),
        X_test=test[MODEL_FEATURES].copy(),
        y_train=train[TARGET_COLUMN].astype(int).copy(),
        y_test=test[TARGET_COLUMN].astype(int).copy(),
        train_order_ids=train[ORDER_ID_COLUMN].copy(),
        test_order_ids=test[ORDER_ID_COLUMN].copy(),
        train_timestamps=train[TIME_COLUMN].copy(),
        test_timestamps=test[TIME_COLUMN].copy(),
        cutoff=pd.Timestamp(cutoff),
    )


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build preprocessing that is learned exclusively from training data."""
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_candidate_models() -> dict[str, Pipeline]:
    """Construct the two model candidates with isolated preprocessing."""
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.07,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def top_fraction_metrics(
    target: pd.Series,
    probabilities: np.ndarray,
    fraction: float = 0.10,
) -> dict[str, float | int]:
    """Evaluate a fixed-capacity intervention group ranked by model risk."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1")

    target_array = np.asarray(target, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    if len(target_array) != len(probability_array):
        raise ValueError("target and probabilities must have the same length")

    flagged_count = max(1, int(np.ceil(len(target_array) * fraction)))
    ranked = np.argsort(-probability_array, kind="stable")
    flagged_index = ranked[:flagged_count]
    flagged_target = target_array[flagged_index]

    positives = int(target_array.sum())
    captured = int(flagged_target.sum())
    base_rate = float(target_array.mean())
    precision = float(flagged_target.mean())

    return {
        "flagged_orders": flagged_count,
        "captured_late_deliveries": captured,
        "capture_rate": captured / positives,
        "precision": precision,
        "lift": precision / base_rate,
    }


def evaluate_probabilities(
    target: pd.Series,
    probabilities: np.ndarray,
    top_fraction: float = 0.10,
) -> dict[str, float | int]:
    """Calculate ranking and operational metrics for predicted probabilities."""
    operational = top_fraction_metrics(target, probabilities, top_fraction)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "base_rate": float(np.asarray(target).mean()),
        "top_fraction": float(top_fraction),
        **operational,
    }


def fit_and_evaluate_candidates(
    split: TemporalSplit,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Fit candidate models and return held-out evaluation metrics."""
    models = build_candidate_models()
    rows = []

    for name, model in models.items():
        model.fit(split.X_train, split.y_train)
        train_probability = model.predict_proba(split.X_train)[:, 1]
        test_probability = model.predict_proba(split.X_test)[:, 1]
        metrics = evaluate_probabilities(split.y_test, test_probability)
        metrics.update(
            {
                "model": name,
                "train_roc_auc": float(
                    roc_auc_score(split.y_train, train_probability)
                ),
            }
        )
        rows.append(metrics)

    columns = [
        "model",
        "train_roc_auc",
        "roc_auc",
        "pr_auc",
        "base_rate",
        "top_fraction",
        "flagged_orders",
        "captured_late_deliveries",
        "capture_rate",
        "precision",
        "lift",
    ]
    return models, pd.DataFrame(rows)[columns]


def score_held_out_orders(
    split: TemporalSplit,
    model: Pipeline,
    n_bands: int = DEFAULT_RISK_BANDS,
) -> pd.DataFrame:
    """Score held-out future orders and assign equal-sized risk bands."""
    if n_bands < 2:
        raise ValueError("n_bands must be at least 2")

    probabilities = model.predict_proba(split.X_test)[:, 1]
    scored = pd.DataFrame(
        {
            ORDER_ID_COLUMN: split.test_order_ids.to_numpy(),
            TIME_COLUMN: split.test_timestamps.to_numpy(),
            TARGET_COLUMN: split.y_test.to_numpy(),
            "risk_score": probabilities,
        }
    )

    score_rank = scored["risk_score"].rank(method="first")
    scored["risk_band"] = pd.qcut(
        score_rank,
        q=n_bands,
        labels=range(1, n_bands + 1),
    ).astype(int)
    return scored.sort_values(TIME_COLUMN).reset_index(drop=True)


def summarize_risk_bands(scored_orders: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed late-delivery rates within held-out risk bands."""
    required = {"risk_band", TARGET_COLUMN}
    require_non_empty(scored_orders, "scored_orders")
    require_columns(scored_orders, required, "scored_orders")

    summary = (
        scored_orders.groupby("risk_band")
        .agg(
            orders=(TARGET_COLUMN, "size"),
            late_deliveries=(TARGET_COLUMN, "sum"),
        )
        .reset_index()
    )
    summary["late_delivery_rate"] = (
        summary["late_deliveries"] / summary["orders"]
    )
    overall_rate = scored_orders[TARGET_COLUMN].mean()
    summary["lift"] = summary["late_delivery_rate"] / overall_rate
    return summary


def permutation_importance_table(
    model: Pipeline,
    split: TemporalSplit,
    repeats: int = 5,
) -> pd.DataFrame:
    """Measure held-out ROC-AUC loss when each original feature is shuffled."""
    importance = permutation_importance(
        model,
        split.X_test,
        split.y_test,
        n_repeats=repeats,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    return (
        pd.DataFrame(
            {
                "feature": split.X_test.columns,
                "drop_in_roc_auc": importance.importances_mean,
                "standard_deviation": importance.importances_std,
            }
        )
        .sort_values("drop_in_roc_auc", ascending=False)
        .reset_index(drop=True)
    )


def logistic_coefficient_table(model: Pipeline) -> pd.DataFrame:
    """Return coefficients and odds ratios for transformed logistic features."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression):
        raise TypeError("Expected a fitted logistic-regression pipeline")

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "odds_ratio": np.exp(coefficients),
                "absolute_coefficient": np.abs(coefficients),
            }
        )
        .sort_values("absolute_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def run_modeling(
    orders: pd.DataFrame,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> ModelingResult:
    """Run chronological candidate evaluation and held-out risk analysis."""
    split = temporal_train_test_split(orders, test_fraction=test_fraction)
    models, metrics = fit_and_evaluate_candidates(split)

    best_model_name = metrics.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    best_model = models[str(best_model_name)]
    risk_scores = score_held_out_orders(split, best_model)
    risk_bands = summarize_risk_bands(risk_scores)

    importance = permutation_importance_table(best_model, split)
    coefficients = logistic_coefficient_table(models["logistic_regression"])

    split_metadata = {
        "strategy": "chronological_holdout",
        "cutoff": split.cutoff.isoformat(),
        "train_orders": int(len(split.y_train)),
        "test_orders": int(len(split.y_test)),
        "train_start": split.train_timestamps.min().isoformat(),
        "train_end": split.train_timestamps.max().isoformat(),
        "test_start": split.test_timestamps.min().isoformat(),
        "test_end": split.test_timestamps.max().isoformat(),
        "train_late_delivery_rate": float(split.y_train.mean()),
        "test_late_delivery_rate": float(split.y_test.mean()),
        "selected_model": str(best_model_name),
    }

    return ModelingResult(
        models=models,
        metrics=metrics,
        risk_scores=risk_scores,
        risk_bands=risk_bands,
        permutation_importance=importance,
        logistic_coefficients=coefficients,
        split_metadata=split_metadata,
    )
