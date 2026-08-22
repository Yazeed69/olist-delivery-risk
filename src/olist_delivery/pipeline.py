"""End-to-end orchestration for the Olist delivery project.

The functions here connect loading, cleaning, validation, feature engineering,
analysis, modelling, and visualization. Domain logic remains in the dedicated
modules so each stage can be tested independently.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import (
    AdjustedModelResult,
    chi_square_association,
    dissatisfaction_threshold_sensitivity,
    dissatisfaction_rates_by_outcome,
    fit_adjusted_dissatisfaction_model,
    monthly_delivery_drift,
    review_score_distribution,
    seller_count_summary,
)
from .cleaning import (
    attach_customer_details,
    attach_geography,
    attach_order_items,
    attach_reviews,
    clean_order_items,
    clean_products,
    clean_reviews,
    keep_coherent_order_timelines,
    keep_delivered_orders,
    keep_plausible_shipping_deadlines,
    keep_reviews_after_purchase,
    keep_single_seller_orders,
    parse_order_dates,
)
from .data import OUTPUTS_DIR, PROCESSED_DIR, RAW_FILES, load_raw_table
from .features import OUTCOME_ORDER, build_features
from .modeling import ModelingResult, run_modeling
from .validation import (
    validate_clean_orders,
    validate_clean_products,
    validate_clean_reviews,
    validate_raw_tables,
)
from .visualization import (
    plot_dissatisfaction_rates,
    plot_calibration,
    plot_delivery_drift,
    plot_review_score_distribution,
    plot_risk_bands,
    save_figure,
)


FEATURES_FILENAME = "olist_delivery_features.csv"
METRICS_FILENAME = "metrics.json"
POPULATION_FLOW_FILENAME = "population_flow.csv"
DELIVERY_DRIFT_FILENAME = "delivery_drift.csv"
SELLER_DIAGNOSTIC_FILENAME = "seller_count_diagnostic.csv"
SENSITIVITY_FILENAME = "dissatisfaction_sensitivity.csv"
MODEL_IMPORTANCE_FILENAME = "model_feature_importance.csv"
LOGISTIC_COEFFICIENTS_FILENAME = "logistic_coefficients.csv"
VALIDATION_METRICS_FILENAME = "model_validation_metrics.csv"
METRIC_INTERVALS_FILENAME = "model_holdout_intervals.csv"


@dataclass(frozen=True)
class PipelineResult:
    """In-memory artifacts produced by a successful pipeline run."""

    features: pd.DataFrame
    population_flow: pd.DataFrame
    dissatisfaction_rates: pd.DataFrame
    review_distribution: pd.DataFrame
    seller_summary: pd.DataFrame
    sensitivity: pd.DataFrame
    delivery_drift: pd.DataFrame
    association_metrics: dict[str, float | int]
    adjusted_model: AdjustedModelResult
    modeling: ModelingResult


def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Load every source table required by this project."""
    return {name: load_raw_table(name) for name in RAW_FILES}


def _population_flow(stages: list[tuple[str, int]]) -> pd.DataFrame:
    """Build an auditable order-count table for sequential cohort filters."""
    raw_orders = stages[0][1]
    rows: list[dict[str, Any]] = []

    for position, (stage, orders) in enumerate(stages):
        previous_orders = stages[position - 1][1] if position else orders
        rows.append(
            {
                "stage": stage,
                "orders": orders,
                "orders_removed": previous_orders - orders,
                "retained_from_previous_pct": 100 * orders / previous_orders,
                "retained_from_raw_pct": 100 * orders / raw_orders,
            }
        )

    return pd.DataFrame(rows)


def build_analysis_population(
    raw_tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the validated single-seller cohort and its population audit.

    The returned seller summary is calculated from the comparable population
    before the single-seller restriction. It documents why that restriction is
    part of the final study design.
    """
    validate_raw_tables(raw_tables)

    products = clean_products(raw_tables["products"])
    reviews = clean_reviews(raw_tables["reviews"])
    validate_clean_products(products)
    validate_clean_reviews(reviews)

    raw_orders = raw_tables["orders"]
    stages: list[tuple[str, int]] = [("raw_orders", len(raw_orders))]

    orders = keep_delivered_orders(raw_orders)
    stages.append(("delivered_orders", len(orders)))

    orders = parse_order_dates(orders)
    stages.append(("complete_order_timestamps", len(orders)))

    orders = keep_coherent_order_timelines(orders)
    stages.append(("coherent_delivery_timelines", len(orders)))

    orders = attach_customer_details(orders, raw_tables["customers"])
    order_items = clean_order_items(
        raw_tables["order_items"],
        products,
        orders["order_id"],
    )
    orders = attach_order_items(orders, order_items)
    stages.append(("orders_with_usable_items", len(orders)))

    # Preserve a pre-restriction comparison before dropping multi-seller orders.
    seller_comparison = attach_reviews(orders, reviews)
    seller_comparison = keep_reviews_after_purchase(seller_comparison)
    seller_comparison = keep_plausible_shipping_deadlines(seller_comparison)
    seller_comparison = build_features(seller_comparison)
    seller_summary = seller_count_summary(seller_comparison)

    orders = keep_single_seller_orders(orders)
    stages.append(("single_seller_orders", len(orders)))

    orders = attach_geography(
        orders,
        raw_tables["sellers"],
        raw_tables["geolocation"],
    )
    orders = attach_reviews(orders, reviews)
    stages.append(("orders_with_usable_reviews", len(orders)))

    orders = keep_reviews_after_purchase(orders)
    stages.append(("reviews_after_purchase", len(orders)))

    orders = keep_plausible_shipping_deadlines(orders)
    stages.append(("final_analysis_population", len(orders)))
    validate_clean_orders(orders)

    features = build_features(orders).sort_values(
        ["order_purchase_timestamp", "order_id"]
    ).reset_index(drop=True)

    return features, _population_flow(stages), seller_summary


def run_analysis(features: pd.DataFrame) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float | int],
    AdjustedModelResult,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run the descriptive and adjusted dissatisfaction analyses."""
    rates = dissatisfaction_rates_by_outcome(features)
    distribution = review_score_distribution(features)
    association = chi_square_association(features)
    adjusted_model = fit_adjusted_dissatisfaction_model(features)
    sensitivity = dissatisfaction_threshold_sensitivity(features)
    drift = monthly_delivery_drift(features)
    return rates, distribution, association, adjusted_model, sensitivity, drift


def execute_pipeline() -> PipelineResult:
    """Execute all computational stages without writing project artifacts."""
    raw_tables = load_raw_tables()
    features, population_flow, seller_summary = build_analysis_population(raw_tables)
    (
        rates,
        distribution,
        association,
        adjusted_model,
        sensitivity,
        drift,
    ) = run_analysis(features)
    modeling = run_modeling(features)

    return PipelineResult(
        features=features,
        population_flow=population_flow,
        dissatisfaction_rates=rates,
        review_distribution=distribution,
        seller_summary=seller_summary,
        sensitivity=sensitivity,
        delivery_drift=drift,
        association_metrics=association,
        adjusted_model=adjusted_model,
        modeling=modeling,
    )


def _json_default(value: Any) -> Any:
    """Convert common scientific-Python scalar types for JSON output."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_metrics(result: PipelineResult) -> dict[str, Any]:
    """Collect the main reproducible findings in a machine-readable object."""
    rates = result.dissatisfaction_rates.copy()
    rate_records = rates[
        [
            "deadline_outcome",
            "orders",
            "dissatisfied",
            "rate",
            "ci_low",
            "ci_high",
            "vs_baseline",
        ]
    ].to_dict(orient="records")

    outcome_terms = set(OUTCOME_ORDER[1:])
    outcome_estimates = result.adjusted_model.estimates.loc[
        result.adjusted_model.estimates["term"].isin(outcome_terms)
    ].to_dict(orient="records")

    return {
        "population": {
            "raw_orders": int(result.population_flow.iloc[0]["orders"]),
            "final_orders": int(len(result.features)),
            "dissatisfied_orders": int(result.features["dissatisfied"].sum()),
            "dissatisfaction_rate": float(result.features["dissatisfied"].mean()),
            "late_deliveries": int(result.features["late_delivery"].sum()),
            "late_delivery_rate": float(result.features["late_delivery"].mean()),
        },
        "dissatisfaction_by_deadline_outcome": rate_records,
        "chi_square_association": result.association_metrics,
        "adjusted_dissatisfaction_model": {
            "converged": result.adjusted_model.converged,
            "n_observations": result.adjusted_model.n_observations,
            "pseudo_r_squared": result.adjusted_model.pseudo_r_squared,
            "log_likelihood": result.adjusted_model.log_likelihood,
            "covariance_type": result.adjusted_model.covariance_type,
            "seller_clusters": result.adjusted_model.n_clusters,
            "deadline_outcome_estimates": outcome_estimates,
        },
        "dissatisfaction_definition_sensitivity": result.sensitivity.to_dict(
            orient="records"
        ),
        "seller_count_scope_check": result.seller_summary.to_dict(
            orient="records"
        ),
        "late_delivery_modeling": {
            "split": result.modeling.split_metadata,
            "validation_metrics": result.modeling.validation_metrics.to_dict(
                orient="records"
            ),
            "holdout_metrics": result.modeling.holdout_metrics,
            "holdout_metric_intervals": (
                result.modeling.metric_intervals.to_dict(orient="records")
            ),
            "risk_band_calibration": result.modeling.risk_bands.to_dict(
                orient="records"
            ),
        },
    }


def write_outputs(
    result: PipelineResult,
    outputs_dir: Path = OUTPUTS_DIR,
    processed_dir: Path = PROCESSED_DIR,
    figures_dir: Path | None = None,
) -> None:
    """Write the processed cohort, metrics, population audit, and figures."""
    if figures_dir is None:
        figures_dir = outputs_dir / "figures"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    result.features.to_csv(processed_dir / FEATURES_FILENAME, index=False)
    result.population_flow.to_csv(
        outputs_dir / POPULATION_FLOW_FILENAME,
        index=False,
        float_format="%.2f",
    )
    result.delivery_drift.to_csv(
        outputs_dir / DELIVERY_DRIFT_FILENAME,
        index=False,
    )
    result.seller_summary.to_csv(
        outputs_dir / SELLER_DIAGNOSTIC_FILENAME,
        index=False,
    )
    result.sensitivity.to_csv(
        outputs_dir / SENSITIVITY_FILENAME,
        index=False,
    )
    result.modeling.permutation_importance.to_csv(
        outputs_dir / MODEL_IMPORTANCE_FILENAME,
        index=False,
    )
    result.modeling.logistic_coefficients.to_csv(
        outputs_dir / LOGISTIC_COEFFICIENTS_FILENAME,
        index=False,
    )
    result.modeling.validation_metrics.to_csv(
        outputs_dir / VALIDATION_METRICS_FILENAME,
        index=False,
    )
    result.modeling.metric_intervals.to_csv(
        outputs_dir / METRIC_INTERVALS_FILENAME,
        index=False,
    )

    metrics = build_metrics(result)
    with (outputs_dir / METRICS_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
        file.write("\n")

    save_figure(
        plot_dissatisfaction_rates(result.dissatisfaction_rates),
        figures_dir / "dissatisfaction_by_deadline_outcome.png",
    )
    save_figure(
        plot_review_score_distribution(result.review_distribution),
        figures_dir / "review_scores_by_deadline_outcome.png",
    )

    split = result.modeling.split_metadata
    evaluation_period = f"{split['test_start'][:10]} to {split['test_end'][:10]}"
    save_figure(
        plot_risk_bands(result.modeling.risk_bands, evaluation_period),
        figures_dir / "late_delivery_risk_bands.png",
    )
    save_figure(
        plot_calibration(result.modeling.risk_bands),
        figures_dir / "late_delivery_calibration.png",
    )
    save_figure(
        plot_delivery_drift(
            result.delivery_drift,
            test_start=split["test_start"],
        ),
        figures_dir / "delivery_target_drift.png",
    )


def main() -> None:
    """Run the complete project and print a compact execution summary."""
    started = time.perf_counter()
    print("Loading, cleaning, and validating data...")
    result = execute_pipeline()

    print("Writing processed data, metrics, and figures...")
    write_outputs(result)

    selected_model = result.modeling.split_metadata["selected_model"]
    selected_metrics = result.modeling.holdout_metrics

    print(
        f"Done: {len(result.features):,} analysis orders in "
        f"{time.perf_counter() - started:.1f}s"
    )
    print(
        f"Selected model: {selected_model} | "
        f"held-out ROC-AUC {selected_metrics['roc_auc']:.3f} | "
        f"PR-AUC {selected_metrics['pr_auc']:.3f}"
    )
    print(f"Outputs: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
