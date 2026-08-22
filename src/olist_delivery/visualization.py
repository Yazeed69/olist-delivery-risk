"""Publication-ready figures for the Olist delivery analysis."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .features import OUTCOME_ORDER
from .validation import require_columns, require_non_empty


OUTCOME_LABELS = {
    "both_deadlines_met": "both deadlines\nmet",
    "late_handoff_recovered": "late handoff,\nrecovered",
    "on_time_handoff_late_delivery": "on-time handoff,\nlate delivery",
    "both_deadlines_missed": "both deadlines\nmissed",
}

PRIMARY_BLUE = "#2F73D2"
REFERENCE_ORANGE = "#E66A35"
TEXT_GREY = "#52514E"
GRID_GREY = "#D9DDE3"

SCORE_COLOURS = {
    1: "#B3262D",
    2: "#E34B4F",
    3: "#C9C7C1",
    4: "#6DA7EC",
    5: "#1C5CAB",
}


def _ordered_outcome_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy ordered from best to worst fulfilment outcome."""
    ordered = table.copy()
    ordered["deadline_outcome"] = pd.Categorical(
        ordered["deadline_outcome"],
        categories=OUTCOME_ORDER,
        ordered=True,
    )
    return ordered.sort_values("deadline_outcome").reset_index(drop=True)


def _clean_axes(axis: plt.Axes, grid_axis: str | None = "y") -> None:
    """Apply the shared minimal chart style."""
    axis.spines[["top", "right"]].set_visible(False)
    if grid_axis is not None:
        axis.grid(axis=grid_axis, color=GRID_GREY, alpha=0.7, linewidth=0.8)
        axis.set_axisbelow(True)


def plot_dissatisfaction_rates(rates: pd.DataFrame) -> Figure:
    """Plot dissatisfaction rates with Wilson confidence intervals."""
    name = "dissatisfaction_rates"
    required = {
        "deadline_outcome",
        "orders",
        "rate",
        "ci_low",
        "ci_high",
        "confidence_level",
    }
    require_non_empty(rates, name)
    require_columns(rates, required, name)
    table = _ordered_outcome_table(rates)

    positions = np.arange(len(table))
    values = 100 * table["rate"].to_numpy()
    lower = 100 * table["ci_low"].to_numpy()
    upper = 100 * table["ci_high"].to_numpy()
    errors = np.vstack([values - lower, upper - values])

    labels = [
        f"{OUTCOME_LABELS[outcome]}\nn = {int(count):,}"
        for outcome, count in zip(table["deadline_outcome"], table["orders"])
    ]

    figure, axis = plt.subplots(figsize=(10, 5.8))
    bars = axis.bar(
        positions,
        values,
        width=0.62,
        color=PRIMARY_BLUE,
        yerr=errors,
        capsize=4,
    )

    offset = max(values.max() * 0.025, 0.8)
    for bar, value, interval_top in zip(bars, values, upper):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            interval_top + offset,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    total_orders = int(table["orders"].sum())
    confidence = 100 * float(table["confidence_level"].iloc[0])
    axis.set_title(
        "Missing the customer delivery promise tracks much higher dissatisfaction",
        loc="left",
        pad=28,
        fontsize=14,
        weight="bold",
    )
    axis.text(
        0,
        1.025,
        f"{total_orders:,} delivered single-seller orders; "
        f"error bars are {confidence:.0f}% Wilson intervals",
        transform=axis.transAxes,
        color=TEXT_GREY,
        fontsize=9,
    )
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Fulfilment outcome")
    axis.set_ylabel("Orders with a 1- or 2-star review (%)")
    axis.set_ylim(0, max(upper) * 1.20)
    _clean_axes(axis)

    figure.tight_layout()
    return figure


def plot_review_score_distribution(distribution: pd.DataFrame) -> Figure:
    """Plot the complete 1-5 review distribution by deadline outcome."""
    name = "review_score_distribution"
    share_columns = {f"score_{score}_share" for score in range(1, 6)}
    required = {"deadline_outcome", "orders", *share_columns}
    require_non_empty(distribution, name)
    require_columns(distribution, required, name)
    table = _ordered_outcome_table(distribution)

    positions = np.arange(len(table))
    left = np.zeros(len(table))
    figure, axis = plt.subplots(figsize=(10, 5.2))

    for score in range(1, 6):
        values = 100 * table[f"score_{score}_share"].to_numpy()
        axis.barh(
            positions,
            values,
            left=left,
            height=0.62,
            color=SCORE_COLOURS[score],
            label=f"{score} star",
        )

        for position, (value, start) in enumerate(zip(values, left)):
            if value >= 5:
                axis.text(
                    start + value / 2,
                    position,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if score in {1, 5} else "black",
                    fontsize=9,
                )
        left += values

    labels = [
        f"{OUTCOME_LABELS[outcome].replace(chr(10), ' ')}\nn = {int(count):,}"
        for outcome, count in zip(table["deadline_outcome"], table["orders"])
    ]
    total_orders = int(table["orders"].sum())

    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of the outcome group's reviews (%)")
    axis.set_ylabel("Fulfilment outcome")
    axis.set_title(
        "A late delivery shifts the most common rating from 5 stars to 1",
        loc="left",
        pad=28,
        fontsize=14,
        weight="bold",
    )
    axis.text(
        0,
        1.025,
        f"{total_orders:,} delivered single-seller orders; all review scores shown",
        transform=axis.transAxes,
        color=TEXT_GREY,
        fontsize=9,
    )
    axis.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        title="Review score",
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(False)

    figure.tight_layout()
    return figure


def plot_risk_bands(
    risk_bands: pd.DataFrame,
    evaluation_period: str | None = None,
) -> Figure:
    """Plot observed late-delivery rates in held-out predicted-risk bands."""
    name = "risk_bands"
    required = {
        "risk_band",
        "orders",
        "late_deliveries",
        "late_delivery_rate",
    }
    require_non_empty(risk_bands, name)
    require_columns(risk_bands, required, name)
    table = risk_bands.sort_values("risk_band").reset_index(drop=True)

    bands = table["risk_band"].to_numpy()
    values = 100 * table["late_delivery_rate"].to_numpy()
    total_orders = int(table["orders"].sum())
    total_late = int(table["late_deliveries"].sum())
    overall_rate = 100 * total_late / total_orders

    figure, axis = plt.subplots(figsize=(10, 5.8))
    bars = axis.bar(bands, values, width=0.7, color=PRIMARY_BLUE)

    offset = max(values.max() * 0.025, 0.15)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.axhline(
        overall_rate,
        color=REFERENCE_ORANGE,
        linestyle="--",
        linewidth=1.6,
    )
    axis.text(
        bands.min() - 0.25,
        overall_rate + offset,
        f"held-out average {overall_rate:.1f}%",
        color=REFERENCE_ORANGE,
        fontsize=9,
    )

    subtitle = f"{total_orders:,} chronologically held-out orders"
    if evaluation_period:
        subtitle += f"; {evaluation_period}"

    axis.set_title(
        "Observed late-delivery rate by predicted risk band",
        loc="left",
        pad=28,
        fontsize=14,
        weight="bold",
    )
    axis.text(
        0,
        1.025,
        subtitle,
        transform=axis.transAxes,
        color=TEXT_GREY,
        fontsize=9,
    )
    axis.set_xticks(bands)
    axis.set_xlabel("Predicted risk band (1 = safest, 10 = riskiest)")
    axis.set_ylabel("Orders delivered late (%)")
    axis.set_ylim(0, max(values.max(), overall_rate) * 1.22)
    _clean_axes(axis)

    figure.tight_layout()
    return figure


def save_figure(
    figure: Figure,
    path: Path,
    dpi: int = 160,
    close: bool = True,
) -> None:
    """Save a figure, creating its parent directory when necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(figure)
