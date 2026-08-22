# Olist Delivery Risk and Customer Dissatisfaction

An end-to-end data science project investigating how fulfilment failures relate
to customer dissatisfaction and whether late-delivery risk can be identified at
checkout.

The project turns seven raw Olist marketplace tables into a validated,
order-level analytical dataset, an adjusted statistical analysis, and a
chronologically evaluated prediction model.

## Results

- **93,306** delivered, reviewed, single-seller orders form the final study
  population.
- Dissatisfaction rises from **8.2%** when both deadlines are met to **62–63%**
  when the customer-facing delivery promise is missed.
- After adjusting for order characteristics, product category, customer state,
  and seller-clustered uncertainty, a late customer delivery is associated with
  roughly **18–20 times** the odds of dissatisfaction compared with meeting
  both deadlines.
- Logistic regression is selected on an intermediate validation period, then
  reaches **0.713 ROC-AUC** and **0.079 PR-AUC** on a final untouched temporal
  holdout. The PR-AUC is about 2.25 times its **3.52%** prevalence floor.
- Its highest-risk 10% captures **23.2%** of held-out late deliveries
  (95% bootstrap CI **20.3–26.4%**) with **2.32× lift**
  (95% CI **2.03–2.64×**).

These are observational associations and held-out predictive results, not
causal estimates.

![Dissatisfaction by fulfilment outcome](outputs/figures/dissatisfaction_by_deadline_outcome.png)

![Review scores by fulfilment outcome](outputs/figures/review_scores_by_deadline_outcome.png)

![Held-out late-delivery risk bands](outputs/figures/late_delivery_risk_bands.png)

## Temporal drift and calibration

Late-delivery prevalence changes sharply across time: **7.77%** in model
training, **6.66%** in validation, and **3.52%** in the final holdout. The drop
is not explained by one stable trend. A wider median promised window coincides
with the unusually low June 2018 rate, but promised windows narrow again in
July and August while lateness rises. The target reflects both fulfilment
performance and Olist's changing promise policy.

![Monthly late-delivery target drift](outputs/figures/delivery_target_drift.png)

The final model has a **0.0336 Brier score** and **1.24 percentage-point
expected calibration error**. Ranking remains useful, but the highest-risk
band averages 13.6% predicted risk against 8.1% observed risk. Operational use
would therefore require ongoing recalibration rather than treating the scores
as stable probabilities.

Held-out permutation importance confirms that the promised delivery window
dominates the model: shuffling it reduces ROC-AUC by 0.268, while every other
feature reduces it by less than 0.004. The model is therefore recovering a
large amount of Olist's existing promise-setting policy rather than discovering
an independent operational signal.

![Held-out probability calibration](outputs/figures/late_delivery_calibration.png)

## Why single-seller orders?

Olist records one customer-delivery timestamp per order, even when an order is
split across multiple sellers. That makes seller-level handoff performance
ambiguous for multi-seller orders. The pipeline preserves a seller-count audit,
then restricts the main analysis to single-seller orders so each observed
delivery can be attributed coherently.

The anomaly is reported rather than hidden: two-seller orders have **46.6%**
dissatisfaction but only **1.0%** recorded lateness, versus 12.2% and 6.7% for
single-seller orders. Their median promised window is only one day longer
(25 versus 24 days), and reviews created before the recorded delivery are less
common, not more common (1.1% versus 5.1%). Those checks do not explain the
pattern. The order-level source schema still cannot identify which parcel a
single delivery timestamp represents, so parcel-level attribution remains
unresolved and these orders stay outside the main deadline analysis.

The dissatisfaction finding is also robust to its definition. With only
1-star reviews flagged, late-delivery groups are about 9.2 times baseline; with
1–3-star reviews flagged, they remain about 4.6 times baseline.

## Method

The pipeline performs the following sequence:

1. Load and validate the raw table schemas and source keys.
2. Resolve duplicate or conflicting reviews before joining them to orders.
3. Clean products, timestamps, order items, customer details, and geography.
4. Restrict the population to coherent, delivered, single-seller orders.
5. Engineer deadline outcomes, dissatisfaction, order controls, and
   checkout-time predictors.
6. Estimate Wilson confidence intervals, a chi-square association, and an
   adjusted binomial logistic model with standard errors clustered by seller.
7. Repeat the dissatisfaction analysis at 1-star, 1–2-star, and 1–3-star
   thresholds.
8. Select between logistic regression and histogram gradient boosting on a
   chronological validation period, refit on all development data, and evaluate
   once on the untouched latest 20% of orders.
9. Measure target drift, calibration, and bootstrap uncertainty before
   exporting auditable tables and publication-ready figures.

All preprocessing for predictive models is fitted on training data only.
Post-checkout fields—including actual delivery timestamps and reviews—are
explicitly excluded from model features. Month, weekday, and hour use cyclical
encodings; purchase year is retained only for drift analysis and is excluded
from the predictive feature matrix.

## Repository layout

```text
src/
    olist_delivery/
        data.py             paths and raw-table loading
        cleaning.py         order-level cleaning transformations
        validation.py       runtime schema and invariant checks
        features.py         analytical and modelling features
        analysis.py         descriptive and adjusted statistics
        modeling.py         temporal evaluation and risk scoring
        visualization.py    publication-ready charts
        pipeline.py         end-to-end orchestration and CLI
tests/                      focused tests for high-risk assumptions
data/
    raw/                    downloaded source CSVs; not committed
    processed/              regenerated analysis cohort; not committed
outputs/
    figures/                generated portfolio figures
    metrics.json            key analysis and model results
    population_flow.csv     row-count audit from raw to final cohort
    delivery_drift.csv      monthly target and promised-window audit
    seller_count_diagnostic.csv
    dissatisfaction_sensitivity.csv
    model_validation_metrics.csv
    model_holdout_intervals.csv
    model_feature_importance.csv
    logistic_coefficients.csv
pyproject.toml              package metadata and dependencies
```

## Reproduce the project

Python 3.11 or newer is required.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the package and test dependencies:

```bash
python -m pip install -e ".[dev]"
```

Download the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
and place these files in `data/raw/`:

```text
olist_customers_dataset.csv
olist_geolocation_dataset.csv
olist_order_items_dataset.csv
olist_order_reviews_dataset.csv
olist_orders_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
```

Run the complete pipeline:

```bash
python -m olist_delivery.pipeline
```

The package also registers the shorter console command on systems that permit
Python-generated executable wrappers:

```bash
olist-delivery
```

Run the tests:

```bash
pytest
```

The full pipeline runs in approximately 75 seconds on the development machine.
It regenerates `data/processed/olist_delivery_features.csv` and every file under
`outputs/`.

## Limitations

- Reviews are available only for orders that received a usable survey response.
- Straight-line seller-to-customer distance is a geographic proxy, not route
  distance.
- The held-out period has a much lower late-delivery rate than training. The
  drift analysis shows that changing promised windows explain part, but not all,
  of this non-stationarity.
- Held-out probabilities are imperfectly calibrated, particularly in the
  highest-risk band; the model is more defensible for ranking than for direct
  probability-based intervention.
- The historical Olist dataset may not reflect current marketplace operations.

## License

The project code is released under the [MIT License](LICENSE). The source data
is distributed separately by Olist through Kaggle and is not included here.
