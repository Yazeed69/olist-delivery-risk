# Late-delivery risk model card

## Intended use

The model ranks Olist marketplace orders by the risk that customer delivery
will occur after the customer-facing estimated delivery date. It is designed
for retrospective portfolio analysis and as a prototype for capacity-constrained
operational outreach. It is not a production service or a causal estimator of
the effect of any intervention.

The saved model is fitted on every eligible delivered, single-seller order
after model selection and holdout evaluation are complete. Review availability
does not determine membership in the modeling cohort.

## Model and input contract

The selected estimator and its preprocessing are serialized together in
`outputs/late_delivery_model.joblib`. The adjacent JSON metadata declares the
feature order and training end date. Inputs must contain the following
checkout-time fields:

- order composition: item count, log price, freight value and ratio, weight;
- promise and geography: promised window, distance, same-state indicator,
  customer state;
- cyclical purchase-time encodings for hour, weekday, and month.

Actual handoff, delivery, and review fields are prohibited. See
`examples/scoring_input.csv` for a two-row schema example.

## Evaluation design

Models are selected on an intermediate chronological validation period and
evaluated once on the latest 20% of orders. The comparison includes a constant
prevalence baseline, a promised-window-only logistic baseline, full logistic
regression, and histogram gradient boosting. Expanding-window monthly
backtests provide a separate stability diagnostic.

The main outputs are:

- `outputs/model_validation_metrics.csv`;
- `outputs/model_holdout_intervals.csv`;
- `outputs/model_rolling_backtest.csv`;
- `outputs/model_feature_importance.csv`;
- `outputs/model_intervention_value.csv`.

## Decision scenario

The intervention table is a scenario, not a claim of realized savings. Its
default currency-neutral assumptions are:

- 25% of targeted late deliveries can be prevented;
- each late delivery costs 30 units;
- each intervention costs 1 unit.

The table reports net value and the break-even intervention cost at capacities
of 1%, 5%, 10%, and 20%. Teams should replace all three assumptions before
making a decision.

## Limitations and failure modes

- The data describes historical Brazilian marketplace activity from 2016–2018.
- Target prevalence and promise-setting policy change materially over time.
- The promised window encodes existing operational policy and is not an
  independent measure of fulfillment capability.
- Holdout probabilities require recalibration; ranking is more defensible than
  treating scores as stable probabilities.
- Monthly backtests show that ranking performance is not stable in every month.
- Straight-line distance is not route distance, and multi-seller orders are
  excluded because the source provides only one delivery timestamp per order.
- A useful score does not prove that outreach will prevent a late delivery.

## Monitoring and retraining

A production implementation should monitor monthly prevalence, PR-AUC, top-k
capture, calibration, feature missingness, and promised-window distribution.
Trigger investigation when performance leaves the historical backtest range or
when policy changes alter the meaning of the promised window. Recalibrate on a
recent labeled window before using probabilities for cost-based decisions, and
repeat temporal validation before replacing the model.
