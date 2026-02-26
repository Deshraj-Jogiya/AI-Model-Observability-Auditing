# Findings

Measured results from this pipeline. The 30,000 inference records are simulated, with drift starting on day 31, so the results
describe the method, not a production model.

## Does retraining after the first drift alert recover accuracy?

**How it was tested.** `analytics/retrain_response.py` compares two responses to the first drift alert over the rest of the 60
days: leaving the deployed model as it is, and retraining it daily on the latest week of outcomes that have arrived (labels lag
predictions by 7 days).

| Metric | Untouched model | Daily retraining |
|---|---|---|
| Accumulated accuracy loss | 3.5396 | 2.3856 |
| Final-day accuracy | 0.66 | 0.83 |
| Reduction in accumulated loss | | 0.326 |

Baseline accuracy before the drift was 0.9988, and the first alert fired on day 40.

**Decision.** Retraining on the delayed labels is triggered by the drift alert, which fires without needing outcome labels.

**Source.** `exports/retrain_response.csv` (written by `analytics/retrain_response.py`).

## How much review does the alert table save?

**How it was tested.** Each daily audit writes an `audit_alerts` row only when a rule crosses its threshold (KS p < 0.01, PSI
> 0.25, disparate impact outside 0.8 to 1.25, true-positive-rate gap over 10 points). The count compares reading only the days and
alerts raised against reading every daily metric. Re-run with `python etl/log_ingestion.py` then `python analytics/audit_engine.py`.

| Metric | Value |
|---|---|
| Days that raised an alert | 21 of 60 |
| Fewer days to review | 65% |
| Alerts to read | 25 instead of 300 metric values |
| Days with a KS drift alert | 21 |
| Days with a major PSI shift (> 0.25) | 4 |

**Decision.** The reviewer works from the `audit_alerts` table, not the full daily metrics. The KS test raised its first alert
on 2026-10-05 and PSI on 2026-10-19, so the KS test is the earlier signal on this drift.

**Source.** Output of `analytics/audit_engine.py` (the "Review load" and "Guardrail alert" lines), `data/fairness_audit_history.csv`.
