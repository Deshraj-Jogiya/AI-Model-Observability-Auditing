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
> 0.25, disparate impact outside 0.8 to 1.25, true-positive-rate gap over 10 points, 7-day approval gap more than 3 standard
errors wider than its baseline). The count compares reading only the days and
alerts raised against reading every daily metric. Re-run with `python etl/log_ingestion.py` then `python analytics/audit_engine.py`.

| Metric | Value |
|---|---|
| Days that raised an alert | 21 of 60 |
| Fewer days to review | 65% |
| Alerts to read | 29 instead of 360 metric values |
| Days with a KS drift alert | 21 |
| Days with a major PSI shift (> 0.25) | 4 |
| Days with an approval-gap alert | 4 |

**Decision.** The reviewer works from the `audit_alerts` table, not the full daily metrics. The KS test raised its first alert
on day 40 and PSI on day 54, so the KS test is the earlier signal on this drift. (The simulated dates move with the run date, so
results are given by day number.)

**Source.** Output of `analytics/audit_engine.py` (the "Review load" and "Guardrail alert" lines), `data/fairness_audit_history.csv`.

## Does the four-fifths rule notice when the model starts to favour one group?

**How it was tested.** The disparate-impact ratio (female approval rate over male) stayed between 0.96 and 1.01 on all 60 days, so
the four-fifths rule never fired, while accuracy fell 33.9 points. The approval rates explain why: almost everyone is approved, so
the ratio stays near 1 even when one group's rate falls several times faster. Both groups were approved 99.8% of the time over the
baseline days; over the 7 days ending on day 55, women were approved 97.4% of the time and men 99.1%. A new check, `approval_gap_shift` in `analytics/audit_engine.py`,
pools the last 7 days, takes the male-minus-female approval gap, subtracts the same gap over the 5 baseline days, and divides by
the standard error of the two gaps. It alerts when that z-score is above 3, and only for a gap widening against women. Tests in
`tests/test_audit_engine.py` cover a case where the ratio passes the four-fifths rule and the new check fires.

| Metric | Value |
|---|---|
| Disparate-impact ratio, lowest day | 0.963 (rule needs < 0.8) |
| Four-fifths alerts | 0 |
| Approval-gap z-score, highest before the drift (days 1-30) | 0.88 |
| Approval-gap alerts | 4 days (days 52-55) |
| Largest gap shift | 1.65 points wider than baseline (day 55) |

**Decision.** The approval-gap check stays alongside the four-fifths rule: the ratio answers the legal test, the gap shift
answers whether the model is drifting toward one group. It fired 12 days after the first KS drift alert and went quiet again on
day 56 as the 7-day gap fell back below 3 standard errors, so it is a second signal for the reviewer, not a replacement for the
drift alert. On simulated data with 250 decisions a day, a smaller shift would need a longer window to separate from noise.

**Source.** `fairness_audit_history.approval_gap_shift` and `approval_gap_z`, and the `audit_alerts` rows with rule
`approval_gap_shift`, written by `analytics/audit_engine.py`.
