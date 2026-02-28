import numpy as np
import pandas as pd

from analytics.audit_engine import (approval_gap_shift, equal_opportunity_difference, guardrail_alerts,
                                    population_stability_index, review_load)


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(42)
    baseline = rng.normal(loc=0.3, scale=0.1, size=2000)
    current = rng.normal(loc=0.3, scale=0.1, size=500)

    assert population_stability_index(baseline, current) < 0.05


def test_psi_is_large_for_a_clearly_shifted_distribution():
    rng = np.random.default_rng(7)
    baseline = rng.normal(loc=0.3, scale=0.05, size=2000)
    shifted = rng.normal(loc=0.9, scale=0.05, size=500)

    assert population_stability_index(baseline, shifted) > 0.25


def test_psi_handles_a_baseline_with_no_spread():
    baseline = np.full(100, 0.5)
    current = np.full(50, 0.5)

    assert population_stability_index(baseline, current) == 0.0


def test_equal_opportunity_difference_is_the_true_positive_rate_gap():
    day = pd.DataFrame({
        'gender':          ['Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Female'],
        'actual_label':    [1, 1, 1, 0, 1, 1, 1, 0],
        'predicted_class': [1, 1, 1, 0, 1, 0, 0, 1],
    })
    # Male: 3 of 3 positives found; Female: 1 of 3
    assert abs(equal_opportunity_difference(day) - (1.0 - 1 / 3)) < 1e-9


def test_guardrail_alerts_fire_only_past_their_thresholds():
    audit = pd.DataFrame([
        {'audit_date': '2026-01-01', 'drift_p_value': 0.5, 'psi_score': 0.02, 'disparate_impact_ratio': 0.97, 'equal_opportunity_difference': 0.01, 'approval_gap_z': 0.4},
        {'audit_date': '2026-01-02', 'drift_p_value': 0.001, 'psi_score': 0.3, 'disparate_impact_ratio': 0.7, 'equal_opportunity_difference': -0.2, 'approval_gap_z': 3.5},
    ])
    alerts = guardrail_alerts(audit)
    assert set(alerts['audit_date']) == {'2026-01-02'}
    assert set(alerts['rule']) == {'ks_drift', 'psi_major_shift', 'disparate_impact', 'equal_opportunity', 'approval_gap_shift'}


def test_review_load_counts_only_alert_days_and_rows():
    audit = pd.DataFrame({'audit_date': ['d1', 'd2', 'd3', 'd4']})
    alerts = pd.DataFrame({'audit_date': ['d2', 'd2', 'd4'], 'rule': ['ks_drift', 'psi_major_shift', 'ks_drift']})
    load = review_load(audit, alerts)
    assert load['alert_days'] == 2 and load['days_saved'] == 0.5
    assert load['metric_values'] == 24 and load['alert_rows'] == 3 and load['values_saved'] == 0.875


def test_retraining_after_the_alert_recovers_a_shifted_boundary():
    from analytics.retrain_response import accuracy_loss
    rng = np.random.default_rng(0)
    rows = []
    for day in range(30):
        x1 = rng.normal(0, 1, 200); x2 = rng.normal(0, 1, 200)
        boundary = 0.0 if day < 10 else 1.0          # the true boundary moves on day 10
        rows.append(pd.DataFrame({"day": day, "input_feature_1": x1, "input_feature_2": x2,
                                  "predicted_class": (x1 > 0).astype(int), "actual_label": (x1 > boundary).astype(int)}))
    df = pd.concat(rows, ignore_index=True)
    result = accuracy_loss(df, first_alert_day=10, lag=2, window=3)
    assert result["loss_with_retraining"] < result["loss_without_response"]
    assert result["final_accuracy_with"] > 0.95


def _decisions(male_approved, male_total, female_approved, female_total):
    return pd.DataFrame({
        'gender': ['Male'] * male_total + ['Female'] * female_total,
        'predicted_class': [1] * male_approved + [0] * (male_total - male_approved)
                           + [1] * female_approved + [0] * (female_total - female_approved),
    })


def test_approval_gap_shift_catches_a_widening_the_four_fifths_ratio_misses():
    baseline = _decisions(1995, 2000, 1994, 2000)   # 99.75% vs 99.70%
    window = _decisions(1990, 2000, 1940, 2000)     # 99.5% vs 97.0%
    assert (1940 / 2000) / (1990 / 2000) > 0.8      # the four-fifths rule still passes
    shift, z = approval_gap_shift(baseline, window)
    assert abs(shift - 2.45) < 1e-9
    assert z > 3


def test_approval_gap_shift_stays_quiet_on_noise_and_when_the_gap_favours_the_unprivileged_group():
    baseline = _decisions(1995, 2000, 1994, 2000)
    assert abs(approval_gap_shift(baseline, _decisions(1996, 2000, 1993, 2000))[1]) < 1
    assert approval_gap_shift(baseline, _decisions(1940, 2000, 1990, 2000))[1] < 0
