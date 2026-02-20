import numpy as np
import pandas as pd

from analytics.audit_engine import equal_opportunity_difference, guardrail_alerts, population_stability_index


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
        {'audit_date': '2026-01-01', 'drift_p_value': 0.5, 'psi_score': 0.02, 'disparate_impact_ratio': 0.97, 'equal_opportunity_difference': 0.01},
        {'audit_date': '2026-01-02', 'drift_p_value': 0.001, 'psi_score': 0.3, 'disparate_impact_ratio': 0.7, 'equal_opportunity_difference': -0.2},
    ])
    alerts = guardrail_alerts(audit)
    assert set(alerts['audit_date']) == {'2026-01-02'}
    assert set(alerts['rule']) == {'ks_drift', 'psi_major_shift', 'disparate_impact', 'equal_opportunity'}
