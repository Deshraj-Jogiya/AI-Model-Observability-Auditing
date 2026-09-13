import numpy as np

from analytics.audit_engine import population_stability_index


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
