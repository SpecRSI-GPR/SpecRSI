import pytest
from specrsi.statistics import aubc, calibration, cost_endpoints, paired_bootstrap


def test_clustered_pairing_uses_replicas_not_search_count():
    result = paired_bootstrap([1, 3, 10], [0, 0, 0], clusters=["a", "a", "b"], resamples=1000)
    assert result.independent_units == 2
    assert result.mean_difference == 6
    assert result.family_interval[0] <= result.interval[0] <= result.interval[1] <= result.family_interval[1]


def test_cost_endpoints_keep_failure_bill():
    assert cost_endpoints(0.7, 2, None) == {"actual_C": .7, "hit_H": 0, "deadline_L": 2}
    with pytest.raises(ValueError):
        cost_endpoints(.7, 2, .8)


def test_aubc_complete_incumbent_steps():
    assert aubc([(0, 1), (.5, 2), (1, 3)], 2) == 4.5
    with pytest.raises(ValueError):
        aubc([(.1, 1)], 2)


def test_unknown_checker_attempt_is_not_a_label():
    assert calibration([0, 1], [0, 1])["brier"] == 0
    with pytest.raises(ValueError):
        calibration([.5, .6], [0, -1])
