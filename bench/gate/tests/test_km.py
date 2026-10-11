"""Kaplan-Meier against a hand-computed sample."""

import math

import pytest

from rggate.km import kaplan_meier


def test_hand_example():
    # times 1,2,3,4,5 with events death, censor, death, death, censor.
    # At risk just before each time: 5, 4, 3, 2, 1.
    # S: 4/5, then unchanged at the censoring, then (4/5)*(2/3), then half of that.
    # Greenwood sum at the median: 1/(5*4) + 1/(3*2) + 1/(2*1) = 43/60.
    result = kaplan_meier([1, 2, 3, 4, 5], [1, 0, 1, 1, 0])
    survival = 1.0
    survival *= 1.0 - 1.0 / 5.0
    expected = [survival, survival]
    survival *= 1.0 - 1.0 / 3.0
    expected.append(survival)
    survival *= 1.0 - 1.0 / 2.0
    expected.extend([survival, survival])
    got = [point[1] for point in result["curve"]]
    assert got == pytest.approx(expected)
    assert [point[5] for point in result["curve"]] == [1, 0, 1, 1, 0]
    assert [point[4] for point in result["curve"]] == [5, 4, 3, 2, 1]
    assert result["median"] == 4
    greenwood = 1.0 / (5 * 4) + 1.0 / (3 * 2) + 1.0 / (2 * 1)
    se = math.sqrt((expected[3] ** 2) * greenwood)
    from scipy.stats import norm

    z = float(norm.ppf(0.975))
    low = max(0.0, expected[3] - z * se)
    high = min(1.0, expected[3] + z * se)
    _t, _s, band_low, band_high, _n, _d = result["curve"][3]
    assert band_low == pytest.approx(low)
    assert band_high == pytest.approx(high)
    assert result["median_low"] == 1
    assert result["median_high"] == 5
