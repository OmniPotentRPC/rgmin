"""Kaplan-Meier against a hand-computed sample."""

import math

import pytest
from scipy.stats import norm

from rggate.km import kaplan_meier


def _loglog(survival, greenwood, z):
    log_s = math.log(survival)
    theta = math.log(-log_s)
    se = math.sqrt(greenwood) / abs(log_s)
    low = math.exp(-math.exp(theta + z * se))
    high = math.exp(-math.exp(theta - z * se))
    return low, high


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
    greenwood = [1.0 / (5 * 4)]
    greenwood.append(greenwood[-1])
    greenwood.append(greenwood[-1] + 1.0 / (3 * 2))
    greenwood.append(greenwood[-1] + 1.0 / (2 * 1))
    greenwood.append(greenwood[-1])
    z = float(norm.ppf(0.975))
    bands = [_loglog(s, g, z) for s, g in zip(expected, greenwood)]
    for point, band in zip(result["curve"], bands):
        assert point[2] == pytest.approx(band[0])
        assert point[3] == pytest.approx(band[1])
    cover = [point[0] for point, band in zip(result["curve"], bands) if band[0] <= 0.5 <= band[1]]
    assert result["median_low"] == cover[0]
    assert result["median_high"] == cover[-1]
