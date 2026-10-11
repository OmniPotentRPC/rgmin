"""BCa coverage on lognormal samples. Marked slow."""

import numpy as np
import pytest

from rggate.bootstrap import ratio_intervals
from rggate.rng import philox


@pytest.mark.slow
def test_bca_coverage_lognormal():
    # Lognormal(0, 0.5) has median and geometric mean equal to 1.
    # 2000 replicates, nominal 95%. Coverage must stay within 2 points.
    data = philox(11, 4)
    median_hits = 0
    geomean_hits = 0
    replicates = 2000
    for index in range(replicates):
        sample = data.lognormal(mean=0.0, sigma=0.5, size=40)
        logs = np.log(sample)
        interval = ratio_intervals(
            logs,
            seed=1,
            n_resamples=999,
            confidence_level=0.95,
            pair_index=index,
        )
        if interval["median_low"] <= 1.0 <= interval["median_high"]:
            median_hits += 1
        if interval["geomean_low"] <= 1.0 <= interval["geomean_high"]:
            geomean_hits += 1
    median_coverage = median_hits / replicates
    geomean_coverage = geomean_hits / replicates
    assert abs(median_coverage - 0.95) <= 0.02
    assert abs(geomean_coverage - 0.95) <= 0.02
