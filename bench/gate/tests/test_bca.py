"""BCa coverage on lognormal samples. Marked slow."""

import numpy as np
import pytest

from rggate.bootstrap import paired_intervals, ratio_intervals
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


@pytest.mark.slow
def test_cluster_bca_coverage_with_a_system_effect():
    # Five starts per system. A system effect of sd 0.5 and a within
    # effect of sd 0.3. The log-ratio mean is 0, so the ratio is 1.
    data = philox(13, 4)
    replicates = 300
    resamples = 999
    floors = {10: 0.85, 20: 0.90}
    for n_systems, floor in floors.items():
        median_hits = 0
        geomean_hits = 0
        for index in range(replicates):
            groups = []
            for _system in range(n_systems):
                effect = float(data.normal()) * 0.5
                groups.append(effect + data.normal(size=5) * 0.3)
            interval = paired_intervals(
                groups,
                seed=1,
                n_resamples=resamples,
                confidence_level=0.95,
                pair_index=n_systems * replicates + index,
            )
            if interval["cluster_median_low"] <= 1.0 <= interval["cluster_median_high"]:
                median_hits += 1
            if interval["cluster_geomean_low"] <= 1.0 <= interval["cluster_geomean_high"]:
                geomean_hits += 1
        assert median_hits / replicates >= floor
        assert geomean_hits / replicates >= floor
