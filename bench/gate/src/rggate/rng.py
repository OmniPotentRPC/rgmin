"""Counter-based Philox streams.

The key is fixed. ``seed`` and ``stream`` are the counter, so a resample
index does not depend on thread count or on how many numbers another
stream has drawn.
"""

import numpy as np
from numpy.random import Generator, Philox

# Two-word Philox key. Do not change it: the golden output is tied to it.
PHILOX_KEY = np.array([0xA341316C4C2F6E9A, 0xC1B54A32B6C7E8D9], dtype=np.uint64)

STREAM_ORDER = 1
STREAM_MEDIAN = 2
STREAM_GEOMEAN = 3


def philox(seed, stream, index=0):
    """Generator for one (seed, stream, index) triple.

    ``index`` is the pair ordinal or the replicate number. The fourth
    counter word stays 0.
    """
    counter = np.zeros(4, dtype=np.uint64)
    counter[0] = np.uint64(seed)
    counter[1] = np.uint64(stream)
    counter[2] = np.uint64(index)
    return Generator(Philox(key=PHILOX_KEY.copy(), counter=counter))
