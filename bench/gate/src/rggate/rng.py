"""Counter-based Philox streams.

The published key is fixed. The seed is mixed into key word 1, because
Philox increments counter word 0: storing the seed there makes seed
``s + 1`` replay seed ``s`` one block later. Stream and pair index sit
in counter words 2 and 3. Words 0 and 1 stay 0 so draws can increment
them. A resample index does not depend on thread count or on how many
numbers another stream has drawn.
"""

import numpy as np
from numpy.random import Generator, Philox

# Two-word Philox key. The seed is XOR-ed into word 1 at use time.
PHILOX_KEY = np.array([0xA341316C4C2F6E9A, 0xC1B54A32B6C7E8D9], dtype=np.uint64)

STREAM_ORDER = 1
STREAM_MEDIAN = 2
STREAM_GEOMEAN = 3
STREAM_CLUSTER_MEDIAN = 4
STREAM_CLUSTER_GEOMEAN = 5
STREAM_WITHIN_MEDIAN = 6
STREAM_WITHIN_GEOMEAN = 7


def philox(seed, stream, index=0):
    """Generator for one (seed, stream, index) triple."""
    key = PHILOX_KEY.copy()
    key[1] = np.uint64(key[1]) ^ np.uint64(seed)
    counter = np.zeros(4, dtype=np.uint64)
    counter[2] = np.uint64(stream)
    counter[3] = np.uint64(index)
    return Generator(Philox(key=key, counter=counter))
