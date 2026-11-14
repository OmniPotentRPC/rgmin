"""Philox keys do not replay a neighbouring seed, stream, or index."""

from rggate.rng import philox


def _raw(seed, stream, index, count=10_000):
    return philox(seed, stream, index).bit_generator.random_raw(count).tolist()


def test_seed_stream_and_index_are_disjoint():
    base = set(_raw(4, 1, 0))
    assert base.isdisjoint(_raw(5, 1, 0))
    assert base.isdisjoint(_raw(4, 2, 0))
    assert base.isdisjoint(_raw(4, 1, 1))
