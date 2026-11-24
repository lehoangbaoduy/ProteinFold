"""Hypothesis property tests (fuzzing) for the HP model core (master prompt, Q5)."""

import itertools

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from hpfold.folds import repair
from hpfold.hp_model import (
    INITIAL_HEADING,
    INITIAL_UP,
    MOVES_2D,
    MOVES_3D,
    decode,
    encode,
    energy,
    is_valid,
    parse_sequence,
    step_frame,
)

moves_2d = st.text(alphabet=MOVES_2D, max_size=60)
moves_3d = st.text(alphabet=MOVES_3D, max_size=60)
any_moves = st.one_of(moves_2d, moves_3d)


def brute_energy(seq, coords):
    e = 0
    for i, j in itertools.combinations(range(len(seq)), 2):
        if j > i + 1 and seq[i] == seq[j] == "H":
            if sum(abs(a - b) for a, b in zip(coords[i], coords[j])) == 1:
                e -= 1
    return e


@given(any_moves)
def test_bonds_have_unit_length(moves):
    coords = decode(moves)
    for a, b in zip(coords, coords[1:]):
        assert sum(abs(p - q) for p, q in zip(a, b)) == 1


@given(any_moves)
def test_no_immediate_reversal(moves):
    coords = decode(moves)
    assert all(coords[i] != coords[i + 2] for i in range(len(coords) - 2))


@given(any_moves)
def test_encode_inverts_decode(moves):
    assert encode(decode(moves)) == moves


@given(any_moves)
def test_frame_stays_orthonormal(moves):
    heading, up = INITIAL_HEADING, INITIAL_UP
    for m in moves:
        heading, up = step_frame(heading, up, m)
        assert sum(h * h for h in heading) == 1
        assert sum(u * u for u in up) == 1
        assert sum(h * u for h, u in zip(heading, up)) == 0


@given(moves_2d)
def test_2d_stays_in_plane(moves):
    assert all(z == 0 for _, _, z in decode(moves))


@given(any_moves, st.data())
def test_energy_nonpositive_and_matches_brute_force(moves, data):
    coords = decode(moves)
    seq = data.draw(st.text(alphabet="HP", min_size=len(coords), max_size=len(coords)))
    e = energy(seq, coords)
    assert e <= 0
    if is_valid(coords):
        assert e == brute_energy(seq, coords)


@settings(max_examples=200, deadline=None)
@given(
    st.one_of(
        st.tuples(st.text(alphabet=MOVES_2D, max_size=150), st.just(2)),
        st.tuples(st.text(alphabet=MOVES_3D, max_size=150), st.just(3)),
    ),
    st.integers(0, 2**32 - 1),
)
def test_repair_always_valid(moves_dim, seed):
    moves, dim = moves_dim
    fixed = repair(moves, np.random.default_rng(seed), dim)
    assert len(fixed) == len(moves)
    assert is_valid(decode(fixed))


@given(st.text(max_size=200))
def test_parse_sequence_never_crashes_unexpectedly(s):
    # Fuzz the input boundary: either a clean H/P string or a ValueError, nothing else.
    try:
        out = parse_sequence(s)
    except ValueError:
        return
    assert out and set(out) <= {"H", "P"}
