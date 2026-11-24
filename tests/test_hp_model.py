"""Unit tests for hpfold.hp_model (PRD FR-1..FR-4, FR-8; edge cases E1-E3)."""

import itertools

import pytest

from hpfold.hp_model import (
    MOVES_2D,
    MOVES_3D,
    collisions,
    contacts,
    decode,
    encode,
    energy,
    fold_coords,
    is_valid,
    parse_sequence,
)


def brute_energy(seq, coords):
    """Independent O(n^2) reference: count H-H pairs at unit distance that are not chain neighbours."""
    e = 0
    for i, j in itertools.combinations(range(len(seq)), 2):
        adjacent = sum(abs(a - b) for a, b in zip(coords[i], coords[j])) == 1
        if j > i + 1 and seq[i] == seq[j] == "H" and adjacent:
            e -= 1
    return e


# Hand-checked example from docs/HANDOFF.md.
SANITY_SEQ = "HPHHHHPHHP"
SANITY_COORDS_2D = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2)]
SANITY_MOVES = "FLLFRRFF"  # bonds +x,+x,+y,-x,-x,+y,+x,+x,+x


# ---------- parse_sequence (FR-8, E1, E2) ----------

def test_parse_accepts_plain_sequence():
    assert parse_sequence("HPPH") == "HPPH"


def test_parse_normalizes_case_and_whitespace():
    assert parse_sequence("  hp ph\nH\tp ") == "HPPHHP"


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_parse_rejects_empty(bad):
    with pytest.raises(ValueError, match="empty"):
        parse_sequence(bad)


@pytest.mark.parametrize("bad", ["HPX", "HP1", "H-P", "HPÉ", "<script>"])
def test_parse_rejects_invalid_characters(bad):
    with pytest.raises(ValueError, match="H and P"):
        parse_sequence(bad)


def test_parse_rejects_non_string():
    with pytest.raises(TypeError):
        parse_sequence(None)  # type: ignore[arg-type]


# ---------- decode / encode (FR-1, FR-2) ----------

def test_alphabets():
    assert MOVES_2D == "FLR"
    assert MOVES_3D == "FLRUD"


def test_decode_fixes_first_bond_along_x():
    assert decode("") == ((0, 0, 0), (1, 0, 0))


def test_sanity_fold_decodes_to_handoff_coordinates():
    assert decode(SANITY_MOVES) == tuple((x, y, 0) for x, y in SANITY_COORDS_2D)


def test_decode_returns_immutable_tuples():
    coords = decode("FL")
    assert isinstance(coords, tuple) and all(isinstance(c, tuple) for c in coords)


def test_decode_rejects_unknown_move():
    with pytest.raises(ValueError, match="move"):
        decode("FLX")


def test_left_and_right_turns_in_2d():
    assert decode("L")[-1] == (1, 1, 0)
    assert decode("R")[-1] == (1, -1, 0)


def test_3d_frame_up_down_and_left_after_pitch():
    # U pitches toward +z; after pitching up, "left" (up x heading) is +y.
    assert decode("U") == ((0, 0, 0), (1, 0, 0), (1, 0, 1))
    assert decode("D")[-1] == (1, 0, -1)
    assert decode("UL")[-1] == (1, 1, 1)
    # Two ups make a U-turn back toward -x, one level higher.
    assert decode("UU")[-1] == (0, 0, 1)


@pytest.mark.parametrize(
    "moves,last",
    [
        # Hand-derived with left = up x heading, U -> (up, -heading), D -> (-up, heading).
        ("DL", (1, 1, -1)),
        ("DR", (1, -1, -1)),
        ("UR", (1, -1, 1)),
        ("LU", (1, 1, 1)),
        ("LD", (1, 1, -1)),
        ("DD", (0, 0, -1)),   # two pitches down make a U-turn one level lower
        ("DDL", (0, 1, -1)),  # ...and the frame is now upside down: left is still +y
        ("UDL", (2, 1, 1)),   # U then D restores the original frame (left = +y)
        ("DUL", (2, 1, -1)),
    ],
)
def test_3d_frame_hand_derived(moves, last):
    assert decode(moves)[-1] == last


def test_decode_with_dim_checks_alphabet():
    assert decode("FLR", dim=2) == decode("FLR")
    with pytest.raises(ValueError, match="not allowed in 2D"):
        decode("FU", dim=2)


def test_2d_moves_never_leave_plane():
    for moves in itertools.product(MOVES_2D, repeat=6):
        assert all(z == 0 for _, _, z in decode("".join(moves)))


def test_encode_matches_hand_derived_moves():
    assert encode(SANITY_COORDS_2D) == SANITY_MOVES


def test_encode_accepts_translated_input():
    shifted = [(x + 5, y - 3) for x, y in SANITY_COORDS_2D]
    assert encode(shifted) == SANITY_MOVES


def test_encode_rejects_non_canonical_first_bond():
    with pytest.raises(ValueError, match=r"\+x"):
        encode([(0, 0), (0, 1), (1, 1)])


def test_encode_rejects_non_unit_bond():
    with pytest.raises(ValueError, match="bond"):
        encode([(0, 0), (1, 0), (3, 0)])


@pytest.mark.parametrize("alphabet", [MOVES_2D, MOVES_3D])
def test_roundtrip_all_short_move_strings(alphabet):
    for moves in itertools.product(alphabet, repeat=4):
        m = "".join(moves)
        assert encode(decode(m)) == m


# ---------- fold_coords (E3) ----------

def test_fold_coords_length_matches_sequence():
    assert len(fold_coords("HPPHH", "FLR")) == 5


@pytest.mark.parametrize("seq", ["H", "HP"])
def test_short_sequences_are_trivial(seq):
    coords = fold_coords(seq, "")
    assert len(coords) == len(seq)
    assert energy(seq, coords) == 0


def test_fold_coords_rejects_wrong_move_count():
    with pytest.raises(ValueError, match="n-2"):
        fold_coords("HPHP", "F")


# ---------- validity, collisions, contacts, energy (FR-3, FR-4) ----------

def test_sanity_fold_energy_is_minus_3():
    coords = decode(SANITY_MOVES)
    assert is_valid(coords)
    assert collisions(coords) == 0
    assert energy(SANITY_SEQ, coords) == -3
    assert contacts(SANITY_SEQ, coords) == ((0, 5), (3, 8), (4, 7))


def test_straight_chain_has_no_contacts():
    assert energy("HHHHH", decode("FFF")) == 0


def test_bonded_neighbours_never_count():
    coords = decode("LL")  # (0,0) (1,0) (1,1) (0,1): only 0-3 is a topological contact
    assert contacts("HHHH", coords) == ((0, 3),)
    assert energy("HHHP", coords) == 0


def test_only_h_h_pairs_count():
    coords = decode("LL")
    assert energy("HPPP", coords) == 0
    assert energy("PHHP", coords) == 0


def test_collision_detected():
    coords = decode("LLL")  # closes a square: bead 4 lands on the origin
    assert coords[4] == coords[0]
    assert not is_valid(coords)
    assert collisions(coords) == 1


def test_energy_matches_brute_force_exhaustively_2d():
    # Every 2D fold of the 10-mer (3^8 = 6561 move strings), valid or not.
    for moves in itertools.product(MOVES_2D, repeat=len(SANITY_SEQ) - 2):
        coords = decode("".join(moves))
        if is_valid(coords):
            assert energy(SANITY_SEQ, coords) == brute_energy(SANITY_SEQ, coords)


def test_energy_matches_brute_force_exhaustively_3d():
    seq = "HHPHPHHP"  # 5^6 = 15625 move strings
    for moves in itertools.product(MOVES_3D, repeat=len(seq) - 2):
        coords = decode("".join(moves))
        if is_valid(coords):
            assert energy(seq, coords) == brute_energy(seq, coords)


def test_sanity_sequence_optimum_is_at_most_minus_3():
    best = min(
        energy(SANITY_SEQ, c)
        for moves in itertools.product(MOVES_2D, repeat=len(SANITY_SEQ) - 2)
        if is_valid(c := decode("".join(moves)))
    )
    assert best <= -3


def test_contacts_requires_matching_lengths():
    with pytest.raises(ValueError, match="length"):
        contacts("HPH", decode("FF"))
