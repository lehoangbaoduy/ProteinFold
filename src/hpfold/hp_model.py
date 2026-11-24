"""HP lattice model: input parsing, move decoding/encoding, validity and energy.

Encoding (PRD §3)
-----------------
A fold of an n-bead chain is a string of n-2 *relative* moves. The first bond is
fixed (bead 0 at the origin, bead 1 at +x), which removes rotational symmetry.
Each later move is relative to the direction of the previous bond:

    F = forward, L = turn left, R = turn right        (2D and 3D)
    U = pitch up, D = pitch down                       (3D only)

To make L/R/U/D meaningful in 3D, the walker carries an orientation frame:
`heading` (direction of the last bond) and `up` (a unit vector perpendicular to
it). "left" = up x heading. 2D is the special case that never uses U/D: `up`
stays +z, so L/R are +/-90 degree turns in the xy-plane and z stays 0.

Energy (PRD §3)
---------------
E = -(number of topological H-H contacts): pairs (i, j) with j > i + 1, both H,
on adjacent lattice sites. Computed in O(n) with a site -> beads hash map.
"""

from __future__ import annotations

from collections.abc import Sequence

MOVES_2D = "FLR"
MOVES_3D = "FLRUD"
HP_ALPHABET = frozenset("HP")

Coord = tuple[int, int, int]

ORIGIN: Coord = (0, 0, 0)
INITIAL_HEADING: Coord = (1, 0, 0)
INITIAL_UP: Coord = (0, 0, 1)
NEIGHBOR_OFFSETS: tuple[Coord, ...] = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
)


# ---------------------------------------------------------------- input ----

def parse_sequence(text: str) -> str:
    """Normalize and validate an HP sequence: drop whitespace, uppercase, allow only H/P."""
    if not isinstance(text, str):
        raise TypeError(f"sequence must be a string, got {type(text).__name__}")
    seq = "".join(text.split()).upper()
    if not seq:
        raise ValueError("sequence is empty")
    bad = sorted(set(seq) - HP_ALPHABET)
    if bad:
        shown = ", ".join(repr(c) for c in bad[:5])
        raise ValueError(f"sequence may contain only H and P (found {shown})")
    return seq


def alphabet_for(dim: int) -> str:
    """Move alphabet for a lattice dimension (2 or 3)."""
    if dim == 2:
        return MOVES_2D
    if dim == 3:
        return MOVES_3D
    raise ValueError(f"dim must be 2 or 3, got {dim!r}")


# ------------------------------------------------------------- geometry ----

def _cross(a: Coord, b: Coord) -> Coord:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _neg(a: Coord) -> Coord:
    return (-a[0], -a[1], -a[2])


def add(a: Coord, b: Coord) -> Coord:
    """Vector sum of two lattice points."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def step_frame(heading: Coord, up: Coord, move: str) -> tuple[Coord, Coord]:
    """Apply one relative move to the (heading, up) frame; returns the new frame."""
    if move == "F":
        return heading, up
    if move == "L":
        return _cross(up, heading), up
    if move == "R":
        return _neg(_cross(up, heading)), up
    if move == "U":  # pitch up: new heading = old up, new up = old backward
        return up, _neg(heading)
    if move == "D":  # pitch down
        return _neg(up), heading
    raise ValueError(f"unknown move {move!r} (expected one of {MOVES_3D})")


def decode(moves: str, dim: int | None = None) -> tuple[Coord, ...]:
    """Relative-move string -> coordinates of len(moves) + 2 beads.

    If `dim` is given, the moves are first checked against that lattice's alphabet.
    The result may contain repeated sites if the fold self-collides; see `is_valid`.
    """
    if dim is not None:
        check_moves(moves, dim)
    heading, up = INITIAL_HEADING, INITIAL_UP
    coords = [ORIGIN, INITIAL_HEADING]
    for move in moves:
        heading, up = step_frame(heading, up, move)
        coords.append(add(coords[-1], heading))
    return tuple(coords)


def fold_coords(sequence: str, moves: str) -> tuple[Coord, ...]:
    """Coordinates for exactly len(sequence) beads; moves must have length max(n-2, 0)."""
    n = len(sequence)
    expected = max(n - 2, 0)
    if len(moves) != expected:
        raise ValueError(f"expected n-2 = {expected} moves for n = {n}, got {len(moves)}")
    return decode(moves)[:n]


def encode(coords: Sequence[Sequence[int]]) -> str:
    """Absolute coordinates (2D or 3D tuples) -> relative-move string. Inverse of `decode`.

    The chain is translated so bead 0 is at the origin; its first bond must point
    along +x (the canonical orientation `decode` produces).
    """
    pts = [tuple(p) + (0,) * (3 - len(p)) for p in coords]
    if len(pts) < 2:
        return ""
    x0, y0, z0 = pts[0]
    pts = [(x - x0, y - y0, z - z0) for x, y, z in pts]
    if pts[1] != INITIAL_HEADING:
        raise ValueError("encode expects the first bond to point along +x")
    heading, up = INITIAL_HEADING, INITIAL_UP
    out = []
    for a, b in zip(pts[1:], pts[2:]):
        bond = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        move = _move_for_bond(heading, up, bond)
        if move is None:
            raise ValueError(f"bond {bond} between {a} and {b} is not a unit forward/turn step")
        heading, up = step_frame(heading, up, move)
        out.append(move)
    return "".join(out)


def _move_for_bond(heading: Coord, up: Coord, bond: Coord) -> str | None:
    for move in MOVES_3D:
        if step_frame(heading, up, move)[0] == bond:
            return move
    return None


def check_moves(moves: str, dim: int) -> None:
    """Raise ValueError if `moves` uses letters outside the alphabet for `dim`."""
    allowed = alphabet_for(dim)
    bad = sorted(set(moves) - set(allowed))
    if bad:
        raise ValueError(f"move(s) {bad} not allowed in {dim}D (alphabet {allowed})")


# --------------------------------------------------------------- scoring ----

def is_valid(coords: Sequence[Coord]) -> bool:
    """Self-avoiding: every bead occupies a distinct lattice site."""
    return len(set(coords)) == len(coords)


def collisions(coords: Sequence[Coord]) -> int:
    """Number of beads placed on an already-occupied site (0 for valid folds)."""
    return len(coords) - len(set(coords))


def contacts(sequence: str, coords: Sequence[Coord]) -> tuple[tuple[int, int], ...]:
    """Topological H-H contacts as sorted (i, j) pairs with j > i + 1.

    For self-colliding folds a site may hold several beads; all are considered,
    so the count stays well defined for penalty-based fitness.
    """
    if len(sequence) != len(coords):
        raise ValueError(f"sequence length {len(sequence)} != number of beads {len(coords)}")
    site: dict[Coord, list[int]] = {}
    for i, c in enumerate(coords):
        site.setdefault(c, []).append(i)
    pairs = [
        (i, j)
        for i, c in enumerate(coords) if sequence[i] == "H"
        for d in NEIGHBOR_OFFSETS
        for j in site.get(add(c, d), ())
        if j > i + 1 and sequence[j] == "H"
    ]
    return tuple(sorted(pairs))


def energy(sequence: str, coords: Sequence[Coord]) -> int:
    """HP energy E = -(number of topological H-H contacts). Lower is better."""
    return -len(contacts(sequence, coords))
