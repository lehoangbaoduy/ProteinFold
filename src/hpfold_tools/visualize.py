"""Fold plots for the paper and talk (PRD FR-A5). Native-only tooling (matplotlib).

Styling: the chain is a grey line; H beads are filled dark circles and P beads
hollow white circles (distinguished by fill, not by hue alone); topological
H-H contacts are dashed red lines; bead 0 (the N-terminus) is ringed. 2D folds
are drawn on an equal-aspect integer grid, 3D folds on 3D axes with an equal
box aspect so lattice bonds all look the same length.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from hpfold.hp_model import (
    check_moves,
    contacts,
    energy,
    fold_coords,
    is_valid,
    parse_sequence,
)

H_FACE = "#1f1f1f"
P_FACE = "#ffffff"
EDGE = "#1f1f1f"
CHAIN = "#7a7a7a"
CONTACT = "#d62728"
START_RING = "#1f77b4"
GRID = "#e6e6e6"

BEAD_SIZE = 150
START_RING_SIZE = 340
BEAD_EDGE_WIDTH = 1.5
CHAIN_WIDTH = 2.5
CONTACT_WIDTH = 2.2
START_RING_WIDTH = 2.0
Z_CONTACT, Z_CHAIN, Z_BEADS, Z_START = 1, 2, 3, 4

AXIS_MARGIN = 1  # empty lattice cells shown around a 2D fold
BOX_PAD = 0.5  # half a cell of padding on each side of a 3D fold
FIG_SIZE = (6, 6)
SAVE_DPI = 200


def plot_fold(sequence: str, moves: str, dim: int, ax=None, title: str | None = None):
    """Draw one fold and return the matplotlib Axes.

    If `ax` is None a new pyplot figure is created; the caller owns it and should
    close it (or use `save_fold`, which does). A provided `ax` must be 3D for dim=3
    and 2D for dim=2.
    """
    seq = parse_sequence(sequence)
    if seq != sequence:
        raise ValueError("sequence must already be uppercase H/P without whitespace")
    check_moves(moves, dim)
    coords = fold_coords(sequence, moves)
    if ax is not None:
        _check_axes(ax, dim)
        _draw(ax, sequence, coords, dim, title)
        return ax

    fig = plt.figure(figsize=FIG_SIZE)
    try:
        ax = fig.add_subplot(projection="3d" if dim == 3 else None)
        _draw(ax, sequence, coords, dim, title)
    except Exception:
        plt.close(fig)
        raise
    return ax


def save_fold(sequence: str, moves: str, dim: int, path: str | Path, title: str | None = None) -> Path:
    """Plot a fold and write it to `path`; the format comes from the suffix (.png, .svg, .pdf).

    Missing parent directories are created. The figure is always closed.
    """
    out = Path(path)
    if not out.suffix:
        raise ValueError(f"output path needs a file suffix such as .png or .svg: {out}")
    ax = plot_fold(sequence, moves, dim, title=title)
    fig = ax.get_figure()
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)
    return out


def _check_axes(ax, dim: int) -> None:
    is_3d = ax.name == "3d"
    if dim == 3 and not is_3d:
        raise ValueError("dim=3 needs 3D axes (projection='3d')")
    if dim == 2 and is_3d:
        raise ValueError("dim=2 needs 2D axes, got 3D axes")


def _draw(ax, sequence, coords, dim, title) -> None:
    pick = (lambda c: c) if dim == 3 else (lambda c: c[:2])
    for i, j in contacts(sequence, coords):
        pts = list(zip(pick(coords[i]), pick(coords[j])))
        ax.plot(*pts, linestyle="--", color=CONTACT, linewidth=CONTACT_WIDTH, zorder=Z_CONTACT, gid="contact")

    xyz = list(zip(*[pick(c) for c in coords]))
    ax.plot(*xyz, "-", color=CHAIN, linewidth=CHAIN_WIDTH, zorder=Z_CHAIN, gid="chain")
    faces = [H_FACE if r == "H" else P_FACE for r in sequence]
    extra = {"depthshade": False} if dim == 3 else {}
    ax.scatter(*xyz, s=BEAD_SIZE, c=faces, edgecolors=EDGE, linewidths=BEAD_EDGE_WIDTH,
               zorder=Z_BEADS, gid="beads", **extra)
    ax.scatter(*[[v[0]] for v in xyz], s=START_RING_SIZE, facecolors="none", edgecolors=START_RING,
               linewidths=START_RING_WIDTH, zorder=Z_START, gid="start", **extra)

    if dim == 2:
        _style_2d(ax, xyz)
    else:
        _style_3d(ax, xyz)
    label = f"E = {energy(sequence, coords)}" + ("" if is_valid(coords) else "  (invalid: self-collision)")
    ax.set_title(f"{title}\n{label}" if title else label)


def _style_2d(ax, xyz) -> None:
    xs, ys = xyz
    ax.set_aspect("equal")
    ax.set_xticks(range(min(xs) - AXIS_MARGIN, max(xs) + AXIS_MARGIN + 1))
    ax.set_yticks(range(min(ys) - AXIS_MARGIN, max(ys) + AXIS_MARGIN + 1))
    ax.grid(True, color=GRID)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)


def _style_3d(ax, xyz) -> None:
    spans = []
    for values, set_lim, axis in zip(xyz, (ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d),
                                     (ax.xaxis, ax.yaxis, ax.zaxis)):
        lo, hi = min(values) - BOX_PAD, max(values) + BOX_PAD
        set_lim(lo, hi)
        spans.append(hi - lo)
        axis.set_major_locator(MaxNLocator(integer=True))
    ax.set_box_aspect(spans)
