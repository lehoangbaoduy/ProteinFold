"""Behaviour tests for hpfold_tools.visualize (PRD FR-A5, plan T1.9)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from hpfold.hp_model import contacts, fold_coords
from hpfold_tools import visualize
from hpfold_tools.visualize import plot_fold, save_fold

SANITY_SEQ = "HPHHHHPHHP"
SANITY_MOVES = "FLLFRRFF"  # E = -3, contacts (0,5) (3,8) (4,7)
SEQ_3D = "HHHHHHHHHH"
MOVES_3D = "UULDRRFF"


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


def _by_gid(ax, gid):
    return [ln for ln in ax.get_lines() if ln.get_gid() == gid]


def _collection(ax, gid):
    return next(c for c in ax.collections if c.get_gid() == gid)


def _segment(line, dim):
    data = line.get_data_3d() if dim == 3 else line.get_xydata().T
    return frozenset(tuple(round(float(v)) for v in point) for point in zip(*data))


def _expected_segments(seq, moves, dim):
    coords = fold_coords(seq, moves)
    pick = (lambda c: c) if dim == 3 else (lambda c: c[:2])
    return {frozenset({pick(coords[i]), pick(coords[j])}) for i, j in contacts(seq, coords)}


@pytest.mark.parametrize("seq,moves,dim", [(SANITY_SEQ, SANITY_MOVES, 2), (SEQ_3D, MOVES_3D, 3)])
def test_contact_lines_join_exactly_the_contact_pairs(seq, moves, dim):
    ax = plot_fold(seq, moves, dim)
    lines = _by_gid(ax, "contact")
    segments = [_segment(ln, dim) for ln in lines]
    assert len(segments) == len(set(segments)), "duplicate contact lines"
    assert set(segments) == _expected_segments(seq, moves, dim)
    assert segments, "test fold should have contacts"
    assert all(ln.get_linestyle() == "--" for ln in lines)


def test_beads_sit_on_fold_coordinates_and_start_is_ringed():
    ax = plot_fold(SANITY_SEQ, SANITY_MOVES, 2)
    expected = np.array([c[:2] for c in fold_coords(SANITY_SEQ, SANITY_MOVES)])
    assert np.array_equal(_collection(ax, "beads").get_offsets(), expected)
    assert np.array_equal(_collection(ax, "start").get_offsets(), expected[:1])


def test_h_is_filled_dark_and_p_is_hollow_white():
    ax = plot_fold(SANITY_SEQ, SANITY_MOVES, 2)
    faces = _collection(ax, "beads").get_facecolors()
    for i, residue in enumerate(SANITY_SEQ):
        expected = visualize.H_FACE if residue == "H" else visualize.P_FACE
        assert np.allclose(faces[i], mcolors.to_rgba(expected)), (i, residue)


def test_title_reports_energy_and_validity():
    ax = plot_fold(SANITY_SEQ, SANITY_MOVES, 2, title="demo")
    assert "demo" in ax.get_title() and "E = -3" in ax.get_title()
    assert "invalid" in plot_fold("PPPPPPP", "LLLFF", 2).get_title().lower()


def test_3d_plot_uses_3d_axes_with_equal_box_aspect():
    ax = plot_fold(SEQ_3D, MOVES_3D, 3)
    assert ax.name == "3d"
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    spans = [hi - lo for lo, hi in (xlim, ylim, zlim)]
    assert np.allclose(ax.get_box_aspect() / max(ax.get_box_aspect()), np.array(spans) / max(spans), atol=0.05)


@pytest.mark.parametrize("bad_seq", ["hphh", "HXPH", ""])
def test_sequence_validated_and_no_figure_leaks(bad_seq):
    with pytest.raises(ValueError, match="sequence"):
        plot_fold(bad_seq, "FF"[: max(len(bad_seq) - 2, 0)], 2)
    assert plt.get_fignums() == []


def test_moves_must_match_dim():
    with pytest.raises(ValueError, match="not allowed in 2D"):
        plot_fold("HHHH", "UU", 2)
    assert plt.get_fignums() == []


def test_wrong_move_count_rejected_without_leaking():
    with pytest.raises(ValueError, match="n-2"):
        plot_fold("HHHHH", "FF", 2)
    assert plt.get_fignums() == []


def test_axes_must_match_dim():
    _, ax2d = plt.subplots()
    with pytest.raises(ValueError, match="3D axes"):
        plot_fold(SEQ_3D, MOVES_3D, 3, ax=ax2d)


def test_plot_into_existing_axes_opens_no_new_figure():
    _, ax = plt.subplots()
    assert plot_fold(SANITY_SEQ, SANITY_MOVES, 2, ax=ax) is ax
    assert len(plt.get_fignums()) == 1


@pytest.mark.parametrize("dim,seq,moves,suffix", [(2, SANITY_SEQ, SANITY_MOVES, ".png"), (3, SEQ_3D, MOVES_3D, ".svg")])
def test_save_fold_writes_file_and_closes_figure(tmp_path, dim, seq, moves, suffix):
    out = save_fold(seq, moves, dim, tmp_path / "nested" / f"fold{suffix}", title="saved")
    assert out == tmp_path / "nested" / f"fold{suffix}"
    assert out.is_file() and out.stat().st_size > 1000
    assert plt.get_fignums() == []


def test_save_fold_requires_a_file_suffix(tmp_path):
    with pytest.raises(ValueError, match="suffix"):
        save_fold(SANITY_SEQ, SANITY_MOVES, 2, tmp_path / "noext")
    assert plt.get_fignums() == []


def test_save_fold_closes_figure_when_writing_fails(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", boom)
    with pytest.raises(OSError):
        save_fold(SANITY_SEQ, SANITY_MOVES, 2, tmp_path / "f.png")
    assert plt.get_fignums() == []
