"""Validate data/benchmarks.json (PRD FR-B1..B5, edge case E11; spec PFOLD-DATA-001)."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from hpfold.hp_model import parse_sequence

PATH = Path(__file__).resolve().parents[1] / "data" / "benchmarks.json"
ENERGY_KINDS = {"best_known", "optimal", "lower_bound"}


@pytest.fixture(scope="module")
def data():
    return json.loads(PATH.read_text(encoding="utf-8"))


def expand(notation: str) -> str:
    """Independent expander for the papers' notation: H3 = HHH, (HP)2 = HPHP."""
    def parse(i):
        out = ""
        while i < len(notation) and notation[i] != ")":
            if notation[i] == "(":
                inner, i = parse(i + 1)
                i += 1
            else:
                inner, i = notation[i], i + 1
            m = re.match(r"\d+", notation[i:])
            if m:
                i += len(m.group())
            out += inner * (int(m.group()) if m else 1)
        return out, i

    result, end = parse(0)
    assert end == len(notation), f"unbalanced notation {notation!r}"
    return result


def contact_upper_bound(seq: str, dim: int) -> int:
    """Max possible topological contacts. Both lattices are bipartite, so contacts only
    join an even-index H with an odd-index H; each H has at most 2 (2D) / 4 (3D)
    non-bonded neighbour sites, plus 1 extra for each chain end."""
    h_even = sum(1 for i, c in enumerate(seq) if c == "H" and i % 2 == 0)
    h_odd = sum(1 for i, c in enumerate(seq) if c == "H" and i % 2 == 1)
    per_h = 2 if dim == 2 else 4
    return per_h * min(h_even, h_odd) + 2


# Golden values: (id, length, E*, sha256(sequence)). Independently verified by the code
# reviewer against Thachuk et al. 2007 Table 1 (Europe PMC full-text XML, PMC2071922).
# Any edit to a sequence or energy must be re-verified against the source before updating.
GOLDEN = [
    ("S1-1", 20, -9, "2b756e5ff288deb25139c2c192f62fcd448644fca040b7eb07f5ac38c3b4e819"),
    ("S1-2", 24, -9, "2698d0599a72e05ba6044a486a67daeeefa3c179b9e98648562f1d78510c8303"),
    ("S1-3", 25, -8, "28323fc80185f526a319c2be4547a7f11c6afe4aed84b9d3d27c01f0e0eeff71"),
    ("S1-4", 36, -14, "9c806ba775fe6432553395503d7c95c1cd9948d6735175f8b6071649fd1f0e04"),
    ("S1-5", 48, -23, "fda06078a9a02a8234345c46fb8b262ae89c26f32a71ba80375358b13e733b44"),
    ("S1-6", 50, -21, "052aae3072d49976b5fc78c8de152b2715003e22c568f9e9196ab9dd4a62bf38"),
    ("S1-7", 60, -36, "f0d2e9a73a43c9c9c68ad4b1e3907049a3a92fc5d61f73241c827915b17c8404"),
    ("S1-8", 64, -42, "6db553ab9e6c6c1806c5a04df6c811e40987d87779bc8f4ce2a7ea094a775111"),
    ("S1-9", 85, -53, "84ab37fca4d4012eb48dcf98e6e4f2104bbe3f34b5dc84983b4d86555d9dc3a7"),
    ("S1-10", 100, -50, "ca534082d678e4a34b1e1353621c2e6e8239a4146ca17a5b3dcdc88e5982ff7a"),
    ("S1-11", 100, -48, "c12edf6dc926ab0a8b9984e41e3f15108449827f85ccb513fd2aa3bc6c0af5ee"),
    ("S2-1", 48, -32, "ec9a470104934bec3897fe2fcce82ae94eb04cde934c0e85ae16b31bda5412da"),
    ("S2-2", 48, -34, "00a7bd60bb55cda3de172b7726531535c43b2e6603fdfe778239ba94c775e05b"),
    ("S2-3", 48, -34, "a30d4c50319ff98c0627f07269699ccbc514364775004b35cc1f79007736fdb9"),
    ("S2-4", 48, -33, "18423cbb6fbaeac4e77d02637d60c5b758604744397e97ecf9fa15c12c69dba6"),
    ("S2-5", 48, -32, "18761fcce7744187ccaedf93f6c51f935614b10cc58f87fa95fd874347fcfa29"),
    ("S2-6", 48, -32, "7bc4caaaea7a49f91b38be6fa8b42b86a84292fe49a01aa30c92e5fd27af1419"),
    ("S2-7", 48, -32, "0d38e4ab126b8e8175845a21f6142178ea8405d63cde34e7fcb0fa6cf6818e1d"),
    ("S2-8", 48, -31, "799f662e08363dd404637d4c1ffc2dde27cfeb9ca2646cea965b43517d750f84"),
    ("S2-9", 48, -34, "ec5d3dbab9164b72f5e93c0a3a3b073bde4e2f40830e71ff2da69b9ff27affde"),
    ("S2-10", 48, -33, "3d1607be95e9f6e61929c804b015eb1be8f779aaf7615d6b5448fb01b1cd49b5"),
]


@pytest.mark.parametrize("rid,length,energy,digest", GOLDEN)
def test_matches_verified_source_values(data, rid, length, energy, digest):
    b = next(x for x in data["instances"] if x["id"] == rid)
    assert b["length"] == length
    assert b["best_known_energy"] == energy
    assert hashlib.sha256(b["sequence"].encode()).hexdigest() == digest


def test_top_level_structure(data):
    assert set(data) >= {"instances", "sources"}
    assert data["instances"] and data["sources"]


def test_expected_instance_sets(data):
    ids = [b["id"] for b in data["instances"]]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert sorted(i for i in ids if i.startswith("S1-")) == sorted(f"S1-{k}" for k in range(1, 12))
    assert sorted(i for i in ids if i.startswith("S2-")) == sorted(f"S2-{k}" for k in range(1, 11))


def test_every_instance_is_well_formed(data):
    for b in data["instances"]:
        assert set(b) >= {"id", "dim", "length", "sequence", "notation", "best_known_energy",
                          "energy_kind", "sources", "notes"}, b["id"]
        assert b["dim"] == (2 if b["id"].startswith("S1-") else 3), b["id"]
        assert parse_sequence(b["sequence"]) == b["sequence"], b["id"]
        assert len(b["sequence"]) == b["length"], b["id"]
        assert isinstance(b["best_known_energy"], int) and b["best_known_energy"] < 0, b["id"]
        assert b["energy_kind"] in ENERGY_KINDS, b["id"]


def test_sequence_matches_printed_notation(data):
    for b in data["instances"]:
        assert expand(b["notation"]) == b["sequence"], b["id"]


def test_every_instance_cites_defined_sources(data):
    for b in data["instances"]:
        assert b["sources"], b["id"]
        for key in b["sources"]:
            assert key in data["sources"], f"{b['id']} cites undefined source {key}"


def test_sources_have_citation_and_url(data):
    for key, src in data["sources"].items():
        assert src.get("citation") and src.get("url", "").startswith("https://"), key


def test_best_known_energy_is_physically_possible(data):
    for b in data["instances"]:
        bound = contact_upper_bound(b["sequence"], b["dim"])
        assert -b["best_known_energy"] <= bound, f"{b['id']}: |E*| exceeds contact bound {bound}"
