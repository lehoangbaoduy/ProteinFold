"""hpfold_tools.web_runtime: self-hosted Pyodide + hpfold bundle for the page (PFOLD-DOM-010, T5.2).

Self-hosting makes the page work offline and pins exactly what runs in the browser. The
numpy wheel is the one file fetched from the network, so its sha256 is checked against
Pyodide's own lock file before anything is written.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from hpfold_tools import web_runtime as wr

ROOT = Path(__file__).resolve().parents[1]
WHEEL_BYTES = b"fake numpy wheel"
WHEEL_NAME = "numpy-2.2.5-cp313-cp313-pyodide_2025_0_wasm32.whl"


def fake_pyodide(tmp_path: Path, version: str = wr.PYODIDE_VERSION, sha: str | None = None) -> Path:
    d = tmp_path / "node_modules" / "pyodide"
    d.mkdir(parents=True)
    (d / "package.json").write_text(json.dumps({"name": "pyodide", "version": version}))
    for name in wr.PYODIDE_FILES:
        if name != "pyodide-lock.json":
            (d / name).write_bytes(b"x" + name.encode())
    lock = {"info": {"version": version}, "packages": {"numpy": {
        "name": "numpy", "file_name": WHEEL_NAME, "version": "2.2.5",
        "sha256": sha or hashlib.sha256(WHEEL_BYTES).hexdigest()}}}
    (d / "pyodide-lock.json").write_text(json.dumps(lock))
    return d


def no_network(url: str) -> bytes:  # pragma: no cover - fails the test if called
    raise AssertionError(f"unexpected download {url}")


def test_zip_is_deterministic_and_contains_only_core_sources(tmp_path):
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    wr.build_hpfold_zip(ROOT / "src", a)
    wr.build_hpfold_zip(ROOT / "src", b)
    assert a.read_bytes() == b.read_bytes()
    with zipfile.ZipFile(a) as z:
        names = z.namelist()
        infos = z.infolist()
    assert names == sorted(names)
    assert all(n.startswith("hpfold/") and n.endswith(".py") for n in names)
    assert not any("__pycache__" in n for n in names)
    assert "hpfold/webjob.py" in names
    assert "hpfold/algorithms/ga.py" in names
    assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in infos)


def test_zip_ignores_non_python_files(tmp_path):
    src = tmp_path / "src"
    (src / "hpfold").mkdir(parents=True)
    (src / "hpfold" / "__init__.py").write_text("")
    (src / "hpfold" / "notes.txt").write_text("x")
    (src / "hpfold" / "__pycache__").mkdir()
    (src / "hpfold" / "__pycache__" / "m.py").write_text("")
    out = tmp_path / "o.zip"
    wr.build_hpfold_zip(src, out)
    with zipfile.ZipFile(out) as z:
        assert z.namelist() == ["hpfold/__init__.py"]


def test_version_mismatch_is_refused(tmp_path):
    d = fake_pyodide(tmp_path, version="0.27.0")
    with pytest.raises(RuntimeError, match="0.28.3"):
        wr.check_pyodide(d)


def test_missing_distribution_file_is_an_error(tmp_path):
    d = fake_pyodide(tmp_path)
    (d / "pyodide.asm.wasm").unlink()
    with pytest.raises(FileNotFoundError):
        wr.copy_pyodide(d, tmp_path / "out")


def test_wheel_url_is_the_pinned_official_cdn():
    assert wr.wheel_url(WHEEL_NAME) == f"https://cdn.jsdelivr.net/pyodide/v0.28.3/full/{WHEEL_NAME}"


@pytest.mark.parametrize("name", ["../evil.whl", "a/b.whl", "x.exe", ""])
def test_wheel_names_are_checked(name):
    with pytest.raises(ValueError):
        wr.wheel_url(name)


def test_fetched_wheel_is_verified_and_written(tmp_path):
    entry = {"file_name": WHEEL_NAME, "sha256": hashlib.sha256(WHEEL_BYTES).hexdigest()}
    urls = []
    path = wr.fetch_wheel(entry, tmp_path, fetch=lambda u: urls.append(u) or WHEEL_BYTES)
    assert path.read_bytes() == WHEEL_BYTES
    assert urls == [wr.wheel_url(WHEEL_NAME)]


def test_hash_mismatch_writes_nothing(tmp_path):
    entry = {"file_name": WHEEL_NAME, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="sha256"):
        wr.fetch_wheel(entry, tmp_path, fetch=lambda _u: WHEEL_BYTES)
    assert not (tmp_path / WHEEL_NAME).exists()


def test_local_wheel_dir_avoids_the_network(tmp_path):
    local = tmp_path / "wheels"
    local.mkdir()
    (local / WHEEL_NAME).write_bytes(WHEEL_BYTES)
    entry = {"file_name": WHEEL_NAME, "sha256": hashlib.sha256(WHEEL_BYTES).hexdigest()}
    path = wr.fetch_wheel(entry, tmp_path / "out", fetch=no_network, wheel_dir=local)
    assert path.read_bytes() == WHEEL_BYTES


def test_already_present_verified_wheel_is_not_downloaded_again(tmp_path):
    (tmp_path / WHEEL_NAME).write_bytes(WHEEL_BYTES)
    entry = {"file_name": WHEEL_NAME, "sha256": hashlib.sha256(WHEEL_BYTES).hexdigest()}
    wr.fetch_wheel(entry, tmp_path, fetch=no_network)


def test_build_writes_runtime_bundle_and_manifest(tmp_path):
    d = fake_pyodide(tmp_path)
    out = tmp_path / "public"
    manifest = wr.build(pyodide_dir=d, src_dir=ROOT / "src", out_dir=out, fetch=lambda _u: WHEEL_BYTES,
                        pinned_sha256=hashlib.sha256(WHEEL_BYTES).hexdigest())
    for name in wr.PYODIDE_FILES:
        assert (out / "pyodide" / name).is_file()
    assert (out / "pyodide" / WHEEL_NAME).read_bytes() == WHEEL_BYTES
    zip_bytes = (out / "py" / "hpfold.zip").read_bytes()
    on_disk = json.loads((out / "py" / "manifest.json").read_text())
    assert on_disk == manifest
    assert manifest == {"pyodide": "0.28.3", "numpy": "2.2.5", "numpy_wheel": WHEEL_NAME,
                        "hpfold_sha256": hashlib.sha256(zip_bytes).hexdigest()}


def test_build_with_bad_wheel_hash_fails(tmp_path):
    d = fake_pyodide(tmp_path, sha="f" * 64)
    with pytest.raises(ValueError, match="sha256"):
        wr.build(pyodide_dir=d, src_dir=ROOT / "src", out_dir=tmp_path / "public",
                 fetch=lambda _u: WHEEL_BYTES, pinned_sha256="f" * 64)


# ---------------------------------------------------------------- review run 34 ----

def test_numpy_hash_is_pinned_independently_of_the_lock(tmp_path):
    """The lock file comes from npm; a second, hard-coded hash guards against a tampered lock."""
    d = fake_pyodide(tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        wr.build(pyodide_dir=d, src_dir=ROOT / "src", out_dir=tmp_path / "public",
                 fetch=lambda _u: WHEEL_BYTES, pinned_sha256=hashlib.sha256(b"other").hexdigest())


def test_real_pin_matches_the_pyodide_lock():
    lock_path = ROOT / "web" / "node_modules" / "pyodide" / "pyodide-lock.json"
    if not lock_path.exists():
        pytest.skip("npm --prefix web ci has not been run")
    numpy = json.loads(lock_path.read_text())["packages"]["numpy"]
    assert numpy["sha256"] == wr.NUMPY_SHA256


def test_rebuild_removes_stale_files(tmp_path):
    d = fake_pyodide(tmp_path)
    out = tmp_path / "public"
    (out / "pyodide").mkdir(parents=True)
    (out / "pyodide" / "numpy-1.0-old.whl").write_bytes(b"old")
    (out / "py").mkdir()
    (out / "py" / "old.zip").write_bytes(b"old")
    (out / "data").mkdir()
    (out / "data" / "keep.json").write_text("{}")
    wr.build(pyodide_dir=d, src_dir=ROOT / "src", out_dir=out, fetch=lambda _u: WHEEL_BYTES,
             pinned_sha256=hashlib.sha256(WHEEL_BYTES).hexdigest())
    assert not (out / "pyodide" / "numpy-1.0-old.whl").exists()
    assert not (out / "py" / "old.zip").exists()
    assert (out / "data" / "keep.json").exists()  # other public files are untouched


def test_failed_build_leaves_the_previous_runtime_in_place(tmp_path):
    d = fake_pyodide(tmp_path, sha="f" * 64)
    out = tmp_path / "public"
    (out / "pyodide").mkdir(parents=True)
    (out / "pyodide" / "pyodide.mjs").write_text("previous")
    with pytest.raises(ValueError):
        wr.build(pyodide_dir=d, src_dir=ROOT / "src", out_dir=out, fetch=lambda _u: WHEEL_BYTES,
                 pinned_sha256="f" * 64)
    assert (out / "pyodide" / "pyodide.mjs").read_text() == "previous"


def test_zip_entries_are_platform_independent(tmp_path):
    out = tmp_path / "a.zip"
    wr.build_hpfold_zip(ROOT / "src", out)
    with zipfile.ZipFile(out) as z:
        for info in z.infolist():
            assert info.create_system == 3
            assert info.compress_type == zipfile.ZIP_STORED  # no dependence on the zlib build


def test_download_size_is_capped(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            assert n == wr.MAX_DOWNLOAD_BYTES + 1  # never an unbounded read
            return b"x" * n

    monkeypatch.setattr(wr.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    with pytest.raises(ValueError, match="too large"):
        wr._download(wr.wheel_url(WHEEL_NAME))


def test_download_refuses_other_hosts():
    with pytest.raises(ValueError):
        wr._download("https://example.com/" + WHEEL_NAME)
