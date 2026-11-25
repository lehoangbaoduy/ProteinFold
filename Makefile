.PHONY: test cov lint sec audit check figures tables web-data web-runtime

# Core package (src/hpfold) must stay >= 90% covered (PRD §6); the suite is the gate.
COV_MIN := 90

test:
	uv run pytest

cov:
	uv run pytest -q --cov=src/hpfold --cov-report=term-missing --cov-fail-under=$(COV_MIN)

# Style / correctness lint (ruff) — separate from the security scan.
lint:
	uv run ruff check src tests

# SAST: Bandit security scan of the source packages.
sec:
	uv run bandit -q -r src

# SCA: audit the locked dependency set (needs network access).
audit:
	uv export --no-emit-project --quiet | uv run pip-audit -r /dev/stdin

check: test lint sec audit

# Paper figures for every experiment CSV in results/ (PRD FR-A3..A5).
figures:
	for f in results/*.csv; do [ -e "$$f" ] || continue; uv run python -m hpfold_tools.figures "$$f" || exit 1; done

# LaTeX tables for every experiment CSV in results/ (PRD FR-A7).
tables:
	for f in results/*.csv; do [ -e "$$f" ] || continue; uv run python -m hpfold_tools.tables "$$f" || exit 1; done

# Static JSON for the web page (PRD FR-E6, FR-T1/T2): defaults, benchmarks, summaries, replay traces.
web-data:
	uv run python -m hpfold_tools.export_web --config configs/main_2d.toml --config configs/ext_3d.toml

# Self-hosted Pyodide + numpy wheel (sha256-checked) + hpfold.zip into web/public/ (T5.2).
# WHEEL_DIR=<dir> uses a local numpy wheel instead of downloading it.
web-runtime:
	uv run python -m hpfold_tools.web_runtime $(if $(WHEEL_DIR),--wheel-dir $(WHEEL_DIR))
