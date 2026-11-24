.PHONY: test cov lint sec audit check

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
