"""Search algorithms for HP folding, all behind the same run() contract (see base.py)."""

from hpfold.algorithms import ga
from hpfold.algorithms.base import Evaluator, Frame, RunContext, RunResult

ALGORITHMS = {"GA": ga}

__all__ = ["ALGORITHMS", "Evaluator", "Frame", "RunContext", "RunResult", "ga"]
