"""Search algorithms for HP folding, all behind the same run() contract (see base.py)."""

from hpfold.algorithms import aco, ga
from hpfold.algorithms.base import Evaluator, Frame, RunContext, RunResult

ALGORITHMS = {"GA": ga, "ACO": aco}

__all__ = ["ALGORITHMS", "Evaluator", "Frame", "RunContext", "RunResult", "aco", "ga"]
