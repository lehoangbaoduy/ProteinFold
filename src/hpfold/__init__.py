"""hpfold: HP lattice protein folding core (pure Python + numpy; runs in Pyodide)."""

from hpfold.fitness import DEFAULT_PENALTY, Evaluation, evaluate
from hpfold.folds import random_valid_fold, repair
from hpfold.hp_model import (
    MOVES_2D,
    MOVES_3D,
    alphabet_for,
    collisions,
    contacts,
    decode,
    encode,
    energy,
    fold_coords,
    is_valid,
    parse_sequence,
)

__all__ = [
    "DEFAULT_PENALTY",
    "MOVES_2D",
    "MOVES_3D",
    "Evaluation",
    "alphabet_for",
    "collisions",
    "contacts",
    "decode",
    "encode",
    "energy",
    "evaluate",
    "fold_coords",
    "is_valid",
    "parse_sequence",
    "random_valid_fold",
    "repair",
]
