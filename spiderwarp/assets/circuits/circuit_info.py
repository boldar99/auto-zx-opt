from enum import Enum


class Methods(Enum):
    FlagAtOrigin = 0
    Miscellaneous = 1
    RL = 2
    SAT = 3
    SteanePermutation = 4



def get_available_circuits(method: Methods | None):
    pass