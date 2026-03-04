import pytest
from qecc import *

def test_to_and_from_stim():
    gadgets = QECCGadgets.from_json("circuits/7_1_3.json")
    c = gadgets.ft_z_state_prep.to_stim(_layer_cnots=False)
    gad = StatePreparationCircuit.from_stim(c, basis=Basis.Z)
    assert c.flattened_operations() == gad.to_stim(_layer_cnots=False).flattened_operations()