"""Destructive logical Z measurement for the [[32, 20, 4]] code.

Classical syndrome correction and logical decoding live in sol_experiment.decoding.
"""

from guppylang import guppy
from guppylang.std.builtins import array, owned
from guppylang.std.quantum import measure_array, qubit

from sol_experiment.decoding import LogicalZResult, decode_z_readout


@guppy
def logical_z_meas(data: array[qubit, 32] @ owned) -> LogicalZResult:
    """Consume a block with transversal Z measurements and decode its readout.

    No additional qubits or entangling gates are needed. All measurements are
    requested before their results are read. This function reports acceptance;
    it does not abort or retry a destructively measured block.
    """
    measurements = measure_array(data)
    readout = array(m.read() for m in measurements)
    return decode_z_readout(readout)
