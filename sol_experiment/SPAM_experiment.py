from guppylang import guppy
from guppylang.std.builtins import array, output
from guppylang.std.quantum import qubit

from sol_experiment.logical_measurement import logical_z_meas
from sol_experiment.state_preparation import state_prep_32_20_4_zero_d3


@guppy
def estimate_SPAM_LER() -> None:
    """Executable single-attempt example; a controller may instead call RUS."""
    data = array(qubit() for _ in range(32))
    flags = state_prep_32_20_4_zero_d3(data)
    output("flags", flags)
    logical_measurement = logical_z_meas(data)
    output("prep_accepted", flags == 0)
    output("readout_accepted", logical_measurement.accepted)
    output("accepted", flags == 0 and logical_measurement.accepted)
    output("syndrome", logical_measurement.syndrome)
    output("correction", logical_measurement.correction)
    output("data", logical_measurement.logicals)


if __name__ == "__main__":
    from guppylang import OptimizationLevel
    from local_sol_example.sol.sim import run_hugr, shot_viewer

    print('You can also run with the hugr pipeline like this:')
    hugr = estimate_SMAP_LER.with_opt_level(OptimizationLevel.Minimal).compile()
    result = run_hugr(hugr, n_shots=100, n_qubits=34, stabilizer=True)
    print(result)

    print('Shot viewer should now be open in your browser like this:')
    shot_viewer(hugr)
