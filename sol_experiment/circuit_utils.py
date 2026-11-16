"""Compile-time emission of the project's Stim-style operation lists.

Call emit_circuit only inside @guppy.comptime functions. It emits quantum
instructions and returns lazy Measurement handles in circuit order. Reading
those handles, interpreting detectors, and decoding flags/syndromes belong to
the caller. There is no operation-list interpreter at quantum runtime.
"""

from guppylang.std.quantum import cx, cz, h, measure, qubit, reset, x, y, z


def emit_circuit(data, operations, *, fresh_data: bool = False):
    """Emit operations on a borrowed data block and locally managed ancillas.

    Indices 0..len(data)-1 refer to the caller's data. Higher indices are local
    ancillas, allocated on their first R/RX and consumed by M/MX. A subsequent
    R/RX at that index starts a new lifetime. Resetting a still-live qubit uses
    reset rather than allocating and leaking its previous value.

    fresh_data=True promises that the caller has allocated all data in |0>.
    Only resets before a data qubit's first use are then omitted; later resets
    are always emitted. The default preserves every explicit data reset, as
    needed for circuits acting on an existing state.

    Supported operations: R, RX, H, X, Y, Z, CX, CZ, M, MX. Two-qubit target
    lists are consecutive ordered pairs. Data qubits cannot be destructively
    measured, and all local ancillas must be measured before the circuit ends.
    """
    n_data = len(data)
    register = dict(enumerate(data))
    pristine = set(range(n_data)) if fresh_data else set()
    measurements = []
    single_gates = {"H": h, "X": x, "Y": y, "Z": z}
    pair_gates = {"CX": cx, "CZ": cz}
    supported = {"R", "RX", "M", "MX", *single_gates, *pair_gates}

    for name, targets in operations:
        if name not in supported:
            raise ValueError(f"Unsupported circuit operation: {name}")
        targets = tuple(targets)
        if any(type(q) is not int or q < 0 for q in targets):
            raise ValueError(f"{name} requires nonnegative integer qubit indices")
        if name in pair_gates and len(targets) % 2:
            raise ValueError(f"{name} requires an even number of targets")

        if name in {"R", "RX"}:
            for q in targets:
                if q not in register:
                    register[q] = qubit()
                elif q not in pristine:
                    reset(register[q])
                if name == "RX":
                    h(register[q])
                pristine.discard(q)
        else:
            if any(q not in register for q in targets):
                raise ValueError(f"{name} uses a qubit before preparation or after measurement")
            if name in {"M", "MX"}:
                if any(q < n_data for q in targets):
                    raise ValueError("Cannot destructively measure borrowed data qubits")
                if len(set(targets)) != len(targets):
                    raise ValueError("Cannot destructively measure the same qubit twice")
                for q in targets:
                    if name == "MX":
                        h(register[q])
                    # Keep the handle lazy; higher-level code decides when to read it.
                    measurements.append(measure(register.pop(q)))
            elif name in pair_gates:
                for control, target in zip(targets[::2], targets[1::2]):
                    if control == target:
                        raise ValueError(f"{name} cannot use the same qubit twice")
                    pair_gates[name](register[control], register[target])
            else:
                for q in targets:
                    single_gates[name](register[q])
            pristine.difference_update(targets)

    remaining = sorted(q for q in register if q >= n_data)
    if remaining:
        raise ValueError(f"Circuit leaves unmeasured ancillas: {remaining}")
    return measurements
