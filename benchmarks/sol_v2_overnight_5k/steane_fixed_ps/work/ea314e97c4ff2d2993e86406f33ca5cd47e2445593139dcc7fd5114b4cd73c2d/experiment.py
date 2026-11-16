from guppylang.std.platform import result
from guppylang import guppy
from guppylang.std.builtins import array, comptime, exit, owned
from guppylang.std import quantum as _quantum
from guppylang.std.quantum import qubit, h, cx, cz, x, y, z, measure

N_DATA = 32


@guppy.struct
class ExtractionResult:
    flags: int
    syndrome: int


@guppy.struct
class Correction:
    accepted: bool
    x_mask: int
    z_mask: int


@guppy.struct
class RecoveryRoute:
    distance: int
    cleanup_distance: int
    decoder_id: int


if hasattr(_quantum, "Measurement"):
    @guppy
    def measure_bit(q: qubit @ owned) -> bool:
        return measure(q).read()
else:
    @guppy
    def measure_bit(q: qubit @ owned) -> bool:
        return measure(q)


@guppy
def append_bit(mask: int, bit: bool, index: int) -> int:
    if bit:
        return mask | (1 << index)
    return mask


# These Python helpers are used only inside @guppy.comptime circuits.
# The register maps physical indices to live qubits; measurement removes an
# ancilla, so a later preparation starts a fresh lifetime at the same index.
def prepare(register, targets):
    for q in targets:
        register[q] = qubit()


def apply(gate, register, targets):
    for target in targets:
        if isinstance(target, tuple):
            gate(*(register[q] for q in target))
        else:
            gate(register[target])


def read(register, measurements, targets):
    measurements.extend(measure_bit(register.pop(q)) for q in targets)


def pack_parities(measurements, groups):
    mask = 0
    for index, group in enumerate(groups):
        parity = False
        for record in group:
            parity ^= measurements[record]
        mask = append_bit(mask, parity, index)
    return mask




@guppy.struct
class CycleResult:
    accepted: bool
    frame_x: int
    frame_z: int

@guppy.struct
class Prepared:
    ancilla: array[qubit, comptime(N_DATA)]
    level: int

@guppy.comptime
def prep_z_d1(ancilla: array[qubit, comptime(N_DATA)]) -> int:
    register = dict(enumerate(ancilla))
    measurements = []
    apply(h, register, (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31))
    apply(cx, register, ((3, 2), (5, 4), (9, 8), (17, 16), (2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9), (18, 16), (19, 17), (4, 0), (5, 1), (6, 2), (7, 3), (12, 8), (13, 9), (14, 10), (15, 11), (20, 16), (21, 17), (22, 18), (23, 19), (8, 0), (9, 1), (10, 2), (11, 3), (12, 4), (13, 5), (14, 6), (15, 7), (24, 16), (25, 17), (26, 18), (27, 19), (28, 20), (29, 21), (30, 22), (31, 23), (16, 0), (17, 1), (18, 2), (19, 3), (20, 4), (21, 5), (22, 6), (23, 7), (24, 8), (25, 9), (26, 10), (27, 11), (28, 12), (29, 13), (30, 14), (31, 15)))
    flags = pack_parities(measurements, ())
    result("_sw_stage_0", flags)
    return flags

@guppy.comptime
def prep_z_d3(ancilla: array[qubit, comptime(N_DATA)]) -> int:
    register = dict(enumerate(ancilla))
    measurements = []
    apply(h, register, (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31))
    apply(cx, register, ((3, 2), (5, 4), (9, 8), (17, 16), (2, 0), (3, 1), (6, 4), (7, 5), (10, 8), (11, 9), (18, 16), (19, 17), (4, 0), (5, 1), (6, 2), (7, 3), (12, 8), (13, 9), (14, 10), (15, 11), (20, 16), (21, 17), (22, 18), (23, 19), (8, 0), (9, 1), (10, 2), (11, 3), (12, 4), (13, 5), (14, 6), (15, 7), (24, 16), (25, 17), (26, 18), (27, 19), (28, 20), (29, 21), (30, 22), (31, 23), (16, 0), (17, 1), (18, 2), (19, 3), (20, 4), (21, 5), (22, 6), (23, 7), (24, 8), (25, 9), (26, 10), (27, 11), (28, 12), (29, 13), (30, 14), (31, 15)))
    prepare(register, (32,))
    apply(h, register, (32,))
    apply(cx, register, ((32, 28), (32, 18), (32, 31), (32, 17)))
    apply(h, register, (32,))
    read(register, measurements, (32,))
    prepare(register, (32, 33))
    apply(h, register, (33,))
    apply(cx, register, ((33, 32), (28, 32), (25, 33), (1, 32), (7, 33), (8, 32), (14, 33), (16, 32), (22, 33), (26, 32), (31, 33), (13, 32), (2, 33), (4, 32), (11, 33), (21, 32), (19, 33), (33, 32)))
    read(register, measurements, (32,))
    apply(h, register, (33,))
    read(register, measurements, (33,))
    prepare(register, (32, 33))
    apply(h, register, (33, 32))
    apply(cz, register, ((33, 32),))
    apply(h, register, (33, 32))
    apply(cx, register, ((32, 4), (32, 5), (32, 22), (32, 23), (33, 24), (33, 27), (33, 29), (33, 30)))
    apply(h, register, (33, 32))
    apply(cz, register, ((33, 32),))
    apply(h, register, (33, 32))
    read(register, measurements, (32, 33))
    prepare(register, (32, 33))
    apply(h, register, (32, 33))
    apply(cz, register, ((32, 33),))
    apply(h, register, (32, 33))
    apply(cx, register, ((33, 16), (33, 25), (33, 26), (33, 19), (32, 6), (32, 15), (32, 30), (32, 17), (32, 21), (32, 19)))
    apply(h, register, (32, 33))
    apply(cz, register, ((32, 33),))
    apply(h, register, (32, 33))
    read(register, measurements, (32, 33))
    prepare(register, (32, 33))
    apply(h, register, (33, 32))
    apply(cz, register, ((33, 32),))
    apply(h, register, (33, 32))
    apply(cx, register, ((32, 15), (32, 27), (32, 28), (32, 21), (32, 4), (32, 25), (33, 24), (33, 31), (33, 23), (33, 16)))
    apply(h, register, (33, 32))
    apply(cz, register, ((33, 32),))
    apply(h, register, (33, 32))
    read(register, measurements, (32, 33))
    flags = pack_parities(measurements, ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)))
    result("_sw_stage_1", flags)
    return flags

@guppy
def prepare_z(level: int) -> Prepared:
    while True:
        ancilla = array(qubit() for _ in range(comptime(N_DATA)))
        flags = prep_z_d3(ancilla) if level == 3 else prep_z_d1(ancilla)
        if flags == 0:
            return Prepared(ancilla, level)
        for q in ancilla:
            measure_bit(q)

@guppy.comptime
def syndrome_z(measurements: array[bool, comptime(N_DATA)]) -> int:
    return pack_parities(measurements, ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23), (0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27), (0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29), (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30)))

@guppy
def read_z(data: array[qubit, comptime(N_DATA)], ancilla: array[qubit, comptime(N_DATA)] @ owned) -> int:
    for i in range(comptime(N_DATA)):
        cx(data[i], ancilla[i])
    measurements = array(measure_bit(q) for q in ancilla)
    result("_sw_stage_2", 0)
    return syndrome_z(measurements)

CHECKS_Z = [4294967295, 65535, 16711935, 252645135, 858993459, 1431655765]
DECODER_Z = [0, 2147483648, 65537, 32768, 257, 8388608, 16777217, 128, 17, 134217728, 1048577, 2048, 4097, 524288, 268435457, 8, 5, 536870912, 262145, 8192, 1025, 2097152, 67108865, 32, 65, 33554432, 4194305, 512, 16385, 131072, 1073741825, 2, 3, 1073741824, 131073, 16384, 513, 4194304, 33554433, 64, 33, 67108864, 2097153, 1024, 8193, 262144, 536870913, 4, 9, 268435456, 524289, 4096, 2049, 1048576, 134217729, 16, 129, 16777216, 8388609, 256, 32769, 65536, 2147483649, 1]

@guppy
def frame_syndrome_z(frame: int) -> int:
    checks = comptime(CHECKS_Z)
    syndrome = 0
    for i in range(comptime(len(CHECKS_Z))):
        parity = False
        for q in range(comptime(N_DATA)):
            if (checks[i] & frame & (1 << q)) != 0:
                parity = not parity
        syndrome = append_bit(syndrome, parity, i)
    return syndrome

UNIQUE_Z = [True, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True]

@guppy.comptime
def prep_x_d1(ancilla: array[qubit, comptime(N_DATA)]) -> int:
    register = dict(enumerate(ancilla))
    measurements = []
    apply(h, register, (0, 1, 2, 4, 8, 16))
    apply(cx, register, ((2, 3), (4, 5), (8, 9), (16, 17), (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (16, 18), (17, 19), (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15), (16, 20), (17, 21), (18, 22), (19, 23), (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (16, 24), (17, 25), (18, 26), (19, 27), (20, 28), (21, 29), (22, 30), (23, 31), (0, 16), (1, 17), (2, 18), (3, 19), (4, 20), (5, 21), (6, 22), (7, 23), (8, 24), (9, 25), (10, 26), (11, 27), (12, 28), (13, 29), (14, 30), (15, 31)))
    flags = pack_parities(measurements, ())
    result("_sw_stage_3", flags)
    return flags

@guppy.comptime
def prep_x_d3(ancilla: array[qubit, comptime(N_DATA)]) -> int:
    register = dict(enumerate(ancilla))
    measurements = []
    apply(h, register, (0, 1, 2, 4, 8, 16))
    apply(cx, register, ((2, 3), (4, 5), (8, 9), (16, 17), (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (16, 18), (17, 19), (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15), (16, 20), (17, 21), (18, 22), (19, 23), (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (16, 24), (17, 25), (18, 26), (19, 27), (20, 28), (21, 29), (22, 30), (23, 31), (0, 16), (1, 17), (2, 18), (3, 19), (4, 20), (5, 21), (6, 22), (7, 23), (8, 24), (9, 25), (10, 26), (11, 27), (12, 28), (13, 29), (14, 30), (15, 31)))
    prepare(register, (32,))
    apply(cx, register, ((28, 32), (18, 32), (31, 32), (17, 32)))
    read(register, measurements, (32,))
    prepare(register, (32,))
    apply(h, register, (32,))
    prepare(register, (33,))
    apply(cx, register, ((32, 33), (32, 28), (33, 25), (32, 1), (33, 7), (32, 8), (33, 14), (32, 16), (33, 22), (32, 26), (33, 31), (32, 13), (33, 2), (32, 4), (33, 11), (32, 21), (33, 19), (32, 33)))
    apply(h, register, (32,))
    read(register, measurements, (32, 33))
    prepare(register, (32, 33))
    apply(h, register, (32, 33))
    apply(cz, register, ((33, 32),))
    apply(cx, register, ((4, 32), (5, 32), (22, 32), (23, 32), (24, 33), (27, 33), (29, 33), (30, 33)))
    apply(cz, register, ((33, 32),))
    apply(h, register, (32, 33))
    read(register, measurements, (32, 33))
    prepare(register, (32, 33))
    apply(h, register, (32, 33))
    apply(cz, register, ((32, 33),))
    apply(cx, register, ((16, 33), (25, 33), (26, 33), (19, 33), (6, 32), (15, 32), (30, 32), (17, 32), (21, 32), (19, 32)))
    apply(cz, register, ((32, 33),))
    apply(h, register, (32, 33))
    read(register, measurements, (32, 33))
    prepare(register, (32, 33))
    apply(h, register, (32, 33))
    apply(cz, register, ((33, 32),))
    apply(cx, register, ((15, 32), (27, 32), (28, 32), (21, 32), (4, 32), (25, 32), (24, 33), (31, 33), (23, 33), (16, 33)))
    apply(cz, register, ((33, 32),))
    apply(h, register, (32, 33))
    read(register, measurements, (32, 33))
    flags = pack_parities(measurements, ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,)))
    result("_sw_stage_4", flags)
    return flags

@guppy
def prepare_x(level: int) -> Prepared:
    while True:
        ancilla = array(qubit() for _ in range(comptime(N_DATA)))
        flags = prep_x_d3(ancilla) if level == 3 else prep_x_d1(ancilla)
        if flags == 0:
            return Prepared(ancilla, level)
        for q in ancilla:
            measure_bit(q)

@guppy.comptime
def syndrome_x(measurements: array[bool, comptime(N_DATA)]) -> int:
    return pack_parities(measurements, ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23), (0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27), (0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29), (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30)))

@guppy
def read_x(data: array[qubit, comptime(N_DATA)], ancilla: array[qubit, comptime(N_DATA)] @ owned) -> int:
    for i in range(comptime(N_DATA)):
        cx(ancilla[i], data[i])
    for i in range(comptime(N_DATA)):
        h(ancilla[i])
    measurements = array(measure_bit(q) for q in ancilla)
    result("_sw_stage_5", 0)
    return syndrome_x(measurements)

CHECKS_X = [4294967295, 65535, 16711935, 252645135, 858993459, 1431655765]
DECODER_X = [0, 2147483648, 65537, 32768, 257, 8388608, 16777217, 128, 17, 134217728, 1048577, 2048, 4097, 524288, 268435457, 8, 5, 536870912, 262145, 8192, 1025, 2097152, 67108865, 32, 65, 33554432, 4194305, 512, 16385, 131072, 1073741825, 2, 3, 1073741824, 131073, 16384, 513, 4194304, 33554433, 64, 33, 67108864, 2097153, 1024, 8193, 262144, 536870913, 4, 9, 268435456, 524289, 4096, 2049, 1048576, 134217729, 16, 129, 16777216, 8388609, 256, 32769, 65536, 2147483649, 1]

@guppy
def frame_syndrome_x(frame: int) -> int:
    checks = comptime(CHECKS_X)
    syndrome = 0
    for i in range(comptime(len(CHECKS_X))):
        parity = False
        for q in range(comptime(N_DATA)):
            if (checks[i] & frame & (1 << q)) != 0:
                parity = not parity
        syndrome = append_bit(syndrome, parity, i)
    return syndrome

UNIQUE_X = [True, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True]

@guppy
def se_cycle_with_frame(data: array[qubit, comptime(N_DATA)], frame_x: int, frame_z: int) -> CycleResult:
    # One data-syndrome read per basis; only ancilla preparation retries.
    prepared = prepare_z(3)
    syndrome = read_z(data, prepared.ancilla)
    adjusted = syndrome ^ frame_syndrome_z(frame_x)
    unique = comptime(UNIQUE_Z)
    if not unique[adjusted]:
        return CycleResult(False, frame_x, frame_z)
    decoder = comptime(DECODER_Z)
    frame_x ^= decoder[syndrome ^ frame_syndrome_z(frame_x)]
    prepared = prepare_x(3)
    syndrome = read_x(data, prepared.ancilla)
    adjusted = syndrome ^ frame_syndrome_x(frame_z)
    unique = comptime(UNIQUE_X)
    if not unique[adjusted]:
        return CycleResult(False, frame_x, frame_z)
    decoder = comptime(DECODER_X)
    frame_z ^= decoder[syndrome ^ frame_syndrome_x(frame_z)]
    return CycleResult(True, frame_x, frame_z)

@guppy
def se_cycle(data: array[qubit, comptime(N_DATA)]) -> CycleResult:
    return se_cycle_with_frame(data, 0, 0)

def compile_se_cycle():
    target = se_cycle
    minimal = getattr(target, "with_minimal_opt", None)
    if minimal is not None:
        target = minimal()
    return target.compile_function()

EXPERIMENT_CYCLES = 10
EXPERIMENT_PLUS = True

from guppylang.std.platform import result
from qtm_platform.ops import sleep
from guppylang.std.quantum import s

@guppy.comptime
def prepare_experiment(data: array[qubit, comptime(N_DATA)]) -> None:
    apply(h, data, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28))
    apply(cx, data, ((0, 15), (0, 23), (0, 27), (0, 29), (0, 30), (1, 15), (1, 23), (1, 27), (1, 29), (1, 31), (2, 15), (2, 23), (2, 27), (2, 30), (2, 31), (3, 15), (3, 23), (3, 27), (4, 15), (4, 23), (4, 29), (4, 30), (4, 31), (5, 15), (5, 23), (5, 29), (6, 15), (6, 23), (6, 30), (7, 15), (7, 23), (7, 31), (8, 15), (8, 27), (8, 29), (8, 30), (8, 31), (9, 15), (9, 27), (9, 29), (10, 15), (10, 27), (10, 30), (11, 15), (11, 27), (11, 31), (12, 15), (12, 29), (12, 30), (13, 15), (13, 29), (13, 31), (14, 15), (14, 30), (14, 31), (16, 23), (16, 27), (16, 29), (16, 30), (16, 31), (17, 23), (17, 27), (17, 29), (18, 23), (18, 27), (18, 30), (19, 23), (19, 27), (19, 31), (20, 23), (20, 29), (20, 30), (21, 23), (21, 29), (21, 31), (22, 23), (22, 30), (22, 31), (24, 27), (24, 29), (24, 30), (25, 27), (25, 29), (25, 31), (26, 27), (26, 30), (26, 31), (28, 29), (28, 30), (28, 31)))

@guppy
def main() -> None:
    data = array(qubit() for _ in range(comptime(N_DATA)))
    prepare_experiment(data)
    synchronize_input(data)
    sleep(data, 1152921504606846977)
    accepted = True
    frame_x = 0
    frame_z = 0
    completed = 0
    for _ in range(comptime(EXPERIMENT_CYCLES)):
        if accepted:
            result("_sw_cycle", 1)
            outcome = se_cycle_with_frame(data, frame_x, frame_z)
            accepted = outcome.accepted
            frame_x = outcome.frame_x
            frame_z = outcome.frame_z
            if accepted:
                completed += 1
    sleep(data, 1152921504606846978)
    result("accepted", accepted)
    result("frame_x", frame_x)
    result("frame_z", frame_z)
    result("completed", completed)
    if comptime(EXPERIMENT_PLUS):
        for i in range(comptime(N_DATA)):
            h(data[i])
    bits = array(measure_bit(q) for q in data)
    result("measurements", bits)

@guppy
def check_input(observed: bool, expected: bool) -> None:
    if observed != expected:
        exit("Ideal Sol input synchronization failed")

@guppy.comptime
def synchronize_input(data: array[qubit, comptime(N_DATA)]) -> None:
    probe = qubit()
    h(probe)
    cx(probe, data[0])
    cx(probe, data[1])
    cx(probe, data[2])
    cx(probe, data[3])
    cx(probe, data[4])
    cx(probe, data[5])
    cx(probe, data[6])
    cx(probe, data[7])
    cx(probe, data[8])
    cx(probe, data[9])
    cx(probe, data[10])
    cx(probe, data[11])
    cx(probe, data[12])
    cx(probe, data[13])
    cx(probe, data[14])
    cx(probe, data[15])
    cx(probe, data[16])
    cx(probe, data[17])
    cx(probe, data[18])
    cx(probe, data[19])
    cx(probe, data[20])
    cx(probe, data[21])
    cx(probe, data[22])
    cx(probe, data[23])
    cx(probe, data[24])
    cx(probe, data[25])
    cx(probe, data[26])
    cx(probe, data[27])
    cx(probe, data[28])
    cx(probe, data[29])
    cx(probe, data[30])
    cx(probe, data[31])
    h(probe)
    observed = measure_bit(probe)
    check_input(observed, False)
